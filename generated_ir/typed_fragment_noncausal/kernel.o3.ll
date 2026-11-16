; ModuleID = 'generated_ir/typed_fragment_noncausal/kernel.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/typed_fragment_noncausal/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %kdma.tid = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !7
  %kdma.wave = and i32 %kdma.tid, 960, !dbg !7
  %kdma.lane = lshr i32 %kdma.tid, 1, !dbg !7
  %5 = and i32 %kdma.lane, 31, !dbg !7
  %kdma.owner = or disjoint i32 %5, %kdma.wave, !dbg !7
  %kdma.owner.hi = or disjoint i32 %kdma.owner, 32, !dbg !7
  %6 = shl nuw nsw i32 %kdma.tid, 3, !dbg !7
  %kdma.intra = and i32 %6, 8, !dbg !7
  %kdma.intra.hi = xor i32 %kdma.intra, 8, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %10 = sext i32 %9 to i64, !dbg !7
  %11 = shl nsw i64 %10, 5, !dbg !7
  %12 = add nsw i64 %11, %8, !dbg !7
  %13 = lshr i64 %12, 5, !dbg !7
  %14 = and i64 %13, 576460752303423456, !dbg !7
  %15 = and i64 %8, 31, !dbg !7
  %16 = or disjoint i64 %14, %15, !dbg !7
  %17 = lshr i64 %12, 10, !dbg !8
  %18 = add nuw nsw i64 %13, %17, !dbg !8
  %19 = shl nuw nsw i64 %16, 7, !dbg !8
  %20 = shl i64 %17, 12, !dbg !8
  %21 = sub i64 %19, %20, !dbg !8
  %22 = shl i64 %18, 19, !dbg !8
  %23 = and i64 %22, 16252928, !dbg !8
  %24 = shl nsw i64 %21, 7, !dbg !8
  %25 = add i64 %24, %23, !dbg !8
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !8, !range !9, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !8
  %address.local404 = shl nuw nsw i64 %address.local, 3, !dbg !8
  %address.local405 = and i64 %address.local404, 1536, !dbg !8
  %address.local406 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local405, !dbg !8
  %address.local409 = shl nuw nsw i64 %address.local, 7, !dbg !8
  %address.local410 = and i64 %address.local409, 16256, !dbg !8
  %address.local411 = lshr i64 %address.local, 4, !dbg !8
  %address.local412 = and i64 %address.local411, 8, !dbg !8
  %address.local413 = or disjoint i64 %address.local410, %address.local412, !dbg !8
  %26 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !8
  %.idx1399 = or disjoint i64 %address.local413, %25, !dbg !8
  %.idx1399.tr = trunc i64 %.idx1399 to i32, !dbg !8
  %dma.buffer.offset1396 = shl i32 %.idx1399.tr, 1, !dbg !8
  %27 = insertelement <2 x i64> poison, i64 %26, i64 0, !dbg !8
  %28 = bitcast <2 x i64> %27 to <4 x i32>, !dbg !8
  %dma.buffer.rsrc1397 = shufflevector <4 x i32> %28, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !8
  %address.local415 = or disjoint i64 %address.local, 256, !dbg !8
  %address.local416 = shl nuw nsw i64 %address.local415, 3, !dbg !8
  %address.local417 = and i64 %address.local416, 3584, !dbg !8
  %address.local418 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local417, !dbg !8
  %address.local423 = lshr i64 %address.local415, 4, !dbg !8
  %address.local424 = and i64 %address.local423, 24, !dbg !8
  %address.local425 = or disjoint i64 %address.local424, %address.local410, !dbg !8
  %.idx1400 = or disjoint i64 %address.local425, %25, !dbg !8
  %.idx1400.tr = trunc i64 %.idx1400 to i32, !dbg !8
  %dma.buffer.offset1393 = shl i32 %.idx1400.tr, 1, !dbg !8
  %address.local427 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local405, !dbg !8
  %29 = or disjoint i64 %25, 32, !dbg !8
  %30 = or disjoint i64 %address.local413, %29, !dbg !8
  %.tr = trunc i64 %30 to i32, !dbg !8
  %dma.buffer.offset1390 = shl i32 %.tr, 1, !dbg !8
  %address.local431 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local417, !dbg !8
  %31 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local406, i64 0, !dbg !8
  %32 = insertelement <4 x ptr addrspace(3)> %31, ptr addrspace(3) %address.local418, i64 1, !dbg !8
  %33 = insertelement <4 x ptr addrspace(3)> %32, ptr addrspace(3) %address.local427, i64 2, !dbg !8
  %34 = insertelement <4 x ptr addrspace(3)> %33, ptr addrspace(3) %address.local431, i64 3, !dbg !8
  %35 = ptrtoint <4 x ptr addrspace(3)> %34 to <4 x i64>, !dbg !8
  %36 = trunc <4 x i64> %35 to <4 x i32>, !dbg !8
  %37 = extractelement <4 x i32> %36, i64 0, !dbg !8
  %38 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %37), !dbg !8
  %dma.buffer.lds1398 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %38), !dbg !8
  %39 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1396, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %40 = extractelement <4 x i32> %36, i64 1, !dbg !8
  %41 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %40), !dbg !8
  %dma.buffer.lds1395 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %41), !dbg !8
  %42 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1393, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %43 = extractelement <4 x i32> %36, i64 2, !dbg !8
  %44 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %43), !dbg !8
  %dma.buffer.lds1392 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %44), !dbg !8
  %45 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1390, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %46 = extractelement <4 x i32> %36, i64 3, !dbg !8
  %47 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %46), !dbg !8
  %48 = or disjoint i64 %address.local425, %29, !dbg !8
  %.tr1401 = trunc i64 %48 to i32, !dbg !8
  %dma.buffer.offset1387 = shl i32 %.tr1401, 1, !dbg !8
  %dma.buffer.lds1389 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %47), !dbg !8
  %49 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1387, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %address.local435 = lshr i32 %address.thread, 2, !dbg !10
  %address.local436 = and i32 %address.local435, 4, !dbg !10
  %address.local437 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local438 = and i32 %address.local437, 384, !dbg !10
  %address.local439 = or disjoint i32 %address.local436, %address.local438, !dbg !10
  %address.local440 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local441 = and i32 %address.local440, 120, !dbg !10
  %address.local442 = or disjoint i32 %address.local439, %address.local441, !dbg !10
  %address.local443 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local444 = and i32 %address.local443, 1024, !dbg !10
  %address.local445 = or disjoint i32 %address.local442, %address.local444, !dbg !10
  %address.local446 = zext nneg i32 %address.local445 to i64, !dbg !10
  %address.local447 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local446, !dbg !10
  %50 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %lds.const.gep = getelementptr i8, ptr addrspace(3) %address.local447, i64 4096, !dbg !10
  %51 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %52 = fmul <4 x half> %50, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1197 = getelementptr i8, ptr addrspace(3) %address.local447, i64 1024, !dbg !10
  %53 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %54 = fmul <4 x half> %51, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1198 = getelementptr i8, ptr addrspace(3) %address.local447, i64 5120, !dbg !10
  %55 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %56 = fmul <4 x half> %53, splat (half 1.275630e-01), !dbg !10
  %57 = fmul <4 x half> %55, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %58 = or disjoint i64 %25, 64, !dbg !8
  %59 = or disjoint i64 %address.local413, %58, !dbg !8
  %.tr1402 = trunc i64 %59 to i32, !dbg !8
  %dma.buffer.offset1384 = shl i32 %.tr1402, 1, !dbg !8
  %60 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %dma.buffer.offset1384, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %61 = or disjoint i64 %address.local425, %58, !dbg !8
  %.tr1403 = trunc i64 %61 to i32, !dbg !8
  %dma.buffer.offset1381 = shl i32 %.tr1403, 1, !dbg !8
  %62 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %dma.buffer.offset1381, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %lds.const.gep1199 = getelementptr i8, ptr addrspace(3) %address.local447, i64 8192, !dbg !10
  %63 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %lds.const.gep1200 = getelementptr i8, ptr addrspace(3) %address.local447, i64 12288, !dbg !10
  %64 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %lds.const.gep1201 = getelementptr i8, ptr addrspace(3) %address.local447, i64 9216, !dbg !10
  %65 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %66 = fmul <4 x half> %63, splat (half 1.275630e-01), !dbg !10
  %lds.const.gep1202 = getelementptr i8, ptr addrspace(3) %address.local447, i64 13312, !dbg !10
  %67 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %68 = fmul <4 x half> %64, splat (half 1.275630e-01), !dbg !10
  %69 = fmul <4 x half> %65, splat (half 1.275630e-01), !dbg !10
  %70 = fmul <4 x half> %67, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %71 = or disjoint i64 %25, 96, !dbg !8
  %72 = or disjoint i64 %address.local413, %71, !dbg !8
  %.tr1404 = trunc i64 %72 to i32, !dbg !8
  %dma.buffer.offset1378 = shl i32 %.tr1404, 1, !dbg !8
  %73 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1392, i32 %dma.buffer.offset1378, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %74 = or disjoint i64 %address.local425, %71, !dbg !8
  %.tr1405 = trunc i64 %74 to i32, !dbg !8
  %dma.buffer.offset1375 = shl i32 %.tr1405, 1, !dbg !8
  %75 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1389, i32 %dma.buffer.offset1375, <4 x i32> %dma.buffer.rsrc1397) #9, !dbg !8
  %76 = load <4 x half>, ptr addrspace(3) %address.local447, align 8, !dbg !10
  %77 = load <4 x half>, ptr addrspace(3) %lds.const.gep, align 8, !dbg !10
  %78 = load <4 x half>, ptr addrspace(3) %lds.const.gep1197, align 8, !dbg !10
  %79 = fmul <4 x half> %76, splat (half 1.275630e-01), !dbg !10
  %80 = load <4 x half>, ptr addrspace(3) %lds.const.gep1198, align 8, !dbg !10
  %81 = fmul <4 x half> %77, splat (half 1.275630e-01), !dbg !10
  %82 = fmul <4 x half> %78, splat (half 1.275630e-01), !dbg !10
  %83 = fmul <4 x half> %80, splat (half 1.275630e-01), !dbg !10
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !8
  fence syncscope("workgroup") release, !dbg !8
  tail call void @llvm.amdgcn.s.barrier(), !dbg !8
  fence syncscope("workgroup") acquire, !dbg !8
  %84 = load <4 x half>, ptr addrspace(3) %lds.const.gep1199, align 8, !dbg !10
  %85 = load <4 x half>, ptr addrspace(3) %lds.const.gep1200, align 8, !dbg !10
  %86 = load <4 x half>, ptr addrspace(3) %lds.const.gep1201, align 8, !dbg !10
  %87 = fmul <4 x half> %84, splat (half 1.275630e-01), !dbg !10
  %88 = load <4 x half>, ptr addrspace(3) %lds.const.gep1202, align 8, !dbg !10
  %89 = fmul <4 x half> %85, splat (half 1.275630e-01), !dbg !10
  %90 = fmul <4 x half> %86, splat (half 1.275630e-01), !dbg !10
  %91 = fmul <4 x half> %88, splat (half 1.275630e-01), !dbg !10
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %92 = shl nuw nsw i32 %kdma.tid, 2, !dbg !12
  %93 = and i32 %92, 3084, !dbg !12
  %94 = and i32 %kdma.tid, 1, !dbg !12
  %95 = add nuw nsw i32 %94, 1, !dbg !12
  %narrow = or disjoint i32 %95, %93, !dbg !12
  %96 = lshr i32 %narrow, 1, !dbg !12
  %97 = zext nneg i32 %96 to i64, !dbg !12
  %k0.prime.1963 = shl i64 %13, 19, !dbg !13
  %k0.prime.1964 = and i64 %k0.prime.1963, 16252928, !dbg !13
  %k0.prime.kdma.source = zext nneg i32 %kdma.owner to i64, !dbg !13
  %k0.prime.kdma.source1506 = lshr i64 %k0.prime.kdma.source, 4, !dbg !13
  %k0.prime.kdma.source1507 = and i64 %k0.prime.kdma.source1506, 8, !dbg !13
  %k0.prime.kdma.source1508 = and i64 %k0.prime.kdma.source, 91, !dbg !13
  %k0.prime.kdma.source1509 = lshr i64 %k0.prime.kdma.source, 2, !dbg !13
  %k0.prime.kdma.source1510 = and i64 %k0.prime.kdma.source1509, 4, !dbg !13
  %k0.prime.kdma.source1511 = add nuw nsw i64 %k0.prime.kdma.source1510, %k0.prime.kdma.source, !dbg !13
  %k0.prime.kdma.source1512 = and i64 %k0.prime.kdma.source1511, 4, !dbg !13
  %k0.prime.kdma.source1513 = or disjoint i64 %k0.prime.kdma.source1512, %k0.prime.kdma.source1508, !dbg !13
  %k0.prime.kdma.source1514 = shl nuw nsw i64 %k0.prime.kdma.source1513, 7, !dbg !13
  %k0.prime.kdma.source1515 = or disjoint i64 %k0.prime.kdma.source1514, %k0.prime.kdma.source1507, !dbg !13
  %k0.prime.kdma.source1516 = or disjoint i64 %k0.prime.kdma.source1515, %k0.prime.1964, !dbg !13
  %k0.prime.kdma.source1517 = trunc nuw nsw i64 %k0.prime.kdma.source1516 to i32, !dbg !13
  %k0.prime.kdma.source1518 = shl nuw nsw i32 %k0.prime.kdma.source1517, 1, !dbg !13
  %k0.prime.kdma.offset = or disjoint i32 %k0.prime.kdma.source1518, %kdma.intra, !dbg !13
  %k0.prime.1967 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !13
  %k0.prime.1968 = insertelement <2 x i64> poison, i64 %k0.prime.1967, i64 0, !dbg !13
  %k0.prime.1969 = bitcast <2 x i64> %k0.prime.1968 to <4 x i32>, !dbg !13
  %k0.prime.dma.buffer.rsrc1373 = shufflevector <4 x i32> %k0.prime.1969, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>, !dbg !13
  %k0.prime.1970 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1398, i32 %k0.prime.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1534 = add i32 %dma.buffer.lds1398, 512, !dbg !13
  %k0.prime.kdma.source1519 = zext nneg i32 %kdma.owner.hi to i64, !dbg !13
  %k0.prime.kdma.source1520 = lshr i64 %k0.prime.kdma.source1519, 4, !dbg !13
  %k0.prime.kdma.source1521 = and i64 %k0.prime.kdma.source1520, 8, !dbg !13
  %k0.prime.kdma.source1522 = and i64 %k0.prime.kdma.source1519, 123, !dbg !13
  %k0.prime.kdma.source1523 = lshr i64 %k0.prime.kdma.source1519, 2, !dbg !13
  %k0.prime.kdma.source1524 = and i64 %k0.prime.kdma.source1523, 4, !dbg !13
  %k0.prime.kdma.source1525 = add nuw nsw i64 %k0.prime.kdma.source1524, %k0.prime.kdma.source1519, !dbg !13
  %k0.prime.kdma.source1526 = and i64 %k0.prime.kdma.source1525, 4, !dbg !13
  %k0.prime.kdma.source1527 = or disjoint i64 %k0.prime.kdma.source1526, %k0.prime.kdma.source1522, !dbg !13
  %k0.prime.kdma.source1528 = shl nuw nsw i64 %k0.prime.kdma.source1527, 7, !dbg !13
  %k0.prime.kdma.source1529 = or disjoint i64 %k0.prime.kdma.source1528, %k0.prime.kdma.source1521, !dbg !13
  %k0.prime.kdma.source1530 = or disjoint i64 %k0.prime.kdma.source1529, %k0.prime.1964, !dbg !13
  %k0.prime.kdma.source1531 = trunc nuw nsw i64 %k0.prime.kdma.source1530 to i32, !dbg !13
  %k0.prime.kdma.source1532 = shl nuw nsw i32 %k0.prime.kdma.source1531, 1, !dbg !13
  %k0.prime.kdma.offset1533 = or disjoint i32 %k0.prime.kdma.source1532, %kdma.intra.hi, !dbg !13
  %k0.prime.1971 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1534, i32 %k0.prime.kdma.offset1533, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.source1546 = or disjoint i64 %k0.prime.kdma.source1515, 16, !dbg !13
  %k0.prime.kdma.source1547 = or disjoint i64 %k0.prime.kdma.source1546, %k0.prime.1964, !dbg !13
  %k0.prime.kdma.source1548 = trunc nuw nsw i64 %k0.prime.kdma.source1547 to i32, !dbg !13
  %k0.prime.kdma.source1549 = shl nuw nsw i32 %k0.prime.kdma.source1548, 1, !dbg !13
  %k0.prime.kdma.offset1550 = or disjoint i32 %k0.prime.kdma.source1549, %kdma.intra, !dbg !13
  %k0.prime.1973 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1395, i32 %k0.prime.kdma.offset1550, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.prime.kdma.lds1568 = add i32 %dma.buffer.lds1395, 512, !dbg !13
  %k0.prime.kdma.source1563 = or disjoint i64 %k0.prime.kdma.source1529, 16, !dbg !13
  %k0.prime.kdma.source1564 = or disjoint i64 %k0.prime.kdma.source1563, %k0.prime.1964, !dbg !13
  %k0.prime.kdma.source1565 = trunc nuw nsw i64 %k0.prime.kdma.source1564 to i32, !dbg !13
  %k0.prime.kdma.source1566 = shl nuw nsw i32 %k0.prime.kdma.source1565, 1, !dbg !13
  %k0.prime.kdma.offset1567 = or disjoint i32 %k0.prime.kdma.source1566, %kdma.intra.hi, !dbg !13
  %k0.prime.1974 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %k0.prime.kdma.lds1568, i32 %k0.prime.kdma.offset1567, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %98 = ptrtoint ptr addrspace(1) %2 to i64
  %99 = insertelement <2 x i64> poison, i64 %98, i64 0
  %100 = bitcast <2 x i64> %99 to <4 x i32>
  %dma.buffer.rsrc1349 = shufflevector <4 x i32> %100, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  br label %101, !dbg !11

101:                                              ; preds = %4, %k0.join
  %102 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1031, %k0.join ]
  %103 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1027, %k0.join ]
  %104 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1023, %k0.join ]
  %105 = phi [2 x <1 x float>] [ [<1 x float> splat (float -inf), <1 x float> splat (float -inf)], %4 ], [ %1019, %k0.join ]
  %106 = phi [2 x <1 x float>] [ zeroinitializer, %4 ], [ %782, %k0.join ]
  %107 = phi [2 x <8 x float>] [ zeroinitializer, %4 ], [ %1017, %k0.join ]
  %108 = phi i64 [ 0, %4 ], [ %939, %k0.join ]
  %109 = shl nuw nsw i64 %108, 14, !dbg !13
  %110 = add nuw nsw i64 %109, %k0.prime.1964, !dbg !13
  %address.thread498 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !13, !range !9, !frisk.register_lifetimes !5
  %address.local499 = zext nneg i32 %address.thread498 to i64, !dbg !8
  %address.local500 = shl nuw nsw i64 %address.local499, 3, !dbg !8
  %address.local501 = and i64 %address.local500, 1536, !dbg !8
  %address.local502 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local501, !dbg !8
  %address.local516 = or disjoint i64 %address.local499, 256, !dbg !8
  %address.local517 = shl nuw nsw i64 %address.local516, 3, !dbg !8
  %address.local518 = and i64 %address.local517, 3584, !dbg !8
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local518, !dbg !8
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %111 = or disjoint i64 %110, 32, !dbg !13
  %address.local533 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local501, !dbg !8
  %kdma.source1580 = or disjoint i64 %111, %k0.prime.kdma.source1515, !dbg !13
  %kdma.source1581 = trunc nuw nsw i64 %kdma.source1580 to i32, !dbg !13
  %kdma.source1582 = shl nuw nsw i32 %kdma.source1581, 1, !dbg !13
  %kdma.offset1583 = or disjoint i32 %kdma.source1582, %kdma.intra, !dbg !13
  %kdma.source1596 = or disjoint i64 %111, %k0.prime.kdma.source1529, !dbg !13
  %kdma.source1597 = trunc nuw nsw i64 %kdma.source1596 to i32, !dbg !13
  %kdma.source1598 = shl nuw nsw i32 %kdma.source1597, 1, !dbg !13
  %kdma.offset1599 = or disjoint i32 %kdma.source1598, %kdma.intra.hi, !dbg !13
  %address.local537 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local518, !dbg !8
  %112 = insertelement <4 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local502, i64 0, !dbg !8
  %113 = insertelement <4 x ptr addrspace(3)> %112, ptr addrspace(3) %address.local519, i64 1, !dbg !8
  %114 = insertelement <4 x ptr addrspace(3)> %113, ptr addrspace(3) %address.local533, i64 2, !dbg !8
  %115 = insertelement <4 x ptr addrspace(3)> %114, ptr addrspace(3) %address.local537, i64 3, !dbg !8
  %116 = ptrtoint <4 x ptr addrspace(3)> %115 to <4 x i64>, !dbg !8
  %117 = trunc <4 x i64> %116 to <4 x i32>, !dbg !8
  %118 = extractelement <4 x i32> %117, i64 0, !dbg !13
  %119 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !13
  %dma.buffer.lds1374 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %119), !dbg !13
  %kdma.lds1534 = add i32 %dma.buffer.lds1374, 512, !dbg !13
  %120 = extractelement <4 x i32> %117, i64 1, !dbg !13
  %121 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %120), !dbg !13
  %dma.buffer.lds1371 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %121), !dbg !13
  %kdma.lds1568 = add i32 %dma.buffer.lds1371, 512, !dbg !13
  %122 = extractelement <4 x i32> %117, i64 2, !dbg !13
  %123 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %122), !dbg !13
  %dma.buffer.lds1368 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %123), !dbg !13
  %124 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1583, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.lds1600 = add i32 %dma.buffer.lds1368, 512, !dbg !13
  %125 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1599, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %126 = extractelement <4 x i32> %117, i64 3, !dbg !13
  %127 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %126), !dbg !13
  %dma.buffer.lds1365 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %127), !dbg !13
  %kdma.source1613 = or disjoint i64 %111, %k0.prime.kdma.source1546, !dbg !13
  %kdma.source1614 = trunc nuw nsw i64 %kdma.source1613 to i32, !dbg !13
  %kdma.source1615 = shl nuw nsw i32 %kdma.source1614, 1, !dbg !13
  %kdma.offset1616 = or disjoint i32 %kdma.source1615, %kdma.intra, !dbg !13
  %128 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1616, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1630 = or disjoint i64 %111, %k0.prime.kdma.source1563, !dbg !13
  %kdma.source1631 = trunc nuw nsw i64 %kdma.source1630 to i32, !dbg !13
  %kdma.source1632 = shl nuw nsw i32 %kdma.source1631, 1, !dbg !13
  %kdma.offset1633 = or disjoint i32 %kdma.source1632, %kdma.intra.hi, !dbg !13
  %kdma.lds1634 = add i32 %dma.buffer.lds1365, 512, !dbg !13
  %129 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1633, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %address.local542 = lshr i64 %address.local499, 2, !dbg !12
  %address.local545 = shl nuw nsw i64 %address.local499, 1, !dbg !12
  %address.local546 = and i64 %address.local545, 24, !dbg !12
  %address.local548 = shl i64 %address.local499, 62, !dbg !12
  %address.local549 = ashr exact i64 %address.local548, 57, !dbg !12
  %address.local550 = and i64 %address.local549, -64, !dbg !12
  %address.local553 = and i64 %address.local542, 12, !dbg !10
  %address.local555 = and i64 %address.local499, 63, !dbg !10
  %address.local556 = lshr i64 %address.local555, 5, !dbg !10
  %address.local557 = mul nuw nsw i64 %address.local556, 1016, !dbg !10
  %.idx = mul nuw nsw i64 %address.local499, 320, !dbg !12
  %130 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %.idx, !dbg !12
  %.idx1410 = mul nsw i64 %address.local542, -1280, !dbg !12
  %131 = getelementptr i8, ptr addrspace(3) %130, i64 %.idx1410, !dbg !12
  %132 = getelementptr [2 x i8], ptr addrspace(3) %131, i64 %address.local546, !dbg !12
  %133 = getelementptr [2 x i8], ptr addrspace(3) %132, i64 %address.local550, !dbg !12
  %134 = getelementptr [2 x i8], ptr addrspace(3) %133, i64 %address.local553, !dbg !12
  %address.local559 = getelementptr [2 x i8], ptr addrspace(3) %134, i64 %address.local557, !dbg !12
  %135 = add nsw i64 %.idx1410, %.idx, !dbg !12
  %.idx1893 = shl nuw nsw i64 %address.local546, 1, !dbg !12
  %136 = or disjoint i64 %135, %.idx1893, !dbg !12
  %137 = or disjoint i64 %address.local550, %address.local553, !dbg !12
  %138 = shl nsw i64 %137, 1, !dbg !12
  %address.local559.idx = mul nuw nsw i64 %address.local556, 2032, !dbg !12
  %139 = add nsw i64 %136, %address.local559.idx, !dbg !12
  %140 = add nsw i64 %139, %138, !dbg !12
  %141 = lshr i64 %140, 6, !dbg !12
  %142 = and i64 %141, 8, !dbg !12
  %143 = xor i64 %142, %140, !dbg !12
  %144 = sub i64 %143, %140, !dbg !12
  %kdma.swizzled = getelementptr i8, ptr addrspace(3) %address.local559, i64 %144, !dbg !12
  %145 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !12
  %146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %145, <4 x float> zeroinitializer) #8, !dbg !12
  %kdma.gep = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4096, !dbg !12
  %147 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !12
  %address.local570 = shl nuw nsw i32 %address.thread498, 1, !dbg !12
  %address.local571 = and i32 %address.local570, 24, !dbg !12
  %address.local572 = shl nuw nsw i32 %address.thread498, 8, !dbg !12
  %address.local573 = and i32 %address.local572, 768, !dbg !12
  %address.local574 = or disjoint i32 %address.local571, %address.local573, !dbg !12
  %address.local575 = shl nuw nsw i32 %address.thread498, 5, !dbg !12
  %address.local576 = and i32 %address.local575, 96, !dbg !12
  %address.local577 = or disjoint i32 %address.local574, %address.local576, !dbg !12
  %address.local578 = zext nneg i32 %address.local577 to i64, !dbg !12
  %address.base = tail call i64 asm sideeffect "", "=v,0"(i64 %97) #8, !dbg !12, !frisk.register_lifetimes !5
  %address.local579 = lshr i32 %address.thread498, 1, !dbg !12
  %address.local580 = and i32 %address.local579, 1, !dbg !12
  %address.local581 = zext nneg i32 %address.local580 to i64, !dbg !12
  %address.local582 = add nsw i64 %address.base, %address.local581, !dbg !12
  %address.local583 = shl i64 %address.local582, 6, !dbg !12
  %address.local584 = sub i64 %address.local578, %address.local583, !dbg !12
  %148 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local584, !dbg !12
  %149 = getelementptr [2 x i8], ptr addrspace(3) %148, i64 %address.local553, !dbg !12
  %150 = getelementptr [2 x i8], ptr addrspace(3) %149, i64 %address.local557, !dbg !12
  %address.local588 = getelementptr i8, ptr addrspace(3) %150, i64 64, !dbg !12
  %.idx18961899 = add i64 %address.local584, %address.local553, !dbg !12
  %151 = shl i64 %.idx18961899, 1, !dbg !12
  %152 = add nuw nsw i64 %address.local559.idx, 64, !dbg !12
  %153 = add i64 %152, %151, !dbg !12
  %154 = lshr i64 %153, 6, !dbg !12
  %155 = and i64 %154, 8, !dbg !12
  %156 = xor i64 %155, %153, !dbg !12
  %157 = sub i64 %156, %153, !dbg !12
  %kdma.swizzled1767 = getelementptr i8, ptr addrspace(3) %address.local588, i64 %157, !dbg !12
  %158 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !12, !frisk.k_lds_pair !14
  %kdma.gep1771 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 128, !dbg !12
  %159 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !12, !frisk.k_lds_pair !14
  %kdma.gep1768 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4096, !dbg !12
  %160 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !12, !frisk.k_lds_pair !15
  %kdma.gep1772 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 4224, !dbg !12
  %161 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !12, !frisk.k_lds_pair !15
  %kdma.gep1769 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 128, !dbg !12
  %162 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !12, !frisk.k_lds_pair !16
  %kdma.gep1773 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1024, !dbg !12
  %163 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !12, !frisk.k_lds_pair !16
  %kdma.gep1770 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 4224, !dbg !12
  %164 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !12, !frisk.k_lds_pair !17
  %kdma.gep1774 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5120, !dbg !12
  %165 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !12, !frisk.k_lds_pair !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %166 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %147, <4 x float> %146) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1775 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1024, !dbg !12
  %167 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !12, !frisk.k_lds_pair !18
  %kdma.gep1779 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 1152, !dbg !12
  %168 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !12, !frisk.k_lds_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %169 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %158, <4 x float> zeroinitializer) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1776 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5120, !dbg !12
  %170 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !12, !frisk.k_lds_pair !19
  %kdma.gep1780 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 5248, !dbg !12
  %171 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !12, !frisk.k_lds_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %160, <4 x float> %169) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1777 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 1152, !dbg !12
  %173 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !12, !frisk.k_lds_pair !20
  %kdma.gep1778 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 5248, !dbg !12
  %174 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !12, !frisk.k_lds_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %175 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %162, <4 x float> zeroinitializer) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %164, <4 x float> %175) #8, !dbg !12
  %177 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %159, <4 x float> zeroinitializer) #8, !dbg !12
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %161, <4 x float> %177) #8, !dbg !12
  %179 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %163, <4 x float> zeroinitializer) #8, !dbg !12
  %180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %165, <4 x float> %179) #8, !dbg !12
  %181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %167, <4 x float> zeroinitializer) #8, !dbg !12
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %170, <4 x float> %181) #8, !dbg !12
  %183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %173, <4 x float> zeroinitializer) #8, !dbg !12
  %184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %174, <4 x float> %183) #8, !dbg !12
  %185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %52, <4 x half> %168, <4 x float> zeroinitializer) #8, !dbg !12
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %54, <4 x half> %171, <4 x float> %185) #8, !dbg !12
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %145, <4 x float> zeroinitializer) #8, !dbg !12
  %188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %147, <4 x float> %187) #8, !dbg !12
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %158, <4 x float> zeroinitializer) #8, !dbg !12
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %160, <4 x float> %189) #8, !dbg !12
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %162, <4 x float> zeroinitializer) #8, !dbg !12
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %164, <4 x float> %191) #8, !dbg !12
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %159, <4 x float> zeroinitializer) #8, !dbg !12
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %161, <4 x float> %193) #8, !dbg !12
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %163, <4 x float> zeroinitializer) #8, !dbg !12
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %165, <4 x float> %195) #8, !dbg !12
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %167, <4 x float> zeroinitializer) #8, !dbg !12
  %198 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %170, <4 x float> %197) #8, !dbg !12
  %199 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %173, <4 x float> zeroinitializer) #8, !dbg !12
  %200 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %174, <4 x float> %199) #8, !dbg !12
  %201 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %56, <4 x half> %168, <4 x float> zeroinitializer) #8, !dbg !12
  %202 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %57, <4 x half> %171, <4 x float> %201) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %203 = or disjoint i64 %110, 64, !dbg !13
  %kdma.source1646 = or disjoint i64 %203, %k0.prime.kdma.source1515, !dbg !13
  %kdma.source1647 = trunc nuw nsw i64 %kdma.source1646 to i32, !dbg !13
  %kdma.source1648 = shl nuw nsw i32 %kdma.source1647, 1, !dbg !13
  %kdma.offset1649 = or disjoint i32 %kdma.source1648, %kdma.intra, !dbg !13
  %204 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %kdma.offset1649, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1662 = or disjoint i64 %203, %k0.prime.kdma.source1529, !dbg !13
  %kdma.source1663 = trunc nuw nsw i64 %kdma.source1662 to i32, !dbg !13
  %kdma.source1664 = shl nuw nsw i32 %kdma.source1663, 1, !dbg !13
  %kdma.offset1665 = or disjoint i32 %kdma.source1664, %kdma.intra.hi, !dbg !13
  %205 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %kdma.offset1665, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1679 = or disjoint i64 %203, %k0.prime.kdma.source1546, !dbg !13
  %kdma.source1680 = trunc nuw nsw i64 %kdma.source1679 to i32, !dbg !13
  %kdma.source1681 = shl nuw nsw i32 %kdma.source1680, 1, !dbg !13
  %kdma.offset1682 = or disjoint i32 %kdma.source1681, %kdma.intra, !dbg !13
  %206 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %kdma.offset1682, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1696 = or disjoint i64 %203, %k0.prime.kdma.source1563, !dbg !13
  %kdma.source1697 = trunc nuw nsw i64 %kdma.source1696 to i32, !dbg !13
  %kdma.source1698 = shl nuw nsw i32 %kdma.source1697, 1, !dbg !13
  %kdma.offset1699 = or disjoint i32 %kdma.source1698, %kdma.intra.hi, !dbg !13
  %207 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %kdma.offset1699, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.gep1797 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8192, !dbg !12
  %208 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !12, !frisk.k_lds_pair !21
  %kdma.gep1801 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 8320, !dbg !12
  %209 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !12, !frisk.k_lds_pair !21
  %kdma.gep1798 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12288, !dbg !12
  %210 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !12, !frisk.k_lds_pair !22
  %kdma.gep1802 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 12416, !dbg !12
  %211 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !12, !frisk.k_lds_pair !22
  %kdma.gep1799 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8192, !dbg !12
  %212 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !12, !frisk.k_lds_pair !23
  %kdma.gep1803 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 8320, !dbg !12
  %213 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !12, !frisk.k_lds_pair !23
  %kdma.gep1800 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12288, !dbg !12
  %214 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !12, !frisk.k_lds_pair !24
  %kdma.gep1804 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 12416, !dbg !12
  %215 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !12, !frisk.k_lds_pair !24
  %kdma.gep1805 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9216, !dbg !12
  %216 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !12, !frisk.k_lds_pair !25
  %kdma.gep1809 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 9344, !dbg !12
  %217 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !12, !frisk.k_lds_pair !25
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %218 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %208, <4 x float> %166) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1806 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13312, !dbg !12
  %219 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !12, !frisk.k_lds_pair !26
  %kdma.gep1810 = getelementptr i8, ptr addrspace(3) %kdma.swizzled, i64 13440, !dbg !12
  %220 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !12, !frisk.k_lds_pair !26
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %221 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %210, <4 x float> %218) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1807 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9216, !dbg !12
  %222 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !12, !frisk.k_lds_pair !27
  %kdma.gep1811 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 9344, !dbg !12
  %223 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !12, !frisk.k_lds_pair !27
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %224 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %212, <4 x float> %172) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %kdma.gep1808 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13312, !dbg !12
  %225 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !12, !frisk.k_lds_pair !28
  %kdma.gep1812 = getelementptr i8, ptr addrspace(3) %kdma.swizzled1767, i64 13440, !dbg !12
  %226 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !12, !frisk.k_lds_pair !28
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %227 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %214, <4 x float> %224) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %228 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %209, <4 x float> %176) #8, !dbg !12
  %229 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %211, <4 x float> %228) #8, !dbg !12
  %230 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %213, <4 x float> %178) #8, !dbg !12
  %231 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %215, <4 x float> %230) #8, !dbg !12
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %216, <4 x float> %180) #8, !dbg !12
  %233 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %219, <4 x float> %232) #8, !dbg !12
  %234 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %222, <4 x float> %182) #8, !dbg !12
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %225, <4 x float> %234) #8, !dbg !12
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %217, <4 x float> %184) #8, !dbg !12
  %237 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %220, <4 x float> %236) #8, !dbg !12
  %238 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %66, <4 x half> %223, <4 x float> %186) #8, !dbg !12
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %68, <4 x half> %226, <4 x float> %238) #8, !dbg !12
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %208, <4 x float> %188) #8, !dbg !12
  %241 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %210, <4 x float> %240) #8, !dbg !12
  %242 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %212, <4 x float> %190) #8, !dbg !12
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %214, <4 x float> %242) #8, !dbg !12
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %209, <4 x float> %192) #8, !dbg !12
  %245 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %211, <4 x float> %244) #8, !dbg !12
  %246 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %213, <4 x float> %194) #8, !dbg !12
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %215, <4 x float> %246) #8, !dbg !12
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %216, <4 x float> %196) #8, !dbg !12
  %249 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %219, <4 x float> %248) #8, !dbg !12
  %250 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %222, <4 x float> %198) #8, !dbg !12
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %225, <4 x float> %250) #8, !dbg !12
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %217, <4 x float> %200) #8, !dbg !12
  %253 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %220, <4 x float> %252) #8, !dbg !12
  %254 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %69, <4 x half> %223, <4 x float> %202) #8, !dbg !12
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %70, <4 x half> %226, <4 x float> %254) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %256 = or disjoint i64 %110, 96, !dbg !13
  %kdma.source1712 = or disjoint i64 %256, %k0.prime.kdma.source1515, !dbg !13
  %kdma.source1713 = trunc nuw nsw i64 %kdma.source1712 to i32, !dbg !13
  %kdma.source1714 = shl nuw nsw i32 %kdma.source1713, 1, !dbg !13
  %kdma.offset1715 = or disjoint i32 %kdma.source1714, %kdma.intra, !dbg !13
  %257 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1368, i32 %kdma.offset1715, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1728 = or disjoint i64 %256, %k0.prime.kdma.source1529, !dbg !13
  %kdma.source1729 = trunc nuw nsw i64 %kdma.source1728 to i32, !dbg !13
  %kdma.source1730 = shl nuw nsw i32 %kdma.source1729, 1, !dbg !13
  %kdma.offset1731 = or disjoint i32 %kdma.source1730, %kdma.intra.hi, !dbg !13
  %258 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1600, i32 %kdma.offset1731, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1745 = or disjoint i64 %256, %k0.prime.kdma.source1546, !dbg !13
  %kdma.source1746 = trunc nuw nsw i64 %kdma.source1745 to i32, !dbg !13
  %kdma.source1747 = shl nuw nsw i32 %kdma.source1746, 1, !dbg !13
  %kdma.offset1748 = or disjoint i32 %kdma.source1747, %kdma.intra, !dbg !13
  %259 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1365, i32 %kdma.offset1748, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %kdma.source1762 = or disjoint i64 %256, %k0.prime.kdma.source1563, !dbg !13
  %kdma.source1763 = trunc nuw nsw i64 %kdma.source1762 to i32, !dbg !13
  %kdma.source1764 = shl nuw nsw i32 %kdma.source1763, 1, !dbg !13
  %kdma.offset1765 = or disjoint i32 %kdma.source1764, %kdma.intra.hi, !dbg !13
  %260 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1634, i32 %kdma.offset1765, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13
  %261 = load <4 x half>, ptr addrspace(3) %kdma.swizzled, align 8, !dbg !12, !frisk.k_lds_pair !29
  %262 = load <4 x half>, ptr addrspace(3) %kdma.gep1769, align 8, !dbg !12, !frisk.k_lds_pair !29
  %263 = load <4 x half>, ptr addrspace(3) %kdma.gep, align 8, !dbg !12, !frisk.k_lds_pair !30
  %264 = load <4 x half>, ptr addrspace(3) %kdma.gep1770, align 8, !dbg !12, !frisk.k_lds_pair !30
  %265 = load <4 x half>, ptr addrspace(3) %kdma.swizzled1767, align 8, !dbg !12, !frisk.k_lds_pair !31
  %266 = load <4 x half>, ptr addrspace(3) %kdma.gep1771, align 8, !dbg !12, !frisk.k_lds_pair !31
  %267 = load <4 x half>, ptr addrspace(3) %kdma.gep1768, align 8, !dbg !12, !frisk.k_lds_pair !32
  %268 = load <4 x half>, ptr addrspace(3) %kdma.gep1772, align 8, !dbg !12, !frisk.k_lds_pair !32
  %269 = load <4 x half>, ptr addrspace(3) %kdma.gep1773, align 8, !dbg !12, !frisk.k_lds_pair !33
  %270 = load <4 x half>, ptr addrspace(3) %kdma.gep1777, align 8, !dbg !12, !frisk.k_lds_pair !33
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %261, <4 x float> %221) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %272 = load <4 x half>, ptr addrspace(3) %kdma.gep1774, align 8, !dbg !12, !frisk.k_lds_pair !34
  %273 = load <4 x half>, ptr addrspace(3) %kdma.gep1778, align 8, !dbg !12, !frisk.k_lds_pair !34
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %263, <4 x float> %271) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %275 = load <4 x half>, ptr addrspace(3) %kdma.gep1775, align 8, !dbg !12, !frisk.k_lds_pair !35
  %276 = load <4 x half>, ptr addrspace(3) %kdma.gep1779, align 8, !dbg !12, !frisk.k_lds_pair !35
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %265, <4 x float> %227) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %278 = load <4 x half>, ptr addrspace(3) %kdma.gep1776, align 8, !dbg !12, !frisk.k_lds_pair !36
  %279 = load <4 x half>, ptr addrspace(3) %kdma.gep1780, align 8, !dbg !12, !frisk.k_lds_pair !36
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %280 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %267, <4 x float> %277) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_lds_prefetch !5
  %281 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %262, <4 x float> %229) #8, !dbg !12
  %282 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %264, <4 x float> %281) #8, !dbg !12
  %283 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %266, <4 x float> %231) #8, !dbg !12
  %284 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %268, <4 x float> %283) #8, !dbg !12
  %285 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %269, <4 x float> %233) #8, !dbg !12
  %286 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %272, <4 x float> %285) #8, !dbg !12
  %287 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %275, <4 x float> %235) #8, !dbg !12
  %288 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %278, <4 x float> %287) #8, !dbg !12
  %289 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %270, <4 x float> %237) #8, !dbg !12
  %290 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %273, <4 x float> %289) #8, !dbg !12
  %291 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %79, <4 x half> %276, <4 x float> %239) #8, !dbg !12
  %292 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %81, <4 x half> %279, <4 x float> %291) #8, !dbg !12
  %293 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %261, <4 x float> %241) #8, !dbg !12
  %294 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %263, <4 x float> %293) #8, !dbg !12
  %295 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %265, <4 x float> %243) #8, !dbg !12
  %296 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %267, <4 x float> %295) #8, !dbg !12
  %297 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %262, <4 x float> %245) #8, !dbg !12
  %298 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %264, <4 x float> %297) #8, !dbg !12
  %299 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %266, <4 x float> %247) #8, !dbg !12
  %300 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %268, <4 x float> %299) #8, !dbg !12
  %301 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %269, <4 x float> %249) #8, !dbg !12
  %302 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %272, <4 x float> %301) #8, !dbg !12
  %303 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %275, <4 x float> %251) #8, !dbg !12
  %304 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %278, <4 x float> %303) #8, !dbg !12
  %305 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %270, <4 x float> %253) #8, !dbg !12
  %306 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %273, <4 x float> %305) #8, !dbg !12
  %307 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %82, <4 x half> %276, <4 x float> %255) #8, !dbg !12
  %308 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %83, <4 x half> %279, <4 x float> %307) #8, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %address.local917 = shl nuw nsw i64 %address.local499, 5, !dbg !37
  %address.local918 = and i64 %address.local917, 8064, !dbg !37
  %address.local920 = and i64 %address.local500, 24, !dbg !37
  %address.local921 = or disjoint i64 %address.local918, %address.local920, !dbg !37
  %.idx14191420 = or disjoint i64 %address.local921, %110, !dbg !37
  %.idx14191420.tr = trunc nuw nsw i64 %.idx14191420 to i32, !dbg !37
  %dma.buffer.offset1348 = shl nuw nsw i32 %.idx14191420.tr, 1, !dbg !37
  %309 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %dma.buffer.offset1348, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %address.local923 = shl nuw nsw i64 %address.local516, 5, !dbg !37
  %address.local924 = and i64 %address.local923, 16256, !dbg !37
  %address.local927 = or disjoint i64 %address.local924, %address.local920, !dbg !37
  %.idx14211422 = or disjoint i64 %address.local927, %110, !dbg !37
  %.idx14211422.tr = trunc nuw nsw i64 %.idx14211422 to i32, !dbg !37
  %dma.buffer.offset1345 = shl nuw nsw i32 %.idx14211422.tr, 1, !dbg !37
  %310 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %dma.buffer.offset1345, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %311 = load <4 x half>, ptr addrspace(3) %kdma.gep1797, align 8, !dbg !12, !frisk.k_tail_pair !14
  %312 = load <4 x half>, ptr addrspace(3) %kdma.gep1801, align 8, !dbg !12, !frisk.k_tail_pair !14
  %313 = load <4 x half>, ptr addrspace(3) %kdma.gep1798, align 8, !dbg !12, !frisk.k_tail_pair !15
  %314 = load <4 x half>, ptr addrspace(3) %kdma.gep1802, align 8, !dbg !12, !frisk.k_tail_pair !15
  %315 = load <4 x half>, ptr addrspace(3) %kdma.gep1799, align 8, !dbg !12, !frisk.k_tail_pair !16
  %316 = load <4 x half>, ptr addrspace(3) %kdma.gep1803, align 8, !dbg !12, !frisk.k_tail_pair !16
  %317 = load <4 x half>, ptr addrspace(3) %kdma.gep1800, align 8, !dbg !12, !frisk.k_tail_pair !17
  %318 = load <4 x half>, ptr addrspace(3) %kdma.gep1804, align 8, !dbg !12, !frisk.k_tail_pair !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %319 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %311, <4 x float> %274) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %313, <4 x float> %319) #8, !dbg !12
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %315, <4 x float> %280) #8, !dbg !12
  %322 = extractelement <4 x float> %320, i64 0, !dbg !38
  %323 = tail call float @llvm.maxnum.f32(float %322, float -inf), !dbg !39, !frisk.local_reduction !5
  %324 = extractelement <4 x float> %320, i64 1, !dbg !38
  %325 = extractelement <4 x float> %320, i64 2, !dbg !38
  %326 = extractelement <4 x float> %320, i64 3, !dbg !38
  %327 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %317, <4 x float> %321) #8, !dbg !12
  %328 = load <4 x half>, ptr addrspace(3) %kdma.gep1805, align 8, !dbg !12, !frisk.k_tail_pair !18
  %329 = load <4 x half>, ptr addrspace(3) %kdma.gep1809, align 8, !dbg !12, !frisk.k_tail_pair !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %330 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %312, <4 x float> %282) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !38, !frisk.k_tail_prefetch !5
  %331 = extractelement <4 x float> %327, i64 0, !dbg !38
  %local.reduce151 = tail call float @llvm.maxnum.f32(float %323, float %331), !dbg !39, !frisk.local_reduction !5
  %332 = extractelement <4 x float> %327, i64 1, !dbg !38
  %local.reduce152 = tail call float @llvm.maxnum.f32(float %324, float %332), !dbg !39, !frisk.local_reduction !5
  %333 = extractelement <4 x float> %327, i64 2, !dbg !38
  %local.reduce153 = tail call float @llvm.maxnum.f32(float %325, float %333), !dbg !39, !frisk.local_reduction !5
  %334 = extractelement <4 x float> %327, i64 3, !dbg !38
  %local.reduce154 = tail call float @llvm.maxnum.f32(float %326, float %334), !dbg !39, !frisk.local_reduction !5
  %335 = load <4 x half>, ptr addrspace(3) %kdma.gep1806, align 8, !dbg !12, !frisk.k_tail_pair !19
  %336 = load <4 x half>, ptr addrspace(3) %kdma.gep1810, align 8, !dbg !12, !frisk.k_tail_pair !19
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %314, <4 x float> %330) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %338 = load <4 x half>, ptr addrspace(3) %kdma.gep1807, align 8, !dbg !12, !frisk.k_tail_pair !20
  %339 = load <4 x half>, ptr addrspace(3) %kdma.gep1811, align 8, !dbg !12, !frisk.k_tail_pair !20
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %316, <4 x float> %284) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !38, !frisk.k_tail_prefetch !5
  %341 = extractelement <4 x float> %337, i64 0, !dbg !38
  %local.reduce155 = tail call float @llvm.maxnum.f32(float %local.reduce151, float %341), !dbg !39, !frisk.local_reduction !5
  %342 = extractelement <4 x float> %337, i64 1, !dbg !38
  %local.reduce156 = tail call float @llvm.maxnum.f32(float %local.reduce152, float %342), !dbg !39, !frisk.local_reduction !5
  %343 = extractelement <4 x float> %337, i64 2, !dbg !38
  %local.reduce157 = tail call float @llvm.maxnum.f32(float %local.reduce153, float %343), !dbg !39, !frisk.local_reduction !5
  %344 = extractelement <4 x float> %337, i64 3, !dbg !38
  %local.reduce158 = tail call float @llvm.maxnum.f32(float %local.reduce154, float %344), !dbg !39, !frisk.local_reduction !5
  %345 = load <4 x half>, ptr addrspace(3) %kdma.gep1808, align 8, !dbg !12, !frisk.k_tail_pair !21
  %346 = load <4 x half>, ptr addrspace(3) %kdma.gep1812, align 8, !dbg !12, !frisk.k_tail_pair !21
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %318, <4 x float> %340) #8, !dbg !12
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !12, !frisk.k_tail_prefetch !5
  %348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %328, <4 x float> %286) #8, !dbg !12
  %349 = extractelement <4 x float> %347, i64 0, !dbg !38
  %local.reduce159 = tail call float @llvm.maxnum.f32(float %local.reduce155, float %349), !dbg !39, !frisk.local_reduction !5
  %350 = extractelement <4 x float> %347, i64 1, !dbg !38
  %local.reduce160 = tail call float @llvm.maxnum.f32(float %local.reduce156, float %350), !dbg !39, !frisk.local_reduction !5
  %351 = extractelement <4 x float> %347, i64 2, !dbg !38
  %local.reduce161 = tail call float @llvm.maxnum.f32(float %local.reduce157, float %351), !dbg !39, !frisk.local_reduction !5
  %352 = extractelement <4 x float> %347, i64 3, !dbg !38
  %local.reduce162 = tail call float @llvm.maxnum.f32(float %local.reduce158, float %352), !dbg !39, !frisk.local_reduction !5
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %335, <4 x float> %348) #8, !dbg !12
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %338, <4 x float> %288) #8, !dbg !12
  %355 = extractelement <4 x float> %353, i64 0, !dbg !38
  %local.reduce163 = tail call float @llvm.maxnum.f32(float %local.reduce159, float %355), !dbg !39, !frisk.local_reduction !5
  %356 = extractelement <4 x float> %353, i64 1, !dbg !38
  %local.reduce164 = tail call float @llvm.maxnum.f32(float %local.reduce160, float %356), !dbg !39, !frisk.local_reduction !5
  %357 = extractelement <4 x float> %353, i64 2, !dbg !38
  %local.reduce165 = tail call float @llvm.maxnum.f32(float %local.reduce161, float %357), !dbg !39, !frisk.local_reduction !5
  %358 = extractelement <4 x float> %353, i64 3, !dbg !38
  %local.reduce166 = tail call float @llvm.maxnum.f32(float %local.reduce162, float %358), !dbg !39, !frisk.local_reduction !5
  %359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %345, <4 x float> %354) #8, !dbg !12
  %360 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %329, <4 x float> %290) #8, !dbg !12
  %361 = extractelement <4 x float> %359, i64 0, !dbg !38
  %local.reduce167 = tail call float @llvm.maxnum.f32(float %local.reduce163, float %361), !dbg !39, !frisk.local_reduction !5
  %362 = extractelement <4 x float> %359, i64 1, !dbg !38
  %local.reduce168 = tail call float @llvm.maxnum.f32(float %local.reduce164, float %362), !dbg !39, !frisk.local_reduction !5
  %363 = extractelement <4 x float> %359, i64 2, !dbg !38
  %local.reduce169 = tail call float @llvm.maxnum.f32(float %local.reduce165, float %363), !dbg !39, !frisk.local_reduction !5
  %364 = extractelement <4 x float> %359, i64 3, !dbg !38
  %local.reduce170 = tail call float @llvm.maxnum.f32(float %local.reduce166, float %364), !dbg !39, !frisk.local_reduction !5
  %365 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %336, <4 x float> %360) #8, !dbg !12
  %366 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %87, <4 x half> %339, <4 x float> %292) #8, !dbg !12
  %367 = extractelement <4 x float> %365, i64 0, !dbg !38
  %local.reduce171 = tail call float @llvm.maxnum.f32(float %local.reduce167, float %367), !dbg !39, !frisk.local_reduction !5
  %368 = extractelement <4 x float> %365, i64 1, !dbg !38
  %local.reduce172 = tail call float @llvm.maxnum.f32(float %local.reduce168, float %368), !dbg !39, !frisk.local_reduction !5
  %369 = extractelement <4 x float> %365, i64 2, !dbg !38
  %local.reduce173 = tail call float @llvm.maxnum.f32(float %local.reduce169, float %369), !dbg !39, !frisk.local_reduction !5
  %370 = extractelement <4 x float> %365, i64 3, !dbg !38
  %local.reduce174 = tail call float @llvm.maxnum.f32(float %local.reduce170, float %370), !dbg !39, !frisk.local_reduction !5
  %371 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %89, <4 x half> %346, <4 x float> %366) #8, !dbg !12
  %372 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %311, <4 x float> %294) #8, !dbg !12
  %373 = extractelement <4 x float> %371, i64 0, !dbg !38
  %local.reduce175 = tail call float @llvm.maxnum.f32(float %local.reduce171, float %373), !dbg !39, !frisk.local_reduction !5
  %374 = extractelement <4 x float> %371, i64 1, !dbg !38
  %local.reduce176 = tail call float @llvm.maxnum.f32(float %local.reduce172, float %374), !dbg !39, !frisk.local_reduction !5
  %375 = extractelement <4 x float> %371, i64 2, !dbg !38
  %local.reduce177 = tail call float @llvm.maxnum.f32(float %local.reduce173, float %375), !dbg !39, !frisk.local_reduction !5
  %376 = extractelement <4 x float> %371, i64 3, !dbg !38
  %local.reduce178 = tail call float @llvm.maxnum.f32(float %local.reduce174, float %376), !dbg !39, !frisk.local_reduction !5
  %local.reduce179 = tail call float @llvm.maxnum.f32(float %local.reduce175, float %local.reduce176), !dbg !39, !frisk.local_reduction !5
  %local.reduce180 = tail call float @llvm.maxnum.f32(float %local.reduce177, float %local.reduce178), !dbg !39, !frisk.local_reduction !5
  %377 = tail call float @llvm.maxnum.f32(float %local.reduce179, float %local.reduce180), !dbg !39, !frisk.local_reduction !5
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %313, <4 x float> %372) #8, !dbg !12
  %379 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %315, <4 x float> %296) #8, !dbg !12
  %380 = extractelement <4 x float> %378, i64 0, !dbg !38
  %381 = extractelement <4 x float> %378, i64 1, !dbg !38
  %382 = extractelement <4 x float> %378, i64 2, !dbg !38
  %383 = extractelement <4 x float> %378, i64 3, !dbg !38
  %384 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %317, <4 x float> %379) #8, !dbg !12
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %312, <4 x float> %298) #8, !dbg !12
  %386 = extractelement <4 x float> %384, i64 0, !dbg !38
  %387 = extractelement <4 x float> %384, i64 1, !dbg !38
  %388 = extractelement <4 x float> %384, i64 2, !dbg !38
  %389 = extractelement <4 x float> %384, i64 3, !dbg !38
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %314, <4 x float> %385) #8, !dbg !12
  %391 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %316, <4 x float> %300) #8, !dbg !12
  %392 = extractelement <4 x float> %390, i64 0, !dbg !38
  %393 = extractelement <4 x float> %390, i64 1, !dbg !38
  %394 = extractelement <4 x float> %390, i64 2, !dbg !38
  %395 = extractelement <4 x float> %390, i64 3, !dbg !38
  %396 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %318, <4 x float> %391) #8, !dbg !12
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %328, <4 x float> %302) #8, !dbg !12
  %398 = extractelement <4 x float> %396, i64 0, !dbg !38
  %399 = extractelement <4 x float> %396, i64 1, !dbg !38
  %400 = extractelement <4 x float> %396, i64 2, !dbg !38
  %401 = extractelement <4 x float> %396, i64 3, !dbg !38
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %335, <4 x float> %397) #8, !dbg !12
  %403 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %338, <4 x float> %304) #8, !dbg !12
  %404 = extractelement <4 x float> %402, i64 0, !dbg !38
  %405 = extractelement <4 x float> %402, i64 1, !dbg !38
  %406 = extractelement <4 x float> %402, i64 2, !dbg !38
  %407 = extractelement <4 x float> %402, i64 3, !dbg !38
  %408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %345, <4 x float> %403) #8, !dbg !12
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %329, <4 x float> %306) #8, !dbg !12
  %410 = extractelement <4 x float> %408, i64 0, !dbg !38
  %411 = extractelement <4 x float> %408, i64 1, !dbg !38
  %412 = extractelement <4 x float> %408, i64 2, !dbg !38
  %413 = extractelement <4 x float> %408, i64 3, !dbg !38
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %336, <4 x float> %409) #8, !dbg !12
  %415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %90, <4 x half> %339, <4 x float> %308) #8, !dbg !12
  %416 = extractelement <4 x float> %414, i64 0, !dbg !38
  %417 = extractelement <4 x float> %414, i64 1, !dbg !38
  %418 = extractelement <4 x float> %414, i64 2, !dbg !38
  %419 = extractelement <4 x float> %414, i64 3, !dbg !38
  %420 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %91, <4 x half> %346, <4 x float> %415) #8, !dbg !12
  %421 = extractelement <4 x float> %420, i64 0, !dbg !38
  %422 = extractelement <4 x float> %420, i64 1, !dbg !38
  %423 = extractelement <4 x float> %420, i64 2, !dbg !38
  %424 = extractelement <4 x float> %420, i64 3, !dbg !38
  %425 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !39
  %426 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %425), !dbg !39
  %427 = add nuw nsw i32 %426, 64, !dbg !39
  %428 = xor i32 %426, 16, !dbg !39
  %429 = and i32 %427, 192, !dbg !39
  %430 = icmp samesign ult i32 %428, %429, !dbg !39
  %431 = select i1 %430, i32 %428, i32 %426, !dbg !39
  %432 = shl nuw nsw i32 %431, 2, !dbg !39
  %433 = bitcast float %377 to i32, !dbg !39
  %434 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %433), !dbg !39
  %435 = bitcast i32 %434 to float, !dbg !39
  %436 = tail call float @llvm.maxnum.f32(float %377, float %435), !dbg !39
  %437 = xor i32 %426, 32, !dbg !39
  %438 = icmp samesign ult i32 %437, %429, !dbg !39
  %439 = select i1 %438, i32 %437, i32 %426, !dbg !39
  %440 = shl nuw nsw i32 %439, 2, !dbg !39
  %441 = bitcast float %436 to i32, !dbg !39
  %442 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %441), !dbg !39
  %443 = bitcast i32 %442 to float, !dbg !39
  %444 = tail call float @llvm.maxnum.f32(float %436, float %443), !dbg !39
  %445 = extractvalue [2 x <1 x float>] %105, 0, !dbg !39
  %446 = insertelement <1 x float> poison, float %444, i64 0, !dbg !39
  %447 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %446, <1 x float> %445), !dbg !39
  %448 = tail call float @llvm.maxnum.f32(float %380, float -inf), !dbg !39, !frisk.local_reduction !5
  %local.reduce120 = tail call float @llvm.maxnum.f32(float %448, float %386), !dbg !39, !frisk.local_reduction !5
  %local.reduce121 = tail call float @llvm.maxnum.f32(float %381, float %387), !dbg !39, !frisk.local_reduction !5
  %local.reduce122 = tail call float @llvm.maxnum.f32(float %382, float %388), !dbg !39, !frisk.local_reduction !5
  %local.reduce123 = tail call float @llvm.maxnum.f32(float %383, float %389), !dbg !39, !frisk.local_reduction !5
  %local.reduce124 = tail call float @llvm.maxnum.f32(float %local.reduce120, float %392), !dbg !39, !frisk.local_reduction !5
  %local.reduce125 = tail call float @llvm.maxnum.f32(float %local.reduce121, float %393), !dbg !39, !frisk.local_reduction !5
  %local.reduce126 = tail call float @llvm.maxnum.f32(float %local.reduce122, float %394), !dbg !39, !frisk.local_reduction !5
  %local.reduce127 = tail call float @llvm.maxnum.f32(float %local.reduce123, float %395), !dbg !39, !frisk.local_reduction !5
  %local.reduce128 = tail call float @llvm.maxnum.f32(float %local.reduce124, float %398), !dbg !39, !frisk.local_reduction !5
  %local.reduce129 = tail call float @llvm.maxnum.f32(float %local.reduce125, float %399), !dbg !39, !frisk.local_reduction !5
  %local.reduce130 = tail call float @llvm.maxnum.f32(float %local.reduce126, float %400), !dbg !39, !frisk.local_reduction !5
  %local.reduce131 = tail call float @llvm.maxnum.f32(float %local.reduce127, float %401), !dbg !39, !frisk.local_reduction !5
  %local.reduce132 = tail call float @llvm.maxnum.f32(float %local.reduce128, float %404), !dbg !39, !frisk.local_reduction !5
  %local.reduce133 = tail call float @llvm.maxnum.f32(float %local.reduce129, float %405), !dbg !39, !frisk.local_reduction !5
  %local.reduce134 = tail call float @llvm.maxnum.f32(float %local.reduce130, float %406), !dbg !39, !frisk.local_reduction !5
  %local.reduce135 = tail call float @llvm.maxnum.f32(float %local.reduce131, float %407), !dbg !39, !frisk.local_reduction !5
  %local.reduce136 = tail call float @llvm.maxnum.f32(float %local.reduce132, float %410), !dbg !39, !frisk.local_reduction !5
  %local.reduce137 = tail call float @llvm.maxnum.f32(float %local.reduce133, float %411), !dbg !39, !frisk.local_reduction !5
  %local.reduce138 = tail call float @llvm.maxnum.f32(float %local.reduce134, float %412), !dbg !39, !frisk.local_reduction !5
  %local.reduce139 = tail call float @llvm.maxnum.f32(float %local.reduce135, float %413), !dbg !39, !frisk.local_reduction !5
  %local.reduce140 = tail call float @llvm.maxnum.f32(float %local.reduce136, float %416), !dbg !39, !frisk.local_reduction !5
  %local.reduce141 = tail call float @llvm.maxnum.f32(float %local.reduce137, float %417), !dbg !39, !frisk.local_reduction !5
  %local.reduce142 = tail call float @llvm.maxnum.f32(float %local.reduce138, float %418), !dbg !39, !frisk.local_reduction !5
  %local.reduce143 = tail call float @llvm.maxnum.f32(float %local.reduce139, float %419), !dbg !39, !frisk.local_reduction !5
  %local.reduce144 = tail call float @llvm.maxnum.f32(float %local.reduce140, float %421), !dbg !39, !frisk.local_reduction !5
  %local.reduce145 = tail call float @llvm.maxnum.f32(float %local.reduce141, float %422), !dbg !39, !frisk.local_reduction !5
  %local.reduce146 = tail call float @llvm.maxnum.f32(float %local.reduce142, float %423), !dbg !39, !frisk.local_reduction !5
  %local.reduce147 = tail call float @llvm.maxnum.f32(float %local.reduce143, float %424), !dbg !39, !frisk.local_reduction !5
  %local.reduce148 = tail call float @llvm.maxnum.f32(float %local.reduce144, float %local.reduce145), !dbg !39, !frisk.local_reduction !5
  %local.reduce149 = tail call float @llvm.maxnum.f32(float %local.reduce146, float %local.reduce147), !dbg !39, !frisk.local_reduction !5
  %449 = tail call float @llvm.maxnum.f32(float %local.reduce148, float %local.reduce149), !dbg !39, !frisk.local_reduction !5
  %450 = bitcast float %449 to i32, !dbg !39
  %451 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %450), !dbg !39
  %452 = bitcast i32 %451 to float, !dbg !39
  %453 = tail call float @llvm.maxnum.f32(float %449, float %452), !dbg !39
  %454 = bitcast float %453 to i32, !dbg !39
  %455 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %454), !dbg !39
  %456 = bitcast i32 %455 to float, !dbg !39
  %457 = tail call float @llvm.maxnum.f32(float %453, float %456), !dbg !39
  %458 = extractvalue [2 x <1 x float>] %105, 1, !dbg !39
  %459 = insertelement <1 x float> poison, float %457, i64 0, !dbg !39
  %460 = tail call <1 x float> @llvm.maxnum.v1f32(<1 x float> %459, <1 x float> %458), !dbg !39
  %461 = shufflevector <1 x float> %447, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %462 = shufflevector <1 x float> %460, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !40
  %463 = fsub <1 x float> %445, %447, !dbg !41
  %464 = extractelement <1 x float> %463, i64 0, !dbg !42
  %465 = tail call float @llvm.amdgcn.exp2.f32(float %464), !dbg !42
  %466 = insertelement <4 x float> poison, float %465, i64 0, !dbg !43
  %467 = shufflevector <4 x float> %466, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %468 = extractvalue [2 x <1 x float>] %106, 0, !dbg !44
  %469 = extractelement <1 x float> %468, i64 0, !dbg !44
  %470 = fmul float %469, %465, !dbg !44
  %471 = fsub <1 x float> %458, %460, !dbg !41
  %472 = extractelement <1 x float> %471, i64 0, !dbg !42
  %473 = tail call float @llvm.amdgcn.exp2.f32(float %472), !dbg !42
  %474 = insertelement <4 x float> poison, float %473, i64 0, !dbg !43
  %475 = shufflevector <4 x float> %474, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !43
  %476 = extractvalue [2 x <1 x float>] %106, 1, !dbg !44
  %477 = extractelement <1 x float> %476, i64 0, !dbg !44
  %478 = fmul float %477, %473, !dbg !44
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %vdb.address1436 = or disjoint i64 %address.local921, %111, !dbg !37
  %vdb.address1437 = trunc nuw nsw i64 %vdb.address1436 to i32, !dbg !37
  %vdb.address1438 = shl nuw nsw i32 %vdb.address1437, 1, !dbg !37
  %vdb.address1445 = or disjoint i64 %address.local927, %111, !dbg !37
  %vdb.address1446 = trunc nuw nsw i64 %vdb.address1445 to i32, !dbg !37
  %vdb.address1447 = shl nuw nsw i32 %vdb.address1446, 1, !dbg !37
  %vdb.dma.base = add i32 %dma.buffer.lds1374, 8192, !dbg !45
  %479 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1438, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %vdb.dma.base1452 = add i32 %dma.buffer.lds1371, 8192, !dbg !45
  %480 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1447, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %address.local929 = shl nuw nsw i32 %address.thread498, 3, !dbg !45
  %address.local930 = and i32 %address.local929, 120, !dbg !45
  %address.local932 = and i32 %address.local575, 1536, !dbg !45
  %address.local933 = or disjoint i32 %address.local930, %address.local932, !dbg !45
  %address.local934 = zext nneg i32 %address.local933 to i64, !dbg !45
  %address.local935 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local934, !dbg !45
  %481 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %482 = bitcast <4 x i32> %481 to <8 x half>, !dbg !45
  %address.local936 = and i64 %address.local499, 60, !dbg !45
  %address.local937 = add nuw nsw i64 %address.local936, %address.local499, !dbg !45
  %address.local939.idx = shl nuw nsw i64 %address.local937, 4, !dbg !45
  %address.local939 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local939.idx, !dbg !45
  %address.local940 = lshr i64 %address.local555, 4, !dbg !45
  %address.local942.idx = mul nuw nsw i64 %address.local940, 768, !dbg !45
  %address.local942 = getelementptr i8, ptr addrspace(3) %address.local939, i64 %address.local942.idx, !dbg !45
  %address.local944.idx = mul nsw i64 %address.local542, -64, !dbg !45
  %address.local944 = getelementptr i8, ptr addrspace(3) %address.local942, i64 %address.local944.idx, !dbg !45
  %address.local945 = getelementptr i8, ptr addrspace(3) %address.local944, i64 256, !dbg !45
  %483 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %484 = bitcast <4 x i32> %483 to <8 x half>, !dbg !45
  %address.local949 = getelementptr i8, ptr addrspace(3) %address.local944, i64 512, !dbg !45
  %485 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %486 = bitcast <4 x i32> %485 to <8 x half>, !dbg !45
  %address.local953 = getelementptr i8, ptr addrspace(3) %address.local944, i64 768, !dbg !45
  %487 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %488 = bitcast <4 x i32> %487 to <8 x half>, !dbg !45
  %address.local955 = getelementptr i8, ptr addrspace(3) %address.local935, i64 4096, !dbg !45
  %489 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %490 = bitcast <4 x i32> %489 to <8 x half>, !dbg !45
  %address.local959 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4352, !dbg !45
  %491 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %492 = bitcast <4 x i32> %491 to <8 x half>, !dbg !45
  %address.local963 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4608, !dbg !45
  %493 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %494 = bitcast <4 x i32> %493 to <8 x half>, !dbg !45
  %address.local967 = getelementptr i8, ptr addrspace(3) %address.local944, i64 4864, !dbg !45
  %495 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %496 = shufflevector <8 x half> %482, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %497 = extractvalue [2 x <8 x float>] %107, 0, !dbg !43
  %498 = shufflevector <8 x float> %497, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %499 = fmul <4 x float> %498, %467, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %500 = fsub <4 x float> %320, %461, !dbg !40
  %501 = extractelement <4 x float> %500, i64 0, !dbg !47
  %502 = tail call float @llvm.amdgcn.exp2.f32(float %501), !dbg !47
  %503 = insertelement <4 x float> poison, float %502, i64 0, !dbg !48
  %504 = extractelement <4 x float> %500, i64 1, !dbg !47
  %505 = tail call float @llvm.amdgcn.exp2.f32(float %504), !dbg !47
  %506 = insertelement <4 x float> %503, float %505, i64 1, !dbg !48
  %507 = extractelement <4 x float> %500, i64 2, !dbg !47
  %508 = tail call float @llvm.amdgcn.exp2.f32(float %507), !dbg !47
  %509 = insertelement <4 x float> %506, float %508, i64 2, !dbg !48
  %510 = extractelement <4 x float> %500, i64 3, !dbg !47
  %511 = tail call float @llvm.amdgcn.exp2.f32(float %510), !dbg !47
  %512 = insertelement <4 x float> %509, float %511, i64 3, !dbg !48
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %512) #8, !dbg !48, !frisk.register_lifetimes !5
  %513 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !48
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %513) #8, !dbg !47, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %514 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %496, <4 x float> %499) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %515 = shufflevector <8 x half> %482, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %516 = shufflevector <8 x float> %497, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %517 = fmul <4 x float> %516, %467, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %518 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %515, <4 x float> %517) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %519 = extractvalue [2 x <8 x float>] %107, 1, !dbg !43
  %520 = shufflevector <8 x float> %519, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %521 = fmul <4 x float> %520, %475, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %522 = fsub <4 x float> %378, %462, !dbg !40
  %523 = extractelement <4 x float> %522, i64 0, !dbg !47
  %524 = tail call float @llvm.amdgcn.exp2.f32(float %523), !dbg !47
  %525 = insertelement <4 x float> poison, float %524, i64 0, !dbg !48
  %526 = extractelement <4 x float> %522, i64 1, !dbg !47
  %527 = tail call float @llvm.amdgcn.exp2.f32(float %526), !dbg !47
  %528 = insertelement <4 x float> %525, float %527, i64 1, !dbg !48
  %529 = extractelement <4 x float> %522, i64 2, !dbg !47
  %530 = tail call float @llvm.amdgcn.exp2.f32(float %529), !dbg !47
  %531 = insertelement <4 x float> %528, float %530, i64 2, !dbg !48
  %532 = extractelement <4 x float> %522, i64 3, !dbg !47
  %533 = tail call float @llvm.amdgcn.exp2.f32(float %532), !dbg !47
  %534 = insertelement <4 x float> %531, float %533, i64 3, !dbg !48
  %math.result388 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %534) #8, !dbg !48, !frisk.register_lifetimes !5
  %535 = fptrunc <4 x float> %math.result388 to <4 x half>, !dbg !48
  %math.packed389 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %535) #8, !dbg !47, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %536 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %496, <4 x float> %521) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %537 = shufflevector <8 x float> %519, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %538 = fmul <4 x float> %537, %475, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %539 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %515, <4 x float> %538) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %540 = shufflevector <8 x half> %484, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %541 = fsub <4 x float> %327, %461, !dbg !40
  %542 = extractelement <4 x float> %541, i64 0, !dbg !47
  %543 = tail call float @llvm.amdgcn.exp2.f32(float %542), !dbg !47
  %local.reduce89 = fadd float %502, %543, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor = tail call float asm sideeffect "", "=v,0"(float %local.reduce89) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %544 = insertelement <4 x float> poison, float %543, i64 0, !dbg !48
  %545 = extractelement <4 x float> %541, i64 1, !dbg !47
  %546 = tail call float @llvm.amdgcn.exp2.f32(float %545), !dbg !47
  %local.reduce90 = fadd float %505, %546, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1900 = tail call float asm sideeffect "", "=v,0"(float %local.reduce90) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %547 = insertelement <4 x float> %544, float %546, i64 1, !dbg !48
  %548 = extractelement <4 x float> %541, i64 2, !dbg !47
  %549 = tail call float @llvm.amdgcn.exp2.f32(float %548), !dbg !47
  %local.reduce91 = fadd float %508, %549, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1901 = tail call float asm sideeffect "", "=v,0"(float %local.reduce91) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %550 = insertelement <4 x float> %547, float %549, i64 2, !dbg !48
  %551 = extractelement <4 x float> %541, i64 3, !dbg !47
  %552 = tail call float @llvm.amdgcn.exp2.f32(float %551), !dbg !47
  %local.reduce92 = fadd float %511, %552, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1902 = tail call float asm sideeffect "", "=v,0"(float %local.reduce92) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %553 = insertelement <4 x float> %550, float %552, i64 3, !dbg !48
  %math.result374 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %553) #8, !dbg !48, !frisk.register_lifetimes !5
  %554 = fptrunc <4 x float> %math.result374 to <4 x half>, !dbg !48
  %math.packed375 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %554) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %555 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %540, <4 x float> %514) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %556 = shufflevector <8 x half> %484, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %557 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %556, <4 x float> %518) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %558 = fsub <4 x float> %384, %462, !dbg !40
  %559 = extractelement <4 x float> %558, i64 0, !dbg !47
  %560 = tail call float @llvm.amdgcn.exp2.f32(float %559), !dbg !47
  %local.reduce = fadd float %524, %560, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1903 = tail call float asm sideeffect "", "=v,0"(float %local.reduce) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %561 = insertelement <4 x float> poison, float %560, i64 0, !dbg !48
  %562 = extractelement <4 x float> %558, i64 1, !dbg !47
  %563 = tail call float @llvm.amdgcn.exp2.f32(float %562), !dbg !47
  %local.reduce59 = fadd float %527, %563, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1904 = tail call float asm sideeffect "", "=v,0"(float %local.reduce59) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %564 = insertelement <4 x float> %561, float %563, i64 1, !dbg !48
  %565 = extractelement <4 x float> %558, i64 2, !dbg !47
  %566 = tail call float @llvm.amdgcn.exp2.f32(float %565), !dbg !47
  %local.reduce60 = fadd float %530, %566, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1905 = tail call float asm sideeffect "", "=v,0"(float %local.reduce60) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %567 = insertelement <4 x float> %564, float %566, i64 2, !dbg !48
  %568 = extractelement <4 x float> %558, i64 3, !dbg !47
  %569 = tail call float @llvm.amdgcn.exp2.f32(float %568), !dbg !47
  %local.reduce61 = fadd float %533, %569, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1906 = tail call float asm sideeffect "", "=v,0"(float %local.reduce61) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %570 = insertelement <4 x float> %567, float %569, i64 3, !dbg !48
  %math.result390 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %570) #8, !dbg !48, !frisk.register_lifetimes !5
  %571 = fptrunc <4 x float> %math.result390 to <4 x half>, !dbg !48
  %math.packed391 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %571) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %572 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %540, <4 x float> %536) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %573 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %556, <4 x float> %539) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %574 = shufflevector <8 x half> %486, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %575 = fsub <4 x float> %337, %461, !dbg !40
  %576 = extractelement <4 x float> %575, i64 0, !dbg !47
  %577 = tail call float @llvm.amdgcn.exp2.f32(float %576), !dbg !47
  %local.reduce93 = fadd float %577, %k0.sum.anchor, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1907 = tail call float asm sideeffect "", "=v,0"(float %local.reduce93) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %578 = insertelement <4 x float> poison, float %577, i64 0, !dbg !48
  %579 = extractelement <4 x float> %575, i64 1, !dbg !47
  %580 = tail call float @llvm.amdgcn.exp2.f32(float %579), !dbg !47
  %local.reduce94 = fadd float %580, %k0.sum.anchor1900, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1908 = tail call float asm sideeffect "", "=v,0"(float %local.reduce94) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %581 = insertelement <4 x float> %578, float %580, i64 1, !dbg !48
  %582 = extractelement <4 x float> %575, i64 2, !dbg !47
  %583 = tail call float @llvm.amdgcn.exp2.f32(float %582), !dbg !47
  %local.reduce95 = fadd float %583, %k0.sum.anchor1901, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1909 = tail call float asm sideeffect "", "=v,0"(float %local.reduce95) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %584 = insertelement <4 x float> %581, float %583, i64 2, !dbg !48
  %585 = extractelement <4 x float> %575, i64 3, !dbg !47
  %586 = tail call float @llvm.amdgcn.exp2.f32(float %585), !dbg !47
  %local.reduce96 = fadd float %586, %k0.sum.anchor1902, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1910 = tail call float asm sideeffect "", "=v,0"(float %local.reduce96) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %587 = insertelement <4 x float> %584, float %586, i64 3, !dbg !48
  %math.result376 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %587) #8, !dbg !48, !frisk.register_lifetimes !5
  %588 = fptrunc <4 x float> %math.result376 to <4 x half>, !dbg !48
  %math.packed377 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %588) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %589 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %574, <4 x float> %555) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %590 = shufflevector <8 x half> %486, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %591 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %590, <4 x float> %557) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %592 = fsub <4 x float> %390, %462, !dbg !40
  %593 = extractelement <4 x float> %592, i64 0, !dbg !47
  %594 = tail call float @llvm.amdgcn.exp2.f32(float %593), !dbg !47
  %local.reduce62 = fadd float %594, %k0.sum.anchor1903, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1911 = tail call float asm sideeffect "", "=v,0"(float %local.reduce62) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %595 = insertelement <4 x float> poison, float %594, i64 0, !dbg !48
  %596 = extractelement <4 x float> %592, i64 1, !dbg !47
  %597 = tail call float @llvm.amdgcn.exp2.f32(float %596), !dbg !47
  %local.reduce63 = fadd float %597, %k0.sum.anchor1904, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1912 = tail call float asm sideeffect "", "=v,0"(float %local.reduce63) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %598 = insertelement <4 x float> %595, float %597, i64 1, !dbg !48
  %599 = extractelement <4 x float> %592, i64 2, !dbg !47
  %600 = tail call float @llvm.amdgcn.exp2.f32(float %599), !dbg !47
  %local.reduce64 = fadd float %600, %k0.sum.anchor1905, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1913 = tail call float asm sideeffect "", "=v,0"(float %local.reduce64) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %601 = insertelement <4 x float> %598, float %600, i64 2, !dbg !48
  %602 = extractelement <4 x float> %592, i64 3, !dbg !47
  %603 = tail call float @llvm.amdgcn.exp2.f32(float %602), !dbg !47
  %local.reduce65 = fadd float %603, %k0.sum.anchor1906, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1914 = tail call float asm sideeffect "", "=v,0"(float %local.reduce65) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %604 = insertelement <4 x float> %601, float %603, i64 3, !dbg !48
  %math.result392 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %604) #8, !dbg !48, !frisk.register_lifetimes !5
  %605 = fptrunc <4 x float> %math.result392 to <4 x half>, !dbg !48
  %math.packed393 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %605) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %606 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %574, <4 x float> %572) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %607 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %590, <4 x float> %573) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %608 = shufflevector <8 x half> %488, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %609 = fsub <4 x float> %347, %461, !dbg !40
  %610 = extractelement <4 x float> %609, i64 0, !dbg !47
  %611 = tail call float @llvm.amdgcn.exp2.f32(float %610), !dbg !47
  %local.reduce97 = fadd float %611, %k0.sum.anchor1907, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1915 = tail call float asm sideeffect "", "=v,0"(float %local.reduce97) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %612 = insertelement <4 x float> poison, float %611, i64 0, !dbg !48
  %613 = extractelement <4 x float> %609, i64 1, !dbg !47
  %614 = tail call float @llvm.amdgcn.exp2.f32(float %613), !dbg !47
  %local.reduce98 = fadd float %614, %k0.sum.anchor1908, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1916 = tail call float asm sideeffect "", "=v,0"(float %local.reduce98) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %615 = insertelement <4 x float> %612, float %614, i64 1, !dbg !48
  %616 = extractelement <4 x float> %609, i64 2, !dbg !47
  %617 = tail call float @llvm.amdgcn.exp2.f32(float %616), !dbg !47
  %local.reduce99 = fadd float %617, %k0.sum.anchor1909, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1917 = tail call float asm sideeffect "", "=v,0"(float %local.reduce99) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %618 = insertelement <4 x float> %615, float %617, i64 2, !dbg !48
  %619 = extractelement <4 x float> %609, i64 3, !dbg !47
  %620 = tail call float @llvm.amdgcn.exp2.f32(float %619), !dbg !47
  %local.reduce100 = fadd float %620, %k0.sum.anchor1910, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1918 = tail call float asm sideeffect "", "=v,0"(float %local.reduce100) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %621 = insertelement <4 x float> %618, float %620, i64 3, !dbg !48
  %math.result378 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %621) #8, !dbg !48, !frisk.register_lifetimes !5
  %622 = fptrunc <4 x float> %math.result378 to <4 x half>, !dbg !48
  %math.packed379 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %622) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %623 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %608, <4 x float> %589) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %624 = shufflevector <8 x half> %488, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %625 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %624, <4 x float> %591) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %626 = fsub <4 x float> %396, %462, !dbg !40
  %627 = extractelement <4 x float> %626, i64 0, !dbg !47
  %628 = tail call float @llvm.amdgcn.exp2.f32(float %627), !dbg !47
  %local.reduce66 = fadd float %628, %k0.sum.anchor1911, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1919 = tail call float asm sideeffect "", "=v,0"(float %local.reduce66) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %629 = insertelement <4 x float> poison, float %628, i64 0, !dbg !48
  %630 = extractelement <4 x float> %626, i64 1, !dbg !47
  %631 = tail call float @llvm.amdgcn.exp2.f32(float %630), !dbg !47
  %local.reduce67 = fadd float %631, %k0.sum.anchor1912, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1920 = tail call float asm sideeffect "", "=v,0"(float %local.reduce67) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %632 = insertelement <4 x float> %629, float %631, i64 1, !dbg !48
  %633 = extractelement <4 x float> %626, i64 2, !dbg !47
  %634 = tail call float @llvm.amdgcn.exp2.f32(float %633), !dbg !47
  %local.reduce68 = fadd float %634, %k0.sum.anchor1913, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1921 = tail call float asm sideeffect "", "=v,0"(float %local.reduce68) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %635 = insertelement <4 x float> %632, float %634, i64 2, !dbg !48
  %636 = extractelement <4 x float> %626, i64 3, !dbg !47
  %637 = tail call float @llvm.amdgcn.exp2.f32(float %636), !dbg !47
  %local.reduce69 = fadd float %637, %k0.sum.anchor1914, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1922 = tail call float asm sideeffect "", "=v,0"(float %local.reduce69) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %638 = insertelement <4 x float> %635, float %637, i64 3, !dbg !48
  %math.result394 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %638) #8, !dbg !48, !frisk.register_lifetimes !5
  %639 = fptrunc <4 x float> %math.result394 to <4 x half>, !dbg !48
  %math.packed395 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %639) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %640 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %608, <4 x float> %606) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %641 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %624, <4 x float> %607) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %642 = shufflevector <8 x half> %490, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %643 = fsub <4 x float> %353, %461, !dbg !40
  %644 = extractelement <4 x float> %643, i64 0, !dbg !47
  %645 = tail call float @llvm.amdgcn.exp2.f32(float %644), !dbg !47
  %local.reduce101 = fadd float %645, %k0.sum.anchor1915, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1923 = tail call float asm sideeffect "", "=v,0"(float %local.reduce101) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %646 = insertelement <4 x float> poison, float %645, i64 0, !dbg !48
  %647 = extractelement <4 x float> %643, i64 1, !dbg !47
  %648 = tail call float @llvm.amdgcn.exp2.f32(float %647), !dbg !47
  %local.reduce102 = fadd float %648, %k0.sum.anchor1916, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1924 = tail call float asm sideeffect "", "=v,0"(float %local.reduce102) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %649 = insertelement <4 x float> %646, float %648, i64 1, !dbg !48
  %650 = extractelement <4 x float> %643, i64 2, !dbg !47
  %651 = tail call float @llvm.amdgcn.exp2.f32(float %650), !dbg !47
  %local.reduce103 = fadd float %651, %k0.sum.anchor1917, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1925 = tail call float asm sideeffect "", "=v,0"(float %local.reduce103) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %652 = insertelement <4 x float> %649, float %651, i64 2, !dbg !48
  %653 = extractelement <4 x float> %643, i64 3, !dbg !47
  %654 = tail call float @llvm.amdgcn.exp2.f32(float %653), !dbg !47
  %local.reduce104 = fadd float %654, %k0.sum.anchor1918, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1926 = tail call float asm sideeffect "", "=v,0"(float %local.reduce104) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %655 = insertelement <4 x float> %652, float %654, i64 3, !dbg !48
  %math.result380 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %655) #8, !dbg !48, !frisk.register_lifetimes !5
  %656 = fptrunc <4 x float> %math.result380 to <4 x half>, !dbg !48
  %math.packed381 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %656) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %657 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %642, <4 x float> %623) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %658 = shufflevector <8 x half> %490, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %659 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %658, <4 x float> %625) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %660 = fsub <4 x float> %402, %462, !dbg !40
  %661 = extractelement <4 x float> %660, i64 0, !dbg !47
  %662 = tail call float @llvm.amdgcn.exp2.f32(float %661), !dbg !47
  %local.reduce70 = fadd float %662, %k0.sum.anchor1919, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1927 = tail call float asm sideeffect "", "=v,0"(float %local.reduce70) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %663 = insertelement <4 x float> poison, float %662, i64 0, !dbg !48
  %664 = extractelement <4 x float> %660, i64 1, !dbg !47
  %665 = tail call float @llvm.amdgcn.exp2.f32(float %664), !dbg !47
  %local.reduce71 = fadd float %665, %k0.sum.anchor1920, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1928 = tail call float asm sideeffect "", "=v,0"(float %local.reduce71) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %666 = insertelement <4 x float> %663, float %665, i64 1, !dbg !48
  %667 = extractelement <4 x float> %660, i64 2, !dbg !47
  %668 = tail call float @llvm.amdgcn.exp2.f32(float %667), !dbg !47
  %local.reduce72 = fadd float %668, %k0.sum.anchor1921, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1929 = tail call float asm sideeffect "", "=v,0"(float %local.reduce72) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %669 = insertelement <4 x float> %666, float %668, i64 2, !dbg !48
  %670 = extractelement <4 x float> %660, i64 3, !dbg !47
  %671 = tail call float @llvm.amdgcn.exp2.f32(float %670), !dbg !47
  %local.reduce73 = fadd float %671, %k0.sum.anchor1922, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1930 = tail call float asm sideeffect "", "=v,0"(float %local.reduce73) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %672 = insertelement <4 x float> %669, float %671, i64 3, !dbg !48
  %math.result396 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %672) #8, !dbg !48, !frisk.register_lifetimes !5
  %673 = fptrunc <4 x float> %math.result396 to <4 x half>, !dbg !48
  %math.packed397 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %673) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %674 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %642, <4 x float> %640) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %675 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %658, <4 x float> %641) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %676 = shufflevector <8 x half> %492, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %677 = fsub <4 x float> %359, %461, !dbg !40
  %678 = extractelement <4 x float> %677, i64 0, !dbg !47
  %679 = tail call float @llvm.amdgcn.exp2.f32(float %678), !dbg !47
  %local.reduce105 = fadd float %679, %k0.sum.anchor1923, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1931 = tail call float asm sideeffect "", "=v,0"(float %local.reduce105) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %680 = insertelement <4 x float> poison, float %679, i64 0, !dbg !48
  %681 = extractelement <4 x float> %677, i64 1, !dbg !47
  %682 = tail call float @llvm.amdgcn.exp2.f32(float %681), !dbg !47
  %local.reduce106 = fadd float %682, %k0.sum.anchor1924, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1932 = tail call float asm sideeffect "", "=v,0"(float %local.reduce106) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %683 = insertelement <4 x float> %680, float %682, i64 1, !dbg !48
  %684 = extractelement <4 x float> %677, i64 2, !dbg !47
  %685 = tail call float @llvm.amdgcn.exp2.f32(float %684), !dbg !47
  %local.reduce107 = fadd float %685, %k0.sum.anchor1925, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1933 = tail call float asm sideeffect "", "=v,0"(float %local.reduce107) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %686 = insertelement <4 x float> %683, float %685, i64 2, !dbg !48
  %687 = extractelement <4 x float> %677, i64 3, !dbg !47
  %688 = tail call float @llvm.amdgcn.exp2.f32(float %687), !dbg !47
  %local.reduce108 = fadd float %688, %k0.sum.anchor1926, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1934 = tail call float asm sideeffect "", "=v,0"(float %local.reduce108) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %689 = insertelement <4 x float> %686, float %688, i64 3, !dbg !48
  %math.result382 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %689) #8, !dbg !48, !frisk.register_lifetimes !5
  %690 = fptrunc <4 x float> %math.result382 to <4 x half>, !dbg !48
  %math.packed383 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %690) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %691 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %676, <4 x float> %657) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %692 = shufflevector <8 x half> %492, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %693 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %692, <4 x float> %659) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %694 = fsub <4 x float> %408, %462, !dbg !40
  %695 = extractelement <4 x float> %694, i64 0, !dbg !47
  %696 = tail call float @llvm.amdgcn.exp2.f32(float %695), !dbg !47
  %local.reduce74 = fadd float %696, %k0.sum.anchor1927, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1935 = tail call float asm sideeffect "", "=v,0"(float %local.reduce74) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %697 = insertelement <4 x float> poison, float %696, i64 0, !dbg !48
  %698 = extractelement <4 x float> %694, i64 1, !dbg !47
  %699 = tail call float @llvm.amdgcn.exp2.f32(float %698), !dbg !47
  %local.reduce75 = fadd float %699, %k0.sum.anchor1928, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1936 = tail call float asm sideeffect "", "=v,0"(float %local.reduce75) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %700 = insertelement <4 x float> %697, float %699, i64 1, !dbg !48
  %701 = extractelement <4 x float> %694, i64 2, !dbg !47
  %702 = tail call float @llvm.amdgcn.exp2.f32(float %701), !dbg !47
  %local.reduce76 = fadd float %702, %k0.sum.anchor1929, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1937 = tail call float asm sideeffect "", "=v,0"(float %local.reduce76) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %703 = insertelement <4 x float> %700, float %702, i64 2, !dbg !48
  %704 = extractelement <4 x float> %694, i64 3, !dbg !47
  %705 = tail call float @llvm.amdgcn.exp2.f32(float %704), !dbg !47
  %local.reduce77 = fadd float %705, %k0.sum.anchor1930, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1938 = tail call float asm sideeffect "", "=v,0"(float %local.reduce77) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %706 = insertelement <4 x float> %703, float %705, i64 3, !dbg !48
  %math.result398 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %706) #8, !dbg !48, !frisk.register_lifetimes !5
  %707 = fptrunc <4 x float> %math.result398 to <4 x half>, !dbg !48
  %math.packed399 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %707) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %708 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %676, <4 x float> %674) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %709 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %692, <4 x float> %675) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %710 = shufflevector <8 x half> %494, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %711 = fsub <4 x float> %365, %461, !dbg !40
  %712 = extractelement <4 x float> %711, i64 0, !dbg !47
  %713 = tail call float @llvm.amdgcn.exp2.f32(float %712), !dbg !47
  %local.reduce109 = fadd float %713, %k0.sum.anchor1931, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1939 = tail call float asm sideeffect "", "=v,0"(float %local.reduce109) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %714 = insertelement <4 x float> poison, float %713, i64 0, !dbg !48
  %715 = extractelement <4 x float> %711, i64 1, !dbg !47
  %716 = tail call float @llvm.amdgcn.exp2.f32(float %715), !dbg !47
  %local.reduce110 = fadd float %716, %k0.sum.anchor1932, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1940 = tail call float asm sideeffect "", "=v,0"(float %local.reduce110) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %717 = insertelement <4 x float> %714, float %716, i64 1, !dbg !48
  %718 = extractelement <4 x float> %711, i64 2, !dbg !47
  %719 = tail call float @llvm.amdgcn.exp2.f32(float %718), !dbg !47
  %local.reduce111 = fadd float %719, %k0.sum.anchor1933, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1941 = tail call float asm sideeffect "", "=v,0"(float %local.reduce111) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %720 = insertelement <4 x float> %717, float %719, i64 2, !dbg !48
  %721 = extractelement <4 x float> %711, i64 3, !dbg !47
  %722 = tail call float @llvm.amdgcn.exp2.f32(float %721), !dbg !47
  %local.reduce112 = fadd float %722, %k0.sum.anchor1934, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1942 = tail call float asm sideeffect "", "=v,0"(float %local.reduce112) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %723 = insertelement <4 x float> %720, float %722, i64 3, !dbg !48
  %math.result384 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %723) #8, !dbg !48, !frisk.register_lifetimes !5
  %724 = fptrunc <4 x float> %math.result384 to <4 x half>, !dbg !48
  %math.packed385 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %724) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %725 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %710, <4 x float> %691) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %726 = shufflevector <8 x half> %494, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %727 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %726, <4 x float> %693) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %728 = fsub <4 x float> %414, %462, !dbg !40
  %729 = extractelement <4 x float> %728, i64 0, !dbg !47
  %730 = tail call float @llvm.amdgcn.exp2.f32(float %729), !dbg !47
  %local.reduce78 = fadd float %730, %k0.sum.anchor1935, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1943 = tail call float asm sideeffect "", "=v,0"(float %local.reduce78) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %731 = insertelement <4 x float> poison, float %730, i64 0, !dbg !48
  %732 = extractelement <4 x float> %728, i64 1, !dbg !47
  %733 = tail call float @llvm.amdgcn.exp2.f32(float %732), !dbg !47
  %local.reduce79 = fadd float %733, %k0.sum.anchor1936, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1944 = tail call float asm sideeffect "", "=v,0"(float %local.reduce79) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %734 = insertelement <4 x float> %731, float %733, i64 1, !dbg !48
  %735 = extractelement <4 x float> %728, i64 2, !dbg !47
  %736 = tail call float @llvm.amdgcn.exp2.f32(float %735), !dbg !47
  %local.reduce80 = fadd float %736, %k0.sum.anchor1937, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1945 = tail call float asm sideeffect "", "=v,0"(float %local.reduce80) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %737 = insertelement <4 x float> %734, float %736, i64 2, !dbg !48
  %738 = extractelement <4 x float> %728, i64 3, !dbg !47
  %739 = tail call float @llvm.amdgcn.exp2.f32(float %738), !dbg !47
  %local.reduce81 = fadd float %739, %k0.sum.anchor1938, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1946 = tail call float asm sideeffect "", "=v,0"(float %local.reduce81) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %740 = insertelement <4 x float> %737, float %739, i64 3, !dbg !48
  %math.result400 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %740) #8, !dbg !48, !frisk.register_lifetimes !5
  %741 = fptrunc <4 x float> %math.result400 to <4 x half>, !dbg !48
  %math.packed401 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %741) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %742 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %710, <4 x float> %708) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %743 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %726, <4 x float> %709) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %744 = bitcast <4 x i32> %495 to <8 x half>, !dbg !45
  %745 = shufflevector <8 x half> %744, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %746 = fsub <4 x float> %371, %461, !dbg !40
  %747 = extractelement <4 x float> %746, i64 0, !dbg !47
  %748 = tail call float @llvm.amdgcn.exp2.f32(float %747), !dbg !47
  %local.reduce113 = fadd float %748, %k0.sum.anchor1939, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1947 = tail call float asm sideeffect "", "=v,0"(float %local.reduce113) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %749 = insertelement <4 x float> poison, float %748, i64 0, !dbg !48
  %750 = extractelement <4 x float> %746, i64 1, !dbg !47
  %751 = tail call float @llvm.amdgcn.exp2.f32(float %750), !dbg !47
  %local.reduce114 = fadd float %751, %k0.sum.anchor1940, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1948 = tail call float asm sideeffect "", "=v,0"(float %local.reduce114) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce117 = fadd float %k0.sum.anchor1947, %k0.sum.anchor1948, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1949 = tail call float asm sideeffect "", "=v,0"(float %local.reduce117) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %752 = insertelement <4 x float> %749, float %751, i64 1, !dbg !48
  %753 = extractelement <4 x float> %746, i64 2, !dbg !47
  %754 = tail call float @llvm.amdgcn.exp2.f32(float %753), !dbg !47
  %local.reduce115 = fadd float %754, %k0.sum.anchor1941, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1950 = tail call float asm sideeffect "", "=v,0"(float %local.reduce115) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %755 = insertelement <4 x float> %752, float %754, i64 2, !dbg !48
  %756 = extractelement <4 x float> %746, i64 3, !dbg !47
  %757 = tail call float @llvm.amdgcn.exp2.f32(float %756), !dbg !47
  %local.reduce116 = fadd float %757, %k0.sum.anchor1942, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1951 = tail call float asm sideeffect "", "=v,0"(float %local.reduce116) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce118 = fadd float %k0.sum.anchor1950, %k0.sum.anchor1951, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1952 = tail call float asm sideeffect "", "=v,0"(float %local.reduce118) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %758 = fadd float %k0.sum.anchor1949, %k0.sum.anchor1952, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1953 = tail call float asm sideeffect "", "=v,0"(float %758) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %759 = fadd float %470, %k0.sum.anchor1953, !dbg !49
  %k0.sum.anchor1954 = tail call float asm sideeffect "", "=v,0"(float %759) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %760 = insertelement <1 x float> poison, float %k0.sum.anchor1954, i64 0, !dbg !11
  %761 = insertvalue [2 x <1 x float>] poison, <1 x float> %760, 0, !dbg !11
  %762 = insertelement <4 x float> %755, float %757, i64 3, !dbg !48
  %math.result386 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %762) #8, !dbg !48, !frisk.register_lifetimes !5
  %763 = fptrunc <4 x float> %math.result386 to <4 x half>, !dbg !48
  %math.packed387 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %763) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %764 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %745, <4 x float> %725) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %765 = shufflevector <8 x half> %744, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %766 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %765, <4 x float> %727) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %767 = fsub <4 x float> %420, %462, !dbg !40
  %768 = extractelement <4 x float> %767, i64 0, !dbg !47
  %769 = tail call float @llvm.amdgcn.exp2.f32(float %768), !dbg !47
  %local.reduce82 = fadd float %769, %k0.sum.anchor1943, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1955 = tail call float asm sideeffect "", "=v,0"(float %local.reduce82) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %770 = insertelement <4 x float> poison, float %769, i64 0, !dbg !48
  %771 = extractelement <4 x float> %767, i64 1, !dbg !47
  %772 = tail call float @llvm.amdgcn.exp2.f32(float %771), !dbg !47
  %local.reduce83 = fadd float %772, %k0.sum.anchor1944, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1956 = tail call float asm sideeffect "", "=v,0"(float %local.reduce83) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce86 = fadd float %k0.sum.anchor1955, %k0.sum.anchor1956, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1957 = tail call float asm sideeffect "", "=v,0"(float %local.reduce86) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %773 = insertelement <4 x float> %770, float %772, i64 1, !dbg !48
  %774 = extractelement <4 x float> %767, i64 2, !dbg !47
  %775 = tail call float @llvm.amdgcn.exp2.f32(float %774), !dbg !47
  %local.reduce84 = fadd float %775, %k0.sum.anchor1945, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1958 = tail call float asm sideeffect "", "=v,0"(float %local.reduce84) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %776 = insertelement <4 x float> %773, float %775, i64 2, !dbg !48
  %777 = extractelement <4 x float> %767, i64 3, !dbg !47
  %778 = tail call float @llvm.amdgcn.exp2.f32(float %777), !dbg !47
  %local.reduce85 = fadd float %778, %k0.sum.anchor1946, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1959 = tail call float asm sideeffect "", "=v,0"(float %local.reduce85) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %local.reduce87 = fadd float %k0.sum.anchor1958, %k0.sum.anchor1959, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1960 = tail call float asm sideeffect "", "=v,0"(float %local.reduce87) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %779 = fadd float %k0.sum.anchor1957, %k0.sum.anchor1960, !dbg !49, !frisk.local_reduction !5
  %k0.sum.anchor1961 = tail call float asm sideeffect "", "=v,0"(float %779) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %780 = fadd float %478, %k0.sum.anchor1961, !dbg !49
  %k0.sum.anchor1962 = tail call float asm sideeffect "", "=v,0"(float %780) #8, !dbg !49, !frisk.cross_bn_sum_anchor !5
  %781 = insertelement <1 x float> poison, float %k0.sum.anchor1962, i64 0, !dbg !11
  %782 = insertvalue [2 x <1 x float>] %761, <1 x float> %781, 1, !dbg !11
  %783 = insertelement <4 x float> %776, float %778, i64 3, !dbg !48
  %math.result402 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %783) #8, !dbg !48, !frisk.register_lifetimes !5
  %784 = fptrunc <4 x float> %math.result402 to <4 x half>, !dbg !48
  %math.packed403 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %784) #8, !dbg !49, !frisk.register_lifetimes !5
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45, !frisk.softmax_fragments !5
  %785 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %745, <4 x float> %742) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %786 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %765, <4 x float> %743) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !37
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %vdb.address1462 = or disjoint i64 %address.local921, %203, !dbg !37
  %vdb.address1463 = trunc nuw nsw i64 %vdb.address1462 to i32, !dbg !37
  %vdb.address1464 = shl nuw nsw i32 %vdb.address1463, 1, !dbg !37
  %vdb.address1471 = or disjoint i64 %address.local927, %203, !dbg !37
  %vdb.address1472 = trunc nuw nsw i64 %vdb.address1471 to i32, !dbg !37
  %vdb.address1473 = shl nuw nsw i32 %vdb.address1472, 1, !dbg !37
  %787 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %vdb.address1464, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %788 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %vdb.address1473, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %vdb.read = getelementptr i8, ptr addrspace(3) %address.local935, i64 8192, !dbg !45
  %789 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %790 = bitcast <4 x i32> %789 to <8 x half>, !dbg !45
  %vdb.read1453 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8448, !dbg !45
  %791 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %792 = bitcast <4 x i32> %791 to <8 x half>, !dbg !45
  %vdb.read1454 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8704, !dbg !45
  %793 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %794 = bitcast <4 x i32> %793 to <8 x half>, !dbg !45
  %vdb.read1455 = getelementptr i8, ptr addrspace(3) %address.local944, i64 8960, !dbg !45
  %795 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %796 = bitcast <4 x i32> %795 to <8 x half>, !dbg !45
  %vdb.read1456 = getelementptr i8, ptr addrspace(3) %address.local935, i64 12288, !dbg !45
  %797 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %798 = bitcast <4 x i32> %797 to <8 x half>, !dbg !45
  %vdb.read1457 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12544, !dbg !45
  %799 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %800 = bitcast <4 x i32> %799 to <8 x half>, !dbg !45
  %vdb.read1458 = getelementptr i8, ptr addrspace(3) %address.local944, i64 12800, !dbg !45
  %801 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %802 = bitcast <4 x i32> %801 to <8 x half>, !dbg !45
  %vdb.read1459 = getelementptr i8, ptr addrspace(3) %address.local944, i64 13056, !dbg !45
  %803 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %804 = shufflevector <8 x half> %790, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %805 = extractvalue [2 x <8 x float>] %104, 0, !dbg !43
  %806 = shufflevector <8 x float> %805, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %807 = fmul <4 x float> %806, %467, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %808 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %804, <4 x float> %807) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %809 = shufflevector <8 x half> %790, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %810 = shufflevector <8 x float> %805, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %811 = fmul <4 x float> %810, %467, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %812 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %809, <4 x float> %811) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %813 = extractvalue [2 x <8 x float>] %104, 1, !dbg !43
  %814 = shufflevector <8 x float> %813, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %815 = fmul <4 x float> %814, %475, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %816 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %804, <4 x float> %815) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %817 = shufflevector <8 x float> %813, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %818 = fmul <4 x float> %817, %475, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %819 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %809, <4 x float> %818) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %820 = shufflevector <8 x half> %792, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %821 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %820, <4 x float> %808) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %822 = shufflevector <8 x half> %792, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %823 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %822, <4 x float> %812) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %824 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %820, <4 x float> %816) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %825 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %822, <4 x float> %819) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %826 = shufflevector <8 x half> %794, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %827 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %826, <4 x float> %821) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %828 = shufflevector <8 x half> %794, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %829 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %828, <4 x float> %823) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %830 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %826, <4 x float> %824) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %831 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %828, <4 x float> %825) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %832 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %833 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %832, <4 x float> %827) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %834 = shufflevector <8 x half> %796, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %835 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %834, <4 x float> %829) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %836 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %832, <4 x float> %830) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %837 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %834, <4 x float> %831) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %838 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %839 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %838, <4 x float> %833) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %840 = shufflevector <8 x half> %798, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %841 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %840, <4 x float> %835) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %842 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %838, <4 x float> %836) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %843 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %840, <4 x float> %837) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %844 = shufflevector <8 x half> %800, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %845 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %844, <4 x float> %839) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %846 = shufflevector <8 x half> %800, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %847 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %846, <4 x float> %841) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %848 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %844, <4 x float> %842) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %849 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %846, <4 x float> %843) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %850 = shufflevector <8 x half> %802, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %851 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %850, <4 x float> %845) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %852 = shufflevector <8 x half> %802, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %853 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %852, <4 x float> %847) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %854 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %850, <4 x float> %848) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %855 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %852, <4 x float> %849) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %856 = bitcast <4 x i32> %803 to <8 x half>, !dbg !45
  %857 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %858 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %857, <4 x float> %851) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %859 = shufflevector <8 x half> %856, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %860 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %859, <4 x float> %853) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %861 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %857, <4 x float> %854) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %862 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %859, <4 x float> %855) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !37
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %vdb.address1480 = or disjoint i64 %address.local921, %256, !dbg !37
  %vdb.address1481 = trunc nuw nsw i64 %vdb.address1480 to i32, !dbg !37
  %vdb.address1482 = shl nuw nsw i32 %vdb.address1481, 1, !dbg !37
  %vdb.address1489 = or disjoint i64 %address.local927, %256, !dbg !37
  %vdb.address1490 = trunc nuw nsw i64 %vdb.address1489 to i32, !dbg !37
  %vdb.address1491 = shl nuw nsw i32 %vdb.address1490, 1, !dbg !37
  %863 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base, i32 %vdb.address1482, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %864 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %vdb.dma.base1452, i32 %vdb.address1491, <4 x i32> %dma.buffer.rsrc1349) #9, !dbg !37
  %865 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local935, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %866 = bitcast <4 x i32> %865 to <8 x half>, !dbg !45
  %867 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local945, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %868 = bitcast <4 x i32> %867 to <8 x half>, !dbg !45
  %869 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local949, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %870 = bitcast <4 x i32> %869 to <8 x half>, !dbg !45
  %871 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local953, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %872 = bitcast <4 x i32> %871 to <8 x half>, !dbg !45
  %873 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local955, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %874 = bitcast <4 x i32> %873 to <8 x half>, !dbg !45
  %875 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local959, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %876 = bitcast <4 x i32> %875 to <8 x half>, !dbg !45
  %877 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local963, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %878 = bitcast <4 x i32> %877 to <8 x half>, !dbg !45
  %879 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local967, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %880 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %881 = extractvalue [2 x <8 x float>] %103, 0, !dbg !43
  %882 = shufflevector <8 x float> %881, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %883 = fmul <4 x float> %882, %467, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %884 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %880, <4 x float> %883) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %885 = shufflevector <8 x half> %866, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %886 = shufflevector <8 x float> %881, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %887 = fmul <4 x float> %886, %467, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %888 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %885, <4 x float> %887) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %889 = extractvalue [2 x <8 x float>] %103, 1, !dbg !43
  %890 = shufflevector <8 x float> %889, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %891 = fmul <4 x float> %890, %475, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %892 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %880, <4 x float> %891) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %893 = shufflevector <8 x float> %889, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %894 = fmul <4 x float> %893, %475, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %885, <4 x float> %894) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %896 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %897 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %896, <4 x float> %884) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %898 = shufflevector <8 x half> %868, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %899 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %898, <4 x float> %888) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %896, <4 x float> %892) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %901 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %898, <4 x float> %895) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %902 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %903 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %902, <4 x float> %897) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %904 = shufflevector <8 x half> %870, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %905 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %904, <4 x float> %899) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %902, <4 x float> %900) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %907 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %904, <4 x float> %901) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %908 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %909 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %908, <4 x float> %903) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %910 = shufflevector <8 x half> %872, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %911 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %910, <4 x float> %905) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %908, <4 x float> %906) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %913 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %910, <4 x float> %907) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %914 = shufflevector <8 x half> %874, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %915 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %914, <4 x float> %909) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %916 = shufflevector <8 x half> %874, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %917 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %916, <4 x float> %911) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %914, <4 x float> %912) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %919 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %916, <4 x float> %913) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %920 = shufflevector <8 x half> %876, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %921 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %920, <4 x float> %915) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %922 = shufflevector <8 x half> %876, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %923 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %922, <4 x float> %917) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %920, <4 x float> %918) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %925 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %922, <4 x float> %919) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %926 = shufflevector <8 x half> %878, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %927 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %926, <4 x float> %921) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %928 = shufflevector <8 x half> %878, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %929 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %928, <4 x float> %923) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %926, <4 x float> %924) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %931 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %928, <4 x float> %925) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %932 = bitcast <4 x i32> %879 to <8 x half>, !dbg !45
  %933 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %934 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %933, <4 x float> %927) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %935 = shufflevector <8 x half> %932, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %936 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %935, <4 x float> %929) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %937 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %933, <4 x float> %930) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %938 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %935, <4 x float> %931) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !37
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !37
  fence syncscope("workgroup") release, !dbg !37
  tail call void @llvm.amdgcn.s.barrier(), !dbg !37
  fence syncscope("workgroup") acquire, !dbg !37
  %939 = add nuw nsw i64 %108, 1, !dbg !11
  %k0.has.next.not = icmp eq i64 %108, 31, !dbg !45
  br i1 %k0.has.next.not, label %k0.join, label %k0.issue, !dbg !45

k0.issue:                                         ; preds = %101
  %k0.next. = shl nuw nsw i64 %939, 14, !dbg !13
  %k0.next.1975 = add nuw nsw i64 %k0.next., %k0.prime.1964, !dbg !13
  %k0.next.kdma.source1516 = or disjoint i64 %k0.next.1975, %k0.prime.kdma.source1515, !dbg !13
  %k0.next.kdma.source1517 = trunc nuw nsw i64 %k0.next.kdma.source1516 to i32, !dbg !13
  %k0.next.kdma.source1518 = shl nuw nsw i32 %k0.next.kdma.source1517, 1, !dbg !13
  %k0.next.kdma.offset = or disjoint i32 %k0.next.kdma.source1518, %kdma.intra, !dbg !13
  %k0.next.1976 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1374, i32 %k0.next.kdma.offset, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1530 = or disjoint i64 %k0.next.1975, %k0.prime.kdma.source1529, !dbg !13
  %k0.next.kdma.source1531 = trunc nuw nsw i64 %k0.next.kdma.source1530 to i32, !dbg !13
  %k0.next.kdma.source1532 = shl nuw nsw i32 %k0.next.kdma.source1531, 1, !dbg !13
  %k0.next.kdma.offset1533 = or disjoint i32 %k0.next.kdma.source1532, %kdma.intra.hi, !dbg !13
  %k0.next.1977 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1534, i32 %k0.next.kdma.offset1533, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1547 = or disjoint i64 %k0.next.1975, %k0.prime.kdma.source1546, !dbg !13
  %k0.next.kdma.source1548 = trunc nuw nsw i64 %k0.next.kdma.source1547 to i32, !dbg !13
  %k0.next.kdma.source1549 = shl nuw nsw i32 %k0.next.kdma.source1548, 1, !dbg !13
  %k0.next.kdma.offset1550 = or disjoint i32 %k0.next.kdma.source1549, %kdma.intra, !dbg !13
  %k0.next.1978 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1371, i32 %k0.next.kdma.offset1550, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  %k0.next.kdma.source1564 = or disjoint i64 %k0.next.1975, %k0.prime.kdma.source1563, !dbg !13
  %k0.next.kdma.source1565 = trunc nuw nsw i64 %k0.next.kdma.source1564 to i32, !dbg !13
  %k0.next.kdma.source1566 = shl nuw nsw i32 %k0.next.kdma.source1565, 1, !dbg !13
  %k0.next.kdma.offset1567 = or disjoint i32 %k0.next.kdma.source1566, %kdma.intra.hi, !dbg !13
  %k0.next.1979 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx2 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %kdma.lds1568, i32 %k0.next.kdma.offset1567, <4 x i32> %k0.prime.dma.buffer.rsrc1373) #9, !dbg !13, !frisk.cross_bn_k0 !5
  br label %k0.join

k0.join:                                          ; preds = %k0.issue, %101
  %940 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %941 = bitcast <4 x i32> %940 to <8 x half>, !dbg !45
  %942 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1453, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %943 = bitcast <4 x i32> %942 to <8 x half>, !dbg !45
  %944 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1454, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %945 = bitcast <4 x i32> %944 to <8 x half>, !dbg !45
  %946 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1455, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %947 = bitcast <4 x i32> %946 to <8 x half>, !dbg !45
  %948 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1456, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %949 = bitcast <4 x i32> %948 to <8 x half>, !dbg !45
  %950 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1457, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %951 = bitcast <4 x i32> %950 to <8 x half>, !dbg !45
  %952 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1458, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %953 = bitcast <4 x i32> %952 to <8 x half>, !dbg !45
  %954 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %vdb.read1459, i32 0) #11, !dbg !45, !frisk.v_double_buffer !46
  %955 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  %956 = extractvalue [2 x <8 x float>] %102, 0, !dbg !43
  %957 = shufflevector <8 x float> %956, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %958 = fmul <4 x float> %957, %467, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %955, <4 x float> %958) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %960 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  %961 = shufflevector <8 x float> %956, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %962 = fmul <4 x float> %961, %467, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %960, <4 x float> %962) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %964 = extractvalue [2 x <8 x float>] %102, 1, !dbg !43
  %965 = shufflevector <8 x float> %964, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !43
  %966 = fmul <4 x float> %965, %475, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %955, <4 x float> %966) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !43
  %968 = shufflevector <8 x float> %964, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !43
  %969 = fmul <4 x float> %968, %475, !dbg !43
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed389, <4 x half> %960, <4 x float> %969) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %971 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %972 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %971, <4 x float> %959) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %973 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %974 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed375, <4 x half> %973, <4 x float> %963) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %975 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %971, <4 x float> %967) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %976 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed391, <4 x half> %973, <4 x float> %970) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %977 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %978 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %977, <4 x float> %972) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %979 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed377, <4 x half> %979, <4 x float> %974) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %981 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %977, <4 x float> %975) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed393, <4 x half> %979, <4 x float> %976) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %983 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %984 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %983, <4 x float> %978) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %985 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %986 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed379, <4 x half> %985, <4 x float> %980) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %983, <4 x float> %981) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed395, <4 x half> %985, <4 x float> %982) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %989 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %989, <4 x float> %984) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %991 = shufflevector <8 x half> %949, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed381, <4 x half> %991, <4 x float> %986) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %993 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %989, <4 x float> %987) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed397, <4 x half> %991, <4 x float> %988) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %995 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %995, <4 x float> %990) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %997 = shufflevector <8 x half> %951, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed383, <4 x half> %997, <4 x float> %992) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %999 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %995, <4 x float> %993) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed399, <4 x half> %997, <4 x float> %994) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1001 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1001, <4 x float> %996) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1003 = shufflevector <8 x half> %953, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1004 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed385, <4 x half> %1003, <4 x float> %998) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1005 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1001, <4 x float> %999) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1006 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed401, <4 x half> %1003, <4 x float> %1000) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1007 = bitcast <4 x i32> %954 to <8 x half>, !dbg !45
  %1008 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1009 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1008, <4 x float> %1002) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1010 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1011 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed387, <4 x half> %1010, <4 x float> %1004) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1012 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1008, <4 x float> %1005) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !45
  %1013 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed403, <4 x half> %1010, <4 x float> %1006) #8, !dbg !45, !frisk.pv_interleave !18
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !11
  %1014 = shufflevector <4 x float> %764, <4 x float> %766, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1015 = insertvalue [2 x <8 x float>] poison, <8 x float> %1014, 0, !dbg !11
  %1016 = shufflevector <4 x float> %785, <4 x float> %786, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1017 = insertvalue [2 x <8 x float>] %1015, <8 x float> %1016, 1, !dbg !11
  %1018 = insertvalue [2 x <1 x float>] poison, <1 x float> %447, 0, !dbg !11
  %1019 = insertvalue [2 x <1 x float>] %1018, <1 x float> %460, 1, !dbg !11
  %1020 = shufflevector <4 x float> %858, <4 x float> %860, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1021 = insertvalue [2 x <8 x float>] poison, <8 x float> %1020, 0, !dbg !11
  %1022 = shufflevector <4 x float> %861, <4 x float> %862, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1023 = insertvalue [2 x <8 x float>] %1021, <8 x float> %1022, 1, !dbg !11
  %1024 = shufflevector <4 x float> %934, <4 x float> %936, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1025 = insertvalue [2 x <8 x float>] poison, <8 x float> %1024, 0, !dbg !11
  %1026 = shufflevector <4 x float> %937, <4 x float> %938, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1027 = insertvalue [2 x <8 x float>] %1025, <8 x float> %1026, 1, !dbg !11
  %1028 = shufflevector <4 x float> %1009, <4 x float> %1011, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1029 = insertvalue [2 x <8 x float>] poison, <8 x float> %1028, 0, !dbg !11
  %1030 = shufflevector <4 x float> %1012, <4 x float> %1013, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !11
  %1031 = insertvalue [2 x <8 x float>] %1029, <8 x float> %1030, 1, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  tail call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %1032 = icmp samesign ult i64 %108, 31, !dbg !11
  br i1 %1032, label %101, label %1033, !dbg !11

1033:                                             ; preds = %k0.join
  %1034 = shl nuw nsw i32 %426, 2, !dbg !11
  %1035 = xor i32 %1034, 64, !dbg !11
  %1036 = bitcast float %k0.sum.anchor1962 to i32, !dbg !11
  %lane.sum.shuffle188 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1035, i32 %1036), !dbg !11
  %1037 = bitcast i32 %lane.sum.shuffle188 to float, !dbg !11
  %lane.sum.final189 = fadd float %k0.sum.anchor1962, %1037, !dbg !11
  %1038 = xor i32 %1034, 128, !dbg !11
  %1039 = bitcast float %lane.sum.final189 to i32, !dbg !11
  %lane.sum.shuffle190 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1038, i32 %1039), !dbg !11
  %1040 = bitcast i32 %lane.sum.shuffle190 to float, !dbg !11
  %lane.sum.final191 = fadd float %lane.sum.final189, %1040, !dbg !11
  %1041 = insertelement <1 x float> poison, float %lane.sum.final191, i64 0, !dbg !11
  %1042 = bitcast float %k0.sum.anchor1954 to i32, !dbg !11
  %lane.sum.shuffle = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1035, i32 %1042), !dbg !11
  %1043 = bitcast i32 %lane.sum.shuffle to float, !dbg !11
  %lane.sum.final = fadd float %k0.sum.anchor1954, %1043, !dbg !11
  %1044 = bitcast float %lane.sum.final to i32, !dbg !11
  %lane.sum.shuffle182 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1038, i32 %1044), !dbg !11
  %1045 = bitcast i32 %lane.sum.shuffle182 to float, !dbg !11
  %lane.sum.final183 = fadd float %lane.sum.final, %1045, !dbg !11
  %1046 = insertelement <1 x float> poison, float %lane.sum.final183, i64 0, !dbg !11
  %1047 = and i32 %kdma.tid, 16, !dbg !11
  %.not = icmp eq i32 %1047, 0, !dbg !11
  %1048 = and i32 %kdma.tid, 32, !dbg !11
  %.not36 = icmp eq i32 %1048, 0, !dbg !11
  %1049 = extractelement <4 x float> %1009, i64 0, !dbg !11
  %1050 = extractelement <4 x float> %1009, i64 1, !dbg !11
  %1051 = extractelement <4 x float> %1009, i64 2, !dbg !11
  %1052 = extractelement <4 x float> %1009, i64 3, !dbg !11
  %1053 = select i1 %.not, float %1050, float %1049, !dbg !11
  %1054 = bitcast float %1053 to i32, !dbg !11
  %1055 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1054), !dbg !11
  %1056 = bitcast i32 %1055 to float, !dbg !11
  %1057 = select i1 %.not, float %1049, float %1056, !dbg !11
  %1058 = select i1 %.not, float %1056, float %1050, !dbg !11
  %1059 = select i1 %.not, float %1052, float %1051, !dbg !11
  %1060 = bitcast float %1059 to i32, !dbg !11
  %1061 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1060), !dbg !11
  %1062 = bitcast i32 %1061 to float, !dbg !11
  %1063 = select i1 %.not, float %1051, float %1062, !dbg !11
  %1064 = select i1 %.not, float %1062, float %1052, !dbg !11
  %1065 = select i1 %.not36, float %1063, float %1057, !dbg !11
  %1066 = bitcast float %1065 to i32, !dbg !11
  %1067 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1066), !dbg !11
  %1068 = bitcast i32 %1067 to float, !dbg !11
  %1069 = select i1 %.not36, float %1064, float %1058, !dbg !11
  %1070 = bitcast float %1069 to i32, !dbg !11
  %1071 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1070), !dbg !11
  %1072 = bitcast i32 %1071 to float, !dbg !11
  %1073 = insertelement <4 x i1> poison, i1 %.not36, i64 0, !dbg !11
  %1074 = shufflevector <4 x i1> %1073, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !11
  %1075 = insertelement <4 x float> poison, float %1057, i64 0, !dbg !11
  %1076 = insertelement <4 x float> %1075, float %1058, i64 1, !dbg !11
  %1077 = insertelement <4 x float> %1076, float %1068, i64 2, !dbg !11
  %1078 = insertelement <4 x float> %1077, float %1072, i64 3, !dbg !11
  %1079 = insertelement <4 x float> poison, float %1068, i64 0, !dbg !11
  %1080 = insertelement <4 x float> %1079, float %1072, i64 1, !dbg !11
  %1081 = insertelement <4 x float> %1080, float %1063, i64 2, !dbg !11
  %1082 = insertelement <4 x float> %1081, float %1064, i64 3, !dbg !11
  %1083 = select <4 x i1> %1074, <4 x float> %1078, <4 x float> %1082, !dbg !11
  %1084 = extractelement <4 x float> %934, i64 0, !dbg !11
  %1085 = extractelement <4 x float> %934, i64 1, !dbg !11
  %1086 = extractelement <4 x float> %934, i64 2, !dbg !11
  %1087 = extractelement <4 x float> %934, i64 3, !dbg !11
  %1088 = select i1 %.not, float %1085, float %1084, !dbg !11
  %1089 = bitcast float %1088 to i32, !dbg !11
  %1090 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1089), !dbg !11
  %1091 = bitcast i32 %1090 to float, !dbg !11
  %1092 = select i1 %.not, float %1084, float %1091, !dbg !11
  %1093 = select i1 %.not, float %1091, float %1085, !dbg !11
  %1094 = select i1 %.not, float %1087, float %1086, !dbg !11
  %1095 = bitcast float %1094 to i32, !dbg !11
  %1096 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1095), !dbg !11
  %1097 = bitcast i32 %1096 to float, !dbg !11
  %1098 = select i1 %.not, float %1086, float %1097, !dbg !11
  %1099 = select i1 %.not, float %1097, float %1087, !dbg !11
  %1100 = select i1 %.not36, float %1098, float %1092, !dbg !11
  %1101 = bitcast float %1100 to i32, !dbg !11
  %1102 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1101), !dbg !11
  %1103 = bitcast i32 %1102 to float, !dbg !11
  %1104 = select i1 %.not36, float %1099, float %1093, !dbg !11
  %1105 = bitcast float %1104 to i32, !dbg !11
  %1106 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1105), !dbg !11
  %1107 = bitcast i32 %1106 to float, !dbg !11
  %1108 = insertelement <4 x float> poison, float %1092, i64 0, !dbg !11
  %1109 = insertelement <4 x float> %1108, float %1093, i64 1, !dbg !11
  %1110 = insertelement <4 x float> %1109, float %1103, i64 2, !dbg !11
  %1111 = insertelement <4 x float> %1110, float %1107, i64 3, !dbg !11
  %1112 = insertelement <4 x float> poison, float %1103, i64 0, !dbg !11
  %1113 = insertelement <4 x float> %1112, float %1107, i64 1, !dbg !11
  %1114 = insertelement <4 x float> %1113, float %1098, i64 2, !dbg !11
  %1115 = insertelement <4 x float> %1114, float %1099, i64 3, !dbg !11
  %1116 = select <4 x i1> %1074, <4 x float> %1111, <4 x float> %1115, !dbg !11
  %1117 = extractelement <4 x float> %858, i64 0, !dbg !11
  %1118 = extractelement <4 x float> %858, i64 1, !dbg !11
  %1119 = extractelement <4 x float> %858, i64 2, !dbg !11
  %1120 = extractelement <4 x float> %858, i64 3, !dbg !11
  %1121 = select i1 %.not, float %1118, float %1117, !dbg !11
  %1122 = bitcast float %1121 to i32, !dbg !11
  %1123 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1122), !dbg !11
  %1124 = bitcast i32 %1123 to float, !dbg !11
  %1125 = select i1 %.not, float %1117, float %1124, !dbg !11
  %1126 = select i1 %.not, float %1124, float %1118, !dbg !11
  %1127 = select i1 %.not, float %1120, float %1119, !dbg !11
  %1128 = bitcast float %1127 to i32, !dbg !11
  %1129 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1128), !dbg !11
  %1130 = bitcast i32 %1129 to float, !dbg !11
  %1131 = select i1 %.not, float %1119, float %1130, !dbg !11
  %1132 = select i1 %.not, float %1130, float %1120, !dbg !11
  %1133 = select i1 %.not36, float %1131, float %1125, !dbg !11
  %1134 = bitcast float %1133 to i32, !dbg !11
  %1135 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1134), !dbg !11
  %1136 = bitcast i32 %1135 to float, !dbg !11
  %1137 = select i1 %.not36, float %1132, float %1126, !dbg !11
  %1138 = bitcast float %1137 to i32, !dbg !11
  %1139 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1138), !dbg !11
  %1140 = bitcast i32 %1139 to float, !dbg !11
  %1141 = insertelement <4 x float> poison, float %1125, i64 0, !dbg !11
  %1142 = insertelement <4 x float> %1141, float %1126, i64 1, !dbg !11
  %1143 = insertelement <4 x float> %1142, float %1136, i64 2, !dbg !11
  %1144 = insertelement <4 x float> %1143, float %1140, i64 3, !dbg !11
  %1145 = insertelement <4 x float> poison, float %1136, i64 0, !dbg !11
  %1146 = insertelement <4 x float> %1145, float %1140, i64 1, !dbg !11
  %1147 = insertelement <4 x float> %1146, float %1131, i64 2, !dbg !11
  %1148 = insertelement <4 x float> %1147, float %1132, i64 3, !dbg !11
  %1149 = select <4 x i1> %1074, <4 x float> %1144, <4 x float> %1148, !dbg !11
  %1150 = extractelement <4 x float> %764, i64 0, !dbg !11
  %1151 = extractelement <4 x float> %764, i64 1, !dbg !11
  %1152 = extractelement <4 x float> %764, i64 2, !dbg !11
  %1153 = extractelement <4 x float> %764, i64 3, !dbg !11
  %1154 = select i1 %.not, float %1151, float %1150, !dbg !11
  %1155 = bitcast float %1154 to i32, !dbg !11
  %1156 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1155), !dbg !11
  %1157 = bitcast i32 %1156 to float, !dbg !11
  %1158 = select i1 %.not, float %1150, float %1157, !dbg !11
  %1159 = select i1 %.not, float %1157, float %1151, !dbg !11
  %1160 = select i1 %.not, float %1153, float %1152, !dbg !11
  %1161 = bitcast float %1160 to i32, !dbg !11
  %1162 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1161), !dbg !11
  %1163 = bitcast i32 %1162 to float, !dbg !11
  %1164 = select i1 %.not, float %1152, float %1163, !dbg !11
  %1165 = select i1 %.not, float %1163, float %1153, !dbg !11
  %1166 = select i1 %.not36, float %1164, float %1158, !dbg !11
  %1167 = bitcast float %1166 to i32, !dbg !11
  %1168 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1167), !dbg !11
  %1169 = select i1 %.not36, float %1165, float %1159, !dbg !11
  %1170 = bitcast float %1169 to i32, !dbg !11
  %1171 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1170), !dbg !11
  %1172 = shufflevector <1 x float> %1046, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !50
  %1173 = extractelement <4 x float> %1011, i64 0, !dbg !11
  %1174 = extractelement <4 x float> %1011, i64 1, !dbg !11
  %1175 = extractelement <4 x float> %1011, i64 2, !dbg !11
  %1176 = extractelement <4 x float> %1011, i64 3, !dbg !11
  %1177 = select i1 %.not, float %1174, float %1173, !dbg !11
  %1178 = bitcast float %1177 to i32, !dbg !11
  %1179 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1178), !dbg !11
  %1180 = bitcast i32 %1179 to float, !dbg !11
  %1181 = select i1 %.not, float %1173, float %1180, !dbg !11
  %1182 = select i1 %.not, float %1180, float %1174, !dbg !11
  %1183 = select i1 %.not, float %1176, float %1175, !dbg !11
  %1184 = bitcast float %1183 to i32, !dbg !11
  %1185 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1184), !dbg !11
  %1186 = bitcast i32 %1185 to float, !dbg !11
  %1187 = select i1 %.not, float %1175, float %1186, !dbg !11
  %1188 = select i1 %.not, float %1186, float %1176, !dbg !11
  %1189 = select i1 %.not36, float %1187, float %1181, !dbg !11
  %1190 = bitcast float %1189 to i32, !dbg !11
  %1191 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1190), !dbg !11
  %1192 = bitcast i32 %1191 to float, !dbg !11
  %1193 = select i1 %.not36, float %1188, float %1182, !dbg !11
  %1194 = bitcast float %1193 to i32, !dbg !11
  %1195 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1194), !dbg !11
  %1196 = bitcast i32 %1195 to float, !dbg !11
  %1197 = insertelement <4 x float> poison, float %1181, i64 0, !dbg !11
  %1198 = insertelement <4 x float> %1197, float %1182, i64 1, !dbg !11
  %1199 = insertelement <4 x float> %1198, float %1192, i64 2, !dbg !11
  %1200 = insertelement <4 x float> %1199, float %1196, i64 3, !dbg !11
  %1201 = insertelement <4 x float> poison, float %1192, i64 0, !dbg !11
  %1202 = insertelement <4 x float> %1201, float %1196, i64 1, !dbg !11
  %1203 = insertelement <4 x float> %1202, float %1187, i64 2, !dbg !11
  %1204 = insertelement <4 x float> %1203, float %1188, i64 3, !dbg !11
  %1205 = select <4 x i1> %1074, <4 x float> %1200, <4 x float> %1204, !dbg !11
  %1206 = extractelement <4 x float> %936, i64 0, !dbg !11
  %1207 = extractelement <4 x float> %936, i64 1, !dbg !11
  %1208 = extractelement <4 x float> %936, i64 2, !dbg !11
  %1209 = extractelement <4 x float> %936, i64 3, !dbg !11
  %1210 = select i1 %.not, float %1207, float %1206, !dbg !11
  %1211 = bitcast float %1210 to i32, !dbg !11
  %1212 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1211), !dbg !11
  %1213 = bitcast i32 %1212 to float, !dbg !11
  %1214 = select i1 %.not, float %1206, float %1213, !dbg !11
  %1215 = select i1 %.not, float %1213, float %1207, !dbg !11
  %1216 = select i1 %.not, float %1209, float %1208, !dbg !11
  %1217 = bitcast float %1216 to i32, !dbg !11
  %1218 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1217), !dbg !11
  %1219 = bitcast i32 %1218 to float, !dbg !11
  %1220 = select i1 %.not, float %1208, float %1219, !dbg !11
  %1221 = select i1 %.not, float %1219, float %1209, !dbg !11
  %1222 = select i1 %.not36, float %1220, float %1214, !dbg !11
  %1223 = bitcast float %1222 to i32, !dbg !11
  %1224 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1223), !dbg !11
  %1225 = bitcast i32 %1224 to float, !dbg !11
  %1226 = select i1 %.not36, float %1221, float %1215, !dbg !11
  %1227 = bitcast float %1226 to i32, !dbg !11
  %1228 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1227), !dbg !11
  %1229 = bitcast i32 %1228 to float, !dbg !11
  %1230 = insertelement <4 x float> poison, float %1214, i64 0, !dbg !11
  %1231 = insertelement <4 x float> %1230, float %1215, i64 1, !dbg !11
  %1232 = insertelement <4 x float> %1231, float %1225, i64 2, !dbg !11
  %1233 = insertelement <4 x float> %1232, float %1229, i64 3, !dbg !11
  %1234 = insertelement <4 x float> poison, float %1225, i64 0, !dbg !11
  %1235 = insertelement <4 x float> %1234, float %1229, i64 1, !dbg !11
  %1236 = insertelement <4 x float> %1235, float %1220, i64 2, !dbg !11
  %1237 = insertelement <4 x float> %1236, float %1221, i64 3, !dbg !11
  %1238 = select <4 x i1> %1074, <4 x float> %1233, <4 x float> %1237, !dbg !11
  %1239 = extractelement <4 x float> %860, i64 0, !dbg !11
  %1240 = extractelement <4 x float> %860, i64 1, !dbg !11
  %1241 = extractelement <4 x float> %860, i64 2, !dbg !11
  %1242 = extractelement <4 x float> %860, i64 3, !dbg !11
  %1243 = select i1 %.not, float %1240, float %1239, !dbg !11
  %1244 = bitcast float %1243 to i32, !dbg !11
  %1245 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1244), !dbg !11
  %1246 = bitcast i32 %1245 to float, !dbg !11
  %1247 = select i1 %.not, float %1239, float %1246, !dbg !11
  %1248 = select i1 %.not, float %1246, float %1240, !dbg !11
  %1249 = select i1 %.not, float %1242, float %1241, !dbg !11
  %1250 = bitcast float %1249 to i32, !dbg !11
  %1251 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1250), !dbg !11
  %1252 = bitcast i32 %1251 to float, !dbg !11
  %1253 = select i1 %.not, float %1241, float %1252, !dbg !11
  %1254 = select i1 %.not, float %1252, float %1242, !dbg !11
  %1255 = select i1 %.not36, float %1253, float %1247, !dbg !11
  %1256 = bitcast float %1255 to i32, !dbg !11
  %1257 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1256), !dbg !11
  %1258 = bitcast i32 %1257 to float, !dbg !11
  %1259 = select i1 %.not36, float %1254, float %1248, !dbg !11
  %1260 = bitcast float %1259 to i32, !dbg !11
  %1261 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1260), !dbg !11
  %1262 = bitcast i32 %1261 to float, !dbg !11
  %1263 = insertelement <4 x float> poison, float %1247, i64 0, !dbg !11
  %1264 = insertelement <4 x float> %1263, float %1248, i64 1, !dbg !11
  %1265 = insertelement <4 x float> %1264, float %1258, i64 2, !dbg !11
  %1266 = insertelement <4 x float> %1265, float %1262, i64 3, !dbg !11
  %1267 = insertelement <4 x float> poison, float %1258, i64 0, !dbg !11
  %1268 = insertelement <4 x float> %1267, float %1262, i64 1, !dbg !11
  %1269 = insertelement <4 x float> %1268, float %1253, i64 2, !dbg !11
  %1270 = insertelement <4 x float> %1269, float %1254, i64 3, !dbg !11
  %1271 = select <4 x i1> %1074, <4 x float> %1266, <4 x float> %1270, !dbg !11
  %1272 = extractelement <4 x float> %766, i64 0, !dbg !11
  %1273 = extractelement <4 x float> %766, i64 1, !dbg !11
  %1274 = extractelement <4 x float> %766, i64 2, !dbg !11
  %1275 = extractelement <4 x float> %766, i64 3, !dbg !11
  %1276 = select i1 %.not, float %1273, float %1272, !dbg !11
  %1277 = bitcast float %1276 to i32, !dbg !11
  %1278 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1277), !dbg !11
  %1279 = bitcast i32 %1278 to float, !dbg !11
  %1280 = select i1 %.not, float %1272, float %1279, !dbg !11
  %1281 = select i1 %.not, float %1279, float %1273, !dbg !11
  %1282 = select i1 %.not, float %1275, float %1274, !dbg !11
  %1283 = bitcast float %1282 to i32, !dbg !11
  %1284 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1283), !dbg !11
  %1285 = bitcast i32 %1284 to float, !dbg !11
  %1286 = select i1 %.not, float %1274, float %1285, !dbg !11
  %1287 = select i1 %.not, float %1285, float %1275, !dbg !11
  %1288 = select i1 %.not36, float %1286, float %1280, !dbg !11
  %1289 = bitcast float %1288 to i32, !dbg !11
  %1290 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1289), !dbg !11
  %1291 = select i1 %.not36, float %1287, float %1281, !dbg !11
  %1292 = bitcast float %1291 to i32, !dbg !11
  %1293 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1292), !dbg !11
  %1294 = extractelement <4 x float> %1012, i64 0, !dbg !11
  %1295 = extractelement <4 x float> %1012, i64 1, !dbg !11
  %1296 = extractelement <4 x float> %1012, i64 2, !dbg !11
  %1297 = extractelement <4 x float> %1012, i64 3, !dbg !11
  %1298 = select i1 %.not, float %1295, float %1294, !dbg !11
  %1299 = bitcast float %1298 to i32, !dbg !11
  %1300 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1299), !dbg !11
  %1301 = bitcast i32 %1300 to float, !dbg !11
  %1302 = select i1 %.not, float %1294, float %1301, !dbg !11
  %1303 = select i1 %.not, float %1301, float %1295, !dbg !11
  %1304 = select i1 %.not, float %1297, float %1296, !dbg !11
  %1305 = bitcast float %1304 to i32, !dbg !11
  %1306 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1305), !dbg !11
  %1307 = bitcast i32 %1306 to float, !dbg !11
  %1308 = select i1 %.not, float %1296, float %1307, !dbg !11
  %1309 = select i1 %.not, float %1307, float %1297, !dbg !11
  %1310 = select i1 %.not36, float %1308, float %1302, !dbg !11
  %1311 = bitcast float %1310 to i32, !dbg !11
  %1312 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1311), !dbg !11
  %1313 = bitcast i32 %1312 to float, !dbg !11
  %1314 = select i1 %.not36, float %1309, float %1303, !dbg !11
  %1315 = bitcast float %1314 to i32, !dbg !11
  %1316 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1315), !dbg !11
  %1317 = bitcast i32 %1316 to float, !dbg !11
  %1318 = insertelement <4 x float> poison, float %1302, i64 0, !dbg !11
  %1319 = insertelement <4 x float> %1318, float %1303, i64 1, !dbg !11
  %1320 = insertelement <4 x float> %1319, float %1313, i64 2, !dbg !11
  %1321 = insertelement <4 x float> %1320, float %1317, i64 3, !dbg !11
  %1322 = insertelement <4 x float> poison, float %1313, i64 0, !dbg !11
  %1323 = insertelement <4 x float> %1322, float %1317, i64 1, !dbg !11
  %1324 = insertelement <4 x float> %1323, float %1308, i64 2, !dbg !11
  %1325 = insertelement <4 x float> %1324, float %1309, i64 3, !dbg !11
  %1326 = select <4 x i1> %1074, <4 x float> %1321, <4 x float> %1325, !dbg !11
  %1327 = extractelement <4 x float> %937, i64 0, !dbg !11
  %1328 = extractelement <4 x float> %937, i64 1, !dbg !11
  %1329 = extractelement <4 x float> %937, i64 2, !dbg !11
  %1330 = extractelement <4 x float> %937, i64 3, !dbg !11
  %1331 = select i1 %.not, float %1328, float %1327, !dbg !11
  %1332 = bitcast float %1331 to i32, !dbg !11
  %1333 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1332), !dbg !11
  %1334 = bitcast i32 %1333 to float, !dbg !11
  %1335 = select i1 %.not, float %1327, float %1334, !dbg !11
  %1336 = select i1 %.not, float %1334, float %1328, !dbg !11
  %1337 = select i1 %.not, float %1330, float %1329, !dbg !11
  %1338 = bitcast float %1337 to i32, !dbg !11
  %1339 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1338), !dbg !11
  %1340 = bitcast i32 %1339 to float, !dbg !11
  %1341 = select i1 %.not, float %1329, float %1340, !dbg !11
  %1342 = select i1 %.not, float %1340, float %1330, !dbg !11
  %1343 = select i1 %.not36, float %1341, float %1335, !dbg !11
  %1344 = bitcast float %1343 to i32, !dbg !11
  %1345 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1344), !dbg !11
  %1346 = bitcast i32 %1345 to float, !dbg !11
  %1347 = select i1 %.not36, float %1342, float %1336, !dbg !11
  %1348 = bitcast float %1347 to i32, !dbg !11
  %1349 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1348), !dbg !11
  %1350 = bitcast i32 %1349 to float, !dbg !11
  %1351 = insertelement <4 x float> poison, float %1335, i64 0, !dbg !11
  %1352 = insertelement <4 x float> %1351, float %1336, i64 1, !dbg !11
  %1353 = insertelement <4 x float> %1352, float %1346, i64 2, !dbg !11
  %1354 = insertelement <4 x float> %1353, float %1350, i64 3, !dbg !11
  %1355 = insertelement <4 x float> poison, float %1346, i64 0, !dbg !11
  %1356 = insertelement <4 x float> %1355, float %1350, i64 1, !dbg !11
  %1357 = insertelement <4 x float> %1356, float %1341, i64 2, !dbg !11
  %1358 = insertelement <4 x float> %1357, float %1342, i64 3, !dbg !11
  %1359 = select <4 x i1> %1074, <4 x float> %1354, <4 x float> %1358, !dbg !11
  %1360 = extractelement <4 x float> %861, i64 0, !dbg !11
  %1361 = extractelement <4 x float> %861, i64 1, !dbg !11
  %1362 = extractelement <4 x float> %861, i64 2, !dbg !11
  %1363 = extractelement <4 x float> %861, i64 3, !dbg !11
  %1364 = select i1 %.not, float %1361, float %1360, !dbg !11
  %1365 = bitcast float %1364 to i32, !dbg !11
  %1366 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1365), !dbg !11
  %1367 = bitcast i32 %1366 to float, !dbg !11
  %1368 = select i1 %.not, float %1360, float %1367, !dbg !11
  %1369 = select i1 %.not, float %1367, float %1361, !dbg !11
  %1370 = select i1 %.not, float %1363, float %1362, !dbg !11
  %1371 = bitcast float %1370 to i32, !dbg !11
  %1372 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1371), !dbg !11
  %1373 = bitcast i32 %1372 to float, !dbg !11
  %1374 = select i1 %.not, float %1362, float %1373, !dbg !11
  %1375 = select i1 %.not, float %1373, float %1363, !dbg !11
  %1376 = select i1 %.not36, float %1374, float %1368, !dbg !11
  %1377 = bitcast float %1376 to i32, !dbg !11
  %1378 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1377), !dbg !11
  %1379 = bitcast i32 %1378 to float, !dbg !11
  %1380 = select i1 %.not36, float %1375, float %1369, !dbg !11
  %1381 = bitcast float %1380 to i32, !dbg !11
  %1382 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1381), !dbg !11
  %1383 = bitcast i32 %1382 to float, !dbg !11
  %1384 = insertelement <4 x float> poison, float %1368, i64 0, !dbg !11
  %1385 = insertelement <4 x float> %1384, float %1369, i64 1, !dbg !11
  %1386 = insertelement <4 x float> %1385, float %1379, i64 2, !dbg !11
  %1387 = insertelement <4 x float> %1386, float %1383, i64 3, !dbg !11
  %1388 = insertelement <4 x float> poison, float %1379, i64 0, !dbg !11
  %1389 = insertelement <4 x float> %1388, float %1383, i64 1, !dbg !11
  %1390 = insertelement <4 x float> %1389, float %1374, i64 2, !dbg !11
  %1391 = insertelement <4 x float> %1390, float %1375, i64 3, !dbg !11
  %1392 = select <4 x i1> %1074, <4 x float> %1387, <4 x float> %1391, !dbg !11
  %1393 = extractelement <4 x float> %785, i64 0, !dbg !11
  %1394 = extractelement <4 x float> %785, i64 1, !dbg !11
  %1395 = extractelement <4 x float> %785, i64 2, !dbg !11
  %1396 = extractelement <4 x float> %785, i64 3, !dbg !11
  %1397 = select i1 %.not, float %1394, float %1393, !dbg !11
  %1398 = bitcast float %1397 to i32, !dbg !11
  %1399 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1398), !dbg !11
  %1400 = bitcast i32 %1399 to float, !dbg !11
  %1401 = select i1 %.not, float %1393, float %1400, !dbg !11
  %1402 = select i1 %.not, float %1400, float %1394, !dbg !11
  %1403 = select i1 %.not, float %1396, float %1395, !dbg !11
  %1404 = bitcast float %1403 to i32, !dbg !11
  %1405 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1404), !dbg !11
  %1406 = bitcast i32 %1405 to float, !dbg !11
  %1407 = select i1 %.not, float %1395, float %1406, !dbg !11
  %1408 = select i1 %.not, float %1406, float %1396, !dbg !11
  %1409 = select i1 %.not36, float %1407, float %1401, !dbg !11
  %1410 = bitcast float %1409 to i32, !dbg !11
  %1411 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1410), !dbg !11
  %1412 = select i1 %.not36, float %1408, float %1402, !dbg !11
  %1413 = bitcast float %1412 to i32, !dbg !11
  %1414 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1413), !dbg !11
  %1415 = shufflevector <1 x float> %1041, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !50
  %1416 = extractelement <4 x float> %1013, i64 0, !dbg !11
  %1417 = extractelement <4 x float> %1013, i64 1, !dbg !11
  %1418 = extractelement <4 x float> %1013, i64 2, !dbg !11
  %1419 = extractelement <4 x float> %1013, i64 3, !dbg !11
  %1420 = select i1 %.not, float %1417, float %1416, !dbg !11
  %1421 = bitcast float %1420 to i32, !dbg !11
  %1422 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1421), !dbg !11
  %1423 = bitcast i32 %1422 to float, !dbg !11
  %1424 = select i1 %.not, float %1416, float %1423, !dbg !11
  %1425 = select i1 %.not, float %1423, float %1417, !dbg !11
  %1426 = select i1 %.not, float %1419, float %1418, !dbg !11
  %1427 = bitcast float %1426 to i32, !dbg !11
  %1428 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1427), !dbg !11
  %1429 = bitcast i32 %1428 to float, !dbg !11
  %1430 = select i1 %.not, float %1418, float %1429, !dbg !11
  %1431 = select i1 %.not, float %1429, float %1419, !dbg !11
  %1432 = select i1 %.not36, float %1430, float %1424, !dbg !11
  %1433 = bitcast float %1432 to i32, !dbg !11
  %1434 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1433), !dbg !11
  %1435 = bitcast i32 %1434 to float, !dbg !11
  %1436 = select i1 %.not36, float %1431, float %1425, !dbg !11
  %1437 = bitcast float %1436 to i32, !dbg !11
  %1438 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1437), !dbg !11
  %1439 = bitcast i32 %1438 to float, !dbg !11
  %1440 = insertelement <4 x float> poison, float %1424, i64 0, !dbg !11
  %1441 = insertelement <4 x float> %1440, float %1425, i64 1, !dbg !11
  %1442 = insertelement <4 x float> %1441, float %1435, i64 2, !dbg !11
  %1443 = insertelement <4 x float> %1442, float %1439, i64 3, !dbg !11
  %1444 = insertelement <4 x float> poison, float %1435, i64 0, !dbg !11
  %1445 = insertelement <4 x float> %1444, float %1439, i64 1, !dbg !11
  %1446 = insertelement <4 x float> %1445, float %1430, i64 2, !dbg !11
  %1447 = insertelement <4 x float> %1446, float %1431, i64 3, !dbg !11
  %1448 = select <4 x i1> %1074, <4 x float> %1443, <4 x float> %1447, !dbg !11
  %1449 = extractelement <4 x float> %938, i64 0, !dbg !11
  %1450 = extractelement <4 x float> %938, i64 1, !dbg !11
  %1451 = extractelement <4 x float> %938, i64 2, !dbg !11
  %1452 = extractelement <4 x float> %938, i64 3, !dbg !11
  %1453 = select i1 %.not, float %1450, float %1449, !dbg !11
  %1454 = bitcast float %1453 to i32, !dbg !11
  %1455 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1454), !dbg !11
  %1456 = bitcast i32 %1455 to float, !dbg !11
  %1457 = select i1 %.not, float %1449, float %1456, !dbg !11
  %1458 = select i1 %.not, float %1456, float %1450, !dbg !11
  %1459 = select i1 %.not, float %1452, float %1451, !dbg !11
  %1460 = bitcast float %1459 to i32, !dbg !11
  %1461 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1460), !dbg !11
  %1462 = bitcast i32 %1461 to float, !dbg !11
  %1463 = select i1 %.not, float %1451, float %1462, !dbg !11
  %1464 = select i1 %.not, float %1462, float %1452, !dbg !11
  %1465 = select i1 %.not36, float %1463, float %1457, !dbg !11
  %1466 = bitcast float %1465 to i32, !dbg !11
  %1467 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1466), !dbg !11
  %1468 = bitcast i32 %1467 to float, !dbg !11
  %1469 = select i1 %.not36, float %1464, float %1458, !dbg !11
  %1470 = bitcast float %1469 to i32, !dbg !11
  %1471 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1470), !dbg !11
  %1472 = bitcast i32 %1471 to float, !dbg !11
  %1473 = insertelement <4 x float> poison, float %1457, i64 0, !dbg !11
  %1474 = insertelement <4 x float> %1473, float %1458, i64 1, !dbg !11
  %1475 = insertelement <4 x float> %1474, float %1468, i64 2, !dbg !11
  %1476 = insertelement <4 x float> %1475, float %1472, i64 3, !dbg !11
  %1477 = insertelement <4 x float> poison, float %1468, i64 0, !dbg !11
  %1478 = insertelement <4 x float> %1477, float %1472, i64 1, !dbg !11
  %1479 = insertelement <4 x float> %1478, float %1463, i64 2, !dbg !11
  %1480 = insertelement <4 x float> %1479, float %1464, i64 3, !dbg !11
  %1481 = select <4 x i1> %1074, <4 x float> %1476, <4 x float> %1480, !dbg !11
  %1482 = extractelement <4 x float> %862, i64 0, !dbg !11
  %1483 = extractelement <4 x float> %862, i64 1, !dbg !11
  %1484 = extractelement <4 x float> %862, i64 2, !dbg !11
  %1485 = extractelement <4 x float> %862, i64 3, !dbg !11
  %1486 = select i1 %.not, float %1483, float %1482, !dbg !11
  %1487 = bitcast float %1486 to i32, !dbg !11
  %1488 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1487), !dbg !11
  %1489 = bitcast i32 %1488 to float, !dbg !11
  %1490 = select i1 %.not, float %1482, float %1489, !dbg !11
  %1491 = select i1 %.not, float %1489, float %1483, !dbg !11
  %1492 = select i1 %.not, float %1485, float %1484, !dbg !11
  %1493 = bitcast float %1492 to i32, !dbg !11
  %1494 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1493), !dbg !11
  %1495 = bitcast i32 %1494 to float, !dbg !11
  %1496 = select i1 %.not, float %1484, float %1495, !dbg !11
  %1497 = select i1 %.not, float %1495, float %1485, !dbg !11
  %1498 = select i1 %.not36, float %1496, float %1490, !dbg !11
  %1499 = bitcast float %1498 to i32, !dbg !11
  %1500 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1499), !dbg !11
  %1501 = bitcast i32 %1500 to float, !dbg !11
  %1502 = select i1 %.not36, float %1497, float %1491, !dbg !11
  %1503 = bitcast float %1502 to i32, !dbg !11
  %1504 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1503), !dbg !11
  %1505 = bitcast i32 %1504 to float, !dbg !11
  %1506 = insertelement <4 x float> poison, float %1490, i64 0, !dbg !11
  %1507 = insertelement <4 x float> %1506, float %1491, i64 1, !dbg !11
  %1508 = insertelement <4 x float> %1507, float %1501, i64 2, !dbg !11
  %1509 = insertelement <4 x float> %1508, float %1505, i64 3, !dbg !11
  %1510 = insertelement <4 x float> poison, float %1501, i64 0, !dbg !11
  %1511 = insertelement <4 x float> %1510, float %1505, i64 1, !dbg !11
  %1512 = insertelement <4 x float> %1511, float %1496, i64 2, !dbg !11
  %1513 = insertelement <4 x float> %1512, float %1497, i64 3, !dbg !11
  %1514 = select <4 x i1> %1074, <4 x float> %1509, <4 x float> %1513, !dbg !11
  %1515 = extractelement <4 x float> %786, i64 0, !dbg !11
  %1516 = extractelement <4 x float> %786, i64 1, !dbg !11
  %1517 = extractelement <4 x float> %786, i64 2, !dbg !11
  %1518 = extractelement <4 x float> %786, i64 3, !dbg !11
  %1519 = select i1 %.not, float %1516, float %1515, !dbg !11
  %1520 = bitcast float %1519 to i32, !dbg !11
  %1521 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1520), !dbg !11
  %1522 = bitcast i32 %1521 to float, !dbg !11
  %1523 = select i1 %.not, float %1515, float %1522, !dbg !11
  %1524 = select i1 %.not, float %1522, float %1516, !dbg !11
  %1525 = select i1 %.not, float %1518, float %1517, !dbg !11
  %1526 = bitcast float %1525 to i32, !dbg !11
  %1527 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %432, i32 %1526), !dbg !11
  %1528 = bitcast i32 %1527 to float, !dbg !11
  %1529 = select i1 %.not, float %1517, float %1528, !dbg !11
  %1530 = select i1 %.not, float %1528, float %1518, !dbg !11
  %1531 = select i1 %.not36, float %1529, float %1523, !dbg !11
  %1532 = bitcast float %1531 to i32, !dbg !11
  %1533 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1532), !dbg !11
  %1534 = select i1 %.not36, float %1530, float %1524, !dbg !11
  %1535 = bitcast float %1534 to i32, !dbg !11
  %1536 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %440, i32 %1535), !dbg !11
  %1537 = bitcast i32 %1533 to float, !dbg !11
  %1538 = bitcast i32 %1536 to float, !dbg !11
  %1539 = insertelement <4 x float> poison, float %1523, i64 0, !dbg !11
  %1540 = insertelement <4 x float> %1539, float %1524, i64 1, !dbg !11
  %1541 = insertelement <4 x float> %1540, float %1537, i64 2, !dbg !11
  %1542 = insertelement <4 x float> %1541, float %1538, i64 3, !dbg !11
  %1543 = insertelement <4 x float> poison, float %1537, i64 0, !dbg !11
  %1544 = insertelement <4 x float> %1543, float %1538, i64 1, !dbg !11
  %1545 = insertelement <4 x float> %1544, float %1529, i64 2, !dbg !11
  %1546 = insertelement <4 x float> %1545, float %1530, i64 3, !dbg !11
  %1547 = select <4 x i1> %1074, <4 x float> %1542, <4 x float> %1546, !dbg !11
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1415, !dbg !50
  %normalized = fmul <4 x float> %row.reciprocal, %1547, !dbg !50
  %1548 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !51
  %1549 = extractelement <4 x half> %1548, i64 3, !dbg !51
  %1550 = extractelement <4 x half> %1548, i64 2, !dbg !51
  %1551 = extractelement <4 x half> %1548, i64 1, !dbg !51
  %1552 = extractelement <4 x half> %1548, i64 0, !dbg !51
  %1553 = bitcast i32 %1411 to float, !dbg !11
  %1554 = bitcast i32 %1414 to float, !dbg !11
  %1555 = insertelement <4 x float> poison, float %1401, i64 0, !dbg !11
  %1556 = insertelement <4 x float> %1555, float %1402, i64 1, !dbg !11
  %1557 = insertelement <4 x float> %1556, float %1553, i64 2, !dbg !11
  %1558 = insertelement <4 x float> %1557, float %1554, i64 3, !dbg !11
  %1559 = insertelement <4 x float> poison, float %1553, i64 0, !dbg !11
  %1560 = insertelement <4 x float> %1559, float %1554, i64 1, !dbg !11
  %1561 = insertelement <4 x float> %1560, float %1407, i64 2, !dbg !11
  %1562 = insertelement <4 x float> %1561, float %1408, i64 3, !dbg !11
  %1563 = select <4 x i1> %1074, <4 x float> %1558, <4 x float> %1562, !dbg !11
  %normalized37 = fmul <4 x float> %row.reciprocal, %1563, !dbg !50
  %1564 = fptrunc <4 x float> %normalized37 to <4 x half>, !dbg !51
  %1565 = extractelement <4 x half> %1564, i64 3, !dbg !51
  %1566 = extractelement <4 x half> %1564, i64 2, !dbg !51
  %1567 = extractelement <4 x half> %1564, i64 1, !dbg !51
  %1568 = extractelement <4 x half> %1564, i64 0, !dbg !51
  %1569 = bitcast i32 %1290 to float, !dbg !11
  %1570 = bitcast i32 %1293 to float, !dbg !11
  %1571 = insertelement <4 x float> poison, float %1280, i64 0, !dbg !11
  %1572 = insertelement <4 x float> %1571, float %1281, i64 1, !dbg !11
  %1573 = insertelement <4 x float> %1572, float %1569, i64 2, !dbg !11
  %1574 = insertelement <4 x float> %1573, float %1570, i64 3, !dbg !11
  %1575 = insertelement <4 x float> poison, float %1569, i64 0, !dbg !11
  %1576 = insertelement <4 x float> %1575, float %1570, i64 1, !dbg !11
  %1577 = insertelement <4 x float> %1576, float %1286, i64 2, !dbg !11
  %1578 = insertelement <4 x float> %1577, float %1287, i64 3, !dbg !11
  %1579 = select <4 x i1> %1074, <4 x float> %1574, <4 x float> %1578, !dbg !11
  %row.reciprocal38 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1172, !dbg !50
  %normalized39 = fmul <4 x float> %row.reciprocal38, %1579, !dbg !50
  %1580 = fptrunc <4 x float> %normalized39 to <4 x half>, !dbg !51
  %1581 = extractelement <4 x half> %1580, i64 3, !dbg !51
  %1582 = extractelement <4 x half> %1580, i64 2, !dbg !51
  %1583 = extractelement <4 x half> %1580, i64 1, !dbg !51
  %1584 = extractelement <4 x half> %1580, i64 0, !dbg !51
  %1585 = bitcast i32 %1168 to float, !dbg !11
  %1586 = bitcast i32 %1171 to float, !dbg !11
  %1587 = insertelement <4 x float> poison, float %1158, i64 0, !dbg !11
  %1588 = insertelement <4 x float> %1587, float %1159, i64 1, !dbg !11
  %1589 = insertelement <4 x float> %1588, float %1585, i64 2, !dbg !11
  %1590 = insertelement <4 x float> %1589, float %1586, i64 3, !dbg !11
  %1591 = insertelement <4 x float> poison, float %1585, i64 0, !dbg !11
  %1592 = insertelement <4 x float> %1591, float %1586, i64 1, !dbg !11
  %1593 = insertelement <4 x float> %1592, float %1164, i64 2, !dbg !11
  %1594 = insertelement <4 x float> %1593, float %1165, i64 3, !dbg !11
  %1595 = select <4 x i1> %1074, <4 x float> %1590, <4 x float> %1594, !dbg !11
  %normalized40 = fmul <4 x float> %row.reciprocal38, %1595, !dbg !50
  %1596 = fptrunc <4 x float> %normalized40 to <4 x half>, !dbg !51
  %1597 = extractelement <4 x half> %1596, i64 3, !dbg !51
  %1598 = extractelement <4 x half> %1596, i64 2, !dbg !51
  %1599 = extractelement <4 x half> %1596, i64 1, !dbg !51
  %1600 = extractelement <4 x half> %1596, i64 0, !dbg !51
  %address.thread974 = tail call i32 asm sideeffect "", "=v,0"(i32 %kdma.tid) #8, !dbg !52, !range !9, !frisk.register_lifetimes !5
  %address.local975 = zext nneg i32 %address.thread974 to i64, !dbg !8
  %1601 = lshr i64 %address.local975, 2, !dbg !52
  %address.local978 = and i64 %1601, 48, !dbg !52
  %address.local980 = and i64 %address.local975, 240, !dbg !52
  %address.local979 = add i64 %21, %address.local975, !dbg !52
  %address.local981 = sub i64 %address.local979, %address.local980, !dbg !52
  %address.local982 = add i64 %address.local981, %address.local978, !dbg !52
  %1602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %23, !dbg !52
  %address.local985.idx = shl nuw nsw i64 %address.local982, 8, !dbg !52
  %address.local985 = getelementptr inbounds nuw i8, ptr addrspace(1) %1602, i64 %address.local985.idx, !dbg !52
  %address.local987 = and i64 %1601, 12, !dbg !10
  %address.local988 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local987, !dbg !52
  store half %1600, ptr addrspace(1) %address.local988, align 2, !dbg !52
  %address.local990 = or disjoint i64 %address.local987, 1, !dbg !52
  %address.local991 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local990, !dbg !52
  store half %1599, ptr addrspace(1) %address.local991, align 2, !dbg !52
  %address.local993 = or disjoint i64 %address.local987, 2, !dbg !52
  %address.local994 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local993, !dbg !52
  store half %1598, ptr addrspace(1) %address.local994, align 2, !dbg !52
  %address.local996 = or disjoint i64 %address.local987, 3, !dbg !52
  %address.local997 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local996, !dbg !52
  store half %1597, ptr addrspace(1) %address.local997, align 2, !dbg !52
  %address.local999 = or disjoint i64 %address.local987, 16, !dbg !52
  %address.local1000 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local999, !dbg !52
  store half %1584, ptr addrspace(1) %address.local1000, align 2, !dbg !52
  %address.local1002 = or disjoint i64 %address.local987, 17, !dbg !52
  %address.local1003 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1002, !dbg !52
  store half %1583, ptr addrspace(1) %address.local1003, align 2, !dbg !52
  %address.local1005 = or disjoint i64 %address.local987, 18, !dbg !52
  %address.local1006 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1005, !dbg !52
  store half %1582, ptr addrspace(1) %address.local1006, align 2, !dbg !52
  %address.local1008 = or disjoint i64 %address.local987, 19, !dbg !52
  %address.local1009 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1008, !dbg !52
  store half %1581, ptr addrspace(1) %address.local1009, align 2, !dbg !52
  %address.local1013 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local985, i64 16384, !dbg !52
  %address.local1014 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local987, !dbg !52
  store half %1568, ptr addrspace(1) %address.local1014, align 2, !dbg !52
  %address.local1016 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local990, !dbg !52
  store half %1567, ptr addrspace(1) %address.local1016, align 2, !dbg !52
  %address.local1018 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local993, !dbg !52
  store half %1566, ptr addrspace(1) %address.local1018, align 2, !dbg !52
  %address.local1020 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local996, !dbg !52
  store half %1565, ptr addrspace(1) %address.local1020, align 2, !dbg !52
  %address.local1022 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local999, !dbg !52
  store half %1552, ptr addrspace(1) %address.local1022, align 2, !dbg !52
  %address.local1024 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1002, !dbg !52
  store half %1551, ptr addrspace(1) %address.local1024, align 2, !dbg !52
  %address.local1026 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1005, !dbg !52
  store half %1550, ptr addrspace(1) %address.local1026, align 2, !dbg !52
  %address.local1028 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1008, !dbg !52
  store half %1549, ptr addrspace(1) %address.local1028, align 2, !dbg !52
  %normalized42 = fmul <4 x float> %row.reciprocal, %1514, !dbg !50
  %1603 = fptrunc <4 x float> %normalized42 to <4 x half>, !dbg !51
  %1604 = extractelement <4 x half> %1603, i64 3, !dbg !51
  %1605 = extractelement <4 x half> %1603, i64 2, !dbg !51
  %1606 = extractelement <4 x half> %1603, i64 1, !dbg !51
  %1607 = extractelement <4 x half> %1603, i64 0, !dbg !51
  %normalized43 = fmul <4 x float> %row.reciprocal, %1392, !dbg !50
  %1608 = fptrunc <4 x float> %normalized43 to <4 x half>, !dbg !51
  %1609 = extractelement <4 x half> %1608, i64 3, !dbg !51
  %1610 = extractelement <4 x half> %1608, i64 2, !dbg !51
  %1611 = extractelement <4 x half> %1608, i64 1, !dbg !51
  %1612 = extractelement <4 x half> %1608, i64 0, !dbg !51
  %normalized45 = fmul <4 x float> %row.reciprocal38, %1271, !dbg !50
  %1613 = fptrunc <4 x float> %normalized45 to <4 x half>, !dbg !51
  %1614 = extractelement <4 x half> %1613, i64 3, !dbg !51
  %1615 = extractelement <4 x half> %1613, i64 2, !dbg !51
  %1616 = extractelement <4 x half> %1613, i64 1, !dbg !51
  %1617 = extractelement <4 x half> %1613, i64 0, !dbg !51
  %normalized46 = fmul <4 x float> %row.reciprocal38, %1149, !dbg !50
  %1618 = fptrunc <4 x float> %normalized46 to <4 x half>, !dbg !51
  %1619 = extractelement <4 x half> %1618, i64 3, !dbg !51
  %1620 = extractelement <4 x half> %1618, i64 2, !dbg !51
  %1621 = extractelement <4 x half> %1618, i64 1, !dbg !51
  %1622 = extractelement <4 x half> %1618, i64 0, !dbg !51
  %address.local1043 = or disjoint i64 %address.local987, 32, !dbg !52
  %address.local1044 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1043, !dbg !52
  store half %1622, ptr addrspace(1) %address.local1044, align 2, !dbg !52
  %address.local1046 = or disjoint i64 %address.local987, 33, !dbg !52
  %address.local1047 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1046, !dbg !52
  store half %1621, ptr addrspace(1) %address.local1047, align 2, !dbg !52
  %address.local1049 = or disjoint i64 %address.local987, 34, !dbg !52
  %address.local1050 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1049, !dbg !52
  store half %1620, ptr addrspace(1) %address.local1050, align 2, !dbg !52
  %address.local1052 = or disjoint i64 %address.local987, 35, !dbg !52
  %address.local1053 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1052, !dbg !52
  store half %1619, ptr addrspace(1) %address.local1053, align 2, !dbg !52
  %address.local1055 = or disjoint i64 %address.local987, 48, !dbg !52
  %address.local1056 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1055, !dbg !52
  store half %1617, ptr addrspace(1) %address.local1056, align 2, !dbg !52
  %address.local1058 = or disjoint i64 %address.local987, 49, !dbg !52
  %address.local1059 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1058, !dbg !52
  store half %1616, ptr addrspace(1) %address.local1059, align 2, !dbg !52
  %address.local1061 = or disjoint i64 %address.local987, 50, !dbg !52
  %address.local1062 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1061, !dbg !52
  store half %1615, ptr addrspace(1) %address.local1062, align 2, !dbg !52
  %address.local1064 = or i64 %1601, 51, !dbg !52
  %address.local1065 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1064, !dbg !52
  store half %1614, ptr addrspace(1) %address.local1065, align 2, !dbg !52
  %address.local1070 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1043, !dbg !52
  store half %1612, ptr addrspace(1) %address.local1070, align 2, !dbg !52
  %address.local1072 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1046, !dbg !52
  store half %1611, ptr addrspace(1) %address.local1072, align 2, !dbg !52
  %address.local1074 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1049, !dbg !52
  store half %1610, ptr addrspace(1) %address.local1074, align 2, !dbg !52
  %address.local1076 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1052, !dbg !52
  store half %1609, ptr addrspace(1) %address.local1076, align 2, !dbg !52
  %address.local1078 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1055, !dbg !52
  store half %1607, ptr addrspace(1) %address.local1078, align 2, !dbg !52
  %address.local1080 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1058, !dbg !52
  store half %1606, ptr addrspace(1) %address.local1080, align 2, !dbg !52
  %address.local1082 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1061, !dbg !52
  store half %1605, ptr addrspace(1) %address.local1082, align 2, !dbg !52
  %address.local1084 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1064, !dbg !52
  store half %1604, ptr addrspace(1) %address.local1084, align 2, !dbg !52
  %normalized48 = fmul <4 x float> %row.reciprocal, %1481, !dbg !50
  %1623 = fptrunc <4 x float> %normalized48 to <4 x half>, !dbg !51
  %1624 = extractelement <4 x half> %1623, i64 3, !dbg !51
  %1625 = extractelement <4 x half> %1623, i64 2, !dbg !51
  %1626 = extractelement <4 x half> %1623, i64 1, !dbg !51
  %1627 = extractelement <4 x half> %1623, i64 0, !dbg !51
  %normalized49 = fmul <4 x float> %row.reciprocal, %1359, !dbg !50
  %1628 = fptrunc <4 x float> %normalized49 to <4 x half>, !dbg !51
  %1629 = extractelement <4 x half> %1628, i64 3, !dbg !51
  %1630 = extractelement <4 x half> %1628, i64 2, !dbg !51
  %1631 = extractelement <4 x half> %1628, i64 1, !dbg !51
  %1632 = extractelement <4 x half> %1628, i64 0, !dbg !51
  %normalized51 = fmul <4 x float> %row.reciprocal38, %1238, !dbg !50
  %1633 = fptrunc <4 x float> %normalized51 to <4 x half>, !dbg !51
  %1634 = extractelement <4 x half> %1633, i64 3, !dbg !51
  %1635 = extractelement <4 x half> %1633, i64 2, !dbg !51
  %1636 = extractelement <4 x half> %1633, i64 1, !dbg !51
  %1637 = extractelement <4 x half> %1633, i64 0, !dbg !51
  %normalized52 = fmul <4 x float> %row.reciprocal38, %1116, !dbg !50
  %1638 = fptrunc <4 x float> %normalized52 to <4 x half>, !dbg !51
  %1639 = extractelement <4 x half> %1638, i64 3, !dbg !51
  %1640 = extractelement <4 x half> %1638, i64 2, !dbg !51
  %1641 = extractelement <4 x half> %1638, i64 1, !dbg !51
  %1642 = extractelement <4 x half> %1638, i64 0, !dbg !51
  %address.local1099 = or disjoint i64 %address.local987, 64, !dbg !52
  %address.local1100 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1099, !dbg !52
  store half %1642, ptr addrspace(1) %address.local1100, align 2, !dbg !52
  %address.local1102 = or disjoint i64 %address.local987, 65, !dbg !52
  %address.local1103 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1102, !dbg !52
  store half %1641, ptr addrspace(1) %address.local1103, align 2, !dbg !52
  %address.local1105 = or disjoint i64 %address.local987, 66, !dbg !52
  %address.local1106 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1105, !dbg !52
  store half %1640, ptr addrspace(1) %address.local1106, align 2, !dbg !52
  %address.local1108 = or disjoint i64 %address.local987, 67, !dbg !52
  %address.local1109 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1108, !dbg !52
  store half %1639, ptr addrspace(1) %address.local1109, align 2, !dbg !52
  %address.local1111 = or disjoint i64 %address.local987, 80, !dbg !52
  %address.local1112 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1111, !dbg !52
  store half %1637, ptr addrspace(1) %address.local1112, align 2, !dbg !52
  %address.local1114 = or disjoint i64 %address.local987, 81, !dbg !52
  %address.local1115 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1114, !dbg !52
  store half %1636, ptr addrspace(1) %address.local1115, align 2, !dbg !52
  %address.local1117 = or disjoint i64 %address.local987, 82, !dbg !52
  %address.local1118 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1117, !dbg !52
  store half %1635, ptr addrspace(1) %address.local1118, align 2, !dbg !52
  %address.local1120 = or disjoint i64 %address.local987, 83, !dbg !52
  %address.local1121 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1120, !dbg !52
  store half %1634, ptr addrspace(1) %address.local1121, align 2, !dbg !52
  %address.local1126 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1099, !dbg !52
  store half %1632, ptr addrspace(1) %address.local1126, align 2, !dbg !52
  %address.local1128 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1102, !dbg !52
  store half %1631, ptr addrspace(1) %address.local1128, align 2, !dbg !52
  %address.local1130 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1105, !dbg !52
  store half %1630, ptr addrspace(1) %address.local1130, align 2, !dbg !52
  %address.local1132 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1108, !dbg !52
  store half %1629, ptr addrspace(1) %address.local1132, align 2, !dbg !52
  %address.local1134 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1111, !dbg !52
  store half %1627, ptr addrspace(1) %address.local1134, align 2, !dbg !52
  %address.local1136 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1114, !dbg !52
  store half %1626, ptr addrspace(1) %address.local1136, align 2, !dbg !52
  %address.local1138 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1117, !dbg !52
  store half %1625, ptr addrspace(1) %address.local1138, align 2, !dbg !52
  %address.local1140 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1120, !dbg !52
  store half %1624, ptr addrspace(1) %address.local1140, align 2, !dbg !52
  %normalized54 = fmul <4 x float> %row.reciprocal, %1448, !dbg !50
  %1643 = fptrunc <4 x float> %normalized54 to <4 x half>, !dbg !51
  %1644 = extractelement <4 x half> %1643, i64 3, !dbg !51
  %1645 = extractelement <4 x half> %1643, i64 2, !dbg !51
  %1646 = extractelement <4 x half> %1643, i64 1, !dbg !51
  %1647 = extractelement <4 x half> %1643, i64 0, !dbg !51
  %normalized55 = fmul <4 x float> %row.reciprocal, %1326, !dbg !50
  %1648 = fptrunc <4 x float> %normalized55 to <4 x half>, !dbg !51
  %1649 = extractelement <4 x half> %1648, i64 3, !dbg !51
  %1650 = extractelement <4 x half> %1648, i64 2, !dbg !51
  %1651 = extractelement <4 x half> %1648, i64 1, !dbg !51
  %1652 = extractelement <4 x half> %1648, i64 0, !dbg !51
  %normalized57 = fmul <4 x float> %row.reciprocal38, %1205, !dbg !50
  %1653 = fptrunc <4 x float> %normalized57 to <4 x half>, !dbg !51
  %1654 = extractelement <4 x half> %1653, i64 3, !dbg !51
  %1655 = extractelement <4 x half> %1653, i64 2, !dbg !51
  %1656 = extractelement <4 x half> %1653, i64 1, !dbg !51
  %1657 = extractelement <4 x half> %1653, i64 0, !dbg !51
  %normalized58 = fmul <4 x float> %row.reciprocal38, %1083, !dbg !50
  %1658 = fptrunc <4 x float> %normalized58 to <4 x half>, !dbg !51
  %1659 = extractelement <4 x half> %1658, i64 3, !dbg !51
  %1660 = extractelement <4 x half> %1658, i64 2, !dbg !51
  %1661 = extractelement <4 x half> %1658, i64 1, !dbg !51
  %1662 = extractelement <4 x half> %1658, i64 0, !dbg !51
  %address.local1155 = or disjoint i64 %address.local987, 96, !dbg !52
  %address.local1156 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1155, !dbg !52
  store half %1662, ptr addrspace(1) %address.local1156, align 2, !dbg !52
  %address.local1158 = or disjoint i64 %address.local987, 97, !dbg !52
  %address.local1159 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1158, !dbg !52
  store half %1661, ptr addrspace(1) %address.local1159, align 2, !dbg !52
  %address.local1161 = or disjoint i64 %address.local987, 98, !dbg !52
  %address.local1162 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1161, !dbg !52
  store half %1660, ptr addrspace(1) %address.local1162, align 2, !dbg !52
  %address.local1164 = or disjoint i64 %address.local987, 99, !dbg !52
  %address.local1165 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1164, !dbg !52
  store half %1659, ptr addrspace(1) %address.local1165, align 2, !dbg !52
  %address.local1167 = or disjoint i64 %address.local987, 112, !dbg !52
  %address.local1168 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1167, !dbg !52
  store half %1657, ptr addrspace(1) %address.local1168, align 2, !dbg !52
  %address.local1170 = or disjoint i64 %address.local987, 113, !dbg !52
  %address.local1171 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1170, !dbg !52
  store half %1656, ptr addrspace(1) %address.local1171, align 2, !dbg !52
  %address.local1173 = or disjoint i64 %address.local987, 114, !dbg !52
  %address.local1174 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1173, !dbg !52
  store half %1655, ptr addrspace(1) %address.local1174, align 2, !dbg !52
  %address.local1176 = or i64 %1601, 115, !dbg !52
  %address.local1177 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local985, i64 %address.local1176, !dbg !52
  store half %1654, ptr addrspace(1) %address.local1177, align 2, !dbg !52
  %address.local1182 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1155, !dbg !52
  store half %1652, ptr addrspace(1) %address.local1182, align 2, !dbg !52
  %address.local1184 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1158, !dbg !52
  store half %1651, ptr addrspace(1) %address.local1184, align 2, !dbg !52
  %address.local1186 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1161, !dbg !52
  store half %1650, ptr addrspace(1) %address.local1186, align 2, !dbg !52
  %address.local1188 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1164, !dbg !52
  store half %1649, ptr addrspace(1) %address.local1188, align 2, !dbg !52
  %address.local1190 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1167, !dbg !52
  store half %1647, ptr addrspace(1) %address.local1190, align 2, !dbg !52
  %address.local1192 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1170, !dbg !52
  store half %1646, ptr addrspace(1) %address.local1192, align 2, !dbg !52
  %address.local1194 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1173, !dbg !52
  store half %1645, ptr addrspace(1) %address.local1194, align 2, !dbg !52
  %address.local1196 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1013, i64 %address.local1176, !dbg !52
  store half %1644, ptr addrspace(1) %address.local1196, align 2, !dbg !52
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #4

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #5

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <1 x float> @llvm.maxnum.v1f32(<1 x float>, <1 x float>) #7

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #4

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.cross_bn_k0"="sum-anchor-v1" "frisk.global_prefetch"="8" "frisk.k_dma"="64-swizzle:1" "frisk.k_lds_prefetch"="8" "frisk.k_tail_prefetch"="4" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "frisk.softmax_fragments"="pv" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #5 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/typed_fragment_noncausal")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 23, type: !4, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 23, column: 3, scope: !3)
!8 = !DILocation(line: 67, column: 22, scope: !3)
!9 = !{i32 0, i32 256}
!10 = !DILocation(line: 69, column: 24, scope: !3)
!11 = !DILocation(line: 57, column: 24, scope: !3)
!12 = !DILocation(line: 70, column: 26, scope: !3)
!13 = !DILocation(line: 68, column: 22, scope: !3)
!14 = !{i32 0}
!15 = !{i32 1}
!16 = !{i32 2}
!17 = !{i32 3}
!18 = !{i32 4}
!19 = !{i32 5}
!20 = !{i32 6}
!21 = !{i32 7}
!22 = !{i32 8}
!23 = !{i32 9}
!24 = !{i32 10}
!25 = !{i32 11}
!26 = !{i32 12}
!27 = !{i32 13}
!28 = !{i32 14}
!29 = !{i32 15}
!30 = !{i32 16}
!31 = !{i32 17}
!32 = !{i32 18}
!33 = !{i32 19}
!34 = !{i32 20}
!35 = !{i32 21}
!36 = !{i32 22}
!37 = !DILocation(line: 86, column: 20, scope: !3)
!38 = !DILocation(line: 71, column: 23, scope: !3)
!39 = !DILocation(line: 74, column: 20, scope: !3)
!40 = !DILocation(line: 75, column: 22, scope: !3)
!41 = !DILocation(line: 76, column: 23, scope: !3)
!42 = !DILocation(line: 77, column: 20, scope: !3)
!43 = !DILocation(line: 88, column: 22, scope: !3)
!44 = !DILocation(line: 78, column: 22, scope: !3)
!45 = !DILocation(line: 87, column: 23, scope: !3)
!46 = !{i64 8192}
!47 = !DILocation(line: 79, column: 22, scope: !3)
!48 = !DILocation(line: 81, column: 16, scope: !3)
!49 = !DILocation(line: 80, column: 20, scope: !3)
!50 = !DILocation(line: 92, column: 23, scope: !3)
!51 = !DILocation(line: 93, column: 19, scope: !3)
!52 = !DILocation(line: 94, column: 9, scope: !3)
