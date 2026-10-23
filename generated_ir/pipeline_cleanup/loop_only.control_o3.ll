; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup/loop_only.bc'
source_filename = "3rd/deepgengraph/test/test_input_spidD.mlir"

@shm_0 = addrspace(3) global [32768 x i8] undef, align 16

; Function Attrs: convergent mustprogress nofree willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) nofree writeonly captures(none) %3) local_unnamed_addr #1 !dbg !3 !reqd_work_group_size !6 {
  %5 = tail call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !7
  %6 = sext i32 %5 to i64, !dbg !7
  %7 = tail call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !7
  %8 = sext i32 %7 to i64, !dbg !7
  %9 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !7
  %10 = sext i32 %9 to i64, !dbg !7
  %11 = shl nsw i64 %8, 7, !dbg !8
  %.lobit = ashr i32 %7, 31, !dbg !9
  %12 = xor i32 %.lobit, %7, !dbg !9
  %13 = sdiv i32 %12, 32, !dbg !9
  %14 = xor i32 %13, %.lobit, !dbg !9
  %15 = sext i32 %14 to i64, !dbg !9
  %16 = add nsw i64 %15, %10, !dbg !9
  %17 = and i64 %16, 31, !dbg !9
  %18 = mul nsw i64 %15, -4096, !dbg !9
  %19 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !10, !range !11
  %20 = shl nsw i64 %10, 7, !dbg !12
  %21 = shl nsw i64 %6, 12, !dbg !12
  %22 = add nsw i64 %20, %21, !dbg !12
  %.not94 = icmp slt i32 %7, 0, !dbg !13
  br i1 %.not94, label %.._crit_edge_crit_edge, label %.lr.ph, !dbg !13

.._crit_edge_crit_edge:                           ; preds = %4
  %.pre = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !13
  %.pre108 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %.pre), !dbg !13
  %.pre110 = add nuw nsw i32 %.pre108, 64, !dbg !13
  %.pre112 = xor i32 %.pre108, 16, !dbg !13
  %.pre114 = and i32 %.pre110, 192, !dbg !13
  %.pre116 = xor i32 %.pre108, 32, !dbg !13
  %.pre118 = add nsw i64 %18, %11, !dbg !14
  %.pre119 = shl nuw nsw i64 %17, 20, !dbg !14
  br label %._crit_edge, !dbg !13

.lr.ph:                                           ; preds = %4
  %23 = shl nsw i64 %6, 17, !dbg !12
  %24 = shl nsw i64 %10, 12, !dbg !12
  %25 = add nsw i64 %18, %11, !dbg !9
  %.idx3 = shl nsw i64 %25, 8, !dbg !9
  %.idx = shl nuw nsw i64 %17, 20, !dbg !9
  %26 = zext nneg i32 %19 to i64, !dbg !10
  %27 = ptrtoint ptr addrspace(1) %0 to i64
  %28 = add nsw i64 %.idx, %.idx3
  %29 = insertelement <2 x i64> poison, i64 %27, i64 0
  %30 = bitcast <2 x i64> %29 to <4 x i32>
  %dma.buffer.rsrc1155 = shufflevector <4 x i32> %30, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %31 = ptrtoint ptr addrspace(1) %1 to i64
  %32 = insertelement <2 x i64> poison, i64 %31, i64 0
  %33 = bitcast <2 x i64> %32 to <4 x i32>
  %dma.buffer.rsrc1149 = shufflevector <4 x i32> %33, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %invariant.op = or disjoint i64 %28, 64, !dbg !13
  %34 = add nsw i64 %23, %24
  %invariant.op99 = or disjoint i64 %28, 128, !dbg !13
  %35 = ptrtoint ptr addrspace(1) %2 to i64
  %36 = insertelement <2 x i64> poison, i64 %35, i64 0
  %37 = bitcast <2 x i64> %36 to <4 x i32>
  %dma.buffer.rsrc1107 = shufflevector <4 x i32> %37, <4 x i32> <i32 poison, i32 poison, i32 33554432, i32 131072>, <4 x i32> <i32 0, i32 1, i32 6, i32 7>
  %38 = add nuw nsw i64 %11, %26
  %39 = lshr i64 %26, 2
  %40 = and i64 %39, 48
  %41 = and i32 %19, 240
  %index.compact = zext nneg i32 %41 to i64
  %42 = sub nsw i64 %38, %index.compact
  %43 = add nsw i64 %42, %40
  %44 = and i64 %26, 48
  %45 = add nsw i64 %43, 1
  %46 = lshr i64 %45, 32
  %compare.high = trunc nuw i64 %46 to i32
  %compare.low.base = trunc i64 %45 to i32
  %47 = add nsw i64 %43, 65
  %48 = lshr i64 %47, 32
  %compare.high956 = trunc nuw i64 %48 to i32
  %compare.low.base960 = trunc i64 %47 to i32
  %49 = insertelement <32 x i32> poison, i32 %compare.low.base960, i64 0, !dbg !15
  %50 = shufflevector <32 x i32> %49, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !15
  %51 = insertelement <32 x i32> poison, i32 %compare.low.base, i64 0, !dbg !15
  %52 = shufflevector <32 x i32> %51, <32 x i32> poison, <32 x i32> zeroinitializer, !dbg !15
  %53 = insertelement <2 x i64> poison, i64 %22, i64 0, !dbg !12
  %54 = insertelement <2 x i64> poison, i64 %34, i64 1
  br label %55, !dbg !13

55:                                               ; preds = %.lr.ph, %510
  %56 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1164, %510 ]
  %57 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1160, %510 ]
  %58 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1156, %510 ]
  %59 = phi [2 x <1 x float>] [ zeroinitializer, %.lr.ph ], [ %853, %510 ]
  %60 = phi [2 x <8 x float>] [ zeroinitializer, %.lr.ph ], [ %1152, %510 ]
  %61 = phi i64 [ 0, %.lr.ph ], [ %1165, %510 ]
  %62 = shl nuw nsw i64 %61, 7, !dbg !13
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !9, !range !11, !frisk.register_lifetimes !5
  %address.local = zext nneg i32 %address.thread to i64, !dbg !10
  %address.local174 = shl nuw nsw i64 %address.local, 3, !dbg !9
  %address.local175 = and i64 %address.local174, 1536, !dbg !9
  %address.local176 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local175, !dbg !9
  %address.local179 = shl nuw nsw i64 %address.local, 7, !dbg !9
  %address.local180 = and i64 %address.local179, 16256, !dbg !9
  %address.local181 = lshr i64 %address.local, 4, !dbg !9
  %address.local182 = and i64 %address.local181, 8, !dbg !9
  %address.local183 = or disjoint i64 %address.local180, %address.local182, !dbg !9
  %address.local184.idx = shl nuw nsw i64 %address.local183, 1, !dbg !9
  %63 = or disjoint i64 %address.local184.idx, %28, !dbg !9
  %dma.buffer.offset1154 = trunc i64 %63 to i32, !dbg !9
  %address.local185 = or disjoint i64 %address.local, 256, !dbg !9
  %address.local186 = shl nuw nsw i64 %address.local185, 3, !dbg !9
  %address.local187 = and i64 %address.local186, 3584, !dbg !9
  %address.local188 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local187, !dbg !9
  %address.local193 = lshr i64 %address.local185, 4, !dbg !9
  %address.local194 = and i64 %address.local193, 24, !dbg !9
  %address.local195 = or disjoint i64 %address.local194, %address.local180, !dbg !9
  %address.local196.idx = shl nuw nsw i64 %address.local195, 1, !dbg !9
  %64 = or disjoint i64 %address.local196.idx, %28, !dbg !9
  %dma.buffer.offset1151 = trunc i64 %64 to i32, !dbg !9
  %65 = lshr i64 %61, 12, !dbg !12
  %66 = add nsw i64 %65, %10, !dbg !12
  %67 = lshr i64 %61, 5, !dbg !12
  %68 = and i64 %67, 127, !dbg !12
  %69 = shl i64 %66, 19, !dbg !12
  %70 = and i64 %69, 16252928, !dbg !12
  %71 = shl i64 %67, 19, !dbg !12
  %72 = shl nuw nsw i64 %61, 14, !dbg !12
  %73 = sub i64 %72, %71, !dbg !12
  %address.local197 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local175, !dbg !12
  %.idx47 = or disjoint i64 %70, %68, !dbg !12
  %74 = add i64 %.idx47, %73, !dbg !12
  %75 = add i64 %address.local183, %74, !dbg !12
  %.tr = trunc i64 %75 to i32, !dbg !12
  %dma.buffer.offset1148 = shl i32 %.tr, 1, !dbg !12
  %address.local201 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local187, !dbg !12
  %76 = add i64 %address.local195, %74, !dbg !12
  %.tr12 = trunc i64 %76 to i32, !dbg !12
  %dma.buffer.offset1145 = shl i32 %.tr12, 1, !dbg !12
  %address.local205 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local175, !dbg !9
  %dma.buffer.offset1142 = or disjoint i32 %dma.buffer.offset1154, 64, !dbg !9
  %address.local209 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local187, !dbg !9
  %.reass = or disjoint i64 %address.local196.idx, %invariant.op
  %dma.buffer.offset1139 = trunc i64 %.reass to i32, !dbg !9
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local175, !dbg !12
  %.idx1519 = add i64 %73, %67, !dbg !12
  %77 = add i64 %.idx1519, %22, !dbg !12
  %78 = insertelement <2 x i64> %54, i64 %67, i64 0, !dbg !12
  %79 = insertelement <2 x i64> %53, i64 %61, i64 1, !dbg !12
  %80 = add nsw <2 x i64> %78, %79, !dbg !12
  %81 = extractelement <2 x i64> %80, i64 1, !dbg !12
  %82 = extractelement <2 x i64> %80, i64 0, !dbg !12
  %83 = add nsw i64 %81, 1024, !dbg !12
  %84 = add nsw <2 x i64> %80, <i64 32, i64 1024>, !dbg !12
  %.lobit4 = ashr i64 %83, 63, !dbg !12
  %85 = icmp slt <2 x i64> %80, <i64 -32, i64 -1024>, !dbg !12
  %86 = icmp slt i64 %82, -32, !dbg !12
  %87 = sub <2 x i64> <i64 -33, i64 -1025>, %80, !dbg !12
  %88 = select <2 x i1> %85, <2 x i64> %87, <2 x i64> %84, !dbg !12
  %89 = sext i1 %86 to i64, !dbg !12
  %90 = insertelement <2 x i64> poison, i64 %89, i64 0, !dbg !12
  %91 = insertelement <2 x i64> %90, i64 %.lobit4, i64 1, !dbg !12
  %address.local217 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local187, !dbg !12
  %92 = insertelement <8 x ptr addrspace(3)> poison, ptr addrspace(3) %address.local176, i64 0, !dbg !9
  %93 = insertelement <8 x ptr addrspace(3)> %92, ptr addrspace(3) %address.local188, i64 1, !dbg !9
  %94 = insertelement <8 x ptr addrspace(3)> %93, ptr addrspace(3) %address.local197, i64 2, !dbg !9
  %95 = insertelement <8 x ptr addrspace(3)> %94, ptr addrspace(3) %address.local201, i64 3, !dbg !9
  %96 = insertelement <8 x ptr addrspace(3)> %95, ptr addrspace(3) %address.local205, i64 4, !dbg !9
  %97 = insertelement <8 x ptr addrspace(3)> %96, ptr addrspace(3) %address.local209, i64 5, !dbg !9
  %98 = insertelement <8 x ptr addrspace(3)> %97, ptr addrspace(3) %address.local213, i64 6, !dbg !9
  %99 = insertelement <8 x ptr addrspace(3)> %98, ptr addrspace(3) %address.local217, i64 7, !dbg !9
  %100 = ptrtoint <8 x ptr addrspace(3)> %99 to <8 x i64>, !dbg !9
  %101 = trunc <8 x i64> %100 to <8 x i32>, !dbg !9
  %102 = extractelement <8 x i32> %101, i64 0, !dbg !9
  %103 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %102), !dbg !9
  %dma.buffer.lds1156 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %103), !dbg !9
  %104 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1156, i32 %dma.buffer.offset1154, <4 x i32> %dma.buffer.rsrc1155) #9, !dbg !9
  %105 = extractelement <8 x i32> %101, i64 1, !dbg !9
  %106 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %105), !dbg !9
  %dma.buffer.lds1153 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %106), !dbg !9
  %107 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1153, i32 %dma.buffer.offset1151, <4 x i32> %dma.buffer.rsrc1155) #9, !dbg !9
  %108 = extractelement <8 x i32> %101, i64 2, !dbg !12
  %109 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %108), !dbg !12
  %dma.buffer.lds1150 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %109), !dbg !12
  %110 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1150, i32 %dma.buffer.offset1148, <4 x i32> %dma.buffer.rsrc1149) #9, !dbg !12
  %111 = extractelement <8 x i32> %101, i64 3, !dbg !12
  %112 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %111), !dbg !12
  %dma.buffer.lds1147 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %112), !dbg !12
  %113 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1147, i32 %dma.buffer.offset1145, <4 x i32> %dma.buffer.rsrc1149) #9, !dbg !12
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %114 = extractelement <8 x i32> %101, i64 4, !dbg !9
  %115 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %114), !dbg !9
  %dma.buffer.lds1144 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %115), !dbg !9
  %116 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1144, i32 %dma.buffer.offset1142, <4 x i32> %dma.buffer.rsrc1155) #9, !dbg !9
  %117 = extractelement <8 x i32> %101, i64 5, !dbg !9
  %118 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %117), !dbg !9
  %dma.buffer.lds1141 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %118), !dbg !9
  %119 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1141, i32 %dma.buffer.offset1139, <4 x i32> %dma.buffer.rsrc1155) #9, !dbg !9
  %120 = extractelement <8 x i32> %101, i64 6, !dbg !12
  %121 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %120), !dbg !12
  %122 = sdiv <2 x i64> %88, <i64 128, i64 4096>, !dbg !12
  %123 = xor <2 x i64> %122, %91, !dbg !12
  %124 = shl <2 x i64> %123, <i64 7, i64 19>, !dbg !12
  %125 = extractelement <2 x i64> %124, i64 0, !dbg !12
  %126 = sub i64 %77, %125, !dbg !12
  %127 = extractelement <2 x i64> %124, i64 1, !dbg !12
  %128 = and i64 %127, 16252928, !dbg !12
  %129 = add i64 %126, %128, !dbg !12
  %130 = add i64 %address.local183, %129, !dbg !12
  %.tr20 = trunc i64 %130 to i32, !dbg !12
  %131 = shl i32 %.tr20, 1, !dbg !12
  %dma.buffer.offset1136 = add i32 %131, 64, !dbg !12
  %dma.buffer.lds1138 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %121), !dbg !12
  %132 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1138, i32 %dma.buffer.offset1136, <4 x i32> %dma.buffer.rsrc1149) #9, !dbg !12
  %133 = extractelement <8 x i32> %101, i64 7, !dbg !12
  %134 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %133), !dbg !12
  %135 = add i64 %address.local195, %129, !dbg !12
  %.tr26 = trunc i64 %135 to i32, !dbg !12
  %136 = shl i32 %.tr26, 1, !dbg !12
  %dma.buffer.offset1133 = add i32 %136, 64, !dbg !12
  %dma.buffer.lds1135 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %134), !dbg !12
  %137 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1135, i32 %dma.buffer.offset1133, <4 x i32> %dma.buffer.rsrc1149) #9, !dbg !12
  %address.local221 = lshr i32 %address.thread, 2, !dbg !10
  %address.local222 = and i32 %address.local221, 4, !dbg !10
  %address.local223 = shl nuw nsw i32 %address.thread, 1, !dbg !10
  %address.local224 = and i32 %address.local223, 384, !dbg !10
  %address.local226 = shl nuw nsw i32 %address.thread, 3, !dbg !10
  %address.local227 = and i32 %address.local226, 120, !dbg !10
  %address.local229 = shl nuw nsw i32 %address.thread, 5, !dbg !10
  %address.local230 = and i32 %address.local229, 1024, !dbg !10
  %138 = or disjoint i32 %address.local227, %address.local224, !dbg !10
  %139 = or disjoint i32 %138, %address.local222, !dbg !10
  %address.local231 = or disjoint i32 %139, %address.local230, !dbg !10
  %address.local232 = zext nneg i32 %address.local231 to i64, !dbg !10
  %address.local233 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local232, !dbg !10
  %140 = load <4 x half>, ptr addrspace(3) %address.local233, align 8, !dbg !10
  %address.local245 = or disjoint i32 %address.local231, 2048, !dbg !10
  %address.local246 = zext nneg i32 %address.local245 to i64, !dbg !10
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local246, !dbg !10
  %141 = load <4 x half>, ptr addrspace(3) %address.local247, align 2, !dbg !10
  %address.local259 = or disjoint i32 %address.local231, 512, !dbg !10
  %address.local260 = zext nneg i32 %address.local259 to i64, !dbg !10
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local260, !dbg !10
  %142 = load <4 x half>, ptr addrspace(3) %address.local261, align 2, !dbg !10
  %143 = fmul <4 x half> %140, splat (half 1.275630e-01), !dbg !10
  %address.local273 = or disjoint i32 %address.local231, 2560, !dbg !10
  %address.local274 = zext nneg i32 %address.local273 to i64, !dbg !10
  %address.local275 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local274, !dbg !10
  %144 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !10
  %145 = fmul <4 x half> %141, splat (half 1.275630e-01), !dbg !10
  %address.local279 = and i32 %address.local223, 24, !dbg !17
  %address.local284 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local285 = and i32 %address.local284, 384, !dbg !17
  %146 = or disjoint i32 %address.local285, %address.local279, !dbg !17
  %147 = or disjoint i32 %146, %address.local222, !dbg !17
  %address.local286 = or disjoint i32 %147, %address.local230, !dbg !17
  %address.local287 = zext nneg i32 %address.local286 to i64, !dbg !17
  %address.local288 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local287, !dbg !17
  %148 = load <4 x half>, ptr addrspace(3) %address.local288, align 8, !dbg !17
  %149 = fmul <4 x half> %142, splat (half 1.275630e-01), !dbg !10
  %address.local300 = or disjoint i32 %address.local286, 2048, !dbg !17
  %address.local301 = zext nneg i32 %address.local300 to i64, !dbg !17
  %address.local302 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local301, !dbg !17
  %150 = load <4 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !17
  %151 = fmul <4 x half> %144, splat (half 1.275630e-01), !dbg !10
  %address.local314 = or disjoint i32 %address.local286, 32, !dbg !17
  %address.local315 = zext nneg i32 %address.local314 to i64, !dbg !17
  %address.local316 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local315, !dbg !17
  %152 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !17
  %153 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %148, <4 x float> zeroinitializer) #8, !dbg !17
  %154 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %150, <4 x float> %153) #8, !dbg !17
  %address.local328 = or disjoint i32 %address.local286, 2080, !dbg !17
  %address.local329 = zext nneg i32 %address.local328 to i64, !dbg !17
  %address.local330 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local329, !dbg !17
  %155 = load <4 x half>, ptr addrspace(3) %address.local330, align 2, !dbg !17
  %156 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %152, <4 x float> zeroinitializer) #8, !dbg !17
  %157 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %155, <4 x float> %156) #8, !dbg !17
  %address.local342 = or disjoint i32 %address.local286, 64, !dbg !17
  %address.local343 = zext nneg i32 %address.local342 to i64, !dbg !17
  %address.local344 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local343, !dbg !17
  %158 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !17
  %address.local356 = or disjoint i32 %address.local286, 2112, !dbg !17
  %address.local357 = zext nneg i32 %address.local356 to i64, !dbg !17
  %address.local358 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local357, !dbg !17
  %159 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !17
  %160 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %158, <4 x float> zeroinitializer) #8, !dbg !17
  %address.local370 = or disjoint i32 %address.local286, 96, !dbg !17
  %address.local371 = zext nneg i32 %address.local370 to i64, !dbg !17
  %address.local372 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local371, !dbg !17
  %161 = load <4 x half>, ptr addrspace(3) %address.local372, align 2, !dbg !17
  %162 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %159, <4 x float> %160) #8, !dbg !17
  %address.local384 = or disjoint i32 %address.local286, 2144, !dbg !17
  %address.local385 = zext nneg i32 %address.local384 to i64, !dbg !17
  %address.local386 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local385, !dbg !17
  %163 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !17
  %164 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %161, <4 x float> zeroinitializer) #8, !dbg !17
  %165 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %163, <4 x float> %164) #8, !dbg !17
  %address.local398 = or disjoint i32 %address.local286, 512, !dbg !17
  %address.local399 = zext nneg i32 %address.local398 to i64, !dbg !17
  %address.local400 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local399, !dbg !17
  %166 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !17
  %address.local412 = or disjoint i32 %address.local286, 2560, !dbg !17
  %address.local413 = zext nneg i32 %address.local412 to i64, !dbg !17
  %address.local414 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local413, !dbg !17
  %167 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !17
  %168 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %166, <4 x float> zeroinitializer) #8, !dbg !17
  %address.local426 = or disjoint i32 %address.local286, 544, !dbg !17
  %address.local427 = zext nneg i32 %address.local426 to i64, !dbg !17
  %address.local428 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local427, !dbg !17
  %169 = load <4 x half>, ptr addrspace(3) %address.local428, align 2, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %170 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %167, <4 x float> %168) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %address.local440 = or disjoint i32 %address.local286, 2592, !dbg !17
  %address.local441 = zext nneg i32 %address.local440 to i64, !dbg !17
  %address.local442 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local441, !dbg !17
  %171 = load <4 x half>, ptr addrspace(3) %address.local442, align 2, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %172 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %169, <4 x float> zeroinitializer) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %173 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %171, <4 x float> %172) #8, !dbg !17
  %address.local454 = or disjoint i32 %address.local286, 576, !dbg !17
  %address.local455 = zext nneg i32 %address.local454 to i64, !dbg !17
  %address.local456 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local455, !dbg !17
  %174 = load <4 x half>, ptr addrspace(3) %address.local456, align 2, !dbg !17
  %address.local468 = or disjoint i32 %address.local286, 2624, !dbg !17
  %address.local469 = zext nneg i32 %address.local468 to i64, !dbg !17
  %address.local470 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local469, !dbg !17
  %175 = load <4 x half>, ptr addrspace(3) %address.local470, align 2, !dbg !17
  %176 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %174, <4 x float> zeroinitializer) #8, !dbg !17
  %address.local482 = or disjoint i32 %address.local286, 608, !dbg !17
  %address.local483 = zext nneg i32 %address.local482 to i64, !dbg !17
  %address.local484 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local483, !dbg !17
  %177 = load <4 x half>, ptr addrspace(3) %address.local484, align 2, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %178 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %175, <4 x float> %176) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %address.local496 = or disjoint i32 %address.local286, 2656, !dbg !17
  %address.local497 = zext nneg i32 %address.local496 to i64, !dbg !17
  %address.local498 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 16384), i64 %address.local497, !dbg !17
  %179 = load <4 x half>, ptr addrspace(3) %address.local498, align 2, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %180 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %143, <4 x half> %177, <4 x float> zeroinitializer) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %181 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %145, <4 x half> %179, <4 x float> %180) #8, !dbg !17
  %182 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %148, <4 x float> zeroinitializer) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %183 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %150, <4 x float> %182) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %184 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %152, <4 x float> zeroinitializer) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %185 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %155, <4 x float> %184) #8, !dbg !17
  %186 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %158, <4 x float> zeroinitializer) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %187 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %159, <4 x float> %186) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %188 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %161, <4 x float> zeroinitializer) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %189 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %163, <4 x float> %188) #8, !dbg !17
  %190 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %166, <4 x float> zeroinitializer) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %191 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %167, <4 x float> %190) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %192 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %169, <4 x float> zeroinitializer) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %193 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %171, <4 x float> %192) #8, !dbg !17
  %194 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %174, <4 x float> zeroinitializer) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %195 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %175, <4 x float> %194) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %196 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %149, <4 x half> %177, <4 x float> zeroinitializer) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %197 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %151, <4 x half> %179, <4 x float> %196) #8, !dbg !17
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %dma.buffer.offset1130 = or disjoint i32 %dma.buffer.offset1154, 128, !dbg !9
  %198 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1156, i32 %dma.buffer.offset1130, <4 x i32> %dma.buffer.rsrc1155) #9, !dbg !9
  %.reass100 = or disjoint i64 %address.local196.idx, %invariant.op99
  %dma.buffer.offset1127 = trunc i64 %.reass100 to i32, !dbg !9
  %199 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1153, i32 %dma.buffer.offset1127, <4 x i32> %dma.buffer.rsrc1155) #9, !dbg !9
  %200 = add nsw i64 %81, 2048, !dbg !12
  %201 = add nsw <2 x i64> %80, <i64 64, i64 2048>, !dbg !12
  %.lobit5 = ashr i64 %200, 63, !dbg !12
  %202 = icmp slt <2 x i64> %80, <i64 -64, i64 -2048>, !dbg !12
  %203 = icmp slt i64 %82, -64, !dbg !12
  %204 = sub <2 x i64> <i64 -65, i64 -2049>, %80, !dbg !12
  %205 = select <2 x i1> %202, <2 x i64> %204, <2 x i64> %201, !dbg !12
  %206 = sdiv <2 x i64> %205, <i64 128, i64 4096>, !dbg !12
  %207 = sext i1 %203 to i64, !dbg !12
  %208 = insertelement <2 x i64> poison, i64 %207, i64 0, !dbg !12
  %209 = insertelement <2 x i64> %208, i64 %.lobit5, i64 1, !dbg !12
  %210 = xor <2 x i64> %206, %209, !dbg !12
  %211 = shl <2 x i64> %210, <i64 7, i64 19>, !dbg !12
  %212 = extractelement <2 x i64> %211, i64 0, !dbg !12
  %213 = sub i64 %77, %212, !dbg !12
  %214 = extractelement <2 x i64> %211, i64 1, !dbg !12
  %215 = and i64 %214, 16252928, !dbg !12
  %216 = add i64 %213, %215, !dbg !12
  %217 = add i64 %address.local183, %216, !dbg !12
  %.tr34 = trunc i64 %217 to i32, !dbg !12
  %218 = shl i32 %.tr34, 1, !dbg !12
  %dma.buffer.offset1124 = add i32 %218, 128, !dbg !12
  %219 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1150, i32 %dma.buffer.offset1124, <4 x i32> %dma.buffer.rsrc1149) #9, !dbg !12
  %220 = add i64 %address.local195, %216, !dbg !12
  %.tr40 = trunc i64 %220 to i32, !dbg !12
  %221 = shl i32 %.tr40, 1, !dbg !12
  %dma.buffer.offset1121 = add i32 %221, 128, !dbg !12
  %222 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1147, i32 %dma.buffer.offset1121, <4 x i32> %dma.buffer.rsrc1149) #9, !dbg !12
  %address.local503 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local232, !dbg !10
  %223 = load <4 x half>, ptr addrspace(3) %address.local503, align 8, !dbg !10
  %224 = fmul <4 x half> %223, splat (half 1.275630e-01), !dbg !10
  %address.local504 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local246, !dbg !10
  %225 = load <4 x half>, ptr addrspace(3) %address.local504, align 2, !dbg !10
  %address.local505 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local260, !dbg !10
  %226 = load <4 x half>, ptr addrspace(3) %address.local505, align 2, !dbg !10
  %227 = fmul <4 x half> %225, splat (half 1.275630e-01), !dbg !10
  %228 = fmul <4 x half> %226, splat (half 1.275630e-01), !dbg !10
  %address.local506 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local274, !dbg !10
  %229 = load <4 x half>, ptr addrspace(3) %address.local506, align 2, !dbg !10
  %address.local507 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local287, !dbg !17
  %230 = load <4 x half>, ptr addrspace(3) %address.local507, align 8, !dbg !17
  %231 = fmul <4 x half> %229, splat (half 1.275630e-01), !dbg !10
  %232 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %230, <4 x float> %154) #8, !dbg !17
  %address.local508 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local301, !dbg !17
  %233 = load <4 x half>, ptr addrspace(3) %address.local508, align 2, !dbg !17
  %address.local509 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local315, !dbg !17
  %234 = load <4 x half>, ptr addrspace(3) %address.local509, align 2, !dbg !17
  %235 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %233, <4 x float> %232) #8, !dbg !17
  %236 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %234, <4 x float> %157) #8, !dbg !17
  %address.local510 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local329, !dbg !17
  %237 = load <4 x half>, ptr addrspace(3) %address.local510, align 2, !dbg !17
  %address.local511 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local343, !dbg !17
  %238 = load <4 x half>, ptr addrspace(3) %address.local511, align 2, !dbg !17
  %239 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %237, <4 x float> %236) #8, !dbg !17
  %240 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %238, <4 x float> %162) #8, !dbg !17
  %address.local512 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local357, !dbg !17
  %241 = load <4 x half>, ptr addrspace(3) %address.local512, align 2, !dbg !17
  %address.local513 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local371, !dbg !17
  %242 = load <4 x half>, ptr addrspace(3) %address.local513, align 2, !dbg !17
  %243 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %241, <4 x float> %240) #8, !dbg !17
  %244 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %242, <4 x float> %165) #8, !dbg !17
  %address.local514 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local385, !dbg !17
  %245 = load <4 x half>, ptr addrspace(3) %address.local514, align 2, !dbg !17
  %address.local515 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local399, !dbg !17
  %246 = load <4 x half>, ptr addrspace(3) %address.local515, align 2, !dbg !17
  %247 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %245, <4 x float> %244) #8, !dbg !17
  %248 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %246, <4 x float> %170) #8, !dbg !17
  %address.local516 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local413, !dbg !17
  %249 = load <4 x half>, ptr addrspace(3) %address.local516, align 2, !dbg !17
  %address.local517 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local427, !dbg !17
  %250 = load <4 x half>, ptr addrspace(3) %address.local517, align 2, !dbg !17
  %251 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %249, <4 x float> %248) #8, !dbg !17
  %252 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %250, <4 x float> %173) #8, !dbg !17
  %address.local518 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local441, !dbg !17
  %253 = load <4 x half>, ptr addrspace(3) %address.local518, align 2, !dbg !17
  %address.local519 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local455, !dbg !17
  %254 = load <4 x half>, ptr addrspace(3) %address.local519, align 2, !dbg !17
  %255 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %253, <4 x float> %252) #8, !dbg !17
  %256 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %254, <4 x float> %178) #8, !dbg !17
  %address.local520 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local469, !dbg !17
  %257 = load <4 x half>, ptr addrspace(3) %address.local520, align 2, !dbg !17
  %address.local521 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local483, !dbg !17
  %258 = load <4 x half>, ptr addrspace(3) %address.local521, align 2, !dbg !17
  %259 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %257, <4 x float> %256) #8, !dbg !17
  %260 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %224, <4 x half> %258, <4 x float> %181) #8, !dbg !17
  %address.local522 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 24576), i64 %address.local497, !dbg !17
  %261 = load <4 x half>, ptr addrspace(3) %address.local522, align 2, !dbg !17
  %262 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %227, <4 x half> %261, <4 x float> %260) #8, !dbg !17
  %263 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %230, <4 x float> %183) #8, !dbg !17
  %264 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %231, <4 x half> %233, <4 x float> %263) #8, !dbg !17
  %265 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %234, <4 x float> %185) #8, !dbg !17
  %266 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %231, <4 x half> %237, <4 x float> %265) #8, !dbg !17
  %267 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %238, <4 x float> %187) #8, !dbg !17
  %268 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %231, <4 x half> %241, <4 x float> %267) #8, !dbg !17
  %269 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %242, <4 x float> %189) #8, !dbg !17
  %270 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %231, <4 x half> %245, <4 x float> %269) #8, !dbg !17
  %271 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %246, <4 x float> %191) #8, !dbg !17
  %272 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %231, <4 x half> %249, <4 x float> %271) #8, !dbg !17
  %273 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %250, <4 x float> %193) #8, !dbg !17
  %274 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %231, <4 x half> %253, <4 x float> %273) #8, !dbg !17
  %275 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %254, <4 x float> %195) #8, !dbg !17
  %276 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %231, <4 x half> %257, <4 x float> %275) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %277 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %228, <4 x half> %258, <4 x float> %197) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %278 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %231, <4 x half> %261, <4 x float> %277) #8, !dbg !17
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %dma.buffer.offset1118 = or disjoint i32 %dma.buffer.offset1154, 192, !dbg !9
  %279 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1144, i32 %dma.buffer.offset1118, <4 x i32> %dma.buffer.rsrc1155) #9, !dbg !9
  %dma.buffer.offset1115 = or disjoint i32 %dma.buffer.offset1151, 192, !dbg !9
  %280 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1141, i32 %dma.buffer.offset1115, <4 x i32> %dma.buffer.rsrc1155) #9, !dbg !9
  %281 = add nsw i64 %81, 3072, !dbg !12
  %282 = add nsw <2 x i64> %80, <i64 96, i64 3072>, !dbg !12
  %.lobit6 = ashr i64 %281, 63, !dbg !12
  %283 = icmp slt <2 x i64> %80, <i64 -96, i64 -3072>, !dbg !12
  %284 = icmp slt i64 %82, -96, !dbg !12
  %285 = sub <2 x i64> <i64 -97, i64 -3073>, %80, !dbg !12
  %286 = select <2 x i1> %283, <2 x i64> %285, <2 x i64> %282, !dbg !12
  %287 = sdiv <2 x i64> %286, <i64 128, i64 4096>, !dbg !12
  %288 = sext i1 %284 to i64, !dbg !12
  %289 = insertelement <2 x i64> poison, i64 %288, i64 0, !dbg !12
  %290 = insertelement <2 x i64> %289, i64 %.lobit6, i64 1, !dbg !12
  %291 = xor <2 x i64> %287, %290, !dbg !12
  %292 = shl <2 x i64> %291, <i64 7, i64 19>, !dbg !12
  %293 = extractelement <2 x i64> %292, i64 0, !dbg !12
  %294 = sub i64 %77, %293, !dbg !12
  %295 = extractelement <2 x i64> %292, i64 1, !dbg !12
  %296 = and i64 %295, 16252928, !dbg !12
  %297 = add i64 %294, %296, !dbg !12
  %298 = add i64 %address.local183, %297, !dbg !12
  %.tr48 = trunc i64 %298 to i32, !dbg !12
  %299 = shl i32 %.tr48, 1, !dbg !12
  %dma.buffer.offset1112 = add i32 %299, 192, !dbg !12
  %300 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1138, i32 %dma.buffer.offset1112, <4 x i32> %dma.buffer.rsrc1149) #9, !dbg !12
  %301 = add i64 %address.local195, %297, !dbg !12
  %.tr54 = trunc i64 %301 to i32, !dbg !12
  %302 = shl i32 %.tr54, 1, !dbg !12
  %dma.buffer.offset1109 = add i32 %302, 192, !dbg !12
  %303 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1135, i32 %dma.buffer.offset1109, <4 x i32> %dma.buffer.rsrc1149) #9, !dbg !12
  %304 = load <4 x half>, ptr addrspace(3) %address.local233, align 8, !dbg !10
  %305 = fmul <4 x half> %304, splat (half 1.275630e-01), !dbg !10
  %306 = load <4 x half>, ptr addrspace(3) %address.local247, align 2, !dbg !10
  %307 = load <4 x half>, ptr addrspace(3) %address.local261, align 2, !dbg !10
  %308 = fmul <4 x half> %306, splat (half 1.275630e-01), !dbg !10
  %309 = fmul <4 x half> %307, splat (half 1.275630e-01), !dbg !10
  %310 = load <4 x half>, ptr addrspace(3) %address.local275, align 2, !dbg !10
  %311 = load <4 x half>, ptr addrspace(3) %address.local288, align 8, !dbg !17
  %312 = fmul <4 x half> %310, splat (half 1.275630e-01), !dbg !10
  %313 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %305, <4 x half> %311, <4 x float> %235) #8, !dbg !17
  %314 = load <4 x half>, ptr addrspace(3) %address.local302, align 2, !dbg !17
  %315 = load <4 x half>, ptr addrspace(3) %address.local316, align 2, !dbg !17
  %316 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %308, <4 x half> %314, <4 x float> %313) #8, !dbg !17
  %317 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %305, <4 x half> %315, <4 x float> %239) #8, !dbg !17
  %318 = load <4 x half>, ptr addrspace(3) %address.local330, align 2, !dbg !17
  %319 = load <4 x half>, ptr addrspace(3) %address.local344, align 2, !dbg !17
  %320 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %308, <4 x half> %318, <4 x float> %317) #8, !dbg !17
  %321 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %305, <4 x half> %319, <4 x float> %243) #8, !dbg !17
  %322 = load <4 x half>, ptr addrspace(3) %address.local358, align 2, !dbg !17
  %323 = load <4 x half>, ptr addrspace(3) %address.local372, align 2, !dbg !17
  %324 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %308, <4 x half> %322, <4 x float> %321) #8, !dbg !17
  %325 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %305, <4 x half> %323, <4 x float> %247) #8, !dbg !17
  %326 = load <4 x half>, ptr addrspace(3) %address.local386, align 2, !dbg !17
  %327 = load <4 x half>, ptr addrspace(3) %address.local400, align 2, !dbg !17
  %328 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %308, <4 x half> %326, <4 x float> %325) #8, !dbg !17
  %329 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %305, <4 x half> %327, <4 x float> %251) #8, !dbg !17
  %330 = load <4 x half>, ptr addrspace(3) %address.local414, align 2, !dbg !17
  %331 = load <4 x half>, ptr addrspace(3) %address.local428, align 2, !dbg !17
  %332 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %308, <4 x half> %330, <4 x float> %329) #8, !dbg !17
  %333 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %305, <4 x half> %331, <4 x float> %255) #8, !dbg !17
  %334 = load <4 x half>, ptr addrspace(3) %address.local442, align 2, !dbg !17
  %335 = load <4 x half>, ptr addrspace(3) %address.local456, align 2, !dbg !17
  %336 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %308, <4 x half> %334, <4 x float> %333) #8, !dbg !17
  %337 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %305, <4 x half> %335, <4 x float> %259) #8, !dbg !17
  %338 = load <4 x half>, ptr addrspace(3) %address.local470, align 2, !dbg !17
  %339 = load <4 x half>, ptr addrspace(3) %address.local484, align 2, !dbg !17
  %340 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %308, <4 x half> %338, <4 x float> %337) #8, !dbg !17
  %341 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %305, <4 x half> %339, <4 x float> %262) #8, !dbg !17
  %342 = load <4 x half>, ptr addrspace(3) %address.local498, align 2, !dbg !17
  %343 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %308, <4 x half> %342, <4 x float> %341) #8, !dbg !17
  %344 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %309, <4 x half> %311, <4 x float> %264) #8, !dbg !17
  %345 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %314, <4 x float> %344) #8, !dbg !17
  %346 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %309, <4 x half> %315, <4 x float> %266) #8, !dbg !17
  %347 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %318, <4 x float> %346) #8, !dbg !17
  %348 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %309, <4 x half> %319, <4 x float> %268) #8, !dbg !17
  %349 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %322, <4 x float> %348) #8, !dbg !17
  %350 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %309, <4 x half> %323, <4 x float> %270) #8, !dbg !17
  %351 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %326, <4 x float> %350) #8, !dbg !17
  %352 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %309, <4 x half> %327, <4 x float> %272) #8, !dbg !17
  %353 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %330, <4 x float> %352) #8, !dbg !17
  %354 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %309, <4 x half> %331, <4 x float> %274) #8, !dbg !17
  %355 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %334, <4 x float> %354) #8, !dbg !17
  %356 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %309, <4 x half> %335, <4 x float> %276) #8, !dbg !17
  %357 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %338, <4 x float> %356) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %358 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %309, <4 x half> %339, <4 x float> %278) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %359 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %312, <4 x half> %342, <4 x float> %358) #8, !dbg !17
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  tail call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %360 = add nsw i64 %67, %10, !dbg !18
  %361 = shl i64 %360, 19, !dbg !18
  %362 = and i64 %361, 16252928, !dbg !18
  %address.local527 = shl nuw nsw i64 %address.local, 6, !dbg !18
  %address.local528 = and i64 %address.local527, 16128, !dbg !18
  %address.local530 = and i64 %address.local174, 24, !dbg !18
  %.idx5558 = add i64 %362, %73, !dbg !18
  %363 = or disjoint i64 %address.local530, %.idx5558, !dbg !18
  %364 = shl i64 %363, 1, !dbg !18
  %365 = or disjoint i64 %364, %address.local528, !dbg !18
  %dma.buffer.offset1106 = trunc i64 %365 to i32, !dbg !18
  %366 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1156, i32 %dma.buffer.offset1106, <4 x i32> %dma.buffer.rsrc1107) #9, !dbg !18
  %address.local533 = shl nuw nsw i64 %address.local185, 6, !dbg !18
  %address.local534 = and i64 %address.local533, 32512, !dbg !18
  %367 = or disjoint i64 %364, %address.local534, !dbg !18
  %dma.buffer.offset1103 = trunc i64 %367 to i32, !dbg !18
  %368 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1153, i32 %dma.buffer.offset1103, <4 x i32> %dma.buffer.rsrc1107) #9, !dbg !18
  %369 = load <4 x half>, ptr addrspace(3) %address.local503, align 8, !dbg !10
  %370 = fmul <4 x half> %369, splat (half 1.275630e-01), !dbg !10
  %371 = load <4 x half>, ptr addrspace(3) %address.local504, align 2, !dbg !10
  %372 = load <4 x half>, ptr addrspace(3) %address.local505, align 2, !dbg !10
  %373 = fmul <4 x half> %371, splat (half 1.275630e-01), !dbg !10
  %374 = fmul <4 x half> %372, splat (half 1.275630e-01), !dbg !10
  %375 = load <4 x half>, ptr addrspace(3) %address.local506, align 2, !dbg !10
  %376 = load <4 x half>, ptr addrspace(3) %address.local507, align 8, !dbg !17
  %377 = fmul <4 x half> %375, splat (half 1.275630e-01), !dbg !10
  %378 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %370, <4 x half> %376, <4 x float> %316) #8, !dbg !17
  %379 = load <4 x half>, ptr addrspace(3) %address.local508, align 2, !dbg !17
  %380 = load <4 x half>, ptr addrspace(3) %address.local509, align 2, !dbg !17
  %381 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %373, <4 x half> %379, <4 x float> %378) #8, !dbg !17
  %382 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %370, <4 x half> %380, <4 x float> %320) #8, !dbg !17
  %383 = load <4 x half>, ptr addrspace(3) %address.local510, align 2, !dbg !17
  %384 = load <4 x half>, ptr addrspace(3) %address.local511, align 2, !dbg !17
  %385 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %373, <4 x half> %383, <4 x float> %382) #8, !dbg !17
  %386 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %370, <4 x half> %384, <4 x float> %324) #8, !dbg !17
  %387 = load <4 x half>, ptr addrspace(3) %address.local512, align 2, !dbg !17
  %388 = load <4 x half>, ptr addrspace(3) %address.local513, align 2, !dbg !17
  %389 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %373, <4 x half> %387, <4 x float> %386) #8, !dbg !17
  %390 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %370, <4 x half> %388, <4 x float> %328) #8, !dbg !17
  %391 = load <4 x half>, ptr addrspace(3) %address.local514, align 2, !dbg !17
  %392 = load <4 x half>, ptr addrspace(3) %address.local515, align 2, !dbg !17
  %393 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %373, <4 x half> %391, <4 x float> %390) #8, !dbg !17
  %394 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %370, <4 x half> %392, <4 x float> %332) #8, !dbg !17
  %395 = load <4 x half>, ptr addrspace(3) %address.local516, align 2, !dbg !17
  %396 = load <4 x half>, ptr addrspace(3) %address.local517, align 2, !dbg !17
  %397 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %373, <4 x half> %395, <4 x float> %394) #8, !dbg !17
  %398 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %370, <4 x half> %396, <4 x float> %336) #8, !dbg !17
  %399 = load <4 x half>, ptr addrspace(3) %address.local518, align 2, !dbg !17
  %400 = load <4 x half>, ptr addrspace(3) %address.local519, align 2, !dbg !17
  %401 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %373, <4 x half> %399, <4 x float> %398) #8, !dbg !17
  %402 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %370, <4 x half> %400, <4 x float> %340) #8, !dbg !17
  %403 = load <4 x half>, ptr addrspace(3) %address.local520, align 2, !dbg !17
  %404 = load <4 x half>, ptr addrspace(3) %address.local521, align 2, !dbg !17
  %405 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %373, <4 x half> %403, <4 x float> %402) #8, !dbg !17
  %406 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %370, <4 x half> %404, <4 x float> %343) #8, !dbg !17
  %407 = load <4 x half>, ptr addrspace(3) %address.local522, align 2, !dbg !17
  %408 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %373, <4 x half> %407, <4 x float> %406) #8, !dbg !17
  %409 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %374, <4 x half> %376, <4 x float> %345) #8, !dbg !17
  %410 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %379, <4 x float> %409) #8, !dbg !17
  %411 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %374, <4 x half> %380, <4 x float> %347) #8, !dbg !17
  %412 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %383, <4 x float> %411) #8, !dbg !17
  %413 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %374, <4 x half> %384, <4 x float> %349) #8, !dbg !17
  %414 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %387, <4 x float> %413) #8, !dbg !17
  %415 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %374, <4 x half> %388, <4 x float> %351) #8, !dbg !17
  %416 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %391, <4 x float> %415) #8, !dbg !17
  %417 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %374, <4 x half> %392, <4 x float> %353) #8, !dbg !17
  %418 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %395, <4 x float> %417) #8, !dbg !17
  %419 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %374, <4 x half> %396, <4 x float> %355) #8, !dbg !17
  %420 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %399, <4 x float> %419) #8, !dbg !17
  %421 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %374, <4 x half> %400, <4 x float> %357) #8, !dbg !17
  %422 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %403, <4 x float> %421) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !17
  %423 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %374, <4 x half> %404, <4 x float> %359) #8, !dbg !17
  tail call void @llvm.amdgcn.sched.barrier(i32 0) #10, !dbg !19
  %424 = or disjoint i64 %62, 127, !dbg !19
  %.not102 = icmp ult i64 %11, %424, !dbg !19
  %425 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %377, <4 x half> %407, <4 x float> %423) #8, !dbg !17
  br i1 %.not102, label %426, label %510, !dbg !19

426:                                              ; preds = %55
  %427 = or disjoint i64 %62, %44, !dbg !19
  %428 = lshr i64 %61, 25, !dbg !15
  %compare.high832 = trunc nuw i64 %428 to i32, !dbg !15
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high832, !dbg !15
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high832, !dbg !15
  %compare.low.base833 = trunc i64 %427 to i32, !dbg !15
  %compare.local.base = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 1) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base836 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 2) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base840 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 3) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base844 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 4) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base848 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 5) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base852 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 6) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base856 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 7) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base860 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 8) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base864 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 9) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base868 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 10) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base872 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 11) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base876 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 12) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base880 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 13) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base884 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 14) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base888 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 15) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base892 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 16) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base896 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 17) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base900 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 18) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base904 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 19) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base908 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 20) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base912 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 21) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base916 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 22) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base920 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 23) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base924 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 24) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base928 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 25) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base932 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 26) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base936 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 27) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base940 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 28) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base944 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 29) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base948 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 30) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base952 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 31) #8, !dbg !15, !frisk.register_lifetimes !5
  %429 = insertelement <32 x i32> poison, i32 %compare.low.base833, i64 0, !dbg !15
  %430 = insertelement <32 x i32> %429, i32 %compare.local.base, i64 1, !dbg !15
  %431 = insertelement <32 x i32> %430, i32 %compare.local.base836, i64 2, !dbg !15
  %432 = insertelement <32 x i32> %431, i32 %compare.local.base840, i64 3, !dbg !15
  %433 = insertelement <32 x i32> %432, i32 %compare.local.base844, i64 4, !dbg !15
  %434 = insertelement <32 x i32> %433, i32 %compare.local.base848, i64 5, !dbg !15
  %435 = insertelement <32 x i32> %434, i32 %compare.local.base852, i64 6, !dbg !15
  %436 = insertelement <32 x i32> %435, i32 %compare.local.base856, i64 7, !dbg !15
  %437 = insertelement <32 x i32> %436, i32 %compare.local.base860, i64 8, !dbg !15
  %438 = insertelement <32 x i32> %437, i32 %compare.local.base864, i64 9, !dbg !15
  %439 = insertelement <32 x i32> %438, i32 %compare.local.base868, i64 10, !dbg !15
  %440 = insertelement <32 x i32> %439, i32 %compare.local.base872, i64 11, !dbg !15
  %441 = insertelement <32 x i32> %440, i32 %compare.local.base876, i64 12, !dbg !15
  %442 = insertelement <32 x i32> %441, i32 %compare.local.base880, i64 13, !dbg !15
  %443 = insertelement <32 x i32> %442, i32 %compare.local.base884, i64 14, !dbg !15
  %444 = insertelement <32 x i32> %443, i32 %compare.local.base888, i64 15, !dbg !15
  %445 = insertelement <32 x i32> %444, i32 %compare.local.base892, i64 16, !dbg !15
  %446 = insertelement <32 x i32> %445, i32 %compare.local.base896, i64 17, !dbg !15
  %447 = insertelement <32 x i32> %446, i32 %compare.local.base900, i64 18, !dbg !15
  %448 = insertelement <32 x i32> %447, i32 %compare.local.base904, i64 19, !dbg !15
  %449 = insertelement <32 x i32> %448, i32 %compare.local.base908, i64 20, !dbg !15
  %450 = insertelement <32 x i32> %449, i32 %compare.local.base912, i64 21, !dbg !15
  %451 = insertelement <32 x i32> %450, i32 %compare.local.base916, i64 22, !dbg !15
  %452 = insertelement <32 x i32> %451, i32 %compare.local.base920, i64 23, !dbg !15
  %453 = insertelement <32 x i32> %452, i32 %compare.local.base924, i64 24, !dbg !15
  %454 = insertelement <32 x i32> %453, i32 %compare.local.base928, i64 25, !dbg !15
  %455 = insertelement <32 x i32> %454, i32 %compare.local.base932, i64 26, !dbg !15
  %456 = insertelement <32 x i32> %455, i32 %compare.local.base936, i64 27, !dbg !15
  %457 = insertelement <32 x i32> %456, i32 %compare.local.base940, i64 28, !dbg !15
  %458 = insertelement <32 x i32> %457, i32 %compare.local.base944, i64 29, !dbg !15
  %459 = insertelement <32 x i32> %458, i32 %compare.local.base948, i64 30, !dbg !15
  %460 = insertelement <32 x i32> %459, i32 %compare.local.base952, i64 31, !dbg !15
  %461 = or <32 x i32> %460, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !15
  %462 = icmp ugt <32 x i32> %52, %461, !dbg !15
  %463 = insertelement <32 x i1> poison, i1 %compare.high.equal, i64 0, !dbg !15
  %464 = shufflevector <32 x i1> %463, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !15
  %465 = insertelement <32 x i1> poison, i1 %compare.high.order, i64 0, !dbg !15
  %466 = shufflevector <32 x i1> %465, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !15
  %467 = select <32 x i1> %464, <32 x i1> %462, <32 x i1> %466, !dbg !15
  %468 = select <32 x i1> %467, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !20
  %compare.high.equal958 = icmp eq i32 %compare.high956, %compare.high832, !dbg !15
  %compare.high.order959 = icmp ugt i32 %compare.high956, %compare.high832, !dbg !15
  %compare.local.base964 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 32) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base968 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 33) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base972 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 34) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base976 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 35) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base980 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 36) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base984 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 37) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base988 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 38) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base992 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 39) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base996 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 40) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1000 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 41) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1004 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 42) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1008 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 43) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1012 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 44) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1016 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 45) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1020 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 46) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1024 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 47) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1028 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 48) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1032 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 49) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1036 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 50) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1040 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 51) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1044 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 52) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1048 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 53) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1052 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 54) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1056 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 55) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1060 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 56) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1064 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 57) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1068 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 58) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1072 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 59) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1076 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 60) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1080 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 61) #8, !dbg !15, !frisk.register_lifetimes !5
  %compare.local.base1084 = tail call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base833, i32 62) #8, !dbg !15, !frisk.register_lifetimes !5
  %469 = insertelement <32 x i32> %429, i32 %compare.local.base964, i64 1, !dbg !15
  %470 = insertelement <32 x i32> %469, i32 %compare.local.base968, i64 2, !dbg !15
  %471 = insertelement <32 x i32> %470, i32 %compare.local.base972, i64 3, !dbg !15
  %472 = insertelement <32 x i32> %471, i32 %compare.local.base976, i64 4, !dbg !15
  %473 = insertelement <32 x i32> %472, i32 %compare.local.base980, i64 5, !dbg !15
  %474 = insertelement <32 x i32> %473, i32 %compare.local.base984, i64 6, !dbg !15
  %475 = insertelement <32 x i32> %474, i32 %compare.local.base988, i64 7, !dbg !15
  %476 = insertelement <32 x i32> %475, i32 %compare.local.base992, i64 8, !dbg !15
  %477 = insertelement <32 x i32> %476, i32 %compare.local.base996, i64 9, !dbg !15
  %478 = insertelement <32 x i32> %477, i32 %compare.local.base1000, i64 10, !dbg !15
  %479 = insertelement <32 x i32> %478, i32 %compare.local.base1004, i64 11, !dbg !15
  %480 = insertelement <32 x i32> %479, i32 %compare.local.base1008, i64 12, !dbg !15
  %481 = insertelement <32 x i32> %480, i32 %compare.local.base1012, i64 13, !dbg !15
  %482 = insertelement <32 x i32> %481, i32 %compare.local.base1016, i64 14, !dbg !15
  %483 = insertelement <32 x i32> %482, i32 %compare.local.base1020, i64 15, !dbg !15
  %484 = insertelement <32 x i32> %483, i32 %compare.local.base1024, i64 16, !dbg !15
  %485 = insertelement <32 x i32> %484, i32 %compare.local.base1028, i64 17, !dbg !15
  %486 = insertelement <32 x i32> %485, i32 %compare.local.base1032, i64 18, !dbg !15
  %487 = insertelement <32 x i32> %486, i32 %compare.local.base1036, i64 19, !dbg !15
  %488 = insertelement <32 x i32> %487, i32 %compare.local.base1040, i64 20, !dbg !15
  %489 = insertelement <32 x i32> %488, i32 %compare.local.base1044, i64 21, !dbg !15
  %490 = insertelement <32 x i32> %489, i32 %compare.local.base1048, i64 22, !dbg !15
  %491 = insertelement <32 x i32> %490, i32 %compare.local.base1052, i64 23, !dbg !15
  %492 = insertelement <32 x i32> %491, i32 %compare.local.base1056, i64 24, !dbg !15
  %493 = insertelement <32 x i32> %492, i32 %compare.local.base1060, i64 25, !dbg !15
  %494 = insertelement <32 x i32> %493, i32 %compare.local.base1064, i64 26, !dbg !15
  %495 = insertelement <32 x i32> %494, i32 %compare.local.base1068, i64 27, !dbg !15
  %496 = insertelement <32 x i32> %495, i32 %compare.local.base1072, i64 28, !dbg !15
  %497 = insertelement <32 x i32> %496, i32 %compare.local.base1076, i64 29, !dbg !15
  %498 = insertelement <32 x i32> %497, i32 %compare.local.base1080, i64 30, !dbg !15
  %499 = insertelement <32 x i32> %498, i32 %compare.local.base1084, i64 31, !dbg !15
  %500 = or <32 x i32> %499, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 64, i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72, i32 73, i32 74, i32 75, i32 76, i32 77, i32 78, i32 79>, !dbg !15
  %501 = icmp ugt <32 x i32> %50, %500, !dbg !15
  %502 = insertelement <32 x i1> poison, i1 %compare.high.equal958, i64 0, !dbg !15
  %503 = shufflevector <32 x i1> %502, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !15
  %504 = insertelement <32 x i1> poison, i1 %compare.high.order959, i64 0, !dbg !15
  %505 = shufflevector <32 x i1> %504, <32 x i1> poison, <32 x i32> zeroinitializer, !dbg !15
  %506 = select <32 x i1> %503, <32 x i1> %501, <32 x i1> %505, !dbg !15
  %507 = select <32 x i1> %506, <32 x float> zeroinitializer, <32 x float> splat (float -inf), !dbg !20
  %508 = insertvalue [2 x <32 x float>] poison, <32 x float> %468, 0, !dbg !19
  %509 = insertvalue [2 x <32 x float>] %508, <32 x float> %507, 1, !dbg !19
  br label %510, !dbg !19

510:                                              ; preds = %426, %55
  %511 = phi [2 x <32 x float>] [ %509, %426 ], [ zeroinitializer, %55 ], !dbg !19
  %512 = extractvalue [2 x <32 x float>] %511, 0, !dbg !21
  %513 = shufflevector <32 x float> %512, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %514 = fadd <4 x float> %381, %513, !dbg !21
  %515 = extractelement <4 x float> %514, i64 0, !dbg !22
  %516 = tail call float @llvm.amdgcn.exp2.f32(float %515), !dbg !22
  %517 = insertelement <4 x float> poison, float %516, i64 0, !dbg !23
  %518 = extractelement <4 x float> %514, i64 1, !dbg !22
  %519 = tail call float @llvm.amdgcn.exp2.f32(float %518), !dbg !22
  %520 = insertelement <4 x float> %517, float %519, i64 1, !dbg !23
  %521 = fadd float %516, %519, !dbg !24
  %522 = extractelement <4 x float> %514, i64 2, !dbg !22
  %523 = tail call float @llvm.amdgcn.exp2.f32(float %522), !dbg !22
  %524 = insertelement <4 x float> %520, float %523, i64 2, !dbg !23
  %525 = fadd float %523, %521, !dbg !24
  %526 = extractelement <4 x float> %514, i64 3, !dbg !22
  %527 = tail call float @llvm.amdgcn.exp2.f32(float %526), !dbg !22
  %528 = insertelement <4 x float> %524, float %527, i64 3, !dbg !23
  %math.result = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %528) #8, !dbg !23, !frisk.register_lifetimes !5
  %529 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !23
  %math.packed = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %529) #8, !dbg !24, !frisk.register_lifetimes !5
  %530 = fadd float %527, %525, !dbg !24
  %531 = shufflevector <32 x float> %512, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %532 = fadd <4 x float> %385, %531, !dbg !21
  %533 = extractelement <4 x float> %532, i64 0, !dbg !22
  %534 = tail call float @llvm.amdgcn.exp2.f32(float %533), !dbg !22
  %535 = insertelement <4 x float> poison, float %534, i64 0, !dbg !23
  %536 = fadd float %534, %530, !dbg !24
  %537 = extractelement <4 x float> %532, i64 1, !dbg !22
  %538 = tail call float @llvm.amdgcn.exp2.f32(float %537), !dbg !22
  %539 = insertelement <4 x float> %535, float %538, i64 1, !dbg !23
  %540 = fadd float %538, %536, !dbg !24
  %541 = extractelement <4 x float> %532, i64 2, !dbg !22
  %542 = tail call float @llvm.amdgcn.exp2.f32(float %541), !dbg !22
  %543 = insertelement <4 x float> %539, float %542, i64 2, !dbg !23
  %544 = fadd float %542, %540, !dbg !24
  %545 = extractelement <4 x float> %532, i64 3, !dbg !22
  %546 = tail call float @llvm.amdgcn.exp2.f32(float %545), !dbg !22
  %547 = insertelement <4 x float> %543, float %546, i64 3, !dbg !23
  %math.result144 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %547) #8, !dbg !23, !frisk.register_lifetimes !5
  %548 = fptrunc <4 x float> %math.result144 to <4 x half>, !dbg !23
  %math.packed145 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %548) #8, !dbg !24, !frisk.register_lifetimes !5
  %549 = fadd float %546, %544, !dbg !24
  %550 = shufflevector <32 x float> %512, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !21
  %551 = fadd <4 x float> %389, %550, !dbg !21
  %552 = extractelement <4 x float> %551, i64 0, !dbg !22
  %553 = tail call float @llvm.amdgcn.exp2.f32(float %552), !dbg !22
  %554 = insertelement <4 x float> poison, float %553, i64 0, !dbg !23
  %555 = fadd float %553, %549, !dbg !24
  %556 = extractelement <4 x float> %551, i64 1, !dbg !22
  %557 = tail call float @llvm.amdgcn.exp2.f32(float %556), !dbg !22
  %558 = insertelement <4 x float> %554, float %557, i64 1, !dbg !23
  %559 = fadd float %557, %555, !dbg !24
  %560 = extractelement <4 x float> %551, i64 2, !dbg !22
  %561 = tail call float @llvm.amdgcn.exp2.f32(float %560), !dbg !22
  %562 = insertelement <4 x float> %558, float %561, i64 2, !dbg !23
  %563 = fadd float %561, %559, !dbg !24
  %564 = extractelement <4 x float> %551, i64 3, !dbg !22
  %565 = tail call float @llvm.amdgcn.exp2.f32(float %564), !dbg !22
  %566 = insertelement <4 x float> %562, float %565, i64 3, !dbg !23
  %math.result146 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %566) #8, !dbg !23, !frisk.register_lifetimes !5
  %567 = fptrunc <4 x float> %math.result146 to <4 x half>, !dbg !23
  %math.packed147 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %567) #8, !dbg !24, !frisk.register_lifetimes !5
  %568 = fadd float %565, %563, !dbg !24
  %569 = shufflevector <32 x float> %512, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !21
  %570 = fadd <4 x float> %393, %569, !dbg !21
  %571 = extractelement <4 x float> %570, i64 0, !dbg !22
  %572 = tail call float @llvm.amdgcn.exp2.f32(float %571), !dbg !22
  %573 = insertelement <4 x float> poison, float %572, i64 0, !dbg !23
  %574 = fadd float %572, %568, !dbg !24
  %575 = extractelement <4 x float> %570, i64 1, !dbg !22
  %576 = tail call float @llvm.amdgcn.exp2.f32(float %575), !dbg !22
  %577 = insertelement <4 x float> %573, float %576, i64 1, !dbg !23
  %578 = fadd float %576, %574, !dbg !24
  %579 = extractelement <4 x float> %570, i64 2, !dbg !22
  %580 = tail call float @llvm.amdgcn.exp2.f32(float %579), !dbg !22
  %581 = insertelement <4 x float> %577, float %580, i64 2, !dbg !23
  %582 = fadd float %580, %578, !dbg !24
  %583 = extractelement <4 x float> %570, i64 3, !dbg !22
  %584 = tail call float @llvm.amdgcn.exp2.f32(float %583), !dbg !22
  %585 = insertelement <4 x float> %581, float %584, i64 3, !dbg !23
  %math.result148 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %585) #8, !dbg !23, !frisk.register_lifetimes !5
  %586 = fptrunc <4 x float> %math.result148 to <4 x half>, !dbg !23
  %math.packed149 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %586) #8, !dbg !24, !frisk.register_lifetimes !5
  %587 = fadd float %584, %582, !dbg !24
  %588 = shufflevector <32 x float> %512, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !21
  %589 = fadd <4 x float> %397, %588, !dbg !21
  %590 = extractelement <4 x float> %589, i64 0, !dbg !22
  %591 = tail call float @llvm.amdgcn.exp2.f32(float %590), !dbg !22
  %592 = insertelement <4 x float> poison, float %591, i64 0, !dbg !23
  %593 = fadd float %591, %587, !dbg !24
  %594 = extractelement <4 x float> %589, i64 1, !dbg !22
  %595 = tail call float @llvm.amdgcn.exp2.f32(float %594), !dbg !22
  %596 = insertelement <4 x float> %592, float %595, i64 1, !dbg !23
  %597 = fadd float %595, %593, !dbg !24
  %598 = extractelement <4 x float> %589, i64 2, !dbg !22
  %599 = tail call float @llvm.amdgcn.exp2.f32(float %598), !dbg !22
  %600 = insertelement <4 x float> %596, float %599, i64 2, !dbg !23
  %601 = fadd float %599, %597, !dbg !24
  %602 = extractelement <4 x float> %589, i64 3, !dbg !22
  %603 = tail call float @llvm.amdgcn.exp2.f32(float %602), !dbg !22
  %604 = insertelement <4 x float> %600, float %603, i64 3, !dbg !23
  %math.result150 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %604) #8, !dbg !23, !frisk.register_lifetimes !5
  %605 = fptrunc <4 x float> %math.result150 to <4 x half>, !dbg !23
  %math.packed151 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %605) #8, !dbg !24, !frisk.register_lifetimes !5
  %606 = fadd float %603, %601, !dbg !24
  %607 = shufflevector <32 x float> %512, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !21
  %608 = fadd <4 x float> %401, %607, !dbg !21
  %609 = extractelement <4 x float> %608, i64 0, !dbg !22
  %610 = tail call float @llvm.amdgcn.exp2.f32(float %609), !dbg !22
  %611 = insertelement <4 x float> poison, float %610, i64 0, !dbg !23
  %612 = fadd float %610, %606, !dbg !24
  %613 = extractelement <4 x float> %608, i64 1, !dbg !22
  %614 = tail call float @llvm.amdgcn.exp2.f32(float %613), !dbg !22
  %615 = insertelement <4 x float> %611, float %614, i64 1, !dbg !23
  %616 = fadd float %614, %612, !dbg !24
  %617 = extractelement <4 x float> %608, i64 2, !dbg !22
  %618 = tail call float @llvm.amdgcn.exp2.f32(float %617), !dbg !22
  %619 = insertelement <4 x float> %615, float %618, i64 2, !dbg !23
  %620 = fadd float %618, %616, !dbg !24
  %621 = extractelement <4 x float> %608, i64 3, !dbg !22
  %622 = tail call float @llvm.amdgcn.exp2.f32(float %621), !dbg !22
  %623 = insertelement <4 x float> %619, float %622, i64 3, !dbg !23
  %math.result152 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %623) #8, !dbg !23, !frisk.register_lifetimes !5
  %624 = fptrunc <4 x float> %math.result152 to <4 x half>, !dbg !23
  %math.packed153 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %624) #8, !dbg !24, !frisk.register_lifetimes !5
  %625 = fadd float %622, %620, !dbg !24
  %626 = shufflevector <32 x float> %512, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !21
  %627 = fadd <4 x float> %405, %626, !dbg !21
  %628 = extractelement <4 x float> %627, i64 0, !dbg !22
  %629 = tail call float @llvm.amdgcn.exp2.f32(float %628), !dbg !22
  %630 = insertelement <4 x float> poison, float %629, i64 0, !dbg !23
  %631 = fadd float %629, %625, !dbg !24
  %632 = extractelement <4 x float> %627, i64 1, !dbg !22
  %633 = tail call float @llvm.amdgcn.exp2.f32(float %632), !dbg !22
  %634 = insertelement <4 x float> %630, float %633, i64 1, !dbg !23
  %635 = fadd float %633, %631, !dbg !24
  %636 = extractelement <4 x float> %627, i64 2, !dbg !22
  %637 = tail call float @llvm.amdgcn.exp2.f32(float %636), !dbg !22
  %638 = insertelement <4 x float> %634, float %637, i64 2, !dbg !23
  %639 = fadd float %637, %635, !dbg !24
  %640 = extractelement <4 x float> %627, i64 3, !dbg !22
  %641 = tail call float @llvm.amdgcn.exp2.f32(float %640), !dbg !22
  %642 = insertelement <4 x float> %638, float %641, i64 3, !dbg !23
  %math.result154 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %642) #8, !dbg !23, !frisk.register_lifetimes !5
  %643 = fptrunc <4 x float> %math.result154 to <4 x half>, !dbg !23
  %math.packed155 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %643) #8, !dbg !24, !frisk.register_lifetimes !5
  %644 = fadd float %641, %639, !dbg !24
  %645 = shufflevector <32 x float> %512, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !21
  %646 = fadd <4 x float> %408, %645, !dbg !21
  %647 = extractelement <4 x float> %646, i64 0, !dbg !22
  %648 = tail call float @llvm.amdgcn.exp2.f32(float %647), !dbg !22
  %649 = insertelement <4 x float> poison, float %648, i64 0, !dbg !23
  %650 = fadd float %648, %644, !dbg !24
  %651 = extractelement <4 x float> %646, i64 1, !dbg !22
  %652 = tail call float @llvm.amdgcn.exp2.f32(float %651), !dbg !22
  %653 = insertelement <4 x float> %649, float %652, i64 1, !dbg !23
  %654 = fadd float %652, %650, !dbg !24
  %655 = extractelement <4 x float> %646, i64 2, !dbg !22
  %656 = tail call float @llvm.amdgcn.exp2.f32(float %655), !dbg !22
  %657 = insertelement <4 x float> %653, float %656, i64 2, !dbg !23
  %658 = fadd float %656, %654, !dbg !24
  %659 = extractelement <4 x float> %646, i64 3, !dbg !22
  %660 = tail call float @llvm.amdgcn.exp2.f32(float %659), !dbg !22
  %661 = insertelement <4 x float> %657, float %660, i64 3, !dbg !23
  %math.result156 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %661) #8, !dbg !23, !frisk.register_lifetimes !5
  %662 = fptrunc <4 x float> %math.result156 to <4 x half>, !dbg !23
  %math.packed157 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %662) #8, !dbg !24, !frisk.register_lifetimes !5
  %663 = fadd float %660, %658, !dbg !24
  %664 = extractvalue [2 x <32 x float>] %511, 1, !dbg !21
  %665 = shufflevector <32 x float> %664, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !21
  %666 = fadd <4 x float> %410, %665, !dbg !21
  %667 = extractelement <4 x float> %666, i64 0, !dbg !22
  %668 = tail call float @llvm.amdgcn.exp2.f32(float %667), !dbg !22
  %669 = insertelement <4 x float> poison, float %668, i64 0, !dbg !23
  %670 = extractelement <4 x float> %666, i64 1, !dbg !22
  %671 = tail call float @llvm.amdgcn.exp2.f32(float %670), !dbg !22
  %672 = insertelement <4 x float> %669, float %671, i64 1, !dbg !23
  %673 = fadd float %668, %671, !dbg !24
  %674 = extractelement <4 x float> %666, i64 2, !dbg !22
  %675 = tail call float @llvm.amdgcn.exp2.f32(float %674), !dbg !22
  %676 = insertelement <4 x float> %672, float %675, i64 2, !dbg !23
  %677 = fadd float %675, %673, !dbg !24
  %678 = extractelement <4 x float> %666, i64 3, !dbg !22
  %679 = tail call float @llvm.amdgcn.exp2.f32(float %678), !dbg !22
  %680 = insertelement <4 x float> %676, float %679, i64 3, !dbg !23
  %math.result158 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %680) #8, !dbg !23, !frisk.register_lifetimes !5
  %681 = fptrunc <4 x float> %math.result158 to <4 x half>, !dbg !23
  %math.packed159 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %681) #8, !dbg !24, !frisk.register_lifetimes !5
  %682 = fadd float %679, %677, !dbg !24
  %683 = shufflevector <32 x float> %664, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !21
  %684 = fadd <4 x float> %412, %683, !dbg !21
  %685 = extractelement <4 x float> %684, i64 0, !dbg !22
  %686 = tail call float @llvm.amdgcn.exp2.f32(float %685), !dbg !22
  %687 = insertelement <4 x float> poison, float %686, i64 0, !dbg !23
  %688 = fadd float %686, %682, !dbg !24
  %689 = extractelement <4 x float> %684, i64 1, !dbg !22
  %690 = tail call float @llvm.amdgcn.exp2.f32(float %689), !dbg !22
  %691 = insertelement <4 x float> %687, float %690, i64 1, !dbg !23
  %692 = fadd float %690, %688, !dbg !24
  %693 = extractelement <4 x float> %684, i64 2, !dbg !22
  %694 = tail call float @llvm.amdgcn.exp2.f32(float %693), !dbg !22
  %695 = insertelement <4 x float> %691, float %694, i64 2, !dbg !23
  %696 = fadd float %694, %692, !dbg !24
  %697 = extractelement <4 x float> %684, i64 3, !dbg !22
  %698 = tail call float @llvm.amdgcn.exp2.f32(float %697), !dbg !22
  %699 = insertelement <4 x float> %695, float %698, i64 3, !dbg !23
  %math.result160 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %699) #8, !dbg !23, !frisk.register_lifetimes !5
  %700 = fptrunc <4 x float> %math.result160 to <4 x half>, !dbg !23
  %math.packed161 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %700) #8, !dbg !24, !frisk.register_lifetimes !5
  %701 = fadd float %698, %696, !dbg !24
  %702 = shufflevector <32 x float> %664, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !21
  %703 = fadd <4 x float> %414, %702, !dbg !21
  %704 = extractelement <4 x float> %703, i64 0, !dbg !22
  %705 = tail call float @llvm.amdgcn.exp2.f32(float %704), !dbg !22
  %706 = insertelement <4 x float> poison, float %705, i64 0, !dbg !23
  %707 = fadd float %705, %701, !dbg !24
  %708 = extractelement <4 x float> %703, i64 1, !dbg !22
  %709 = tail call float @llvm.amdgcn.exp2.f32(float %708), !dbg !22
  %710 = insertelement <4 x float> %706, float %709, i64 1, !dbg !23
  %711 = fadd float %709, %707, !dbg !24
  %712 = extractelement <4 x float> %703, i64 2, !dbg !22
  %713 = tail call float @llvm.amdgcn.exp2.f32(float %712), !dbg !22
  %714 = insertelement <4 x float> %710, float %713, i64 2, !dbg !23
  %715 = fadd float %713, %711, !dbg !24
  %716 = extractelement <4 x float> %703, i64 3, !dbg !22
  %717 = tail call float @llvm.amdgcn.exp2.f32(float %716), !dbg !22
  %718 = insertelement <4 x float> %714, float %717, i64 3, !dbg !23
  %math.result162 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %718) #8, !dbg !23, !frisk.register_lifetimes !5
  %719 = fptrunc <4 x float> %math.result162 to <4 x half>, !dbg !23
  %math.packed163 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %719) #8, !dbg !24, !frisk.register_lifetimes !5
  %720 = fadd float %717, %715, !dbg !24
  %721 = shufflevector <32 x float> %664, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !21
  %722 = fadd <4 x float> %416, %721, !dbg !21
  %723 = extractelement <4 x float> %722, i64 0, !dbg !22
  %724 = tail call float @llvm.amdgcn.exp2.f32(float %723), !dbg !22
  %725 = insertelement <4 x float> poison, float %724, i64 0, !dbg !23
  %726 = fadd float %724, %720, !dbg !24
  %727 = extractelement <4 x float> %722, i64 1, !dbg !22
  %728 = tail call float @llvm.amdgcn.exp2.f32(float %727), !dbg !22
  %729 = insertelement <4 x float> %725, float %728, i64 1, !dbg !23
  %730 = fadd float %728, %726, !dbg !24
  %731 = extractelement <4 x float> %722, i64 2, !dbg !22
  %732 = tail call float @llvm.amdgcn.exp2.f32(float %731), !dbg !22
  %733 = insertelement <4 x float> %729, float %732, i64 2, !dbg !23
  %734 = fadd float %732, %730, !dbg !24
  %735 = extractelement <4 x float> %722, i64 3, !dbg !22
  %736 = tail call float @llvm.amdgcn.exp2.f32(float %735), !dbg !22
  %737 = insertelement <4 x float> %733, float %736, i64 3, !dbg !23
  %math.result164 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %737) #8, !dbg !23, !frisk.register_lifetimes !5
  %738 = fptrunc <4 x float> %math.result164 to <4 x half>, !dbg !23
  %math.packed165 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %738) #8, !dbg !24, !frisk.register_lifetimes !5
  %739 = fadd float %736, %734, !dbg !24
  %740 = shufflevector <32 x float> %664, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !21
  %741 = fadd <4 x float> %418, %740, !dbg !21
  %742 = extractelement <4 x float> %741, i64 0, !dbg !22
  %743 = tail call float @llvm.amdgcn.exp2.f32(float %742), !dbg !22
  %744 = insertelement <4 x float> poison, float %743, i64 0, !dbg !23
  %745 = fadd float %743, %739, !dbg !24
  %746 = extractelement <4 x float> %741, i64 1, !dbg !22
  %747 = tail call float @llvm.amdgcn.exp2.f32(float %746), !dbg !22
  %748 = insertelement <4 x float> %744, float %747, i64 1, !dbg !23
  %749 = fadd float %747, %745, !dbg !24
  %750 = extractelement <4 x float> %741, i64 2, !dbg !22
  %751 = tail call float @llvm.amdgcn.exp2.f32(float %750), !dbg !22
  %752 = insertelement <4 x float> %748, float %751, i64 2, !dbg !23
  %753 = fadd float %751, %749, !dbg !24
  %754 = extractelement <4 x float> %741, i64 3, !dbg !22
  %755 = tail call float @llvm.amdgcn.exp2.f32(float %754), !dbg !22
  %756 = insertelement <4 x float> %752, float %755, i64 3, !dbg !23
  %math.result166 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %756) #8, !dbg !23, !frisk.register_lifetimes !5
  %757 = fptrunc <4 x float> %math.result166 to <4 x half>, !dbg !23
  %math.packed167 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %757) #8, !dbg !24, !frisk.register_lifetimes !5
  %758 = fadd float %755, %753, !dbg !24
  %759 = shufflevector <32 x float> %664, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !21
  %760 = fadd <4 x float> %420, %759, !dbg !21
  %761 = extractelement <4 x float> %760, i64 0, !dbg !22
  %762 = tail call float @llvm.amdgcn.exp2.f32(float %761), !dbg !22
  %763 = insertelement <4 x float> poison, float %762, i64 0, !dbg !23
  %764 = fadd float %762, %758, !dbg !24
  %765 = extractelement <4 x float> %760, i64 1, !dbg !22
  %766 = tail call float @llvm.amdgcn.exp2.f32(float %765), !dbg !22
  %767 = insertelement <4 x float> %763, float %766, i64 1, !dbg !23
  %768 = fadd float %766, %764, !dbg !24
  %769 = extractelement <4 x float> %760, i64 2, !dbg !22
  %770 = tail call float @llvm.amdgcn.exp2.f32(float %769), !dbg !22
  %771 = insertelement <4 x float> %767, float %770, i64 2, !dbg !23
  %772 = fadd float %770, %768, !dbg !24
  %773 = extractelement <4 x float> %760, i64 3, !dbg !22
  %774 = tail call float @llvm.amdgcn.exp2.f32(float %773), !dbg !22
  %775 = insertelement <4 x float> %771, float %774, i64 3, !dbg !23
  %math.result168 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %775) #8, !dbg !23, !frisk.register_lifetimes !5
  %776 = fptrunc <4 x float> %math.result168 to <4 x half>, !dbg !23
  %math.packed169 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %776) #8, !dbg !24, !frisk.register_lifetimes !5
  %777 = fadd float %774, %772, !dbg !24
  %778 = shufflevector <32 x float> %664, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !21
  %779 = fadd <4 x float> %422, %778, !dbg !21
  %780 = extractelement <4 x float> %779, i64 0, !dbg !22
  %781 = tail call float @llvm.amdgcn.exp2.f32(float %780), !dbg !22
  %782 = insertelement <4 x float> poison, float %781, i64 0, !dbg !23
  %783 = fadd float %781, %777, !dbg !24
  %784 = extractelement <4 x float> %779, i64 1, !dbg !22
  %785 = tail call float @llvm.amdgcn.exp2.f32(float %784), !dbg !22
  %786 = insertelement <4 x float> %782, float %785, i64 1, !dbg !23
  %787 = fadd float %785, %783, !dbg !24
  %788 = extractelement <4 x float> %779, i64 2, !dbg !22
  %789 = tail call float @llvm.amdgcn.exp2.f32(float %788), !dbg !22
  %790 = insertelement <4 x float> %786, float %789, i64 2, !dbg !23
  %791 = fadd float %789, %787, !dbg !24
  %792 = extractelement <4 x float> %779, i64 3, !dbg !22
  %793 = tail call float @llvm.amdgcn.exp2.f32(float %792), !dbg !22
  %794 = insertelement <4 x float> %790, float %793, i64 3, !dbg !23
  %math.result170 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %794) #8, !dbg !23, !frisk.register_lifetimes !5
  %795 = fptrunc <4 x float> %math.result170 to <4 x half>, !dbg !23
  %math.packed171 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %795) #8, !dbg !24, !frisk.register_lifetimes !5
  %796 = fadd float %793, %791, !dbg !24
  %797 = shufflevector <32 x float> %664, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !21
  %798 = fadd <4 x float> %425, %797, !dbg !21
  %799 = extractelement <4 x float> %798, i64 0, !dbg !22
  %800 = tail call float @llvm.amdgcn.exp2.f32(float %799), !dbg !22
  %801 = insertelement <4 x float> poison, float %800, i64 0, !dbg !23
  %802 = fadd float %800, %796, !dbg !24
  %803 = extractelement <4 x float> %798, i64 1, !dbg !22
  %804 = tail call float @llvm.amdgcn.exp2.f32(float %803), !dbg !22
  %805 = insertelement <4 x float> %801, float %804, i64 1, !dbg !23
  %806 = fadd float %804, %802, !dbg !24
  %807 = extractelement <4 x float> %798, i64 2, !dbg !22
  %808 = tail call float @llvm.amdgcn.exp2.f32(float %807), !dbg !22
  %809 = insertelement <4 x float> %805, float %808, i64 2, !dbg !23
  %810 = fadd float %808, %806, !dbg !24
  %811 = extractelement <4 x float> %798, i64 3, !dbg !22
  %812 = tail call float @llvm.amdgcn.exp2.f32(float %811), !dbg !22
  %813 = insertelement <4 x float> %809, float %812, i64 3, !dbg !23
  %math.result172 = tail call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %813) #8, !dbg !23, !frisk.register_lifetimes !5
  %814 = fptrunc <4 x float> %math.result172 to <4 x half>, !dbg !23
  %math.packed173 = tail call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %814) #8, !dbg !24, !frisk.register_lifetimes !5
  %815 = fadd float %812, %810, !dbg !24
  %816 = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !24
  %817 = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %816), !dbg !24
  %818 = add nuw nsw i32 %817, 64, !dbg !24
  %819 = xor i32 %817, 16, !dbg !24
  %820 = and i32 %818, 192, !dbg !24
  %821 = icmp samesign ult i32 %819, %820, !dbg !24
  %822 = select i1 %821, i32 %819, i32 %817, !dbg !24
  %823 = shl nuw nsw i32 %822, 2, !dbg !24
  %824 = bitcast float %663 to i32, !dbg !24
  %825 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %824), !dbg !24
  %826 = bitcast i32 %825 to float, !dbg !24
  %827 = fadd float %663, %826, !dbg !24
  %828 = xor i32 %817, 32, !dbg !24
  %829 = icmp samesign ult i32 %828, %820, !dbg !24
  %830 = select i1 %829, i32 %828, i32 %817, !dbg !24
  %831 = shl nuw nsw i32 %830, 2, !dbg !24
  %832 = bitcast float %827 to i32, !dbg !24
  %833 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %831, i32 %832), !dbg !24
  %834 = bitcast i32 %833 to float, !dbg !24
  %835 = fadd float %827, %834, !dbg !24
  %836 = extractvalue [2 x <1 x float>] %59, 0, !dbg !24
  %837 = extractelement <1 x float> %836, i64 0, !dbg !24
  %838 = fadd float %837, %835, !dbg !24
  %839 = insertelement <1 x float> poison, float %838, i64 0, !dbg !13
  %840 = insertvalue [2 x <1 x float>] poison, <1 x float> %839, 0, !dbg !13
  %841 = bitcast float %815 to i32, !dbg !24
  %842 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %823, i32 %841), !dbg !24
  %843 = bitcast i32 %842 to float, !dbg !24
  %844 = fadd float %815, %843, !dbg !24
  %845 = bitcast float %844 to i32, !dbg !24
  %846 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %831, i32 %845), !dbg !24
  %847 = bitcast i32 %846 to float, !dbg !24
  %848 = fadd float %844, %847, !dbg !24
  %849 = extractvalue [2 x <1 x float>] %59, 1, !dbg !24
  %850 = extractelement <1 x float> %849, i64 0, !dbg !24
  %851 = fadd float %850, %848, !dbg !24
  %852 = insertelement <1 x float> poison, float %851, i64 0, !dbg !13
  %853 = insertvalue [2 x <1 x float>] %840, <1 x float> %852, 1, !dbg !13
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread539 = tail call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !25, !range !11, !frisk.register_lifetimes !5
  %address.local540 = shl nuw nsw i32 %address.thread539, 3, !dbg !25
  %address.local541 = and i32 %address.local540, 120, !dbg !25
  %address.local542 = shl nuw nsw i32 %address.thread539, 5, !dbg !25
  %address.local543 = and i32 %address.local542, 1536, !dbg !25
  %address.local544 = or disjoint i32 %address.local541, %address.local543, !dbg !25
  %address.local545 = zext nneg i32 %address.local544 to i64, !dbg !25
  %address.local546 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local545, !dbg !25
  %854 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local546, i32 0) #11, !dbg !25
  %855 = bitcast <4 x i32> %854 to <8 x half>, !dbg !25
  %address.local547 = zext nneg i32 %address.thread539 to i64, !dbg !10
  %address.local548 = and i64 %address.local547, 60, !dbg !25
  %address.local549 = add nuw nsw i64 %address.local548, %address.local547, !dbg !25
  %address.local551.idx = shl nuw nsw i64 %address.local549, 4, !dbg !25
  %address.local551 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local551.idx, !dbg !25
  %address.local552 = lshr i64 %address.local547, 4, !dbg !25
  %address.local553 = and i64 %address.local552, 3, !dbg !25
  %address.local555.idx = mul nuw nsw i64 %address.local553, 768, !dbg !25
  %address.local555 = getelementptr i8, ptr addrspace(3) %address.local551, i64 %address.local555.idx, !dbg !25
  %address.local556 = lshr i64 %address.local547, 2, !dbg !17
  %address.local558.idx = mul nsw i64 %address.local556, -64, !dbg !25
  %address.local558 = getelementptr i8, ptr addrspace(3) %address.local555, i64 %address.local558.idx, !dbg !25
  %address.local559 = getelementptr i8, ptr addrspace(3) %address.local558, i64 256, !dbg !25
  %856 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #11, !dbg !25
  %857 = bitcast <4 x i32> %856 to <8 x half>, !dbg !25
  %address.local563 = getelementptr i8, ptr addrspace(3) %address.local558, i64 512, !dbg !25
  %858 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local563, i32 0) #11, !dbg !25
  %859 = bitcast <4 x i32> %858 to <8 x half>, !dbg !25
  %address.local567 = getelementptr i8, ptr addrspace(3) %address.local558, i64 768, !dbg !25
  %860 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local567, i32 0) #11, !dbg !25
  %861 = bitcast <4 x i32> %860 to <8 x half>, !dbg !25
  %address.local569 = getelementptr i8, ptr addrspace(3) %address.local546, i64 4096, !dbg !25
  %862 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local569, i32 0) #11, !dbg !25
  %863 = bitcast <4 x i32> %862 to <8 x half>, !dbg !25
  %address.local573 = getelementptr i8, ptr addrspace(3) %address.local558, i64 4352, !dbg !25
  %864 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #11, !dbg !25
  %865 = bitcast <4 x i32> %864 to <8 x half>, !dbg !25
  %address.local577 = getelementptr i8, ptr addrspace(3) %address.local558, i64 4608, !dbg !25
  %866 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local577, i32 0) #11, !dbg !25
  %867 = bitcast <4 x i32> %866 to <8 x half>, !dbg !25
  %address.local581 = getelementptr i8, ptr addrspace(3) %address.local558, i64 4864, !dbg !25
  %868 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local581, i32 0) #11, !dbg !25
  %869 = bitcast <4 x i32> %868 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.local582 = shl nuw nsw i64 %address.local547, 3, !dbg !9
  %address.local583 = and i64 %address.local582, 1536, !dbg !9
  %address.local584 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local583, !dbg !9
  %address.local585 = ptrtoint ptr addrspace(3) %address.local584 to i64, !dbg !9
  %address.local586 = trunc i64 %address.local585 to i32, !dbg !9
  %870 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local586), !dbg !18
  %address.local587 = shl nuw nsw i64 %address.local547, 5, !dbg !18
  %address.local588 = and i64 %address.local587, 8064, !dbg !18
  %address.local590 = and i64 %address.local582, 24, !dbg !18
  %address.local591 = or disjoint i64 %address.local588, %address.local590, !dbg !18
  %871 = or disjoint i64 %.idx5558, %address.local591, !dbg !18
  %872 = trunc i64 %871 to i32, !dbg !18
  %.tr66 = shl i32 %872, 1, !dbg !18
  %dma.buffer.offset1100 = or i32 %.tr66, 64, !dbg !18
  %dma.buffer.lds1102 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %870), !dbg !18
  %873 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1102, i32 %dma.buffer.offset1100, <4 x i32> %dma.buffer.rsrc1107) #9, !dbg !18
  %address.local593 = or disjoint i64 %address.local547, 256, !dbg !9
  %address.local594 = shl nuw nsw i64 %address.local593, 3, !dbg !9
  %address.local595 = and i64 %address.local594, 3584, !dbg !9
  %address.local596 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local595, !dbg !9
  %address.local597 = ptrtoint ptr addrspace(3) %address.local596 to i64, !dbg !9
  %address.local598 = trunc i64 %address.local597 to i32, !dbg !9
  %874 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local598), !dbg !18
  %address.local599 = shl nuw nsw i64 %address.local593, 5, !dbg !18
  %address.local600 = and i64 %address.local599, 16256, !dbg !18
  %address.local603 = or disjoint i64 %address.local600, %address.local590, !dbg !18
  %875 = or disjoint i64 %.idx5558, %address.local603, !dbg !18
  %876 = trunc i64 %875 to i32, !dbg !18
  %.tr70 = shl i32 %876, 1, !dbg !18
  %dma.buffer.offset1097 = or i32 %.tr70, 64, !dbg !18
  %dma.buffer.lds1099 = tail call i32 @llvm.amdgcn.readfirstlane.i32(i32 %874), !dbg !18
  %877 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1099, i32 %dma.buffer.offset1097, <4 x i32> %dma.buffer.rsrc1107) #9, !dbg !18
  %878 = shufflevector <8 x half> %855, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %879 = extractvalue [2 x <8 x float>] %60, 0, !dbg !26
  %880 = shufflevector <8 x float> %879, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %881 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %878, <4 x float> %880) #8, !dbg !25
  %882 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %883 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %882, <4 x float> %881) #8, !dbg !25
  %884 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %885 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %884, <4 x float> %883) #8, !dbg !25
  %886 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %887 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %886, <4 x float> %885) #8, !dbg !25
  %888 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %889 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %888, <4 x float> %887) #8, !dbg !25
  %890 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %891 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %890, <4 x float> %889) #8, !dbg !25
  %892 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %893 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %892, <4 x float> %891) #8, !dbg !25
  %894 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %895 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %894, <4 x float> %893) #8, !dbg !25
  %896 = shufflevector <8 x half> %855, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %897 = shufflevector <8 x float> %879, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %898 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %896, <4 x float> %897) #8, !dbg !25
  %899 = shufflevector <8 x half> %857, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %900 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %899, <4 x float> %898) #8, !dbg !25
  %901 = shufflevector <8 x half> %859, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %902 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %901, <4 x float> %900) #8, !dbg !25
  %903 = shufflevector <8 x half> %861, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %904 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %903, <4 x float> %902) #8, !dbg !25
  %905 = shufflevector <8 x half> %863, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %906 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %905, <4 x float> %904) #8, !dbg !25
  %907 = shufflevector <8 x half> %865, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %908 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %907, <4 x float> %906) #8, !dbg !25
  %909 = shufflevector <8 x half> %867, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %910 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %909, <4 x float> %908) #8, !dbg !25
  %911 = shufflevector <8 x half> %869, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %912 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %911, <4 x float> %910) #8, !dbg !25
  %913 = extractvalue [2 x <8 x float>] %60, 1, !dbg !26
  %914 = shufflevector <8 x float> %913, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %915 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %878, <4 x float> %914) #8, !dbg !25
  %916 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %882, <4 x float> %915) #8, !dbg !25
  %917 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %884, <4 x float> %916) #8, !dbg !25
  %918 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %886, <4 x float> %917) #8, !dbg !25
  %919 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %888, <4 x float> %918) #8, !dbg !25
  %920 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %890, <4 x float> %919) #8, !dbg !25
  %921 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %892, <4 x float> %920) #8, !dbg !25
  %922 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %894, <4 x float> %921) #8, !dbg !25
  %923 = shufflevector <8 x float> %913, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %924 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %896, <4 x float> %923) #8, !dbg !25
  %925 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %899, <4 x float> %924) #8, !dbg !25
  %926 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %901, <4 x float> %925) #8, !dbg !25
  %927 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %903, <4 x float> %926) #8, !dbg !25
  %928 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %905, <4 x float> %927) #8, !dbg !25
  %929 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %907, <4 x float> %928) #8, !dbg !25
  %930 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %909, <4 x float> %929) #8, !dbg !25
  %931 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %911, <4 x float> %930) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %932 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local546, i32 0) #11, !dbg !25
  %933 = bitcast <4 x i32> %932 to <8 x half>, !dbg !25
  %934 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #11, !dbg !25
  %935 = bitcast <4 x i32> %934 to <8 x half>, !dbg !25
  %936 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local563, i32 0) #11, !dbg !25
  %937 = bitcast <4 x i32> %936 to <8 x half>, !dbg !25
  %938 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local567, i32 0) #11, !dbg !25
  %939 = bitcast <4 x i32> %938 to <8 x half>, !dbg !25
  %940 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local569, i32 0) #11, !dbg !25
  %941 = bitcast <4 x i32> %940 to <8 x half>, !dbg !25
  %942 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #11, !dbg !25
  %943 = bitcast <4 x i32> %942 to <8 x half>, !dbg !25
  %944 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local577, i32 0) #11, !dbg !25
  %945 = bitcast <4 x i32> %944 to <8 x half>, !dbg !25
  %946 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local581, i32 0) #11, !dbg !25
  %947 = bitcast <4 x i32> %946 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %dma.buffer.offset1094 = or i32 %.tr66, 128, !dbg !18
  %948 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1102, i32 %dma.buffer.offset1094, <4 x i32> %dma.buffer.rsrc1107) #9, !dbg !18
  %dma.buffer.offset1091 = or i32 %.tr70, 128, !dbg !18
  %949 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1099, i32 %dma.buffer.offset1091, <4 x i32> %dma.buffer.rsrc1107) #9, !dbg !18
  %950 = shufflevector <8 x half> %933, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %951 = extractvalue [2 x <8 x float>] %58, 0, !dbg !26
  %952 = shufflevector <8 x float> %951, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %953 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %950, <4 x float> %952) #8, !dbg !25
  %954 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %955 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %954, <4 x float> %953) #8, !dbg !25
  %956 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %957 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %956, <4 x float> %955) #8, !dbg !25
  %958 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %959 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %958, <4 x float> %957) #8, !dbg !25
  %960 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %961 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %960, <4 x float> %959) #8, !dbg !25
  %962 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %963 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %962, <4 x float> %961) #8, !dbg !25
  %964 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %965 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %964, <4 x float> %963) #8, !dbg !25
  %966 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %967 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %966, <4 x float> %965) #8, !dbg !25
  %968 = shufflevector <8 x half> %933, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %969 = shufflevector <8 x float> %951, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %970 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %968, <4 x float> %969) #8, !dbg !25
  %971 = shufflevector <8 x half> %935, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %972 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %971, <4 x float> %970) #8, !dbg !25
  %973 = shufflevector <8 x half> %937, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %974 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %973, <4 x float> %972) #8, !dbg !25
  %975 = shufflevector <8 x half> %939, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %976 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %975, <4 x float> %974) #8, !dbg !25
  %977 = shufflevector <8 x half> %941, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %978 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %977, <4 x float> %976) #8, !dbg !25
  %979 = shufflevector <8 x half> %943, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %980 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %979, <4 x float> %978) #8, !dbg !25
  %981 = shufflevector <8 x half> %945, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %982 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %981, <4 x float> %980) #8, !dbg !25
  %983 = shufflevector <8 x half> %947, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %984 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %983, <4 x float> %982) #8, !dbg !25
  %985 = extractvalue [2 x <8 x float>] %58, 1, !dbg !26
  %986 = shufflevector <8 x float> %985, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %987 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %950, <4 x float> %986) #8, !dbg !25
  %988 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %954, <4 x float> %987) #8, !dbg !25
  %989 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %956, <4 x float> %988) #8, !dbg !25
  %990 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %958, <4 x float> %989) #8, !dbg !25
  %991 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %960, <4 x float> %990) #8, !dbg !25
  %992 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %962, <4 x float> %991) #8, !dbg !25
  %993 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %964, <4 x float> %992) #8, !dbg !25
  %994 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %966, <4 x float> %993) #8, !dbg !25
  %995 = shufflevector <8 x float> %985, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %996 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %968, <4 x float> %995) #8, !dbg !25
  %997 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %971, <4 x float> %996) #8, !dbg !25
  %998 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %973, <4 x float> %997) #8, !dbg !25
  %999 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %975, <4 x float> %998) #8, !dbg !25
  %1000 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %977, <4 x float> %999) #8, !dbg !25
  %1001 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %979, <4 x float> %1000) #8, !dbg !25
  %1002 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %981, <4 x float> %1001) #8, !dbg !25
  %1003 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %983, <4 x float> %1002) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1004 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local546, i32 0) #11, !dbg !25
  %1005 = bitcast <4 x i32> %1004 to <8 x half>, !dbg !25
  %1006 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #11, !dbg !25
  %1007 = bitcast <4 x i32> %1006 to <8 x half>, !dbg !25
  %1008 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local563, i32 0) #11, !dbg !25
  %1009 = bitcast <4 x i32> %1008 to <8 x half>, !dbg !25
  %1010 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local567, i32 0) #11, !dbg !25
  %1011 = bitcast <4 x i32> %1010 to <8 x half>, !dbg !25
  %1012 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local569, i32 0) #11, !dbg !25
  %1013 = bitcast <4 x i32> %1012 to <8 x half>, !dbg !25
  %1014 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #11, !dbg !25
  %1015 = bitcast <4 x i32> %1014 to <8 x half>, !dbg !25
  %1016 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local577, i32 0) #11, !dbg !25
  %1017 = bitcast <4 x i32> %1016 to <8 x half>, !dbg !25
  %1018 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local581, i32 0) #11, !dbg !25
  %1019 = bitcast <4 x i32> %1018 to <8 x half>, !dbg !25
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1020 = or disjoint i64 %.idx5558, 96, !dbg !18
  %1021 = or disjoint i64 %address.local591, %1020, !dbg !18
  %.tr82 = trunc i64 %1021 to i32, !dbg !18
  %dma.buffer.offset1088 = shl i32 %.tr82, 1, !dbg !18
  %1022 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1102, i32 %dma.buffer.offset1088, <4 x i32> %dma.buffer.rsrc1107) #9, !dbg !18
  %1023 = or disjoint i64 %address.local603, %1020, !dbg !18
  %.tr86 = trunc i64 %1023 to i32, !dbg !18
  %dma.buffer.offset = shl i32 %.tr86, 1, !dbg !18
  %1024 = tail call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds1099, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc1107) #9, !dbg !18
  %1025 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1026 = extractvalue [2 x <8 x float>] %57, 0, !dbg !26
  %1027 = shufflevector <8 x float> %1026, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1028 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1025, <4 x float> %1027) #8, !dbg !25
  %1029 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1030 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1029, <4 x float> %1028) #8, !dbg !25
  %1031 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1032 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1031, <4 x float> %1030) #8, !dbg !25
  %1033 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1034 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1033, <4 x float> %1032) #8, !dbg !25
  %1035 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1036 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1035, <4 x float> %1034) #8, !dbg !25
  %1037 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1038 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1037, <4 x float> %1036) #8, !dbg !25
  %1039 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1040 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1039, <4 x float> %1038) #8, !dbg !25
  %1041 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1042 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1041, <4 x float> %1040) #8, !dbg !25
  %1043 = shufflevector <8 x half> %1005, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1044 = shufflevector <8 x float> %1026, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1045 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1043, <4 x float> %1044) #8, !dbg !25
  %1046 = shufflevector <8 x half> %1007, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1047 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1046, <4 x float> %1045) #8, !dbg !25
  %1048 = shufflevector <8 x half> %1009, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1049 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1048, <4 x float> %1047) #8, !dbg !25
  %1050 = shufflevector <8 x half> %1011, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1051 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1050, <4 x float> %1049) #8, !dbg !25
  %1052 = shufflevector <8 x half> %1013, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1053 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1052, <4 x float> %1051) #8, !dbg !25
  %1054 = shufflevector <8 x half> %1015, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1055 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1054, <4 x float> %1053) #8, !dbg !25
  %1056 = shufflevector <8 x half> %1017, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1057 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1056, <4 x float> %1055) #8, !dbg !25
  %1058 = shufflevector <8 x half> %1019, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1059 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1058, <4 x float> %1057) #8, !dbg !25
  %1060 = extractvalue [2 x <8 x float>] %57, 1, !dbg !26
  %1061 = shufflevector <8 x float> %1060, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1062 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1025, <4 x float> %1061) #8, !dbg !25
  %1063 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1029, <4 x float> %1062) #8, !dbg !25
  %1064 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1031, <4 x float> %1063) #8, !dbg !25
  %1065 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1033, <4 x float> %1064) #8, !dbg !25
  %1066 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1035, <4 x float> %1065) #8, !dbg !25
  %1067 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1037, <4 x float> %1066) #8, !dbg !25
  %1068 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1039, <4 x float> %1067) #8, !dbg !25
  %1069 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1041, <4 x float> %1068) #8, !dbg !25
  %1070 = shufflevector <8 x float> %1060, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1071 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1043, <4 x float> %1070) #8, !dbg !25
  %1072 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1046, <4 x float> %1071) #8, !dbg !25
  %1073 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1048, <4 x float> %1072) #8, !dbg !25
  %1074 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1050, <4 x float> %1073) #8, !dbg !25
  %1075 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1052, <4 x float> %1074) #8, !dbg !25
  %1076 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1054, <4 x float> %1075) #8, !dbg !25
  %1077 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1056, <4 x float> %1076) #8, !dbg !25
  %1078 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1058, <4 x float> %1077) #8, !dbg !25
  tail call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #9, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  tail call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %1079 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local546, i32 0) #11, !dbg !25
  %1080 = bitcast <4 x i32> %1079 to <8 x half>, !dbg !25
  %1081 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local559, i32 0) #11, !dbg !25
  %1082 = bitcast <4 x i32> %1081 to <8 x half>, !dbg !25
  %1083 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local563, i32 0) #11, !dbg !25
  %1084 = bitcast <4 x i32> %1083 to <8 x half>, !dbg !25
  %1085 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local567, i32 0) #11, !dbg !25
  %1086 = bitcast <4 x i32> %1085 to <8 x half>, !dbg !25
  %1087 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local569, i32 0) #11, !dbg !25
  %1088 = bitcast <4 x i32> %1087 to <8 x half>, !dbg !25
  %1089 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local573, i32 0) #11, !dbg !25
  %1090 = bitcast <4 x i32> %1089 to <8 x half>, !dbg !25
  %1091 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local577, i32 0) #11, !dbg !25
  %1092 = bitcast <4 x i32> %1091 to <8 x half>, !dbg !25
  %1093 = tail call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local581, i32 0) #11, !dbg !25
  %1094 = bitcast <4 x i32> %1093 to <8 x half>, !dbg !25
  %1095 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1096 = extractvalue [2 x <8 x float>] %56, 0, !dbg !26
  %1097 = shufflevector <8 x float> %1096, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1098 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1095, <4 x float> %1097) #8, !dbg !25
  %1099 = shufflevector <8 x half> %1082, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1100 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1099, <4 x float> %1098) #8, !dbg !25
  %1101 = shufflevector <8 x half> %1084, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1102 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1101, <4 x float> %1100) #8, !dbg !25
  %1103 = shufflevector <8 x half> %1086, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1104 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1103, <4 x float> %1102) #8, !dbg !25
  %1105 = shufflevector <8 x half> %1088, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1106 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1105, <4 x float> %1104) #8, !dbg !25
  %1107 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1108 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1107, <4 x float> %1106) #8, !dbg !25
  %1109 = shufflevector <8 x half> %1092, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1110 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1109, <4 x float> %1108) #8, !dbg !25
  %1111 = shufflevector <8 x half> %1094, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !25
  %1112 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1111, <4 x float> %1110) #8, !dbg !25
  %1113 = shufflevector <8 x half> %1080, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1114 = shufflevector <8 x float> %1096, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1115 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1113, <4 x float> %1114) #8, !dbg !25
  %1116 = shufflevector <8 x half> %1082, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1117 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed145, <4 x half> %1116, <4 x float> %1115) #8, !dbg !25
  %1118 = shufflevector <8 x half> %1084, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1119 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed147, <4 x half> %1118, <4 x float> %1117) #8, !dbg !25
  %1120 = shufflevector <8 x half> %1086, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1121 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed149, <4 x half> %1120, <4 x float> %1119) #8, !dbg !25
  %1122 = shufflevector <8 x half> %1088, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1123 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed151, <4 x half> %1122, <4 x float> %1121) #8, !dbg !25
  %1124 = shufflevector <8 x half> %1090, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1125 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed153, <4 x half> %1124, <4 x float> %1123) #8, !dbg !25
  %1126 = shufflevector <8 x half> %1092, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1127 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed155, <4 x half> %1126, <4 x float> %1125) #8, !dbg !25
  %1128 = shufflevector <8 x half> %1094, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !25
  %1129 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed157, <4 x half> %1128, <4 x float> %1127) #8, !dbg !25
  %1130 = extractvalue [2 x <8 x float>] %56, 1, !dbg !26
  %1131 = shufflevector <8 x float> %1130, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %1132 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1095, <4 x float> %1131) #8, !dbg !25
  %1133 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1099, <4 x float> %1132) #8, !dbg !25
  %1134 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1101, <4 x float> %1133) #8, !dbg !25
  %1135 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1103, <4 x float> %1134) #8, !dbg !25
  %1136 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1105, <4 x float> %1135) #8, !dbg !25
  %1137 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1107, <4 x float> %1136) #8, !dbg !25
  %1138 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1109, <4 x float> %1137) #8, !dbg !25
  %1139 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1111, <4 x float> %1138) #8, !dbg !25
  %1140 = shufflevector <8 x float> %1130, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %1141 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed159, <4 x half> %1113, <4 x float> %1140) #8, !dbg !25
  %1142 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed161, <4 x half> %1116, <4 x float> %1141) #8, !dbg !25
  %1143 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed163, <4 x half> %1118, <4 x float> %1142) #8, !dbg !25
  %1144 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed165, <4 x half> %1120, <4 x float> %1143) #8, !dbg !25
  %1145 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed167, <4 x half> %1122, <4 x float> %1144) #8, !dbg !25
  %1146 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed169, <4 x half> %1124, <4 x float> %1145) #8, !dbg !25
  %1147 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed171, <4 x half> %1126, <4 x float> %1146) #8, !dbg !25
  %1148 = tail call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed173, <4 x half> %1128, <4 x float> %1147) #8, !dbg !25
  %1149 = shufflevector <4 x float> %895, <4 x float> %912, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1150 = insertvalue [2 x <8 x float>] poison, <8 x float> %1149, 0, !dbg !13
  %1151 = shufflevector <4 x float> %922, <4 x float> %931, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1152 = insertvalue [2 x <8 x float>] %1150, <8 x float> %1151, 1, !dbg !13
  %1153 = shufflevector <4 x float> %967, <4 x float> %984, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1154 = insertvalue [2 x <8 x float>] poison, <8 x float> %1153, 0, !dbg !13
  %1155 = shufflevector <4 x float> %994, <4 x float> %1003, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1156 = insertvalue [2 x <8 x float>] %1154, <8 x float> %1155, 1, !dbg !13
  %1157 = shufflevector <4 x float> %1042, <4 x float> %1059, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1158 = insertvalue [2 x <8 x float>] poison, <8 x float> %1157, 0, !dbg !13
  %1159 = shufflevector <4 x float> %1069, <4 x float> %1078, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1160 = insertvalue [2 x <8 x float>] %1158, <8 x float> %1159, 1, !dbg !13
  %1161 = shufflevector <4 x float> %1112, <4 x float> %1129, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1162 = insertvalue [2 x <8 x float>] poison, <8 x float> %1161, 0, !dbg !13
  %1163 = shufflevector <4 x float> %1139, <4 x float> %1148, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !13
  %1164 = insertvalue [2 x <8 x float>] %1162, <8 x float> %1163, 1, !dbg !13
  fence syncscope("workgroup") release, !dbg !13
  tail call void @llvm.amdgcn.s.barrier(), !dbg !13
  fence syncscope("workgroup") acquire, !dbg !13
  %1165 = add nuw nsw i64 %61, 1, !dbg !13
  %.not.not = icmp slt i64 %61, %8, !dbg !13
  br i1 %.not.not, label %55, label %._crit_edge, !dbg !13

._crit_edge:                                      ; preds = %510, %.._crit_edge_crit_edge
  %.idx87.pre-phi = phi i64 [ %.pre119, %.._crit_edge_crit_edge ], [ %.idx, %510 ], !dbg !14
  %address.local611.pre-phi = phi i64 [ %.pre118, %.._crit_edge_crit_edge ], [ %25, %510 ], !dbg !14
  %.pre-phi117 = phi i32 [ %.pre116, %.._crit_edge_crit_edge ], [ %828, %510 ], !dbg !13
  %.pre-phi115 = phi i32 [ %.pre114, %.._crit_edge_crit_edge ], [ %820, %510 ], !dbg !13
  %.pre-phi113 = phi i32 [ %.pre112, %.._crit_edge_crit_edge ], [ %819, %510 ], !dbg !13
  %.pre-phi109 = phi i32 [ %.pre108, %.._crit_edge_crit_edge ], [ %817, %510 ], !dbg !13
  %.lcssa93 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1152, %510 ], !dbg !13
  %.lcssa92 = phi [2 x <1 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %853, %510 ], !dbg !13
  %.lcssa91 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1156, %510 ], !dbg !13
  %.lcssa90 = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1160, %510 ], !dbg !13
  %.lcssa = phi [2 x <8 x float>] [ zeroinitializer, %.._crit_edge_crit_edge ], [ %1164, %510 ], !dbg !13
  %1166 = and i32 %19, 16, !dbg !13
  %.not2 = icmp eq i32 %1166, 0, !dbg !13
  %1167 = and i32 %19, 32, !dbg !13
  %.not3 = icmp eq i32 %1167, 0, !dbg !13
  %1168 = extractvalue [2 x <8 x float>] %.lcssa, 0, !dbg !13
  %1169 = extractelement <8 x float> %1168, i64 0, !dbg !13
  %1170 = extractelement <8 x float> %1168, i64 1, !dbg !13
  %1171 = extractelement <8 x float> %1168, i64 2, !dbg !13
  %1172 = extractelement <8 x float> %1168, i64 3, !dbg !13
  %1173 = select i1 %.not2, float %1170, float %1169, !dbg !13
  %1174 = icmp samesign ult i32 %.pre-phi113, %.pre-phi115, !dbg !13
  %1175 = select i1 %1174, i32 %.pre-phi113, i32 %.pre-phi109, !dbg !13
  %1176 = shl nuw nsw i32 %1175, 2, !dbg !13
  %1177 = bitcast float %1173 to i32, !dbg !13
  %1178 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1177), !dbg !13
  %1179 = bitcast i32 %1178 to float, !dbg !13
  %1180 = select i1 %.not2, float %1169, float %1179, !dbg !13
  %1181 = select i1 %.not2, float %1179, float %1170, !dbg !13
  %1182 = select i1 %.not2, float %1172, float %1171, !dbg !13
  %1183 = bitcast float %1182 to i32, !dbg !13
  %1184 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1183), !dbg !13
  %1185 = bitcast i32 %1184 to float, !dbg !13
  %1186 = select i1 %.not2, float %1171, float %1185, !dbg !13
  %1187 = select i1 %.not2, float %1185, float %1172, !dbg !13
  %1188 = select i1 %.not3, float %1186, float %1180, !dbg !13
  %1189 = icmp samesign ult i32 %.pre-phi117, %.pre-phi115, !dbg !13
  %1190 = select i1 %1189, i32 %.pre-phi117, i32 %.pre-phi109, !dbg !13
  %1191 = shl nuw nsw i32 %1190, 2, !dbg !13
  %1192 = bitcast float %1188 to i32, !dbg !13
  %1193 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1192), !dbg !13
  %1194 = bitcast i32 %1193 to float, !dbg !13
  %1195 = select i1 %.not3, float %1187, float %1181, !dbg !13
  %1196 = bitcast float %1195 to i32, !dbg !13
  %1197 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1196), !dbg !13
  %1198 = bitcast i32 %1197 to float, !dbg !13
  %1199 = insertelement <4 x i1> poison, i1 %.not3, i64 0, !dbg !13
  %1200 = shufflevector <4 x i1> %1199, <4 x i1> poison, <4 x i32> zeroinitializer, !dbg !13
  %1201 = insertelement <4 x float> poison, float %1180, i64 0, !dbg !13
  %1202 = insertelement <4 x float> %1201, float %1181, i64 1, !dbg !13
  %1203 = insertelement <4 x float> %1202, float %1194, i64 2, !dbg !13
  %1204 = insertelement <4 x float> %1203, float %1198, i64 3, !dbg !13
  %1205 = insertelement <4 x float> poison, float %1194, i64 0, !dbg !13
  %1206 = insertelement <4 x float> %1205, float %1198, i64 1, !dbg !13
  %1207 = insertelement <4 x float> %1206, float %1186, i64 2, !dbg !13
  %1208 = insertelement <4 x float> %1207, float %1187, i64 3, !dbg !13
  %1209 = select <4 x i1> %1200, <4 x float> %1204, <4 x float> %1208, !dbg !13
  %1210 = extractvalue [2 x <8 x float>] %.lcssa90, 0, !dbg !13
  %1211 = extractelement <8 x float> %1210, i64 0, !dbg !13
  %1212 = extractelement <8 x float> %1210, i64 1, !dbg !13
  %1213 = extractelement <8 x float> %1210, i64 2, !dbg !13
  %1214 = extractelement <8 x float> %1210, i64 3, !dbg !13
  %1215 = select i1 %.not2, float %1212, float %1211, !dbg !13
  %1216 = bitcast float %1215 to i32, !dbg !13
  %1217 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1216), !dbg !13
  %1218 = bitcast i32 %1217 to float, !dbg !13
  %1219 = select i1 %.not2, float %1211, float %1218, !dbg !13
  %1220 = select i1 %.not2, float %1218, float %1212, !dbg !13
  %1221 = select i1 %.not2, float %1214, float %1213, !dbg !13
  %1222 = bitcast float %1221 to i32, !dbg !13
  %1223 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1222), !dbg !13
  %1224 = bitcast i32 %1223 to float, !dbg !13
  %1225 = select i1 %.not2, float %1213, float %1224, !dbg !13
  %1226 = select i1 %.not2, float %1224, float %1214, !dbg !13
  %1227 = select i1 %.not3, float %1225, float %1219, !dbg !13
  %1228 = bitcast float %1227 to i32, !dbg !13
  %1229 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1228), !dbg !13
  %1230 = bitcast i32 %1229 to float, !dbg !13
  %1231 = select i1 %.not3, float %1226, float %1220, !dbg !13
  %1232 = bitcast float %1231 to i32, !dbg !13
  %1233 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1232), !dbg !13
  %1234 = bitcast i32 %1233 to float, !dbg !13
  %1235 = insertelement <4 x float> poison, float %1219, i64 0, !dbg !13
  %1236 = insertelement <4 x float> %1235, float %1220, i64 1, !dbg !13
  %1237 = insertelement <4 x float> %1236, float %1230, i64 2, !dbg !13
  %1238 = insertelement <4 x float> %1237, float %1234, i64 3, !dbg !13
  %1239 = insertelement <4 x float> poison, float %1230, i64 0, !dbg !13
  %1240 = insertelement <4 x float> %1239, float %1234, i64 1, !dbg !13
  %1241 = insertelement <4 x float> %1240, float %1225, i64 2, !dbg !13
  %1242 = insertelement <4 x float> %1241, float %1226, i64 3, !dbg !13
  %1243 = select <4 x i1> %1200, <4 x float> %1238, <4 x float> %1242, !dbg !13
  %1244 = extractvalue [2 x <8 x float>] %.lcssa91, 0, !dbg !13
  %1245 = extractelement <8 x float> %1244, i64 0, !dbg !13
  %1246 = extractelement <8 x float> %1244, i64 1, !dbg !13
  %1247 = extractelement <8 x float> %1244, i64 2, !dbg !13
  %1248 = extractelement <8 x float> %1244, i64 3, !dbg !13
  %1249 = select i1 %.not2, float %1246, float %1245, !dbg !13
  %1250 = bitcast float %1249 to i32, !dbg !13
  %1251 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1250), !dbg !13
  %1252 = bitcast i32 %1251 to float, !dbg !13
  %1253 = select i1 %.not2, float %1245, float %1252, !dbg !13
  %1254 = select i1 %.not2, float %1252, float %1246, !dbg !13
  %1255 = select i1 %.not2, float %1248, float %1247, !dbg !13
  %1256 = bitcast float %1255 to i32, !dbg !13
  %1257 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1256), !dbg !13
  %1258 = bitcast i32 %1257 to float, !dbg !13
  %1259 = select i1 %.not2, float %1247, float %1258, !dbg !13
  %1260 = select i1 %.not2, float %1258, float %1248, !dbg !13
  %1261 = select i1 %.not3, float %1259, float %1253, !dbg !13
  %1262 = bitcast float %1261 to i32, !dbg !13
  %1263 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1262), !dbg !13
  %1264 = bitcast i32 %1263 to float, !dbg !13
  %1265 = select i1 %.not3, float %1260, float %1254, !dbg !13
  %1266 = bitcast float %1265 to i32, !dbg !13
  %1267 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1266), !dbg !13
  %1268 = bitcast i32 %1267 to float, !dbg !13
  %1269 = insertelement <4 x float> poison, float %1253, i64 0, !dbg !13
  %1270 = insertelement <4 x float> %1269, float %1254, i64 1, !dbg !13
  %1271 = insertelement <4 x float> %1270, float %1264, i64 2, !dbg !13
  %1272 = insertelement <4 x float> %1271, float %1268, i64 3, !dbg !13
  %1273 = insertelement <4 x float> poison, float %1264, i64 0, !dbg !13
  %1274 = insertelement <4 x float> %1273, float %1268, i64 1, !dbg !13
  %1275 = insertelement <4 x float> %1274, float %1259, i64 2, !dbg !13
  %1276 = insertelement <4 x float> %1275, float %1260, i64 3, !dbg !13
  %1277 = select <4 x i1> %1200, <4 x float> %1272, <4 x float> %1276, !dbg !13
  %1278 = extractvalue [2 x <8 x float>] %.lcssa93, 0, !dbg !13
  %1279 = extractelement <8 x float> %1278, i64 0, !dbg !13
  %1280 = extractelement <8 x float> %1278, i64 1, !dbg !13
  %1281 = extractelement <8 x float> %1278, i64 2, !dbg !13
  %1282 = extractelement <8 x float> %1278, i64 3, !dbg !13
  %1283 = select i1 %.not2, float %1280, float %1279, !dbg !13
  %1284 = bitcast float %1283 to i32, !dbg !13
  %1285 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1284), !dbg !13
  %1286 = bitcast i32 %1285 to float, !dbg !13
  %1287 = select i1 %.not2, float %1279, float %1286, !dbg !13
  %1288 = select i1 %.not2, float %1286, float %1280, !dbg !13
  %1289 = select i1 %.not2, float %1282, float %1281, !dbg !13
  %1290 = bitcast float %1289 to i32, !dbg !13
  %1291 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1290), !dbg !13
  %1292 = bitcast i32 %1291 to float, !dbg !13
  %1293 = select i1 %.not2, float %1281, float %1292, !dbg !13
  %1294 = select i1 %.not2, float %1292, float %1282, !dbg !13
  %1295 = select i1 %.not3, float %1293, float %1287, !dbg !13
  %1296 = bitcast float %1295 to i32, !dbg !13
  %1297 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1296), !dbg !13
  %1298 = select i1 %.not3, float %1294, float %1288, !dbg !13
  %1299 = bitcast float %1298 to i32, !dbg !13
  %1300 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1299), !dbg !13
  %1301 = extractvalue [2 x <1 x float>] %.lcssa92, 0, !dbg !27
  %1302 = shufflevector <1 x float> %1301, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %1303 = extractelement <8 x float> %1168, i64 4, !dbg !13
  %1304 = extractelement <8 x float> %1168, i64 5, !dbg !13
  %1305 = extractelement <8 x float> %1168, i64 6, !dbg !13
  %1306 = extractelement <8 x float> %1168, i64 7, !dbg !13
  %1307 = select i1 %.not2, float %1304, float %1303, !dbg !13
  %1308 = bitcast float %1307 to i32, !dbg !13
  %1309 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1308), !dbg !13
  %1310 = bitcast i32 %1309 to float, !dbg !13
  %1311 = select i1 %.not2, float %1303, float %1310, !dbg !13
  %1312 = select i1 %.not2, float %1310, float %1304, !dbg !13
  %1313 = select i1 %.not2, float %1306, float %1305, !dbg !13
  %1314 = bitcast float %1313 to i32, !dbg !13
  %1315 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1314), !dbg !13
  %1316 = bitcast i32 %1315 to float, !dbg !13
  %1317 = select i1 %.not2, float %1305, float %1316, !dbg !13
  %1318 = select i1 %.not2, float %1316, float %1306, !dbg !13
  %1319 = select i1 %.not3, float %1317, float %1311, !dbg !13
  %1320 = bitcast float %1319 to i32, !dbg !13
  %1321 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1320), !dbg !13
  %1322 = bitcast i32 %1321 to float, !dbg !13
  %1323 = select i1 %.not3, float %1318, float %1312, !dbg !13
  %1324 = bitcast float %1323 to i32, !dbg !13
  %1325 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1324), !dbg !13
  %1326 = bitcast i32 %1325 to float, !dbg !13
  %1327 = insertelement <4 x float> poison, float %1311, i64 0, !dbg !13
  %1328 = insertelement <4 x float> %1327, float %1312, i64 1, !dbg !13
  %1329 = insertelement <4 x float> %1328, float %1322, i64 2, !dbg !13
  %1330 = insertelement <4 x float> %1329, float %1326, i64 3, !dbg !13
  %1331 = insertelement <4 x float> poison, float %1322, i64 0, !dbg !13
  %1332 = insertelement <4 x float> %1331, float %1326, i64 1, !dbg !13
  %1333 = insertelement <4 x float> %1332, float %1317, i64 2, !dbg !13
  %1334 = insertelement <4 x float> %1333, float %1318, i64 3, !dbg !13
  %1335 = select <4 x i1> %1200, <4 x float> %1330, <4 x float> %1334, !dbg !13
  %1336 = extractelement <8 x float> %1210, i64 4, !dbg !13
  %1337 = extractelement <8 x float> %1210, i64 5, !dbg !13
  %1338 = extractelement <8 x float> %1210, i64 6, !dbg !13
  %1339 = extractelement <8 x float> %1210, i64 7, !dbg !13
  %1340 = select i1 %.not2, float %1337, float %1336, !dbg !13
  %1341 = bitcast float %1340 to i32, !dbg !13
  %1342 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1341), !dbg !13
  %1343 = bitcast i32 %1342 to float, !dbg !13
  %1344 = select i1 %.not2, float %1336, float %1343, !dbg !13
  %1345 = select i1 %.not2, float %1343, float %1337, !dbg !13
  %1346 = select i1 %.not2, float %1339, float %1338, !dbg !13
  %1347 = bitcast float %1346 to i32, !dbg !13
  %1348 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1347), !dbg !13
  %1349 = bitcast i32 %1348 to float, !dbg !13
  %1350 = select i1 %.not2, float %1338, float %1349, !dbg !13
  %1351 = select i1 %.not2, float %1349, float %1339, !dbg !13
  %1352 = select i1 %.not3, float %1350, float %1344, !dbg !13
  %1353 = bitcast float %1352 to i32, !dbg !13
  %1354 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1353), !dbg !13
  %1355 = bitcast i32 %1354 to float, !dbg !13
  %1356 = select i1 %.not3, float %1351, float %1345, !dbg !13
  %1357 = bitcast float %1356 to i32, !dbg !13
  %1358 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1357), !dbg !13
  %1359 = bitcast i32 %1358 to float, !dbg !13
  %1360 = insertelement <4 x float> poison, float %1344, i64 0, !dbg !13
  %1361 = insertelement <4 x float> %1360, float %1345, i64 1, !dbg !13
  %1362 = insertelement <4 x float> %1361, float %1355, i64 2, !dbg !13
  %1363 = insertelement <4 x float> %1362, float %1359, i64 3, !dbg !13
  %1364 = insertelement <4 x float> poison, float %1355, i64 0, !dbg !13
  %1365 = insertelement <4 x float> %1364, float %1359, i64 1, !dbg !13
  %1366 = insertelement <4 x float> %1365, float %1350, i64 2, !dbg !13
  %1367 = insertelement <4 x float> %1366, float %1351, i64 3, !dbg !13
  %1368 = select <4 x i1> %1200, <4 x float> %1363, <4 x float> %1367, !dbg !13
  %1369 = extractelement <8 x float> %1244, i64 4, !dbg !13
  %1370 = extractelement <8 x float> %1244, i64 5, !dbg !13
  %1371 = extractelement <8 x float> %1244, i64 6, !dbg !13
  %1372 = extractelement <8 x float> %1244, i64 7, !dbg !13
  %1373 = select i1 %.not2, float %1370, float %1369, !dbg !13
  %1374 = bitcast float %1373 to i32, !dbg !13
  %1375 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1374), !dbg !13
  %1376 = bitcast i32 %1375 to float, !dbg !13
  %1377 = select i1 %.not2, float %1369, float %1376, !dbg !13
  %1378 = select i1 %.not2, float %1376, float %1370, !dbg !13
  %1379 = select i1 %.not2, float %1372, float %1371, !dbg !13
  %1380 = bitcast float %1379 to i32, !dbg !13
  %1381 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1380), !dbg !13
  %1382 = bitcast i32 %1381 to float, !dbg !13
  %1383 = select i1 %.not2, float %1371, float %1382, !dbg !13
  %1384 = select i1 %.not2, float %1382, float %1372, !dbg !13
  %1385 = select i1 %.not3, float %1383, float %1377, !dbg !13
  %1386 = bitcast float %1385 to i32, !dbg !13
  %1387 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1386), !dbg !13
  %1388 = bitcast i32 %1387 to float, !dbg !13
  %1389 = select i1 %.not3, float %1384, float %1378, !dbg !13
  %1390 = bitcast float %1389 to i32, !dbg !13
  %1391 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1390), !dbg !13
  %1392 = bitcast i32 %1391 to float, !dbg !13
  %1393 = insertelement <4 x float> poison, float %1377, i64 0, !dbg !13
  %1394 = insertelement <4 x float> %1393, float %1378, i64 1, !dbg !13
  %1395 = insertelement <4 x float> %1394, float %1388, i64 2, !dbg !13
  %1396 = insertelement <4 x float> %1395, float %1392, i64 3, !dbg !13
  %1397 = insertelement <4 x float> poison, float %1388, i64 0, !dbg !13
  %1398 = insertelement <4 x float> %1397, float %1392, i64 1, !dbg !13
  %1399 = insertelement <4 x float> %1398, float %1383, i64 2, !dbg !13
  %1400 = insertelement <4 x float> %1399, float %1384, i64 3, !dbg !13
  %1401 = select <4 x i1> %1200, <4 x float> %1396, <4 x float> %1400, !dbg !13
  %1402 = extractelement <8 x float> %1278, i64 4, !dbg !13
  %1403 = extractelement <8 x float> %1278, i64 5, !dbg !13
  %1404 = extractelement <8 x float> %1278, i64 6, !dbg !13
  %1405 = extractelement <8 x float> %1278, i64 7, !dbg !13
  %1406 = select i1 %.not2, float %1403, float %1402, !dbg !13
  %1407 = bitcast float %1406 to i32, !dbg !13
  %1408 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1407), !dbg !13
  %1409 = bitcast i32 %1408 to float, !dbg !13
  %1410 = select i1 %.not2, float %1402, float %1409, !dbg !13
  %1411 = select i1 %.not2, float %1409, float %1403, !dbg !13
  %1412 = select i1 %.not2, float %1405, float %1404, !dbg !13
  %1413 = bitcast float %1412 to i32, !dbg !13
  %1414 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1413), !dbg !13
  %1415 = bitcast i32 %1414 to float, !dbg !13
  %1416 = select i1 %.not2, float %1404, float %1415, !dbg !13
  %1417 = select i1 %.not2, float %1415, float %1405, !dbg !13
  %1418 = select i1 %.not3, float %1416, float %1410, !dbg !13
  %1419 = bitcast float %1418 to i32, !dbg !13
  %1420 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1419), !dbg !13
  %1421 = select i1 %.not3, float %1417, float %1411, !dbg !13
  %1422 = bitcast float %1421 to i32, !dbg !13
  %1423 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1422), !dbg !13
  %1424 = extractvalue [2 x <8 x float>] %.lcssa, 1, !dbg !13
  %1425 = extractelement <8 x float> %1424, i64 0, !dbg !13
  %1426 = extractelement <8 x float> %1424, i64 1, !dbg !13
  %1427 = extractelement <8 x float> %1424, i64 2, !dbg !13
  %1428 = extractelement <8 x float> %1424, i64 3, !dbg !13
  %1429 = select i1 %.not2, float %1426, float %1425, !dbg !13
  %1430 = bitcast float %1429 to i32, !dbg !13
  %1431 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1430), !dbg !13
  %1432 = bitcast i32 %1431 to float, !dbg !13
  %1433 = select i1 %.not2, float %1425, float %1432, !dbg !13
  %1434 = select i1 %.not2, float %1432, float %1426, !dbg !13
  %1435 = select i1 %.not2, float %1428, float %1427, !dbg !13
  %1436 = bitcast float %1435 to i32, !dbg !13
  %1437 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1436), !dbg !13
  %1438 = bitcast i32 %1437 to float, !dbg !13
  %1439 = select i1 %.not2, float %1427, float %1438, !dbg !13
  %1440 = select i1 %.not2, float %1438, float %1428, !dbg !13
  %1441 = select i1 %.not3, float %1439, float %1433, !dbg !13
  %1442 = bitcast float %1441 to i32, !dbg !13
  %1443 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1442), !dbg !13
  %1444 = bitcast i32 %1443 to float, !dbg !13
  %1445 = select i1 %.not3, float %1440, float %1434, !dbg !13
  %1446 = bitcast float %1445 to i32, !dbg !13
  %1447 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1446), !dbg !13
  %1448 = bitcast i32 %1447 to float, !dbg !13
  %1449 = insertelement <4 x float> poison, float %1433, i64 0, !dbg !13
  %1450 = insertelement <4 x float> %1449, float %1434, i64 1, !dbg !13
  %1451 = insertelement <4 x float> %1450, float %1444, i64 2, !dbg !13
  %1452 = insertelement <4 x float> %1451, float %1448, i64 3, !dbg !13
  %1453 = insertelement <4 x float> poison, float %1444, i64 0, !dbg !13
  %1454 = insertelement <4 x float> %1453, float %1448, i64 1, !dbg !13
  %1455 = insertelement <4 x float> %1454, float %1439, i64 2, !dbg !13
  %1456 = insertelement <4 x float> %1455, float %1440, i64 3, !dbg !13
  %1457 = select <4 x i1> %1200, <4 x float> %1452, <4 x float> %1456, !dbg !13
  %1458 = extractvalue [2 x <8 x float>] %.lcssa90, 1, !dbg !13
  %1459 = extractelement <8 x float> %1458, i64 0, !dbg !13
  %1460 = extractelement <8 x float> %1458, i64 1, !dbg !13
  %1461 = extractelement <8 x float> %1458, i64 2, !dbg !13
  %1462 = extractelement <8 x float> %1458, i64 3, !dbg !13
  %1463 = select i1 %.not2, float %1460, float %1459, !dbg !13
  %1464 = bitcast float %1463 to i32, !dbg !13
  %1465 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1464), !dbg !13
  %1466 = bitcast i32 %1465 to float, !dbg !13
  %1467 = select i1 %.not2, float %1459, float %1466, !dbg !13
  %1468 = select i1 %.not2, float %1466, float %1460, !dbg !13
  %1469 = select i1 %.not2, float %1462, float %1461, !dbg !13
  %1470 = bitcast float %1469 to i32, !dbg !13
  %1471 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1470), !dbg !13
  %1472 = bitcast i32 %1471 to float, !dbg !13
  %1473 = select i1 %.not2, float %1461, float %1472, !dbg !13
  %1474 = select i1 %.not2, float %1472, float %1462, !dbg !13
  %1475 = select i1 %.not3, float %1473, float %1467, !dbg !13
  %1476 = bitcast float %1475 to i32, !dbg !13
  %1477 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1476), !dbg !13
  %1478 = bitcast i32 %1477 to float, !dbg !13
  %1479 = select i1 %.not3, float %1474, float %1468, !dbg !13
  %1480 = bitcast float %1479 to i32, !dbg !13
  %1481 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1480), !dbg !13
  %1482 = bitcast i32 %1481 to float, !dbg !13
  %1483 = insertelement <4 x float> poison, float %1467, i64 0, !dbg !13
  %1484 = insertelement <4 x float> %1483, float %1468, i64 1, !dbg !13
  %1485 = insertelement <4 x float> %1484, float %1478, i64 2, !dbg !13
  %1486 = insertelement <4 x float> %1485, float %1482, i64 3, !dbg !13
  %1487 = insertelement <4 x float> poison, float %1478, i64 0, !dbg !13
  %1488 = insertelement <4 x float> %1487, float %1482, i64 1, !dbg !13
  %1489 = insertelement <4 x float> %1488, float %1473, i64 2, !dbg !13
  %1490 = insertelement <4 x float> %1489, float %1474, i64 3, !dbg !13
  %1491 = select <4 x i1> %1200, <4 x float> %1486, <4 x float> %1490, !dbg !13
  %1492 = extractvalue [2 x <8 x float>] %.lcssa91, 1, !dbg !13
  %1493 = extractelement <8 x float> %1492, i64 0, !dbg !13
  %1494 = extractelement <8 x float> %1492, i64 1, !dbg !13
  %1495 = extractelement <8 x float> %1492, i64 2, !dbg !13
  %1496 = extractelement <8 x float> %1492, i64 3, !dbg !13
  %1497 = select i1 %.not2, float %1494, float %1493, !dbg !13
  %1498 = bitcast float %1497 to i32, !dbg !13
  %1499 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1498), !dbg !13
  %1500 = bitcast i32 %1499 to float, !dbg !13
  %1501 = select i1 %.not2, float %1493, float %1500, !dbg !13
  %1502 = select i1 %.not2, float %1500, float %1494, !dbg !13
  %1503 = select i1 %.not2, float %1496, float %1495, !dbg !13
  %1504 = bitcast float %1503 to i32, !dbg !13
  %1505 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1504), !dbg !13
  %1506 = bitcast i32 %1505 to float, !dbg !13
  %1507 = select i1 %.not2, float %1495, float %1506, !dbg !13
  %1508 = select i1 %.not2, float %1506, float %1496, !dbg !13
  %1509 = select i1 %.not3, float %1507, float %1501, !dbg !13
  %1510 = bitcast float %1509 to i32, !dbg !13
  %1511 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1510), !dbg !13
  %1512 = bitcast i32 %1511 to float, !dbg !13
  %1513 = select i1 %.not3, float %1508, float %1502, !dbg !13
  %1514 = bitcast float %1513 to i32, !dbg !13
  %1515 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1514), !dbg !13
  %1516 = bitcast i32 %1515 to float, !dbg !13
  %1517 = insertelement <4 x float> poison, float %1501, i64 0, !dbg !13
  %1518 = insertelement <4 x float> %1517, float %1502, i64 1, !dbg !13
  %1519 = insertelement <4 x float> %1518, float %1512, i64 2, !dbg !13
  %1520 = insertelement <4 x float> %1519, float %1516, i64 3, !dbg !13
  %1521 = insertelement <4 x float> poison, float %1512, i64 0, !dbg !13
  %1522 = insertelement <4 x float> %1521, float %1516, i64 1, !dbg !13
  %1523 = insertelement <4 x float> %1522, float %1507, i64 2, !dbg !13
  %1524 = insertelement <4 x float> %1523, float %1508, i64 3, !dbg !13
  %1525 = select <4 x i1> %1200, <4 x float> %1520, <4 x float> %1524, !dbg !13
  %1526 = extractvalue [2 x <8 x float>] %.lcssa93, 1, !dbg !13
  %1527 = extractelement <8 x float> %1526, i64 0, !dbg !13
  %1528 = extractelement <8 x float> %1526, i64 1, !dbg !13
  %1529 = extractelement <8 x float> %1526, i64 2, !dbg !13
  %1530 = extractelement <8 x float> %1526, i64 3, !dbg !13
  %1531 = select i1 %.not2, float %1528, float %1527, !dbg !13
  %1532 = bitcast float %1531 to i32, !dbg !13
  %1533 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1532), !dbg !13
  %1534 = bitcast i32 %1533 to float, !dbg !13
  %1535 = select i1 %.not2, float %1527, float %1534, !dbg !13
  %1536 = select i1 %.not2, float %1534, float %1528, !dbg !13
  %1537 = select i1 %.not2, float %1530, float %1529, !dbg !13
  %1538 = bitcast float %1537 to i32, !dbg !13
  %1539 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1538), !dbg !13
  %1540 = bitcast i32 %1539 to float, !dbg !13
  %1541 = select i1 %.not2, float %1529, float %1540, !dbg !13
  %1542 = select i1 %.not2, float %1540, float %1530, !dbg !13
  %1543 = select i1 %.not3, float %1541, float %1535, !dbg !13
  %1544 = bitcast float %1543 to i32, !dbg !13
  %1545 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1544), !dbg !13
  %1546 = select i1 %.not3, float %1542, float %1536, !dbg !13
  %1547 = bitcast float %1546 to i32, !dbg !13
  %1548 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1547), !dbg !13
  %1549 = extractvalue [2 x <1 x float>] %.lcssa92, 1, !dbg !27
  %1550 = shufflevector <1 x float> %1549, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !27
  %1551 = extractelement <8 x float> %1424, i64 4, !dbg !13
  %1552 = extractelement <8 x float> %1424, i64 5, !dbg !13
  %1553 = extractelement <8 x float> %1424, i64 6, !dbg !13
  %1554 = extractelement <8 x float> %1424, i64 7, !dbg !13
  %1555 = select i1 %.not2, float %1552, float %1551, !dbg !13
  %1556 = bitcast float %1555 to i32, !dbg !13
  %1557 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1556), !dbg !13
  %1558 = bitcast i32 %1557 to float, !dbg !13
  %1559 = select i1 %.not2, float %1551, float %1558, !dbg !13
  %1560 = select i1 %.not2, float %1558, float %1552, !dbg !13
  %1561 = select i1 %.not2, float %1554, float %1553, !dbg !13
  %1562 = bitcast float %1561 to i32, !dbg !13
  %1563 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1562), !dbg !13
  %1564 = bitcast i32 %1563 to float, !dbg !13
  %1565 = select i1 %.not2, float %1553, float %1564, !dbg !13
  %1566 = select i1 %.not2, float %1564, float %1554, !dbg !13
  %1567 = select i1 %.not3, float %1565, float %1559, !dbg !13
  %1568 = bitcast float %1567 to i32, !dbg !13
  %1569 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1568), !dbg !13
  %1570 = bitcast i32 %1569 to float, !dbg !13
  %1571 = select i1 %.not3, float %1566, float %1560, !dbg !13
  %1572 = bitcast float %1571 to i32, !dbg !13
  %1573 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1572), !dbg !13
  %1574 = bitcast i32 %1573 to float, !dbg !13
  %1575 = insertelement <4 x float> poison, float %1559, i64 0, !dbg !13
  %1576 = insertelement <4 x float> %1575, float %1560, i64 1, !dbg !13
  %1577 = insertelement <4 x float> %1576, float %1570, i64 2, !dbg !13
  %1578 = insertelement <4 x float> %1577, float %1574, i64 3, !dbg !13
  %1579 = insertelement <4 x float> poison, float %1570, i64 0, !dbg !13
  %1580 = insertelement <4 x float> %1579, float %1574, i64 1, !dbg !13
  %1581 = insertelement <4 x float> %1580, float %1565, i64 2, !dbg !13
  %1582 = insertelement <4 x float> %1581, float %1566, i64 3, !dbg !13
  %1583 = select <4 x i1> %1200, <4 x float> %1578, <4 x float> %1582, !dbg !13
  %1584 = extractelement <8 x float> %1458, i64 4, !dbg !13
  %1585 = extractelement <8 x float> %1458, i64 5, !dbg !13
  %1586 = extractelement <8 x float> %1458, i64 6, !dbg !13
  %1587 = extractelement <8 x float> %1458, i64 7, !dbg !13
  %1588 = select i1 %.not2, float %1585, float %1584, !dbg !13
  %1589 = bitcast float %1588 to i32, !dbg !13
  %1590 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1589), !dbg !13
  %1591 = bitcast i32 %1590 to float, !dbg !13
  %1592 = select i1 %.not2, float %1584, float %1591, !dbg !13
  %1593 = select i1 %.not2, float %1591, float %1585, !dbg !13
  %1594 = select i1 %.not2, float %1587, float %1586, !dbg !13
  %1595 = bitcast float %1594 to i32, !dbg !13
  %1596 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1595), !dbg !13
  %1597 = bitcast i32 %1596 to float, !dbg !13
  %1598 = select i1 %.not2, float %1586, float %1597, !dbg !13
  %1599 = select i1 %.not2, float %1597, float %1587, !dbg !13
  %1600 = select i1 %.not3, float %1598, float %1592, !dbg !13
  %1601 = bitcast float %1600 to i32, !dbg !13
  %1602 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1601), !dbg !13
  %1603 = bitcast i32 %1602 to float, !dbg !13
  %1604 = select i1 %.not3, float %1599, float %1593, !dbg !13
  %1605 = bitcast float %1604 to i32, !dbg !13
  %1606 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1605), !dbg !13
  %1607 = bitcast i32 %1606 to float, !dbg !13
  %1608 = insertelement <4 x float> poison, float %1592, i64 0, !dbg !13
  %1609 = insertelement <4 x float> %1608, float %1593, i64 1, !dbg !13
  %1610 = insertelement <4 x float> %1609, float %1603, i64 2, !dbg !13
  %1611 = insertelement <4 x float> %1610, float %1607, i64 3, !dbg !13
  %1612 = insertelement <4 x float> poison, float %1603, i64 0, !dbg !13
  %1613 = insertelement <4 x float> %1612, float %1607, i64 1, !dbg !13
  %1614 = insertelement <4 x float> %1613, float %1598, i64 2, !dbg !13
  %1615 = insertelement <4 x float> %1614, float %1599, i64 3, !dbg !13
  %1616 = select <4 x i1> %1200, <4 x float> %1611, <4 x float> %1615, !dbg !13
  %1617 = extractelement <8 x float> %1492, i64 4, !dbg !13
  %1618 = extractelement <8 x float> %1492, i64 5, !dbg !13
  %1619 = extractelement <8 x float> %1492, i64 6, !dbg !13
  %1620 = extractelement <8 x float> %1492, i64 7, !dbg !13
  %1621 = select i1 %.not2, float %1618, float %1617, !dbg !13
  %1622 = bitcast float %1621 to i32, !dbg !13
  %1623 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1622), !dbg !13
  %1624 = bitcast i32 %1623 to float, !dbg !13
  %1625 = select i1 %.not2, float %1617, float %1624, !dbg !13
  %1626 = select i1 %.not2, float %1624, float %1618, !dbg !13
  %1627 = select i1 %.not2, float %1620, float %1619, !dbg !13
  %1628 = bitcast float %1627 to i32, !dbg !13
  %1629 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1628), !dbg !13
  %1630 = bitcast i32 %1629 to float, !dbg !13
  %1631 = select i1 %.not2, float %1619, float %1630, !dbg !13
  %1632 = select i1 %.not2, float %1630, float %1620, !dbg !13
  %1633 = select i1 %.not3, float %1631, float %1625, !dbg !13
  %1634 = bitcast float %1633 to i32, !dbg !13
  %1635 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1634), !dbg !13
  %1636 = bitcast i32 %1635 to float, !dbg !13
  %1637 = select i1 %.not3, float %1632, float %1626, !dbg !13
  %1638 = bitcast float %1637 to i32, !dbg !13
  %1639 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1638), !dbg !13
  %1640 = bitcast i32 %1639 to float, !dbg !13
  %1641 = insertelement <4 x float> poison, float %1625, i64 0, !dbg !13
  %1642 = insertelement <4 x float> %1641, float %1626, i64 1, !dbg !13
  %1643 = insertelement <4 x float> %1642, float %1636, i64 2, !dbg !13
  %1644 = insertelement <4 x float> %1643, float %1640, i64 3, !dbg !13
  %1645 = insertelement <4 x float> poison, float %1636, i64 0, !dbg !13
  %1646 = insertelement <4 x float> %1645, float %1640, i64 1, !dbg !13
  %1647 = insertelement <4 x float> %1646, float %1631, i64 2, !dbg !13
  %1648 = insertelement <4 x float> %1647, float %1632, i64 3, !dbg !13
  %1649 = select <4 x i1> %1200, <4 x float> %1644, <4 x float> %1648, !dbg !13
  %1650 = extractelement <8 x float> %1526, i64 4, !dbg !13
  %1651 = extractelement <8 x float> %1526, i64 5, !dbg !13
  %1652 = extractelement <8 x float> %1526, i64 6, !dbg !13
  %1653 = extractelement <8 x float> %1526, i64 7, !dbg !13
  %1654 = select i1 %.not2, float %1651, float %1650, !dbg !13
  %1655 = bitcast float %1654 to i32, !dbg !13
  %1656 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1655), !dbg !13
  %1657 = bitcast i32 %1656 to float, !dbg !13
  %1658 = select i1 %.not2, float %1650, float %1657, !dbg !13
  %1659 = select i1 %.not2, float %1657, float %1651, !dbg !13
  %1660 = select i1 %.not2, float %1653, float %1652, !dbg !13
  %1661 = bitcast float %1660 to i32, !dbg !13
  %1662 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1176, i32 %1661), !dbg !13
  %1663 = bitcast i32 %1662 to float, !dbg !13
  %1664 = select i1 %.not2, float %1652, float %1663, !dbg !13
  %1665 = select i1 %.not2, float %1663, float %1653, !dbg !13
  %1666 = select i1 %.not3, float %1664, float %1658, !dbg !13
  %1667 = bitcast float %1666 to i32, !dbg !13
  %1668 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1667), !dbg !13
  %1669 = select i1 %.not3, float %1665, float %1659, !dbg !13
  %1670 = bitcast float %1669 to i32, !dbg !13
  %1671 = tail call i32 @llvm.amdgcn.ds.bpermute(i32 %1191, i32 %1670), !dbg !13
  %1672 = bitcast i32 %1668 to float, !dbg !13
  %1673 = bitcast i32 %1671 to float, !dbg !13
  %1674 = insertelement <4 x float> poison, float %1658, i64 0, !dbg !13
  %1675 = insertelement <4 x float> %1674, float %1659, i64 1, !dbg !13
  %1676 = insertelement <4 x float> %1675, float %1672, i64 2, !dbg !13
  %1677 = insertelement <4 x float> %1676, float %1673, i64 3, !dbg !13
  %1678 = insertelement <4 x float> poison, float %1672, i64 0, !dbg !13
  %1679 = insertelement <4 x float> %1678, float %1673, i64 1, !dbg !13
  %1680 = insertelement <4 x float> %1679, float %1664, i64 2, !dbg !13
  %1681 = insertelement <4 x float> %1680, float %1665, i64 3, !dbg !13
  %1682 = select <4 x i1> %1200, <4 x float> %1677, <4 x float> %1681, !dbg !13
  %row.reciprocal = fdiv arcp <4 x float> splat (float 1.000000e+00), %1550, !dbg !27
  %normalized = fmul <4 x float> %row.reciprocal, %1682, !dbg !27
  %1683 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !28
  %1684 = extractelement <4 x half> %1683, i64 3, !dbg !28
  %1685 = extractelement <4 x half> %1683, i64 2, !dbg !28
  %1686 = extractelement <4 x half> %1683, i64 1, !dbg !28
  %1687 = extractelement <4 x half> %1683, i64 0, !dbg !28
  %1688 = bitcast i32 %1545 to float, !dbg !13
  %1689 = bitcast i32 %1548 to float, !dbg !13
  %1690 = insertelement <4 x float> poison, float %1535, i64 0, !dbg !13
  %1691 = insertelement <4 x float> %1690, float %1536, i64 1, !dbg !13
  %1692 = insertelement <4 x float> %1691, float %1688, i64 2, !dbg !13
  %1693 = insertelement <4 x float> %1692, float %1689, i64 3, !dbg !13
  %1694 = insertelement <4 x float> poison, float %1688, i64 0, !dbg !13
  %1695 = insertelement <4 x float> %1694, float %1689, i64 1, !dbg !13
  %1696 = insertelement <4 x float> %1695, float %1541, i64 2, !dbg !13
  %1697 = insertelement <4 x float> %1696, float %1542, i64 3, !dbg !13
  %1698 = select <4 x i1> %1200, <4 x float> %1693, <4 x float> %1697, !dbg !13
  %normalized71 = fmul <4 x float> %row.reciprocal, %1698, !dbg !27
  %1699 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !28
  %1700 = extractelement <4 x half> %1699, i64 3, !dbg !28
  %1701 = extractelement <4 x half> %1699, i64 2, !dbg !28
  %1702 = extractelement <4 x half> %1699, i64 1, !dbg !28
  %1703 = extractelement <4 x half> %1699, i64 0, !dbg !28
  %1704 = bitcast i32 %1420 to float, !dbg !13
  %1705 = bitcast i32 %1423 to float, !dbg !13
  %1706 = insertelement <4 x float> poison, float %1410, i64 0, !dbg !13
  %1707 = insertelement <4 x float> %1706, float %1411, i64 1, !dbg !13
  %1708 = insertelement <4 x float> %1707, float %1704, i64 2, !dbg !13
  %1709 = insertelement <4 x float> %1708, float %1705, i64 3, !dbg !13
  %1710 = insertelement <4 x float> poison, float %1704, i64 0, !dbg !13
  %1711 = insertelement <4 x float> %1710, float %1705, i64 1, !dbg !13
  %1712 = insertelement <4 x float> %1711, float %1416, i64 2, !dbg !13
  %1713 = insertelement <4 x float> %1712, float %1417, i64 3, !dbg !13
  %1714 = select <4 x i1> %1200, <4 x float> %1709, <4 x float> %1713, !dbg !13
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 1.000000e+00), %1302, !dbg !27
  %normalized73 = fmul <4 x float> %row.reciprocal72, %1714, !dbg !27
  %1715 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !28
  %1716 = extractelement <4 x half> %1715, i64 3, !dbg !28
  %1717 = extractelement <4 x half> %1715, i64 2, !dbg !28
  %1718 = extractelement <4 x half> %1715, i64 1, !dbg !28
  %1719 = extractelement <4 x half> %1715, i64 0, !dbg !28
  %1720 = bitcast i32 %1297 to float, !dbg !13
  %1721 = bitcast i32 %1300 to float, !dbg !13
  %1722 = insertelement <4 x float> poison, float %1287, i64 0, !dbg !13
  %1723 = insertelement <4 x float> %1722, float %1288, i64 1, !dbg !13
  %1724 = insertelement <4 x float> %1723, float %1720, i64 2, !dbg !13
  %1725 = insertelement <4 x float> %1724, float %1721, i64 3, !dbg !13
  %1726 = insertelement <4 x float> poison, float %1720, i64 0, !dbg !13
  %1727 = insertelement <4 x float> %1726, float %1721, i64 1, !dbg !13
  %1728 = insertelement <4 x float> %1727, float %1293, i64 2, !dbg !13
  %1729 = insertelement <4 x float> %1728, float %1294, i64 3, !dbg !13
  %1730 = select <4 x i1> %1200, <4 x float> %1725, <4 x float> %1729, !dbg !13
  %normalized74 = fmul <4 x float> %row.reciprocal72, %1730, !dbg !27
  %1731 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !28
  %1732 = extractelement <4 x half> %1731, i64 3, !dbg !28
  %1733 = extractelement <4 x half> %1731, i64 2, !dbg !28
  %1734 = extractelement <4 x half> %1731, i64 1, !dbg !28
  %1735 = extractelement <4 x half> %1731, i64 0, !dbg !28
  %address.thread609 = tail call i32 asm sideeffect "", "=v,0"(i32 %19) #8, !dbg !14, !range !11, !frisk.register_lifetimes !5
  %address.local610 = zext nneg i32 %address.thread609 to i64, !dbg !10
  %1736 = lshr i64 %address.local610, 2, !dbg !14
  %address.local613 = and i64 %1736, 48, !dbg !14
  %address.local615 = and i64 %address.local610, 240, !dbg !14
  %address.local614 = add nsw i64 %address.local611.pre-phi, %address.local610, !dbg !14
  %address.local616 = sub nsw i64 %address.local614, %address.local615, !dbg !14
  %address.local617 = add nsw i64 %address.local616, %address.local613, !dbg !14
  %1737 = getelementptr inbounds nuw i8, ptr addrspace(1) %3, i64 %.idx87.pre-phi, !dbg !14
  %address.local620.idx = shl nuw nsw i64 %address.local617, 8, !dbg !14
  %address.local620 = getelementptr inbounds nuw i8, ptr addrspace(1) %1737, i64 %address.local620.idx, !dbg !14
  %address.local622 = and i64 %1736, 12, !dbg !10
  %address.local623 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local622, !dbg !14
  store half %1735, ptr addrspace(1) %address.local623, align 2, !dbg !14
  %address.local625 = or disjoint i64 %address.local622, 1, !dbg !14
  %address.local626 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local625, !dbg !14
  store half %1734, ptr addrspace(1) %address.local626, align 2, !dbg !14
  %address.local628 = or disjoint i64 %address.local622, 2, !dbg !14
  %address.local629 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local628, !dbg !14
  store half %1733, ptr addrspace(1) %address.local629, align 2, !dbg !14
  %address.local631 = or disjoint i64 %address.local622, 3, !dbg !14
  %address.local632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local631, !dbg !14
  store half %1732, ptr addrspace(1) %address.local632, align 2, !dbg !14
  %address.local634 = or disjoint i64 %address.local622, 16, !dbg !14
  %address.local635 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local634, !dbg !14
  store half %1719, ptr addrspace(1) %address.local635, align 2, !dbg !14
  %address.local637 = or disjoint i64 %address.local622, 17, !dbg !14
  %address.local638 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local637, !dbg !14
  store half %1718, ptr addrspace(1) %address.local638, align 2, !dbg !14
  %address.local640 = or disjoint i64 %address.local622, 18, !dbg !14
  %address.local641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local640, !dbg !14
  store half %1717, ptr addrspace(1) %address.local641, align 2, !dbg !14
  %address.local643 = or disjoint i64 %address.local622, 19, !dbg !14
  %address.local644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local643, !dbg !14
  store half %1716, ptr addrspace(1) %address.local644, align 2, !dbg !14
  %address.local648 = getelementptr inbounds nuw i8, ptr addrspace(1) %address.local620, i64 16384, !dbg !14
  %address.local649 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local622, !dbg !14
  store half %1703, ptr addrspace(1) %address.local649, align 2, !dbg !14
  %address.local651 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local625, !dbg !14
  store half %1702, ptr addrspace(1) %address.local651, align 2, !dbg !14
  %address.local653 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local628, !dbg !14
  store half %1701, ptr addrspace(1) %address.local653, align 2, !dbg !14
  %address.local655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local631, !dbg !14
  store half %1700, ptr addrspace(1) %address.local655, align 2, !dbg !14
  %address.local657 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local634, !dbg !14
  store half %1687, ptr addrspace(1) %address.local657, align 2, !dbg !14
  %address.local659 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local637, !dbg !14
  store half %1686, ptr addrspace(1) %address.local659, align 2, !dbg !14
  %address.local661 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local640, !dbg !14
  store half %1685, ptr addrspace(1) %address.local661, align 2, !dbg !14
  %address.local663 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local643, !dbg !14
  store half %1684, ptr addrspace(1) %address.local663, align 2, !dbg !14
  %normalized76 = fmul <4 x float> %row.reciprocal, %1649, !dbg !27
  %1738 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !28
  %1739 = extractelement <4 x half> %1738, i64 3, !dbg !28
  %1740 = extractelement <4 x half> %1738, i64 2, !dbg !28
  %1741 = extractelement <4 x half> %1738, i64 1, !dbg !28
  %1742 = extractelement <4 x half> %1738, i64 0, !dbg !28
  %normalized77 = fmul <4 x float> %row.reciprocal, %1525, !dbg !27
  %1743 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !28
  %1744 = extractelement <4 x half> %1743, i64 3, !dbg !28
  %1745 = extractelement <4 x half> %1743, i64 2, !dbg !28
  %1746 = extractelement <4 x half> %1743, i64 1, !dbg !28
  %1747 = extractelement <4 x half> %1743, i64 0, !dbg !28
  %normalized79 = fmul <4 x float> %row.reciprocal72, %1401, !dbg !27
  %1748 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !28
  %1749 = extractelement <4 x half> %1748, i64 3, !dbg !28
  %1750 = extractelement <4 x half> %1748, i64 2, !dbg !28
  %1751 = extractelement <4 x half> %1748, i64 1, !dbg !28
  %1752 = extractelement <4 x half> %1748, i64 0, !dbg !28
  %normalized80 = fmul <4 x float> %row.reciprocal72, %1277, !dbg !27
  %1753 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !28
  %1754 = extractelement <4 x half> %1753, i64 3, !dbg !28
  %1755 = extractelement <4 x half> %1753, i64 2, !dbg !28
  %1756 = extractelement <4 x half> %1753, i64 1, !dbg !28
  %1757 = extractelement <4 x half> %1753, i64 0, !dbg !28
  %address.local678 = or disjoint i64 %address.local622, 32, !dbg !14
  %address.local679 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local678, !dbg !14
  store half %1757, ptr addrspace(1) %address.local679, align 2, !dbg !14
  %address.local681 = or disjoint i64 %address.local622, 33, !dbg !14
  %address.local682 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local681, !dbg !14
  store half %1756, ptr addrspace(1) %address.local682, align 2, !dbg !14
  %address.local684 = or disjoint i64 %address.local622, 34, !dbg !14
  %address.local685 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local684, !dbg !14
  store half %1755, ptr addrspace(1) %address.local685, align 2, !dbg !14
  %address.local687 = or disjoint i64 %address.local622, 35, !dbg !14
  %address.local688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local687, !dbg !14
  store half %1754, ptr addrspace(1) %address.local688, align 2, !dbg !14
  %address.local690 = or disjoint i64 %address.local622, 48, !dbg !14
  %address.local691 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local690, !dbg !14
  store half %1752, ptr addrspace(1) %address.local691, align 2, !dbg !14
  %address.local693 = or disjoint i64 %address.local622, 49, !dbg !14
  %address.local694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local693, !dbg !14
  store half %1751, ptr addrspace(1) %address.local694, align 2, !dbg !14
  %address.local696 = or disjoint i64 %address.local622, 50, !dbg !14
  %address.local697 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local696, !dbg !14
  store half %1750, ptr addrspace(1) %address.local697, align 2, !dbg !14
  %address.local699 = or i64 %1736, 51, !dbg !14
  %address.local700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local699, !dbg !14
  store half %1749, ptr addrspace(1) %address.local700, align 2, !dbg !14
  %address.local705 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local678, !dbg !14
  store half %1747, ptr addrspace(1) %address.local705, align 2, !dbg !14
  %address.local707 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local681, !dbg !14
  store half %1746, ptr addrspace(1) %address.local707, align 2, !dbg !14
  %address.local709 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local684, !dbg !14
  store half %1745, ptr addrspace(1) %address.local709, align 2, !dbg !14
  %address.local711 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local687, !dbg !14
  store half %1744, ptr addrspace(1) %address.local711, align 2, !dbg !14
  %address.local713 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local690, !dbg !14
  store half %1742, ptr addrspace(1) %address.local713, align 2, !dbg !14
  %address.local715 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local693, !dbg !14
  store half %1741, ptr addrspace(1) %address.local715, align 2, !dbg !14
  %address.local717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local696, !dbg !14
  store half %1740, ptr addrspace(1) %address.local717, align 2, !dbg !14
  %address.local719 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local699, !dbg !14
  store half %1739, ptr addrspace(1) %address.local719, align 2, !dbg !14
  %normalized82 = fmul <4 x float> %row.reciprocal, %1616, !dbg !27
  %1758 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !28
  %1759 = extractelement <4 x half> %1758, i64 3, !dbg !28
  %1760 = extractelement <4 x half> %1758, i64 2, !dbg !28
  %1761 = extractelement <4 x half> %1758, i64 1, !dbg !28
  %1762 = extractelement <4 x half> %1758, i64 0, !dbg !28
  %normalized83 = fmul <4 x float> %row.reciprocal, %1491, !dbg !27
  %1763 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !28
  %1764 = extractelement <4 x half> %1763, i64 3, !dbg !28
  %1765 = extractelement <4 x half> %1763, i64 2, !dbg !28
  %1766 = extractelement <4 x half> %1763, i64 1, !dbg !28
  %1767 = extractelement <4 x half> %1763, i64 0, !dbg !28
  %normalized85 = fmul <4 x float> %row.reciprocal72, %1368, !dbg !27
  %1768 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !28
  %1769 = extractelement <4 x half> %1768, i64 3, !dbg !28
  %1770 = extractelement <4 x half> %1768, i64 2, !dbg !28
  %1771 = extractelement <4 x half> %1768, i64 1, !dbg !28
  %1772 = extractelement <4 x half> %1768, i64 0, !dbg !28
  %normalized86 = fmul <4 x float> %row.reciprocal72, %1243, !dbg !27
  %1773 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !28
  %1774 = extractelement <4 x half> %1773, i64 3, !dbg !28
  %1775 = extractelement <4 x half> %1773, i64 2, !dbg !28
  %1776 = extractelement <4 x half> %1773, i64 1, !dbg !28
  %1777 = extractelement <4 x half> %1773, i64 0, !dbg !28
  %address.local734 = or disjoint i64 %address.local622, 64, !dbg !14
  %address.local735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local734, !dbg !14
  store half %1777, ptr addrspace(1) %address.local735, align 2, !dbg !14
  %address.local737 = or disjoint i64 %address.local622, 65, !dbg !14
  %address.local738 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local737, !dbg !14
  store half %1776, ptr addrspace(1) %address.local738, align 2, !dbg !14
  %address.local740 = or disjoint i64 %address.local622, 66, !dbg !14
  %address.local741 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local740, !dbg !14
  store half %1775, ptr addrspace(1) %address.local741, align 2, !dbg !14
  %address.local743 = or disjoint i64 %address.local622, 67, !dbg !14
  %address.local744 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local743, !dbg !14
  store half %1774, ptr addrspace(1) %address.local744, align 2, !dbg !14
  %address.local746 = or disjoint i64 %address.local622, 80, !dbg !14
  %address.local747 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local746, !dbg !14
  store half %1772, ptr addrspace(1) %address.local747, align 2, !dbg !14
  %address.local749 = or disjoint i64 %address.local622, 81, !dbg !14
  %address.local750 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local749, !dbg !14
  store half %1771, ptr addrspace(1) %address.local750, align 2, !dbg !14
  %address.local752 = or disjoint i64 %address.local622, 82, !dbg !14
  %address.local753 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local752, !dbg !14
  store half %1770, ptr addrspace(1) %address.local753, align 2, !dbg !14
  %address.local755 = or disjoint i64 %address.local622, 83, !dbg !14
  %address.local756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local755, !dbg !14
  store half %1769, ptr addrspace(1) %address.local756, align 2, !dbg !14
  %address.local761 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local734, !dbg !14
  store half %1767, ptr addrspace(1) %address.local761, align 2, !dbg !14
  %address.local763 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local737, !dbg !14
  store half %1766, ptr addrspace(1) %address.local763, align 2, !dbg !14
  %address.local765 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local740, !dbg !14
  store half %1765, ptr addrspace(1) %address.local765, align 2, !dbg !14
  %address.local767 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local743, !dbg !14
  store half %1764, ptr addrspace(1) %address.local767, align 2, !dbg !14
  %address.local769 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local746, !dbg !14
  store half %1762, ptr addrspace(1) %address.local769, align 2, !dbg !14
  %address.local771 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local749, !dbg !14
  store half %1761, ptr addrspace(1) %address.local771, align 2, !dbg !14
  %address.local773 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local752, !dbg !14
  store half %1760, ptr addrspace(1) %address.local773, align 2, !dbg !14
  %address.local775 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local755, !dbg !14
  store half %1759, ptr addrspace(1) %address.local775, align 2, !dbg !14
  %normalized88 = fmul <4 x float> %row.reciprocal, %1583, !dbg !27
  %1778 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !28
  %1779 = extractelement <4 x half> %1778, i64 3, !dbg !28
  %1780 = extractelement <4 x half> %1778, i64 2, !dbg !28
  %1781 = extractelement <4 x half> %1778, i64 1, !dbg !28
  %1782 = extractelement <4 x half> %1778, i64 0, !dbg !28
  %normalized89 = fmul <4 x float> %row.reciprocal, %1457, !dbg !27
  %1783 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !28
  %1784 = extractelement <4 x half> %1783, i64 3, !dbg !28
  %1785 = extractelement <4 x half> %1783, i64 2, !dbg !28
  %1786 = extractelement <4 x half> %1783, i64 1, !dbg !28
  %1787 = extractelement <4 x half> %1783, i64 0, !dbg !28
  %normalized91 = fmul <4 x float> %row.reciprocal72, %1335, !dbg !27
  %1788 = fptrunc <4 x float> %normalized91 to <4 x half>, !dbg !28
  %1789 = extractelement <4 x half> %1788, i64 3, !dbg !28
  %1790 = extractelement <4 x half> %1788, i64 2, !dbg !28
  %1791 = extractelement <4 x half> %1788, i64 1, !dbg !28
  %1792 = extractelement <4 x half> %1788, i64 0, !dbg !28
  %normalized92 = fmul <4 x float> %row.reciprocal72, %1209, !dbg !27
  %1793 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !28
  %1794 = extractelement <4 x half> %1793, i64 3, !dbg !28
  %1795 = extractelement <4 x half> %1793, i64 2, !dbg !28
  %1796 = extractelement <4 x half> %1793, i64 1, !dbg !28
  %1797 = extractelement <4 x half> %1793, i64 0, !dbg !28
  %address.local790 = or disjoint i64 %address.local622, 96, !dbg !14
  %address.local791 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local790, !dbg !14
  store half %1797, ptr addrspace(1) %address.local791, align 2, !dbg !14
  %address.local793 = or disjoint i64 %address.local622, 97, !dbg !14
  %address.local794 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local793, !dbg !14
  store half %1796, ptr addrspace(1) %address.local794, align 2, !dbg !14
  %address.local796 = or disjoint i64 %address.local622, 98, !dbg !14
  %address.local797 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local796, !dbg !14
  store half %1795, ptr addrspace(1) %address.local797, align 2, !dbg !14
  %address.local799 = or disjoint i64 %address.local622, 99, !dbg !14
  %address.local800 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local799, !dbg !14
  store half %1794, ptr addrspace(1) %address.local800, align 2, !dbg !14
  %address.local802 = or disjoint i64 %address.local622, 112, !dbg !14
  %address.local803 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local802, !dbg !14
  store half %1792, ptr addrspace(1) %address.local803, align 2, !dbg !14
  %address.local805 = or disjoint i64 %address.local622, 113, !dbg !14
  %address.local806 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local805, !dbg !14
  store half %1791, ptr addrspace(1) %address.local806, align 2, !dbg !14
  %address.local808 = or disjoint i64 %address.local622, 114, !dbg !14
  %address.local809 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local808, !dbg !14
  store half %1790, ptr addrspace(1) %address.local809, align 2, !dbg !14
  %address.local811 = or i64 %1736, 115, !dbg !14
  %address.local812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local620, i64 %address.local811, !dbg !14
  store half %1789, ptr addrspace(1) %address.local812, align 2, !dbg !14
  %address.local817 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local790, !dbg !14
  store half %1787, ptr addrspace(1) %address.local817, align 2, !dbg !14
  %address.local819 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local793, !dbg !14
  store half %1786, ptr addrspace(1) %address.local819, align 2, !dbg !14
  %address.local821 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local796, !dbg !14
  store half %1785, ptr addrspace(1) %address.local821, align 2, !dbg !14
  %address.local823 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local799, !dbg !14
  store half %1784, ptr addrspace(1) %address.local823, align 2, !dbg !14
  %address.local825 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local802, !dbg !14
  store half %1782, ptr addrspace(1) %address.local825, align 2, !dbg !14
  %address.local827 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local805, !dbg !14
  store half %1781, ptr addrspace(1) %address.local827, align 2, !dbg !14
  %address.local829 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local808, !dbg !14
  store half %1780, ptr addrspace(1) %address.local829, align 2, !dbg !14
  %address.local831 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local648, i64 %address.local811, !dbg !14
  store half %1779, ptr addrspace(1) %address.local831, align 2, !dbg !14
  ret void, !dbg !7
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #2

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #3

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #3

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #4

; Function Attrs: convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #5

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #6

; Function Attrs: mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.amdgcn.exp2.f32(float) #7

; Function Attrs: convergent mustprogress nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #6

attributes #0 = { convergent mustprogress nofree willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.frag_prefetch"="4" "frisk.global_prefetch"="8" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #3 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #4 = { convergent mustprogress nocallback nofree nounwind willreturn memory(none) }
attributes #5 = { convergent mustprogress nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #6 = { convergent mustprogress nocallback nofree nounwind willreturn }
attributes #7 = { mustprogress nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #8 = { convergent nounwind willreturn memory(none) }
attributes #9 = { convergent nounwind }
attributes #10 = { memory(inaccessiblemem: readwrite) }
attributes #11 = { convergent willreturn memory(read) }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test_input_spidD.mlir", directory: "3rd/deepgengraph/test")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !1, file: !1, line: 22, type: !4, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !5)
!5 = !{}
!6 = !{i32 256, i32 1, i32 1}
!7 = !DILocation(line: 22, column: 3, scope: !3)
!8 = !DILocation(line: 46, column: 17, scope: !3)
!9 = !DILocation(line: 66, column: 22, scope: !3)
!10 = !DILocation(line: 68, column: 24, scope: !3)
!11 = !{i32 0, i32 256}
!12 = !DILocation(line: 67, column: 22, scope: !3)
!13 = !DILocation(line: 56, column: 24, scope: !3)
!14 = !DILocation(line: 99, column: 9, scope: !3)
!15 = !DILocation(line: 76, column: 23, scope: !3)
!16 = !DILocation(line: 59, column: 17, scope: !3)
!17 = !DILocation(line: 69, column: 26, scope: !3)
!18 = !DILocation(line: 92, column: 20, scope: !3)
!19 = !DILocation(line: 73, column: 19, scope: !3)
!20 = !DILocation(line: 77, column: 26, scope: !3)
!21 = !DILocation(line: 84, column: 21, scope: !3)
!22 = !DILocation(line: 85, column: 22, scope: !3)
!23 = !DILocation(line: 87, column: 16, scope: !3)
!24 = !DILocation(line: 86, column: 20, scope: !3)
!25 = !DILocation(line: 93, column: 23, scope: !3)
!26 = !DILocation(line: 94, column: 20, scope: !3)
!27 = !DILocation(line: 97, column: 23, scope: !3)
!28 = !DILocation(line: 98, column: 19, scope: !3)
