; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup/nohint_unrolled.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup/nohint_unrolled.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !31 {
  %qr.n5 = call i32 @llvm.amdgcn.workgroup.id.z(), !dbg !5
  %qr.n6 = sext i32 %qr.n5 to i64, !dbg !5
  %bo.x = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %bo.y = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %bo.row = shl i32 %bo.y, 5
  %bo.linear = add i32 %bo.row, %bo.x
  %bo.within = and i32 %bo.linear, 7
  %bo.chunk = lshr i32 %bo.linear, 8
  %bo.start = shl i32 %bo.chunk, 3
  %bo.query = add i32 %bo.start, %bo.within
  %bo.head_chunk = lshr i32 %bo.linear, 3
  %bo.head = and i32 %bo.head_chunk, 31
  %bo.reverse = sub i32 31, %bo.query
  %qr.n7 = add i32 %bo.reverse, 0, !dbg !5
  %qr.n8 = sext i32 %qr.n7 to i64, !dbg !5
  %qr.n9 = add i32 %bo.head, 0, !dbg !5
  %qr.n10 = sext i32 %qr.n9 to i64, !dbg !5
  %qr.n11 = shl nsw i64 %qr.n8, 7, !dbg !9
  %.lobit = ashr i32 %qr.n7, 31, !dbg !11
  %qr.n12 = xor i32 %qr.n7, %.lobit, !dbg !11
  %qr.n13 = sdiv i32 %qr.n12, 32, !dbg !11
  %.lobit1 = ashr i32 %qr.n7, 31, !dbg !11
  %qr.n14 = xor i32 %qr.n13, %.lobit1, !dbg !11
  %qr.n15 = sext i32 %qr.n14 to i64, !dbg !11
  %qr.n16 = add nsw i64 %qr.n10, %qr.n15, !dbg !11
  %qr.n17 = and i64 %qr.n16, 31, !dbg !11
  %qr.n18 = shl nsw i64 %qr.n8, 7, !dbg !11
  %qr.n19 = mul nsw i64 %qr.n15, -4096, !dbg !11
  %qr.n20 = add nsw i64 %qr.n18, %qr.n19, !dbg !11
  %qr.n21 = shl nuw nsw i64 %qr.n17, 19, !dbg !11
  %qr.n22 = shl nsw i64 %qr.n20, 7, !dbg !11
  %qr.n23 = add nsw i64 %qr.n21, %qr.n22, !dbg !11
  %qr.n24 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !12, !range !33
  %ksw.dma.bit = and i32 %qr.n24, 16
  %ksw.dma.mask = shl i32 %ksw.dma.bit, 6
  %ksw.read.bit = and i32 %qr.n24, 1
  %ksw.read.mask = shl i32 %ksw.read.bit, 6
  %qr.n25 = zext nneg i32 %qr.n24 to i64, !dbg !12
  %qr.n26 = lshr i64 %qr.n25, 6, !dbg !12
  %qr.n27 = or disjoint i64 %qr.n11, 127, !dbg !13
  %qr.n28 = add nsw i64 %qr.n11, 128, !dbg !13
  %qr.n29 = icmp uge i64 %qr.n28, %qr.n27, !dbg !13
  %qr.n30 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %qr.n23, !dbg !11
  %qr.n31 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %qr.n23, !dbg !11
  %qr.n32 = getelementptr i8, ptr addrspace(1) %qr.n31, i64 64, !dbg !11
  %qr.n33 = shl nsw i64 %qr.n10, 7, !dbg !16
  %qr.n34 = shl nsw i64 %qr.n6, 12, !dbg !16
  %qr.n35 = add nsw i64 %qr.n33, %qr.n34, !dbg !16
  %qr.n36 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %qr.n23, !dbg !11
  %qr.n37 = getelementptr i8, ptr addrspace(1) %qr.n36, i64 128, !dbg !11
  %qr.n38 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %qr.n23, !dbg !11
  %qr.n39 = getelementptr i8, ptr addrspace(1) %qr.n38, i64 192, !dbg !11
  ; Load and scale Q[:, 0:32] once before the BN loop.
  %qkeep.address.thread = call i32 asm sideeffect "", "=v,0"(i32 %qr.n24) #9, !dbg !11, !range !33, !frisk.register_lifetimes !1
  %qkeep.address.local = zext nneg i32 %qkeep.address.thread to i64, !dbg !12
  %qkeep.address.local129 = shl nuw nsw i64 %qkeep.address.local, 3, !dbg !11
  %qkeep.address.local130 = and i64 %qkeep.address.local129, 1536, !dbg !11
  %qkeep.address.local131 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %qkeep.address.local130, !dbg !11
  %qkeep.address.local132 = ptrtoint ptr addrspace(3) %qkeep.address.local131 to i64, !dbg !11
  %qkeep.address.local133 = trunc i64 %qkeep.address.local132 to i32, !dbg !11
  %qkeep.qr.n49 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qkeep.address.local133), !dbg !11
  %qkeep.qr.n50 = zext i32 %qkeep.qr.n49 to i64, !dbg !11
  %qkeep.qr.n51 = inttoptr i64 %qkeep.qr.n50 to ptr addrspace(3), !dbg !11
  %qkeep.qr.n61 = ptrtoint ptr addrspace(3) %qkeep.qr.n51 to i32, !dbg !11
  %qkeep.dma.buffer.lds936 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qkeep.qr.n61), !dbg !11
  %qkeep.address.local134 = shl nuw nsw i64 %qkeep.address.local, 7, !dbg !11
  %qkeep.address.local135 = and i64 %qkeep.address.local134, 16256, !dbg !11
  %qkeep.address.local136 = lshr i64 %qkeep.address.local, 4, !dbg !11
  %qkeep.address.local137 = and i64 %qkeep.address.local136, 8, !dbg !11
  %qkeep.address.local138 = or disjoint i64 %qkeep.address.local135, %qkeep.address.local137, !dbg !11
  %qkeep.address.local139 = getelementptr [2 x i8], ptr addrspace(1) %qr.n30, i64 %qkeep.address.local138, !dbg !11
  %qkeep.qr.n53 = ptrtoint ptr addrspace(1) %qkeep.address.local139 to i64, !dbg !11
  %qkeep.qr.n52 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %qkeep.qr.n54 = sub i64 %qkeep.qr.n53, %qkeep.qr.n52, !dbg !11
  %qkeep.dma.buffer.offset934 = trunc i64 %qkeep.qr.n54 to i32, !dbg !11
  %qkeep.qr.n55 = trunc i64 %qkeep.qr.n52 to i32, !dbg !11
  %qkeep.qr.n56 = insertelement <4 x i32> poison, i32 %qkeep.qr.n55, i64 0, !dbg !11
  %qkeep.qr.n57 = lshr i64 %qkeep.qr.n52, 32, !dbg !11
  %qkeep.qr.n58 = trunc i64 %qkeep.qr.n57 to i32, !dbg !11
  %qkeep.qr.n59 = insertelement <4 x i32> %qkeep.qr.n56, i32 %qkeep.qr.n58, i64 1, !dbg !11
  %qkeep.qr.n60 = insertelement <4 x i32> %qkeep.qr.n59, i32 33554432, i64 2, !dbg !11
  %qkeep.dma.buffer.rsrc935 = insertelement <4 x i32> %qkeep.qr.n60, i32 131072, i64 3, !dbg !11
  %qkeep.qr.n62 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %qkeep.dma.buffer.lds936, i32 %qkeep.dma.buffer.offset934, <4 x i32> %qkeep.dma.buffer.rsrc935) #10, !dbg !11
  %qkeep.address.local140 = or disjoint i64 %qkeep.address.local, 256, !dbg !11
  %qkeep.address.local141 = shl nuw nsw i64 %qkeep.address.local140, 3, !dbg !11
  %qkeep.address.local142 = and i64 %qkeep.address.local141, 3584, !dbg !11
  %qkeep.address.local143 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %qkeep.address.local142, !dbg !11
  %qkeep.address.local144 = ptrtoint ptr addrspace(3) %qkeep.address.local143 to i64, !dbg !11
  %qkeep.address.local145 = trunc i64 %qkeep.address.local144 to i32, !dbg !11
  %qkeep.qr.n63 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qkeep.address.local145), !dbg !11
  %qkeep.qr.n64 = zext i32 %qkeep.qr.n63 to i64, !dbg !11
  %qkeep.qr.n65 = inttoptr i64 %qkeep.qr.n64 to ptr addrspace(3), !dbg !11
  %qkeep.qr.n75 = ptrtoint ptr addrspace(3) %qkeep.qr.n65 to i32, !dbg !11
  %qkeep.dma.buffer.lds933 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qkeep.qr.n75), !dbg !11
  %qkeep.address.local146 = shl nuw nsw i64 %qkeep.address.local, 7, !dbg !11
  %qkeep.address.local147 = and i64 %qkeep.address.local146, 16256, !dbg !11
  %qkeep.address.local148 = lshr i64 %qkeep.address.local140, 4, !dbg !11
  %qkeep.address.local149 = and i64 %qkeep.address.local148, 24, !dbg !11
  %qkeep.address.local150 = or disjoint i64 %qkeep.address.local147, %qkeep.address.local149, !dbg !11
  %qkeep.address.local151 = getelementptr [2 x i8], ptr addrspace(1) %qr.n30, i64 %qkeep.address.local150, !dbg !11
  %qkeep.qr.n67 = ptrtoint ptr addrspace(1) %qkeep.address.local151 to i64, !dbg !11
  %qkeep.qr.n66 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %qkeep.qr.n68 = sub i64 %qkeep.qr.n67, %qkeep.qr.n66, !dbg !11
  %qkeep.dma.buffer.offset931 = trunc i64 %qkeep.qr.n68 to i32, !dbg !11
  %qkeep.qr.n69 = trunc i64 %qkeep.qr.n66 to i32, !dbg !11
  %qkeep.qr.n70 = insertelement <4 x i32> poison, i32 %qkeep.qr.n69, i64 0, !dbg !11
  %qkeep.qr.n71 = lshr i64 %qkeep.qr.n66, 32, !dbg !11
  %qkeep.qr.n72 = trunc i64 %qkeep.qr.n71 to i32, !dbg !11
  %qkeep.qr.n73 = insertelement <4 x i32> %qkeep.qr.n70, i32 %qkeep.qr.n72, i64 1, !dbg !11
  %qkeep.qr.n74 = insertelement <4 x i32> %qkeep.qr.n73, i32 33554432, i64 2, !dbg !11
  %qkeep.dma.buffer.rsrc932 = insertelement <4 x i32> %qkeep.qr.n74, i32 131072, i64 3, !dbg !11
  %qkeep.qr.n76 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %qkeep.dma.buffer.lds933, i32 %qkeep.dma.buffer.offset931, <4 x i32> %qkeep.dma.buffer.rsrc932) #10, !dbg !11
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10
  fence syncscope("workgroup") release
  call void @llvm.amdgcn.s.barrier()
  fence syncscope("workgroup") acquire
  %qkeep.address.local160 = lshr i32 %qkeep.address.thread, 2, !dbg !12
  %qkeep.address.local161 = and i32 %qkeep.address.local160, 4, !dbg !12
  %qkeep.address.local162 = shl nuw nsw i32 %qkeep.address.thread, 1, !dbg !12
  %qkeep.address.local163 = and i32 %qkeep.address.local162, 384, !dbg !12
  %qkeep.address.local164 = or disjoint i32 %qkeep.address.local161, %qkeep.address.local163, !dbg !12
  %qkeep.address.local165 = shl nuw nsw i32 %qkeep.address.thread, 3, !dbg !12
  %qkeep.address.local166 = and i32 %qkeep.address.local165, 120, !dbg !12
  %qkeep.address.local167 = or disjoint i32 %qkeep.address.local164, %qkeep.address.local166, !dbg !12
  %qkeep.address.local168 = shl nuw nsw i32 %qkeep.address.thread, 5, !dbg !12
  %qkeep.address.local169 = and i32 %qkeep.address.local168, 1024, !dbg !12
  %qkeep.address.local170 = or disjoint i32 %qkeep.address.local167, %qkeep.address.local169, !dbg !12
  %qkeep.address.local171 = zext nneg i32 %qkeep.address.local170 to i64, !dbg !12
  %qkeep.address.local172 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %qkeep.address.local171, !dbg !12
  %qkeep.qr.n117 = load <4 x half>, ptr addrspace(3) %qkeep.address.local172, align 2, !dbg !12
  %qkeep.qr.n120 = fmul <4 x half> %qkeep.qr.n117, splat (half 0xH3015), !dbg !12
  %qkeep.address.local173 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %qkeep.address.local171, !dbg !12
  %qkeep.address.local174 = getelementptr i8, ptr addrspace(3) %qkeep.address.local173, i64 4096, !dbg !12
  %qkeep.qr.n118 = load <4 x half>, ptr addrspace(3) %qkeep.address.local174, align 2, !dbg !12
  %qkeep.qr.n122 = fmul <4 x half> %qkeep.qr.n118, splat (half 0xH3015), !dbg !12
  %qkeep.address.local175 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %qkeep.address.local171, !dbg !12
  %qkeep.address.local176 = getelementptr i8, ptr addrspace(3) %qkeep.address.local175, i64 1024, !dbg !12
  %qkeep.qr.n119 = load <4 x half>, ptr addrspace(3) %qkeep.address.local176, align 2, !dbg !12
  %qkeep.qr.n123 = fmul <4 x half> %qkeep.qr.n119, splat (half 0xH3015), !dbg !12
  %qkeep.address.local177 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %qkeep.address.local171, !dbg !12
  %qkeep.address.local178 = getelementptr i8, ptr addrspace(3) %qkeep.address.local177, i64 5120, !dbg !12
  %qkeep.qr.n121 = load <4 x half>, ptr addrspace(3) %qkeep.address.local178, align 2, !dbg !12
  %qkeep.qr.n124 = fmul <4 x half> %qkeep.qr.n121, splat (half 0xH3015), !dbg !12
  ; Four <4 x half> fragments remain live across BN iterations.
  br label %qr.n40, !dbg !15

qr.n40:                                               ; preds = %qr.n826, %4
  %qr.n41 = phi i64 [ %qr.n1559, %qr.n826 ], [ 0, %4 ], !dbg !15
  %qr.n42 = phi [2 x <8 x float>] [ %qr.n1546, %qr.n826 ], [ zeroinitializer, %4 ], !dbg !15
  %qr.n43 = phi [2 x <1 x float>] [ %qr.n1168, %qr.n826 ], [ zeroinitializer, %4 ], !dbg !15
  %qr.n44 = phi [2 x <8 x float>] [ %qr.n1550, %qr.n826 ], [ zeroinitializer, %4 ], !dbg !15
  %qr.n45 = phi [2 x <8 x float>] [ %qr.n1554, %qr.n826 ], [ zeroinitializer, %4 ], !dbg !15
  %qr.n46 = phi [2 x <8 x float>] [ %qr.n1558, %qr.n826 ], [ zeroinitializer, %4 ], !dbg !15
  %.not = icmp sgt i64 %qr.n41, %qr.n8, !dbg !15
  br i1 %.not, label %qr.n1560, label %qr.n47, !dbg !15

qr.n47:                                               ; preds = %qr.n40
  %qr.n48 = shl nsw i64 %qr.n41, 7, !dbg !15
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %qr.n24) #9, !dbg !11, !range !33, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !12
  %address.local129 = shl nuw nsw i64 %address.local, 3, !dbg !11
  %address.local130 = and i64 %address.local129, 1536, !dbg !11
  %address.local131 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local130, !dbg !11
  %address.local132 = ptrtoint ptr addrspace(3) %address.local131 to i64, !dbg !11
  %address.local133 = trunc i64 %address.local132 to i32, !dbg !11
  %qr.n49 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %qr.n50 = zext i32 %qr.n49 to i64, !dbg !11
  %qr.n51 = inttoptr i64 %qr.n50 to ptr addrspace(3), !dbg !11
  %address.local134 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local135 = and i64 %address.local134, 16256, !dbg !11
  %address.local136 = lshr i64 %address.local, 4, !dbg !11
  %address.local137 = and i64 %address.local136, 8, !dbg !11
  %address.local138 = or disjoint i64 %address.local135, %address.local137, !dbg !11
  %address.local139 = getelementptr [2 x i8], ptr addrspace(1) %qr.n30, i64 %address.local138, !dbg !11
  %qr.n52 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %qr.n53 = ptrtoint ptr addrspace(1) %address.local139 to i64, !dbg !11
  %qr.n54 = sub i64 %qr.n53, %qr.n52, !dbg !11
  %dma.buffer.offset934 = trunc i64 %qr.n54 to i32, !dbg !11
  %qr.n55 = trunc i64 %qr.n52 to i32, !dbg !11
  %qr.n56 = insertelement <4 x i32> poison, i32 %qr.n55, i64 0, !dbg !11
  %qr.n57 = lshr i64 %qr.n52, 32, !dbg !11
  %qr.n58 = trunc i64 %qr.n57 to i32, !dbg !11
  %qr.n59 = insertelement <4 x i32> %qr.n56, i32 %qr.n58, i64 1, !dbg !11
  %qr.n60 = insertelement <4 x i32> %qr.n59, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc935 = insertelement <4 x i32> %qr.n60, i32 131072, i64 3, !dbg !11
  %qr.n61 = ptrtoint ptr addrspace(3) %qr.n51 to i32, !dbg !11
  %dma.buffer.lds936 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n61), !dbg !11
  %address.local140 = or disjoint i64 %address.local, 256, !dbg !11
  %address.local141 = shl nuw nsw i64 %address.local140, 3, !dbg !11
  %address.local142 = and i64 %address.local141, 3584, !dbg !11
  %address.local143 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local142, !dbg !11
  %address.local144 = ptrtoint ptr addrspace(3) %address.local143 to i64, !dbg !11
  %address.local145 = trunc i64 %address.local144 to i32, !dbg !11
  %qr.n63 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !11
  %qr.n64 = zext i32 %qr.n63 to i64, !dbg !11
  %qr.n65 = inttoptr i64 %qr.n64 to ptr addrspace(3), !dbg !11
  %address.local146 = shl nuw nsw i64 %address.local, 7, !dbg !11
  %address.local147 = and i64 %address.local146, 16256, !dbg !11
  %address.local148 = lshr i64 %address.local140, 4, !dbg !11
  %address.local149 = and i64 %address.local148, 24, !dbg !11
  %address.local150 = or disjoint i64 %address.local147, %address.local149, !dbg !11
  %address.local151 = getelementptr [2 x i8], ptr addrspace(1) %qr.n30, i64 %address.local150, !dbg !11
  %qr.n66 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %qr.n67 = ptrtoint ptr addrspace(1) %address.local151 to i64, !dbg !11
  %qr.n68 = sub i64 %qr.n67, %qr.n66, !dbg !11
  %dma.buffer.offset931 = trunc i64 %qr.n68 to i32, !dbg !11
  %qr.n69 = trunc i64 %qr.n66 to i32, !dbg !11
  %qr.n70 = insertelement <4 x i32> poison, i32 %qr.n69, i64 0, !dbg !11
  %qr.n71 = lshr i64 %qr.n66, 32, !dbg !11
  %qr.n72 = trunc i64 %qr.n71 to i32, !dbg !11
  %qr.n73 = insertelement <4 x i32> %qr.n70, i32 %qr.n72, i64 1, !dbg !11
  %qr.n74 = insertelement <4 x i32> %qr.n73, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc932 = insertelement <4 x i32> %qr.n74, i32 131072, i64 3, !dbg !11
  %qr.n75 = ptrtoint ptr addrspace(3) %qr.n65 to i32, !dbg !11
  %dma.buffer.lds933 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n75), !dbg !11
  %qr.n77 = lshr i64 %qr.n41, 12, !dbg !16
  %qr.n78 = add nsw i64 %qr.n77, %qr.n10, !dbg !16
  %qr.n79 = lshr i64 %qr.n41, 5, !dbg !16
  %qr.n80 = and i64 %qr.n79, 127, !dbg !16
  %qr.n81 = shl i64 %qr.n78, 19, !dbg !16
  %qr.n82 = and i64 %qr.n81, 16252928, !dbg !16
  %qr.n83 = shl i64 %qr.n79, 19, !dbg !16
  %qr.n84 = shl i64 %qr.n41, 14, !dbg !16
  %qr.n85 = sub i64 %qr.n84, %qr.n83, !dbg !16
  %qr.n86 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %qr.n82, !dbg !16
  %qr.n87 = getelementptr [2 x i8], ptr addrspace(1) %qr.n86, i64 %qr.n80, !dbg !16
  %qr.n88 = getelementptr [2 x i8], ptr addrspace(1) %qr.n87, i64 %qr.n85, !dbg !16
  %address.local152 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local130, !dbg !16
  %address.local153 = ptrtoint ptr addrspace(3) %address.local152 to i64, !dbg !16
  %address.local154 = trunc i64 %address.local153 to i32, !dbg !16
  %qr.n89 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local154), !dbg !16
  %qr.n90 = zext i32 %qr.n89 to i64, !dbg !16
  %qr.n91 = inttoptr i64 %qr.n90 to ptr addrspace(3), !dbg !16
  %address.local155 = getelementptr [2 x i8], ptr addrspace(1) %qr.n88, i64 %address.local138, !dbg !16
  %qr.n92 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %qr.n93 = ptrtoint ptr addrspace(1) %address.local155 to i64, !dbg !16
  %qr.n94 = sub i64 %qr.n93, %qr.n92, !dbg !16
  %dma.buffer.offset928 = trunc i64 %qr.n94 to i32, !dbg !16
  %qr.n95 = trunc i64 %qr.n92 to i32, !dbg !16
  %qr.n96 = insertelement <4 x i32> poison, i32 %qr.n95, i64 0, !dbg !16
  %qr.n97 = lshr i64 %qr.n92, 32, !dbg !16
  %qr.n98 = trunc i64 %qr.n97 to i32, !dbg !16
  %qr.n99 = insertelement <4 x i32> %qr.n96, i32 %qr.n98, i64 1, !dbg !16
  %qr.n100 = insertelement <4 x i32> %qr.n99, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc929 = insertelement <4 x i32> %qr.n100, i32 131072, i64 3, !dbg !16
  %qr.n101 = ptrtoint ptr addrspace(3) %qr.n91 to i32, !dbg !16
  %dma.buffer.lds930 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n101), !dbg !16
  %ksw.dma.offset0 = xor i32 %dma.buffer.offset928, %ksw.dma.mask, !dbg !16
  %qr.n102 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds930, i32 %ksw.dma.offset0, <4 x i32> %dma.buffer.rsrc929) #10, !dbg !16
  %address.local156 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local142, !dbg !16
  %address.local157 = ptrtoint ptr addrspace(3) %address.local156 to i64, !dbg !16
  %address.local158 = trunc i64 %address.local157 to i32, !dbg !16
  %qr.n103 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !16
  %qr.n104 = zext i32 %qr.n103 to i64, !dbg !16
  %qr.n105 = inttoptr i64 %qr.n104 to ptr addrspace(3), !dbg !16
  %address.local159 = getelementptr [2 x i8], ptr addrspace(1) %qr.n88, i64 %address.local150, !dbg !16
  %qr.n106 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %qr.n107 = ptrtoint ptr addrspace(1) %address.local159 to i64, !dbg !16
  %qr.n108 = sub i64 %qr.n107, %qr.n106, !dbg !16
  %dma.buffer.offset925 = trunc i64 %qr.n108 to i32, !dbg !16
  %qr.n109 = trunc i64 %qr.n106 to i32, !dbg !16
  %qr.n110 = insertelement <4 x i32> poison, i32 %qr.n109, i64 0, !dbg !16
  %qr.n111 = lshr i64 %qr.n106, 32, !dbg !16
  %qr.n112 = trunc i64 %qr.n111 to i32, !dbg !16
  %qr.n113 = insertelement <4 x i32> %qr.n110, i32 %qr.n112, i64 1, !dbg !16
  %qr.n114 = insertelement <4 x i32> %qr.n113, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc926 = insertelement <4 x i32> %qr.n114, i32 131072, i64 3, !dbg !16
  %qr.n115 = ptrtoint ptr addrspace(3) %qr.n105 to i32, !dbg !16
  %dma.buffer.lds927 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n115), !dbg !16
  %ksw.dma.offset1 = xor i32 %dma.buffer.offset925, %ksw.dma.mask, !dbg !16
  %qr.n116 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds927, i32 %ksw.dma.offset1, <4 x i32> %dma.buffer.rsrc926) #10, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local160 = lshr i32 %address.thread, 2, !dbg !12
  %address.local161 = and i32 %address.local160, 4, !dbg !12
  %address.local162 = shl nuw nsw i32 %address.thread, 1, !dbg !12
  %address.local163 = and i32 %address.local162, 384, !dbg !12
  %address.local164 = or disjoint i32 %address.local161, %address.local163, !dbg !12
  %address.local165 = shl nuw nsw i32 %address.thread, 3, !dbg !12
  %address.local166 = and i32 %address.local165, 120, !dbg !12
  %address.local167 = or disjoint i32 %address.local164, %address.local166, !dbg !12
  %address.local168 = shl nuw nsw i32 %address.thread, 5, !dbg !12
  %address.local169 = and i32 %address.local168, 1024, !dbg !12
  %address.local170 = or disjoint i32 %address.local167, %address.local169, !dbg !12
  %address.local171 = zext nneg i32 %address.local170 to i64, !dbg !12
  %address.local172 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !12
  %address.local173 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !12
  %address.local174 = getelementptr i8, ptr addrspace(3) %address.local173, i64 4096, !dbg !12
  %address.local175 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !12
  %address.local176 = getelementptr i8, ptr addrspace(3) %address.local175, i64 1024, !dbg !12
  %address.local177 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local171, !dbg !12
  %address.local178 = getelementptr i8, ptr addrspace(3) %address.local177, i64 5120, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %address.local179 = lshr i32 %address.thread, 2, !dbg !17
  %address.local180 = and i32 %address.local179, 4, !dbg !17
  %address.local181 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local182 = and i32 %address.local181, 24, !dbg !17
  %address.local183 = or disjoint i32 %address.local180, %address.local182, !dbg !17
  %address.local184 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local185 = and i32 %address.local184, 1024, !dbg !17
  %address.local186 = or disjoint i32 %address.local183, %address.local185, !dbg !17
  %address.local187 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local188 = and i32 %address.local187, 384, !dbg !17
  %address.local189 = or disjoint i32 %address.local186, %address.local188, !dbg !17
  %address.local190 = zext nneg i32 %address.local189 to i64, !dbg !17
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local190, !dbg !17
  %ksw.read0.address = ptrtoint ptr addrspace(3) %address.local191 to i32, !dbg !17
  %ksw.read0.xor = xor i32 %ksw.read0.address, %ksw.read.mask, !dbg !17
  %ksw.read0.pointer = inttoptr i32 %ksw.read0.xor to ptr addrspace(3), !dbg !17
  %kgep.base1.bits = xor i32 %ksw.read0.xor, 64, !dbg !17
  %kgep.base1 = inttoptr i32 %kgep.base1.bits to ptr addrspace(3), !dbg !17
  %qr.n125 = load <4 x half>, ptr addrspace(3) %ksw.read0.pointer, align 8, !dbg !17
  %qr.n126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n120, <4 x half> %qr.n125, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local192 = lshr i32 %address.thread, 2, !dbg !17
  %address.local193 = and i32 %address.local192, 4, !dbg !17
  %address.local194 = shl nuw nsw i32 %address.thread, 1, !dbg !17
  %address.local195 = and i32 %address.local194, 24, !dbg !17
  %address.local196 = or disjoint i32 %address.local193, %address.local195, !dbg !17
  %address.local197 = shl nuw nsw i32 %address.thread, 5, !dbg !17
  %address.local198 = and i32 %address.local197, 1024, !dbg !17
  %address.local199 = or disjoint i32 %address.local196, %address.local198, !dbg !17
  %address.local200 = shl nuw nsw i32 %address.thread, 7, !dbg !17
  %address.local201 = and i32 %address.local200, 384, !dbg !17
  %address.local202 = or disjoint i32 %address.local199, %address.local201, !dbg !17
  %address.local203 = zext nneg i32 %address.local202 to i64, !dbg !17
  %address.local204 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203, !dbg !17
  %address.local205 = getelementptr i8, ptr addrspace(3) %address.local204, i64 4096, !dbg !17
  %ksw.read1.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 4096, !dbg !17
  %qr.n127 = load <4 x half>, ptr addrspace(3) %ksw.read1.pointer, align 8, !dbg !17
  %address.local206 = shl nuw nsw i64 %address.local, 7, !dbg !17
  %address.local207 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %address.local208 = lshr i64 %address.local, 2, !dbg !17
  %address.local209 = mul nsw i64 %address.local208, -512, !dbg !17
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) %address.local207, i64 %address.local209, !dbg !17
  %address.local211 = shl nuw nsw i64 %address.local, 1, !dbg !17
  %address.local212 = and i64 %address.local211, 24, !dbg !17
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) %address.local210, i64 %address.local212, !dbg !17
  %address.local214 = lshr i64 %address.local, 2, !dbg !12
  %address.local215 = and i64 %address.local214, 12, !dbg !12
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) %address.local213, i64 %address.local215, !dbg !17
  %address.local217 = and i64 %address.local, 63, !dbg !12
  %address.local218 = lshr i64 %address.local217, 5, !dbg !12
  %address.local219 = mul nuw nsw i64 %address.local218, 1016, !dbg !12
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local216, i64 %address.local219, !dbg !17
  %address.local221 = getelementptr i8, ptr addrspace(3) %address.local220, i64 64, !dbg !17
  %ksw.read2.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 0, !dbg !17
  %qr.n128 = load <4 x half>, ptr addrspace(3) %ksw.read2.pointer, align 8, !dbg !17
  %qr.n129 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n122, <4 x half> %qr.n127, <4 x float> %qr.n126) #9, !dbg !17
  %qr.n130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n120, <4 x half> %qr.n128, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local222 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local222, i64 %address.local215, !dbg !17
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i64 %address.local219, !dbg !17
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) %address.local224, i64 %address.local209, !dbg !17
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i64 %address.local212, !dbg !17
  %address.local227 = getelementptr i8, ptr addrspace(3) %address.local226, i64 4160, !dbg !17
  %ksw.read3.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 4096, !dbg !17
  %qr.n131 = load <4 x half>, ptr addrspace(3) %ksw.read3.pointer, align 8, !dbg !17
  %address.local228 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %address.local229 = getelementptr [2 x i8], ptr addrspace(3) %address.local228, i64 %address.local209, !dbg !17
  %address.local230 = getelementptr [2 x i8], ptr addrspace(3) %address.local229, i64 %address.local212, !dbg !17
  %address.local231 = getelementptr [2 x i8], ptr addrspace(3) %address.local230, i64 %address.local215, !dbg !17
  %address.local232 = getelementptr [2 x i8], ptr addrspace(3) %address.local231, i64 %address.local219, !dbg !17
  %address.local233 = getelementptr i8, ptr addrspace(3) %address.local232, i64 128, !dbg !17
  %ksw.read4.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 128, !dbg !17
  %qr.n132 = load <4 x half>, ptr addrspace(3) %ksw.read4.pointer, align 8, !dbg !17
  %qr.n133 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n122, <4 x half> %qr.n131, <4 x float> %qr.n130) #9, !dbg !17
  %qr.n134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n120, <4 x half> %qr.n132, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local234 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %address.local235 = getelementptr [2 x i8], ptr addrspace(3) %address.local234, i64 %address.local215, !dbg !17
  %address.local236 = getelementptr [2 x i8], ptr addrspace(3) %address.local235, i64 %address.local219, !dbg !17
  %address.local237 = getelementptr [2 x i8], ptr addrspace(3) %address.local236, i64 %address.local209, !dbg !17
  %address.local238 = getelementptr [2 x i8], ptr addrspace(3) %address.local237, i64 %address.local212, !dbg !17
  %address.local239 = getelementptr i8, ptr addrspace(3) %address.local238, i64 4224, !dbg !17
  %ksw.read5.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 4224, !dbg !17
  %qr.n135 = load <4 x half>, ptr addrspace(3) %ksw.read5.pointer, align 8, !dbg !17
  %address.local240 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %address.local241 = getelementptr [2 x i8], ptr addrspace(3) %address.local240, i64 %address.local209, !dbg !17
  %address.local242 = getelementptr [2 x i8], ptr addrspace(3) %address.local241, i64 %address.local212, !dbg !17
  %address.local243 = getelementptr [2 x i8], ptr addrspace(3) %address.local242, i64 %address.local215, !dbg !17
  %address.local244 = getelementptr [2 x i8], ptr addrspace(3) %address.local243, i64 %address.local219, !dbg !17
  %address.local245 = getelementptr i8, ptr addrspace(3) %address.local244, i64 192, !dbg !17
  %ksw.read6.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 128, !dbg !17
  %qr.n136 = load <4 x half>, ptr addrspace(3) %ksw.read6.pointer, align 8, !dbg !17
  %qr.n137 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n122, <4 x half> %qr.n135, <4 x float> %qr.n134) #9, !dbg !17
  %qr.n138 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n120, <4 x half> %qr.n136, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local246 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local206, !dbg !17
  %address.local247 = getelementptr [2 x i8], ptr addrspace(3) %address.local246, i64 %address.local215, !dbg !17
  %address.local248 = getelementptr [2 x i8], ptr addrspace(3) %address.local247, i64 %address.local219, !dbg !17
  %address.local249 = getelementptr [2 x i8], ptr addrspace(3) %address.local248, i64 %address.local209, !dbg !17
  %address.local250 = getelementptr [2 x i8], ptr addrspace(3) %address.local249, i64 %address.local212, !dbg !17
  %address.local251 = getelementptr i8, ptr addrspace(3) %address.local250, i64 4288, !dbg !17
  %ksw.read7.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 4224, !dbg !17
  %qr.n139 = load <4 x half>, ptr addrspace(3) %ksw.read7.pointer, align 8, !dbg !17
  %address.local252 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local190, !dbg !17
  %address.local253 = getelementptr i8, ptr addrspace(3) %address.local252, i64 1024, !dbg !17
  %ksw.read8.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 1024, !dbg !17
  %qr.n140 = load <4 x half>, ptr addrspace(3) %ksw.read8.pointer, align 8, !dbg !17
  %qr.n141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n122, <4 x half> %qr.n139, <4 x float> %qr.n138) #9, !dbg !17
  %qr.n142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n120, <4 x half> %qr.n140, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local254 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local203, !dbg !17
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local254, i64 5120, !dbg !17
  %ksw.read9.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 5120, !dbg !17
  %qr.n143 = load <4 x half>, ptr addrspace(3) %ksw.read9.pointer, align 8, !dbg !17
  %address.local256 = add nuw nsw i64 %address.local206, 512, !dbg !17
  %address.local257 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local256, !dbg !17
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) %address.local257, i64 %address.local209, !dbg !17
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) %address.local258, i64 %address.local212, !dbg !17
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) %address.local259, i64 %address.local215, !dbg !17
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i64 %address.local219, !dbg !17
  %address.local262 = getelementptr i8, ptr addrspace(3) %address.local261, i64 64, !dbg !17
  %ksw.read10.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 1024, !dbg !17
  %qr.n144 = load <4 x half>, ptr addrspace(3) %ksw.read10.pointer, align 8, !dbg !17
  %qr.n145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n122, <4 x half> %qr.n143, <4 x float> %qr.n142) #9, !dbg !17
  %qr.n146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n120, <4 x half> %qr.n144, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local263 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local256, !dbg !17
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local263, i64 %address.local215, !dbg !17
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i64 %address.local219, !dbg !17
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) %address.local265, i64 %address.local209, !dbg !17
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %address.local266, i64 %address.local212, !dbg !17
  %address.local268 = getelementptr i8, ptr addrspace(3) %address.local267, i64 4160, !dbg !17
  %ksw.read11.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 5120, !dbg !17
  %qr.n147 = load <4 x half>, ptr addrspace(3) %ksw.read11.pointer, align 8, !dbg !17
  %address.local269 = add nuw nsw i64 %address.local206, 512, !dbg !17
  %address.local270 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local269, !dbg !17
  %address.local271 = getelementptr [2 x i8], ptr addrspace(3) %address.local270, i64 %address.local209, !dbg !17
  %address.local272 = getelementptr [2 x i8], ptr addrspace(3) %address.local271, i64 %address.local212, !dbg !17
  %address.local273 = getelementptr [2 x i8], ptr addrspace(3) %address.local272, i64 %address.local215, !dbg !17
  %address.local274 = getelementptr [2 x i8], ptr addrspace(3) %address.local273, i64 %address.local219, !dbg !17
  %address.local275 = getelementptr i8, ptr addrspace(3) %address.local274, i64 128, !dbg !17
  %ksw.read12.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 1152, !dbg !17
  %qr.n148 = load <4 x half>, ptr addrspace(3) %ksw.read12.pointer, align 8, !dbg !17
  %qr.n149 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n122, <4 x half> %qr.n147, <4 x float> %qr.n146) #9, !dbg !17
  %qr.n150 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n120, <4 x half> %qr.n148, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local276 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local269, !dbg !17
  %address.local277 = getelementptr [2 x i8], ptr addrspace(3) %address.local276, i64 %address.local215, !dbg !17
  %address.local278 = getelementptr [2 x i8], ptr addrspace(3) %address.local277, i64 %address.local219, !dbg !17
  %address.local279 = getelementptr [2 x i8], ptr addrspace(3) %address.local278, i64 %address.local209, !dbg !17
  %address.local280 = getelementptr [2 x i8], ptr addrspace(3) %address.local279, i64 %address.local212, !dbg !17
  %address.local281 = getelementptr i8, ptr addrspace(3) %address.local280, i64 4224, !dbg !17
  %ksw.read13.pointer = getelementptr i8, ptr addrspace(3) %ksw.read0.pointer, i32 5248, !dbg !17
  %qr.n151 = load <4 x half>, ptr addrspace(3) %ksw.read13.pointer, align 8, !dbg !17
  %address.local282 = add nuw nsw i64 %address.local206, 512, !dbg !17
  %address.local283 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local282, !dbg !17
  %address.local284 = getelementptr [2 x i8], ptr addrspace(3) %address.local283, i64 %address.local209, !dbg !17
  %address.local285 = getelementptr [2 x i8], ptr addrspace(3) %address.local284, i64 %address.local212, !dbg !17
  %address.local286 = getelementptr [2 x i8], ptr addrspace(3) %address.local285, i64 %address.local215, !dbg !17
  %address.local287 = getelementptr [2 x i8], ptr addrspace(3) %address.local286, i64 %address.local219, !dbg !17
  %address.local288 = getelementptr i8, ptr addrspace(3) %address.local287, i64 192, !dbg !17
  %ksw.read14.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 1152, !dbg !17
  %qr.n152 = load <4 x half>, ptr addrspace(3) %ksw.read14.pointer, align 8, !dbg !17
  %qr.n153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n122, <4 x half> %qr.n151, <4 x float> %qr.n150) #9, !dbg !17
  %qr.n154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n120, <4 x half> %qr.n152, <4 x float> zeroinitializer) #9, !dbg !17
  %address.local289 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local282, !dbg !17
  %address.local290 = getelementptr [2 x i8], ptr addrspace(3) %address.local289, i64 %address.local215, !dbg !17
  %address.local291 = getelementptr [2 x i8], ptr addrspace(3) %address.local290, i64 %address.local219, !dbg !17
  %address.local292 = getelementptr [2 x i8], ptr addrspace(3) %address.local291, i64 %address.local209, !dbg !17
  %address.local293 = getelementptr [2 x i8], ptr addrspace(3) %address.local292, i64 %address.local212, !dbg !17
  %address.local294 = getelementptr i8, ptr addrspace(3) %address.local293, i64 4288, !dbg !17
  %ksw.read15.pointer = getelementptr i8, ptr addrspace(3) %kgep.base1, i32 5248, !dbg !17
  %qr.n155 = load <4 x half>, ptr addrspace(3) %ksw.read15.pointer, align 8, !dbg !17
  %qr.n156 = load <4 x half>, ptr addrspace(3) %ksw.read0.pointer, align 8, !dbg !17
  %qr.n157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n122, <4 x half> %qr.n155, <4 x float> %qr.n154) #9, !dbg !17
  %qr.n158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n123, <4 x half> %qr.n156, <4 x float> zeroinitializer) #9, !dbg !17
  %qr.n159 = load <4 x half>, ptr addrspace(3) %ksw.read1.pointer, align 8, !dbg !17
  %qr.n160 = load <4 x half>, ptr addrspace(3) %ksw.read2.pointer, align 8, !dbg !17
  %qr.n161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n124, <4 x half> %qr.n159, <4 x float> %qr.n158) #9, !dbg !17
  %qr.n162 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n123, <4 x half> %qr.n160, <4 x float> zeroinitializer) #9, !dbg !17
  %qr.n163 = load <4 x half>, ptr addrspace(3) %ksw.read3.pointer, align 8, !dbg !17
  %qr.n164 = load <4 x half>, ptr addrspace(3) %ksw.read4.pointer, align 8, !dbg !17
  %qr.n165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n124, <4 x half> %qr.n163, <4 x float> %qr.n162) #9, !dbg !17
  %qr.n166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n123, <4 x half> %qr.n164, <4 x float> zeroinitializer) #9, !dbg !17
  %qr.n167 = load <4 x half>, ptr addrspace(3) %ksw.read5.pointer, align 8, !dbg !17
  %qr.n168 = load <4 x half>, ptr addrspace(3) %ksw.read6.pointer, align 8, !dbg !17
  %qr.n169 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n124, <4 x half> %qr.n167, <4 x float> %qr.n166) #9, !dbg !17
  %qr.n170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n123, <4 x half> %qr.n168, <4 x float> zeroinitializer) #9, !dbg !17
  %qr.n171 = load <4 x half>, ptr addrspace(3) %ksw.read7.pointer, align 8, !dbg !17
  %qr.n172 = load <4 x half>, ptr addrspace(3) %ksw.read8.pointer, align 8, !dbg !17
  %qr.n173 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n124, <4 x half> %qr.n171, <4 x float> %qr.n170) #9, !dbg !17
  %qr.n174 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n123, <4 x half> %qr.n172, <4 x float> zeroinitializer) #9, !dbg !17
  %qr.n175 = load <4 x half>, ptr addrspace(3) %ksw.read9.pointer, align 8, !dbg !17
  %qr.n176 = load <4 x half>, ptr addrspace(3) %ksw.read10.pointer, align 8, !dbg !17
  %qr.n177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n124, <4 x half> %qr.n175, <4 x float> %qr.n174) #9, !dbg !17
  %qr.n178 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n123, <4 x half> %qr.n176, <4 x float> zeroinitializer) #9, !dbg !17
  %qr.n179 = load <4 x half>, ptr addrspace(3) %ksw.read11.pointer, align 8, !dbg !17
  %qr.n180 = load <4 x half>, ptr addrspace(3) %ksw.read12.pointer, align 8, !dbg !17
  %qr.n181 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n124, <4 x half> %qr.n179, <4 x float> %qr.n178) #9, !dbg !17
  %qr.n182 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n123, <4 x half> %qr.n180, <4 x float> zeroinitializer) #9, !dbg !17
  %qr.n183 = load <4 x half>, ptr addrspace(3) %ksw.read13.pointer, align 8, !dbg !17
  %qr.n184 = load <4 x half>, ptr addrspace(3) %ksw.read14.pointer, align 8, !dbg !17
  %qr.n185 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n124, <4 x half> %qr.n183, <4 x float> %qr.n182) #9, !dbg !17
  %qr.n186 = load <4 x half>, ptr addrspace(3) %ksw.read15.pointer, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %qr.n187 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n123, <4 x half> %qr.n184, <4 x float> zeroinitializer) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %qr.n188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qkeep.qr.n124, <4 x half> %qr.n186, <4 x float> %qr.n187) #9, !dbg !17
  %qr.n189 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %qr.n190 = zext i32 %qr.n189 to i64, !dbg !11
  %qr.n191 = inttoptr i64 %qr.n190 to ptr addrspace(3), !dbg !11
  %address.local295 = getelementptr [2 x i8], ptr addrspace(1) %qr.n32, i64 %address.local138, !dbg !11
  %qr.n192 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %qr.n193 = ptrtoint ptr addrspace(1) %address.local295 to i64, !dbg !11
  %qr.n194 = sub i64 %qr.n193, %qr.n192, !dbg !11
  %dma.buffer.offset922 = trunc i64 %qr.n194 to i32, !dbg !11
  %qr.n195 = trunc i64 %qr.n192 to i32, !dbg !11
  %qr.n196 = insertelement <4 x i32> poison, i32 %qr.n195, i64 0, !dbg !11
  %qr.n197 = lshr i64 %qr.n192, 32, !dbg !11
  %qr.n198 = trunc i64 %qr.n197 to i32, !dbg !11
  %qr.n199 = insertelement <4 x i32> %qr.n196, i32 %qr.n198, i64 1, !dbg !11
  %qr.n200 = insertelement <4 x i32> %qr.n199, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc923 = insertelement <4 x i32> %qr.n200, i32 131072, i64 3, !dbg !11
  %qr.n201 = ptrtoint ptr addrspace(3) %qr.n191 to i32, !dbg !11
  %dma.buffer.lds924 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n201), !dbg !11
  %qr.n202 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds924, i32 %dma.buffer.offset922, <4 x i32> %dma.buffer.rsrc923) #10, !dbg !11
  %qr.n203 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !11
  %qr.n204 = zext i32 %qr.n203 to i64, !dbg !11
  %qr.n205 = inttoptr i64 %qr.n204 to ptr addrspace(3), !dbg !11
  %address.local296 = getelementptr [2 x i8], ptr addrspace(1) %qr.n32, i64 %address.local150, !dbg !11
  %qr.n206 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %qr.n207 = ptrtoint ptr addrspace(1) %address.local296 to i64, !dbg !11
  %qr.n208 = sub i64 %qr.n207, %qr.n206, !dbg !11
  %dma.buffer.offset919 = trunc i64 %qr.n208 to i32, !dbg !11
  %qr.n209 = trunc i64 %qr.n206 to i32, !dbg !11
  %qr.n210 = insertelement <4 x i32> poison, i32 %qr.n209, i64 0, !dbg !11
  %qr.n211 = lshr i64 %qr.n206, 32, !dbg !11
  %qr.n212 = trunc i64 %qr.n211 to i32, !dbg !11
  %qr.n213 = insertelement <4 x i32> %qr.n210, i32 %qr.n212, i64 1, !dbg !11
  %qr.n214 = insertelement <4 x i32> %qr.n213, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc920 = insertelement <4 x i32> %qr.n214, i32 131072, i64 3, !dbg !11
  %qr.n215 = ptrtoint ptr addrspace(3) %qr.n205 to i32, !dbg !11
  %dma.buffer.lds921 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n215), !dbg !11
  %qr.n216 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds921, i32 %dma.buffer.offset919, <4 x i32> %dma.buffer.rsrc920) #10, !dbg !11
  %qr.n217 = add nuw i64 %qr.n41, 1024, !dbg !16
  %qr.n218 = icmp slt i64 %qr.n217, 0, !dbg !16
  %qr.n219 = sub nuw i64 -1025, %qr.n41, !dbg !16
  %qr.n220 = select i1 %qr.n218, i64 %qr.n219, i64 %qr.n217, !dbg !16
  %qr.n221 = sdiv i64 %qr.n220, 4096, !dbg !16
  %.lobit4 = ashr i64 %qr.n217, 63, !dbg !16
  %qr.n222 = xor i64 %qr.n221, %.lobit4, !dbg !16
  %qr.n223 = add nsw i64 %qr.n222, %qr.n10, !dbg !16
  %qr.n224 = add nsw i64 %qr.n79, %qr.n35, !dbg !16
  %qr.n225 = add nsw i64 %qr.n224, 32, !dbg !16
  %qr.n226 = icmp slt i64 %qr.n224, -32, !dbg !16
  %qr.n227 = sub nsw i64 -33, %qr.n224, !dbg !16
  %qr.n228 = select i1 %qr.n226, i64 %qr.n227, i64 %qr.n225, !dbg !16
  %qr.n229 = sdiv i64 %qr.n228, 128, !dbg !16
  %qr.n230 = sext i1 %qr.n226 to i64, !dbg !16
  %qr.n231 = xor i64 %qr.n229, %qr.n230, !dbg !16
  %qr.n232 = shl nsw i64 %qr.n231, 7, !dbg !16
  %qr.n233 = sub i64 %qr.n35, %qr.n232, !dbg !16
  %qr.n234 = shl i64 %qr.n223, 19, !dbg !16
  %qr.n235 = and i64 %qr.n234, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %qr.n236 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %qr.n235, !dbg !16
  %qr.n237 = getelementptr [2 x i8], ptr addrspace(1) %qr.n236, i64 %qr.n79, !dbg !16
  %qr.n238 = getelementptr [2 x i8], ptr addrspace(1) %qr.n237, i64 %qr.n233, !dbg !16
  %qr.n239 = getelementptr i8, ptr addrspace(1) %qr.n238, i64 64, !dbg !16
  %qr.n240 = getelementptr [2 x i8], ptr addrspace(1) %qr.n239, i64 %qr.n85, !dbg !16
  %qr.n241 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local154), !dbg !16
  %qr.n242 = zext i32 %qr.n241 to i64, !dbg !16
  %qr.n243 = inttoptr i64 %qr.n242 to ptr addrspace(3), !dbg !16
  %address.local297 = getelementptr [2 x i8], ptr addrspace(1) %qr.n240, i64 %address.local138, !dbg !16
  %qr.n244 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %qr.n245 = ptrtoint ptr addrspace(1) %address.local297 to i64, !dbg !16
  %qr.n246 = sub i64 %qr.n245, %qr.n244, !dbg !16
  %dma.buffer.offset916 = trunc i64 %qr.n246 to i32, !dbg !16
  %qr.n247 = trunc i64 %qr.n244 to i32, !dbg !16
  %qr.n248 = insertelement <4 x i32> poison, i32 %qr.n247, i64 0, !dbg !16
  %qr.n249 = lshr i64 %qr.n244, 32, !dbg !16
  %qr.n250 = trunc i64 %qr.n249 to i32, !dbg !16
  %qr.n251 = insertelement <4 x i32> %qr.n248, i32 %qr.n250, i64 1, !dbg !16
  %qr.n252 = insertelement <4 x i32> %qr.n251, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc917 = insertelement <4 x i32> %qr.n252, i32 131072, i64 3, !dbg !16
  %qr.n253 = ptrtoint ptr addrspace(3) %qr.n243 to i32, !dbg !16
  %dma.buffer.lds918 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n253), !dbg !16
  %ksw.dma.offset2 = xor i32 %dma.buffer.offset916, %ksw.dma.mask, !dbg !16
  %qr.n254 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds918, i32 %ksw.dma.offset2, <4 x i32> %dma.buffer.rsrc917) #10, !dbg !16
  %qr.n255 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !16
  %qr.n256 = zext i32 %qr.n255 to i64, !dbg !16
  %qr.n257 = inttoptr i64 %qr.n256 to ptr addrspace(3), !dbg !16
  %address.local298 = getelementptr [2 x i8], ptr addrspace(1) %qr.n240, i64 %address.local150, !dbg !16
  %qr.n258 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %qr.n259 = ptrtoint ptr addrspace(1) %address.local298 to i64, !dbg !16
  %qr.n260 = sub i64 %qr.n259, %qr.n258, !dbg !16
  %dma.buffer.offset913 = trunc i64 %qr.n260 to i32, !dbg !16
  %qr.n261 = trunc i64 %qr.n258 to i32, !dbg !16
  %qr.n262 = insertelement <4 x i32> poison, i32 %qr.n261, i64 0, !dbg !16
  %qr.n263 = lshr i64 %qr.n258, 32, !dbg !16
  %qr.n264 = trunc i64 %qr.n263 to i32, !dbg !16
  %qr.n265 = insertelement <4 x i32> %qr.n262, i32 %qr.n264, i64 1, !dbg !16
  %qr.n266 = insertelement <4 x i32> %qr.n265, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc914 = insertelement <4 x i32> %qr.n266, i32 131072, i64 3, !dbg !16
  %qr.n267 = ptrtoint ptr addrspace(3) %qr.n257 to i32, !dbg !16
  %dma.buffer.lds915 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n267), !dbg !16
  %ksw.dma.offset3 = xor i32 %dma.buffer.offset913, %ksw.dma.mask, !dbg !16
  %qr.n268 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds915, i32 %ksw.dma.offset3, <4 x i32> %dma.buffer.rsrc914) #10, !dbg !16
  %qr.n269 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !12
  %qr.n270 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !12
  %qr.n271 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !12
  %qr.n272 = fmul <4 x half> %qr.n269, splat (half 0xH3015), !dbg !12
  %qr.n273 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !12
  %qr.n274 = fmul <4 x half> %qr.n270, splat (half 0xH3015), !dbg !12
  %qr.n275 = fmul <4 x half> %qr.n271, splat (half 0xH3015), !dbg !12
  %qr.n276 = fmul <4 x half> %qr.n273, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %qr.n277 = load <4 x half>, ptr addrspace(3) %ksw.read0.pointer, align 8, !dbg !17
  %qr.n278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n272, <4 x half> %qr.n277, <4 x float> %qr.n129) #9, !dbg !17
  %qr.n279 = load <4 x half>, ptr addrspace(3) %ksw.read1.pointer, align 8, !dbg !17
  %qr.n280 = load <4 x half>, ptr addrspace(3) %ksw.read2.pointer, align 8, !dbg !17
  %qr.n281 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n274, <4 x half> %qr.n279, <4 x float> %qr.n278) #9, !dbg !17
  %qr.n282 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n272, <4 x half> %qr.n280, <4 x float> %qr.n133) #9, !dbg !17
  %qr.n283 = load <4 x half>, ptr addrspace(3) %ksw.read3.pointer, align 8, !dbg !17
  %qr.n284 = load <4 x half>, ptr addrspace(3) %ksw.read4.pointer, align 8, !dbg !17
  %qr.n285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n274, <4 x half> %qr.n283, <4 x float> %qr.n282) #9, !dbg !17
  %qr.n286 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n272, <4 x half> %qr.n284, <4 x float> %qr.n137) #9, !dbg !17
  %qr.n287 = load <4 x half>, ptr addrspace(3) %ksw.read5.pointer, align 8, !dbg !17
  %qr.n288 = load <4 x half>, ptr addrspace(3) %ksw.read6.pointer, align 8, !dbg !17
  %qr.n289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n274, <4 x half> %qr.n287, <4 x float> %qr.n286) #9, !dbg !17
  %qr.n290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n272, <4 x half> %qr.n288, <4 x float> %qr.n141) #9, !dbg !17
  %qr.n291 = load <4 x half>, ptr addrspace(3) %ksw.read7.pointer, align 8, !dbg !17
  %qr.n292 = load <4 x half>, ptr addrspace(3) %ksw.read8.pointer, align 8, !dbg !17
  %qr.n293 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n274, <4 x half> %qr.n291, <4 x float> %qr.n290) #9, !dbg !17
  %qr.n294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n272, <4 x half> %qr.n292, <4 x float> %qr.n145) #9, !dbg !17
  %qr.n295 = load <4 x half>, ptr addrspace(3) %ksw.read9.pointer, align 8, !dbg !17
  %qr.n296 = load <4 x half>, ptr addrspace(3) %ksw.read10.pointer, align 8, !dbg !17
  %qr.n297 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n274, <4 x half> %qr.n295, <4 x float> %qr.n294) #9, !dbg !17
  %qr.n298 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n272, <4 x half> %qr.n296, <4 x float> %qr.n149) #9, !dbg !17
  %qr.n299 = load <4 x half>, ptr addrspace(3) %ksw.read11.pointer, align 8, !dbg !17
  %qr.n300 = load <4 x half>, ptr addrspace(3) %ksw.read12.pointer, align 8, !dbg !17
  %qr.n301 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n274, <4 x half> %qr.n299, <4 x float> %qr.n298) #9, !dbg !17
  %qr.n302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n272, <4 x half> %qr.n300, <4 x float> %qr.n153) #9, !dbg !17
  %qr.n303 = load <4 x half>, ptr addrspace(3) %ksw.read13.pointer, align 8, !dbg !17
  %qr.n304 = load <4 x half>, ptr addrspace(3) %ksw.read14.pointer, align 8, !dbg !17
  %qr.n305 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n274, <4 x half> %qr.n303, <4 x float> %qr.n302) #9, !dbg !17
  %qr.n306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n272, <4 x half> %qr.n304, <4 x float> %qr.n157) #9, !dbg !17
  %qr.n307 = load <4 x half>, ptr addrspace(3) %ksw.read15.pointer, align 8, !dbg !17
  %qr.n308 = load <4 x half>, ptr addrspace(3) %ksw.read0.pointer, align 8, !dbg !17
  %qr.n309 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n274, <4 x half> %qr.n307, <4 x float> %qr.n306) #9, !dbg !17
  %qr.n310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n275, <4 x half> %qr.n308, <4 x float> %qr.n161) #9, !dbg !17
  %qr.n311 = load <4 x half>, ptr addrspace(3) %ksw.read1.pointer, align 8, !dbg !17
  %qr.n312 = load <4 x half>, ptr addrspace(3) %ksw.read2.pointer, align 8, !dbg !17
  %qr.n313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n276, <4 x half> %qr.n311, <4 x float> %qr.n310) #9, !dbg !17
  %qr.n314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n275, <4 x half> %qr.n312, <4 x float> %qr.n165) #9, !dbg !17
  %qr.n315 = load <4 x half>, ptr addrspace(3) %ksw.read3.pointer, align 8, !dbg !17
  %qr.n316 = load <4 x half>, ptr addrspace(3) %ksw.read4.pointer, align 8, !dbg !17
  %qr.n317 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n276, <4 x half> %qr.n315, <4 x float> %qr.n314) #9, !dbg !17
  %qr.n318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n275, <4 x half> %qr.n316, <4 x float> %qr.n169) #9, !dbg !17
  %qr.n319 = load <4 x half>, ptr addrspace(3) %ksw.read5.pointer, align 8, !dbg !17
  %qr.n320 = load <4 x half>, ptr addrspace(3) %ksw.read6.pointer, align 8, !dbg !17
  %qr.n321 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n276, <4 x half> %qr.n319, <4 x float> %qr.n318) #9, !dbg !17
  %qr.n322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n275, <4 x half> %qr.n320, <4 x float> %qr.n173) #9, !dbg !17
  %qr.n323 = load <4 x half>, ptr addrspace(3) %ksw.read7.pointer, align 8, !dbg !17
  %qr.n324 = load <4 x half>, ptr addrspace(3) %ksw.read8.pointer, align 8, !dbg !17
  %qr.n325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n276, <4 x half> %qr.n323, <4 x float> %qr.n322) #9, !dbg !17
  %qr.n326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n275, <4 x half> %qr.n324, <4 x float> %qr.n177) #9, !dbg !17
  %qr.n327 = load <4 x half>, ptr addrspace(3) %ksw.read9.pointer, align 8, !dbg !17
  %qr.n328 = load <4 x half>, ptr addrspace(3) %ksw.read10.pointer, align 8, !dbg !17
  %qr.n329 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n276, <4 x half> %qr.n327, <4 x float> %qr.n326) #9, !dbg !17
  %qr.n330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n275, <4 x half> %qr.n328, <4 x float> %qr.n181) #9, !dbg !17
  %qr.n331 = load <4 x half>, ptr addrspace(3) %ksw.read11.pointer, align 8, !dbg !17
  %qr.n332 = load <4 x half>, ptr addrspace(3) %ksw.read12.pointer, align 8, !dbg !17
  %qr.n333 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n276, <4 x half> %qr.n331, <4 x float> %qr.n330) #9, !dbg !17
  %qr.n334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n275, <4 x half> %qr.n332, <4 x float> %qr.n185) #9, !dbg !17
  %qr.n335 = load <4 x half>, ptr addrspace(3) %ksw.read13.pointer, align 8, !dbg !17
  %qr.n336 = load <4 x half>, ptr addrspace(3) %ksw.read14.pointer, align 8, !dbg !17
  %qr.n337 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n276, <4 x half> %qr.n335, <4 x float> %qr.n334) #9, !dbg !17
  %qr.n338 = load <4 x half>, ptr addrspace(3) %ksw.read15.pointer, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %qr.n339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n275, <4 x half> %qr.n336, <4 x float> %qr.n188) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %qr.n340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n276, <4 x half> %qr.n338, <4 x float> %qr.n339) #9, !dbg !17
  %qr.n341 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %qr.n342 = zext i32 %qr.n341 to i64, !dbg !11
  %qr.n343 = inttoptr i64 %qr.n342 to ptr addrspace(3), !dbg !11
  %address.local299 = getelementptr [2 x i8], ptr addrspace(1) %qr.n37, i64 %address.local138, !dbg !11
  %qr.n344 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %qr.n345 = ptrtoint ptr addrspace(1) %address.local299 to i64, !dbg !11
  %qr.n346 = sub i64 %qr.n345, %qr.n344, !dbg !11
  %dma.buffer.offset910 = trunc i64 %qr.n346 to i32, !dbg !11
  %qr.n347 = trunc i64 %qr.n344 to i32, !dbg !11
  %qr.n348 = insertelement <4 x i32> poison, i32 %qr.n347, i64 0, !dbg !11
  %qr.n349 = lshr i64 %qr.n344, 32, !dbg !11
  %qr.n350 = trunc i64 %qr.n349 to i32, !dbg !11
  %qr.n351 = insertelement <4 x i32> %qr.n348, i32 %qr.n350, i64 1, !dbg !11
  %qr.n352 = insertelement <4 x i32> %qr.n351, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc911 = insertelement <4 x i32> %qr.n352, i32 131072, i64 3, !dbg !11
  %qr.n353 = ptrtoint ptr addrspace(3) %qr.n343 to i32, !dbg !11
  %dma.buffer.lds912 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n353), !dbg !11
  %qr.n354 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds912, i32 %dma.buffer.offset910, <4 x i32> %dma.buffer.rsrc911) #10, !dbg !11
  %qr.n355 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !11
  %qr.n356 = zext i32 %qr.n355 to i64, !dbg !11
  %qr.n357 = inttoptr i64 %qr.n356 to ptr addrspace(3), !dbg !11
  %address.local300 = getelementptr [2 x i8], ptr addrspace(1) %qr.n37, i64 %address.local150, !dbg !11
  %qr.n358 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %qr.n359 = ptrtoint ptr addrspace(1) %address.local300 to i64, !dbg !11
  %qr.n360 = sub i64 %qr.n359, %qr.n358, !dbg !11
  %dma.buffer.offset907 = trunc i64 %qr.n360 to i32, !dbg !11
  %qr.n361 = trunc i64 %qr.n358 to i32, !dbg !11
  %qr.n362 = insertelement <4 x i32> poison, i32 %qr.n361, i64 0, !dbg !11
  %qr.n363 = lshr i64 %qr.n358, 32, !dbg !11
  %qr.n364 = trunc i64 %qr.n363 to i32, !dbg !11
  %qr.n365 = insertelement <4 x i32> %qr.n362, i32 %qr.n364, i64 1, !dbg !11
  %qr.n366 = insertelement <4 x i32> %qr.n365, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc908 = insertelement <4 x i32> %qr.n366, i32 131072, i64 3, !dbg !11
  %qr.n367 = ptrtoint ptr addrspace(3) %qr.n357 to i32, !dbg !11
  %dma.buffer.lds909 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n367), !dbg !11
  %qr.n368 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds909, i32 %dma.buffer.offset907, <4 x i32> %dma.buffer.rsrc908) #10, !dbg !11
  %qr.n369 = add nuw i64 %qr.n41, 2048, !dbg !16
  %qr.n370 = icmp slt i64 %qr.n369, 0, !dbg !16
  %qr.n371 = sub nuw i64 -2049, %qr.n41, !dbg !16
  %qr.n372 = select i1 %qr.n370, i64 %qr.n371, i64 %qr.n369, !dbg !16
  %qr.n373 = sdiv i64 %qr.n372, 4096, !dbg !16
  %.lobit5 = ashr i64 %qr.n369, 63, !dbg !16
  %qr.n374 = xor i64 %qr.n373, %.lobit5, !dbg !16
  %qr.n375 = add nsw i64 %qr.n374, %qr.n10, !dbg !16
  %qr.n376 = add nsw i64 %qr.n224, 64, !dbg !16
  %qr.n377 = icmp slt i64 %qr.n224, -64, !dbg !16
  %qr.n378 = sub nsw i64 -65, %qr.n224, !dbg !16
  %qr.n379 = select i1 %qr.n377, i64 %qr.n378, i64 %qr.n376, !dbg !16
  %qr.n380 = sdiv i64 %qr.n379, 128, !dbg !16
  %qr.n381 = sext i1 %qr.n377 to i64, !dbg !16
  %qr.n382 = xor i64 %qr.n380, %qr.n381, !dbg !16
  %qr.n383 = shl nsw i64 %qr.n382, 7, !dbg !16
  %qr.n384 = sub i64 %qr.n35, %qr.n383, !dbg !16
  %qr.n385 = shl i64 %qr.n375, 19, !dbg !16
  %qr.n386 = and i64 %qr.n385, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %qr.n387 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %qr.n386, !dbg !16
  %qr.n388 = getelementptr [2 x i8], ptr addrspace(1) %qr.n387, i64 %qr.n79, !dbg !16
  %qr.n389 = getelementptr [2 x i8], ptr addrspace(1) %qr.n388, i64 %qr.n384, !dbg !16
  %qr.n390 = getelementptr i8, ptr addrspace(1) %qr.n389, i64 128, !dbg !16
  %qr.n391 = getelementptr [2 x i8], ptr addrspace(1) %qr.n390, i64 %qr.n85, !dbg !16
  %qr.n392 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local154), !dbg !16
  %qr.n393 = zext i32 %qr.n392 to i64, !dbg !16
  %qr.n394 = inttoptr i64 %qr.n393 to ptr addrspace(3), !dbg !16
  %address.local301 = getelementptr [2 x i8], ptr addrspace(1) %qr.n391, i64 %address.local138, !dbg !16
  %qr.n395 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %qr.n396 = ptrtoint ptr addrspace(1) %address.local301 to i64, !dbg !16
  %qr.n397 = sub i64 %qr.n396, %qr.n395, !dbg !16
  %dma.buffer.offset904 = trunc i64 %qr.n397 to i32, !dbg !16
  %qr.n398 = trunc i64 %qr.n395 to i32, !dbg !16
  %qr.n399 = insertelement <4 x i32> poison, i32 %qr.n398, i64 0, !dbg !16
  %qr.n400 = lshr i64 %qr.n395, 32, !dbg !16
  %qr.n401 = trunc i64 %qr.n400 to i32, !dbg !16
  %qr.n402 = insertelement <4 x i32> %qr.n399, i32 %qr.n401, i64 1, !dbg !16
  %qr.n403 = insertelement <4 x i32> %qr.n402, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc905 = insertelement <4 x i32> %qr.n403, i32 131072, i64 3, !dbg !16
  %qr.n404 = ptrtoint ptr addrspace(3) %qr.n394 to i32, !dbg !16
  %dma.buffer.lds906 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n404), !dbg !16
  %ksw.dma.offset4 = xor i32 %dma.buffer.offset904, %ksw.dma.mask, !dbg !16
  %qr.n405 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds906, i32 %ksw.dma.offset4, <4 x i32> %dma.buffer.rsrc905) #10, !dbg !16
  %qr.n406 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !16
  %qr.n407 = zext i32 %qr.n406 to i64, !dbg !16
  %qr.n408 = inttoptr i64 %qr.n407 to ptr addrspace(3), !dbg !16
  %address.local302 = getelementptr [2 x i8], ptr addrspace(1) %qr.n391, i64 %address.local150, !dbg !16
  %qr.n409 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %qr.n410 = ptrtoint ptr addrspace(1) %address.local302 to i64, !dbg !16
  %qr.n411 = sub i64 %qr.n410, %qr.n409, !dbg !16
  %dma.buffer.offset901 = trunc i64 %qr.n411 to i32, !dbg !16
  %qr.n412 = trunc i64 %qr.n409 to i32, !dbg !16
  %qr.n413 = insertelement <4 x i32> poison, i32 %qr.n412, i64 0, !dbg !16
  %qr.n414 = lshr i64 %qr.n409, 32, !dbg !16
  %qr.n415 = trunc i64 %qr.n414 to i32, !dbg !16
  %qr.n416 = insertelement <4 x i32> %qr.n413, i32 %qr.n415, i64 1, !dbg !16
  %qr.n417 = insertelement <4 x i32> %qr.n416, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc902 = insertelement <4 x i32> %qr.n417, i32 131072, i64 3, !dbg !16
  %qr.n418 = ptrtoint ptr addrspace(3) %qr.n408 to i32, !dbg !16
  %dma.buffer.lds903 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n418), !dbg !16
  %ksw.dma.offset5 = xor i32 %dma.buffer.offset901, %ksw.dma.mask, !dbg !16
  %qr.n419 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds903, i32 %ksw.dma.offset5, <4 x i32> %dma.buffer.rsrc902) #10, !dbg !16
  %qr.n420 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !12
  %qr.n421 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !12
  %qr.n422 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !12
  %qr.n423 = fmul <4 x half> %qr.n420, splat (half 0xH3015), !dbg !12
  %qr.n424 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !12
  %qr.n425 = fmul <4 x half> %qr.n421, splat (half 0xH3015), !dbg !12
  %qr.n426 = fmul <4 x half> %qr.n422, splat (half 0xH3015), !dbg !12
  %qr.n427 = fmul <4 x half> %qr.n424, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %qr.n428 = load <4 x half>, ptr addrspace(3) %ksw.read0.pointer, align 8, !dbg !17
  %qr.n429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n423, <4 x half> %qr.n428, <4 x float> %qr.n281) #9, !dbg !17
  %qr.n430 = load <4 x half>, ptr addrspace(3) %ksw.read1.pointer, align 8, !dbg !17
  %qr.n431 = load <4 x half>, ptr addrspace(3) %ksw.read2.pointer, align 8, !dbg !17
  %qr.n432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n425, <4 x half> %qr.n430, <4 x float> %qr.n429) #9, !dbg !17
  %qr.n433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n423, <4 x half> %qr.n431, <4 x float> %qr.n285) #9, !dbg !17
  %qr.n434 = load <4 x half>, ptr addrspace(3) %ksw.read3.pointer, align 8, !dbg !17
  %qr.n435 = load <4 x half>, ptr addrspace(3) %ksw.read4.pointer, align 8, !dbg !17
  %qr.n436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n425, <4 x half> %qr.n434, <4 x float> %qr.n433) #9, !dbg !17
  %qr.n437 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n423, <4 x half> %qr.n435, <4 x float> %qr.n289) #9, !dbg !17
  %qr.n438 = load <4 x half>, ptr addrspace(3) %ksw.read5.pointer, align 8, !dbg !17
  %qr.n439 = load <4 x half>, ptr addrspace(3) %ksw.read6.pointer, align 8, !dbg !17
  %qr.n440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n425, <4 x half> %qr.n438, <4 x float> %qr.n437) #9, !dbg !17
  %qr.n441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n423, <4 x half> %qr.n439, <4 x float> %qr.n293) #9, !dbg !17
  %qr.n442 = load <4 x half>, ptr addrspace(3) %ksw.read7.pointer, align 8, !dbg !17
  %qr.n443 = load <4 x half>, ptr addrspace(3) %ksw.read8.pointer, align 8, !dbg !17
  %qr.n444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n425, <4 x half> %qr.n442, <4 x float> %qr.n441) #9, !dbg !17
  %qr.n445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n423, <4 x half> %qr.n443, <4 x float> %qr.n297) #9, !dbg !17
  %qr.n446 = load <4 x half>, ptr addrspace(3) %ksw.read9.pointer, align 8, !dbg !17
  %qr.n447 = load <4 x half>, ptr addrspace(3) %ksw.read10.pointer, align 8, !dbg !17
  %qr.n448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n425, <4 x half> %qr.n446, <4 x float> %qr.n445) #9, !dbg !17
  %qr.n449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n423, <4 x half> %qr.n447, <4 x float> %qr.n301) #9, !dbg !17
  %qr.n450 = load <4 x half>, ptr addrspace(3) %ksw.read11.pointer, align 8, !dbg !17
  %qr.n451 = load <4 x half>, ptr addrspace(3) %ksw.read12.pointer, align 8, !dbg !17
  %qr.n452 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n425, <4 x half> %qr.n450, <4 x float> %qr.n449) #9, !dbg !17
  %qr.n453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n423, <4 x half> %qr.n451, <4 x float> %qr.n305) #9, !dbg !17
  %qr.n454 = load <4 x half>, ptr addrspace(3) %ksw.read13.pointer, align 8, !dbg !17
  %qr.n455 = load <4 x half>, ptr addrspace(3) %ksw.read14.pointer, align 8, !dbg !17
  %qr.n456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n425, <4 x half> %qr.n454, <4 x float> %qr.n453) #9, !dbg !17
  %qr.n457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n423, <4 x half> %qr.n455, <4 x float> %qr.n309) #9, !dbg !17
  %qr.n458 = load <4 x half>, ptr addrspace(3) %ksw.read15.pointer, align 8, !dbg !17
  %qr.n459 = load <4 x half>, ptr addrspace(3) %ksw.read0.pointer, align 8, !dbg !17
  %qr.n460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n425, <4 x half> %qr.n458, <4 x float> %qr.n457) #9, !dbg !17
  %qr.n461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n426, <4 x half> %qr.n459, <4 x float> %qr.n313) #9, !dbg !17
  %qr.n462 = load <4 x half>, ptr addrspace(3) %ksw.read1.pointer, align 8, !dbg !17
  %qr.n463 = load <4 x half>, ptr addrspace(3) %ksw.read2.pointer, align 8, !dbg !17
  %qr.n464 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n427, <4 x half> %qr.n462, <4 x float> %qr.n461) #9, !dbg !17
  %qr.n465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n426, <4 x half> %qr.n463, <4 x float> %qr.n317) #9, !dbg !17
  %qr.n466 = load <4 x half>, ptr addrspace(3) %ksw.read3.pointer, align 8, !dbg !17
  %qr.n467 = load <4 x half>, ptr addrspace(3) %ksw.read4.pointer, align 8, !dbg !17
  %qr.n468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n427, <4 x half> %qr.n466, <4 x float> %qr.n465) #9, !dbg !17
  %qr.n469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n426, <4 x half> %qr.n467, <4 x float> %qr.n321) #9, !dbg !17
  %qr.n470 = load <4 x half>, ptr addrspace(3) %ksw.read5.pointer, align 8, !dbg !17
  %qr.n471 = load <4 x half>, ptr addrspace(3) %ksw.read6.pointer, align 8, !dbg !17
  %qr.n472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n427, <4 x half> %qr.n470, <4 x float> %qr.n469) #9, !dbg !17
  %qr.n473 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n426, <4 x half> %qr.n471, <4 x float> %qr.n325) #9, !dbg !17
  %qr.n474 = load <4 x half>, ptr addrspace(3) %ksw.read7.pointer, align 8, !dbg !17
  %qr.n475 = load <4 x half>, ptr addrspace(3) %ksw.read8.pointer, align 8, !dbg !17
  %qr.n476 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n427, <4 x half> %qr.n474, <4 x float> %qr.n473) #9, !dbg !17
  %qr.n477 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n426, <4 x half> %qr.n475, <4 x float> %qr.n329) #9, !dbg !17
  %qr.n478 = load <4 x half>, ptr addrspace(3) %ksw.read9.pointer, align 8, !dbg !17
  %qr.n479 = load <4 x half>, ptr addrspace(3) %ksw.read10.pointer, align 8, !dbg !17
  %qr.n480 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n427, <4 x half> %qr.n478, <4 x float> %qr.n477) #9, !dbg !17
  %qr.n481 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n426, <4 x half> %qr.n479, <4 x float> %qr.n333) #9, !dbg !17
  %qr.n482 = load <4 x half>, ptr addrspace(3) %ksw.read11.pointer, align 8, !dbg !17
  %qr.n483 = load <4 x half>, ptr addrspace(3) %ksw.read12.pointer, align 8, !dbg !17
  %qr.n484 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n427, <4 x half> %qr.n482, <4 x float> %qr.n481) #9, !dbg !17
  %qr.n485 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n426, <4 x half> %qr.n483, <4 x float> %qr.n337) #9, !dbg !17
  %qr.n486 = load <4 x half>, ptr addrspace(3) %ksw.read13.pointer, align 8, !dbg !17
  %qr.n487 = load <4 x half>, ptr addrspace(3) %ksw.read14.pointer, align 8, !dbg !17
  %qr.n488 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n427, <4 x half> %qr.n486, <4 x float> %qr.n485) #9, !dbg !17
  %qr.n489 = load <4 x half>, ptr addrspace(3) %ksw.read15.pointer, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %qr.n490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n426, <4 x half> %qr.n487, <4 x float> %qr.n340) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %qr.n491 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n427, <4 x half> %qr.n489, <4 x float> %qr.n490) #9, !dbg !17
  %qr.n492 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !11
  %qr.n493 = zext i32 %qr.n492 to i64, !dbg !11
  %qr.n494 = inttoptr i64 %qr.n493 to ptr addrspace(3), !dbg !11
  %address.local303 = getelementptr [2 x i8], ptr addrspace(1) %qr.n39, i64 %address.local138, !dbg !11
  %qr.n495 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %qr.n496 = ptrtoint ptr addrspace(1) %address.local303 to i64, !dbg !11
  %qr.n497 = sub i64 %qr.n496, %qr.n495, !dbg !11
  %dma.buffer.offset898 = trunc i64 %qr.n497 to i32, !dbg !11
  %qr.n498 = trunc i64 %qr.n495 to i32, !dbg !11
  %qr.n499 = insertelement <4 x i32> poison, i32 %qr.n498, i64 0, !dbg !11
  %qr.n500 = lshr i64 %qr.n495, 32, !dbg !11
  %qr.n501 = trunc i64 %qr.n500 to i32, !dbg !11
  %qr.n502 = insertelement <4 x i32> %qr.n499, i32 %qr.n501, i64 1, !dbg !11
  %qr.n503 = insertelement <4 x i32> %qr.n502, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc899 = insertelement <4 x i32> %qr.n503, i32 131072, i64 3, !dbg !11
  %qr.n504 = ptrtoint ptr addrspace(3) %qr.n494 to i32, !dbg !11
  %dma.buffer.lds900 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n504), !dbg !11
  %qr.n505 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds900, i32 %dma.buffer.offset898, <4 x i32> %dma.buffer.rsrc899) #10, !dbg !11
  %qr.n506 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !11
  %qr.n507 = zext i32 %qr.n506 to i64, !dbg !11
  %qr.n508 = inttoptr i64 %qr.n507 to ptr addrspace(3), !dbg !11
  %address.local304 = getelementptr [2 x i8], ptr addrspace(1) %qr.n39, i64 %address.local150, !dbg !11
  %qr.n509 = ptrtoint ptr addrspace(1) %0 to i64, !dbg !11
  %qr.n510 = ptrtoint ptr addrspace(1) %address.local304 to i64, !dbg !11
  %qr.n511 = sub i64 %qr.n510, %qr.n509, !dbg !11
  %dma.buffer.offset895 = trunc i64 %qr.n511 to i32, !dbg !11
  %qr.n512 = trunc i64 %qr.n509 to i32, !dbg !11
  %qr.n513 = insertelement <4 x i32> poison, i32 %qr.n512, i64 0, !dbg !11
  %qr.n514 = lshr i64 %qr.n509, 32, !dbg !11
  %qr.n515 = trunc i64 %qr.n514 to i32, !dbg !11
  %qr.n516 = insertelement <4 x i32> %qr.n513, i32 %qr.n515, i64 1, !dbg !11
  %qr.n517 = insertelement <4 x i32> %qr.n516, i32 33554432, i64 2, !dbg !11
  %dma.buffer.rsrc896 = insertelement <4 x i32> %qr.n517, i32 131072, i64 3, !dbg !11
  %qr.n518 = ptrtoint ptr addrspace(3) %qr.n508 to i32, !dbg !11
  %dma.buffer.lds897 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n518), !dbg !11
  %qr.n519 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds897, i32 %dma.buffer.offset895, <4 x i32> %dma.buffer.rsrc896) #10, !dbg !11
  %qr.n520 = add nuw i64 %qr.n41, 3072, !dbg !16
  %qr.n521 = icmp slt i64 %qr.n520, 0, !dbg !16
  %qr.n522 = sub nuw i64 -3073, %qr.n41, !dbg !16
  %qr.n523 = select i1 %qr.n521, i64 %qr.n522, i64 %qr.n520, !dbg !16
  %qr.n524 = sdiv i64 %qr.n523, 4096, !dbg !16
  %.lobit6 = ashr i64 %qr.n520, 63, !dbg !16
  %qr.n525 = xor i64 %qr.n524, %.lobit6, !dbg !16
  %qr.n526 = add nsw i64 %qr.n525, %qr.n10, !dbg !16
  %qr.n527 = add nsw i64 %qr.n224, 96, !dbg !16
  %qr.n528 = icmp slt i64 %qr.n224, -96, !dbg !16
  %qr.n529 = sub nsw i64 -97, %qr.n224, !dbg !16
  %qr.n530 = select i1 %qr.n528, i64 %qr.n529, i64 %qr.n527, !dbg !16
  %qr.n531 = sdiv i64 %qr.n530, 128, !dbg !16
  %qr.n532 = sext i1 %qr.n528 to i64, !dbg !16
  %qr.n533 = xor i64 %qr.n531, %qr.n532, !dbg !16
  %qr.n534 = shl nsw i64 %qr.n533, 7, !dbg !16
  %qr.n535 = sub i64 %qr.n35, %qr.n534, !dbg !16
  %qr.n536 = shl i64 %qr.n526, 19, !dbg !16
  %qr.n537 = and i64 %qr.n536, 16252928, !dbg !16
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %qr.n538 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %qr.n537, !dbg !16
  %qr.n539 = getelementptr [2 x i8], ptr addrspace(1) %qr.n538, i64 %qr.n79, !dbg !16
  %qr.n540 = getelementptr [2 x i8], ptr addrspace(1) %qr.n539, i64 %qr.n535, !dbg !16
  %qr.n541 = getelementptr i8, ptr addrspace(1) %qr.n540, i64 192, !dbg !16
  %qr.n542 = getelementptr [2 x i8], ptr addrspace(1) %qr.n541, i64 %qr.n85, !dbg !16
  %qr.n543 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local154), !dbg !16
  %qr.n544 = zext i32 %qr.n543 to i64, !dbg !16
  %qr.n545 = inttoptr i64 %qr.n544 to ptr addrspace(3), !dbg !16
  %address.local305 = getelementptr [2 x i8], ptr addrspace(1) %qr.n542, i64 %address.local138, !dbg !16
  %qr.n546 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %qr.n547 = ptrtoint ptr addrspace(1) %address.local305 to i64, !dbg !16
  %qr.n548 = sub i64 %qr.n547, %qr.n546, !dbg !16
  %dma.buffer.offset892 = trunc i64 %qr.n548 to i32, !dbg !16
  %qr.n549 = trunc i64 %qr.n546 to i32, !dbg !16
  %qr.n550 = insertelement <4 x i32> poison, i32 %qr.n549, i64 0, !dbg !16
  %qr.n551 = lshr i64 %qr.n546, 32, !dbg !16
  %qr.n552 = trunc i64 %qr.n551 to i32, !dbg !16
  %qr.n553 = insertelement <4 x i32> %qr.n550, i32 %qr.n552, i64 1, !dbg !16
  %qr.n554 = insertelement <4 x i32> %qr.n553, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc893 = insertelement <4 x i32> %qr.n554, i32 131072, i64 3, !dbg !16
  %qr.n555 = ptrtoint ptr addrspace(3) %qr.n545 to i32, !dbg !16
  %dma.buffer.lds894 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n555), !dbg !16
  %ksw.dma.offset6 = xor i32 %dma.buffer.offset892, %ksw.dma.mask, !dbg !16
  %qr.n556 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds894, i32 %ksw.dma.offset6, <4 x i32> %dma.buffer.rsrc893) #10, !dbg !16
  %qr.n557 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local158), !dbg !16
  %qr.n558 = zext i32 %qr.n557 to i64, !dbg !16
  %qr.n559 = inttoptr i64 %qr.n558 to ptr addrspace(3), !dbg !16
  %address.local306 = getelementptr [2 x i8], ptr addrspace(1) %qr.n542, i64 %address.local150, !dbg !16
  %qr.n560 = ptrtoint ptr addrspace(1) %1 to i64, !dbg !16
  %qr.n561 = ptrtoint ptr addrspace(1) %address.local306 to i64, !dbg !16
  %qr.n562 = sub i64 %qr.n561, %qr.n560, !dbg !16
  %dma.buffer.offset889 = trunc i64 %qr.n562 to i32, !dbg !16
  %qr.n563 = trunc i64 %qr.n560 to i32, !dbg !16
  %qr.n564 = insertelement <4 x i32> poison, i32 %qr.n563, i64 0, !dbg !16
  %qr.n565 = lshr i64 %qr.n560, 32, !dbg !16
  %qr.n566 = trunc i64 %qr.n565 to i32, !dbg !16
  %qr.n567 = insertelement <4 x i32> %qr.n564, i32 %qr.n566, i64 1, !dbg !16
  %qr.n568 = insertelement <4 x i32> %qr.n567, i32 33554432, i64 2, !dbg !16
  %dma.buffer.rsrc890 = insertelement <4 x i32> %qr.n568, i32 131072, i64 3, !dbg !16
  %qr.n569 = ptrtoint ptr addrspace(3) %qr.n559 to i32, !dbg !16
  %dma.buffer.lds891 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n569), !dbg !16
  %ksw.dma.offset7 = xor i32 %dma.buffer.offset889, %ksw.dma.mask, !dbg !16
  %qr.n570 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds891, i32 %ksw.dma.offset7, <4 x i32> %dma.buffer.rsrc890) #10, !dbg !16
  %qr.n571 = load <4 x half>, ptr addrspace(3) %address.local172, align 2, !dbg !12
  %qr.n572 = load <4 x half>, ptr addrspace(3) %address.local174, align 2, !dbg !12
  %qr.n573 = load <4 x half>, ptr addrspace(3) %address.local176, align 2, !dbg !12
  %qr.n574 = fmul <4 x half> %qr.n571, splat (half 0xH3015), !dbg !12
  %qr.n575 = load <4 x half>, ptr addrspace(3) %address.local178, align 2, !dbg !12
  %qr.n576 = fmul <4 x half> %qr.n572, splat (half 0xH3015), !dbg !12
  %qr.n577 = fmul <4 x half> %qr.n573, splat (half 0xH3015), !dbg !12
  %qr.n578 = fmul <4 x half> %qr.n575, splat (half 0xH3015), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !16
  fence syncscope("workgroup") release, !dbg !16
  call void @llvm.amdgcn.s.barrier(), !dbg !16
  fence syncscope("workgroup") acquire, !dbg !16
  %qr.n579 = add nsw i64 %qr.n79, %qr.n10, !dbg !18
  %qr.n580 = shl i64 %qr.n579, 19, !dbg !18
  %qr.n581 = and i64 %qr.n580, 16252928, !dbg !18
  %qr.n582 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %qr.n581, !dbg !18
  %qr.n583 = getelementptr [2 x i8], ptr addrspace(1) %qr.n582, i64 %qr.n85, !dbg !18
  %qr.n584 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local133), !dbg !18
  %qr.n585 = zext i32 %qr.n584 to i64, !dbg !18
  %qr.n586 = inttoptr i64 %qr.n585 to ptr addrspace(3), !dbg !18
  %address.local307 = shl nuw nsw i64 %address.local, 5, !dbg !18
  %address.local308 = and i64 %address.local307, 8064, !dbg !18
  %address.local309 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local310 = and i64 %address.local309, 24, !dbg !18
  %address.local311 = or disjoint i64 %address.local308, %address.local310, !dbg !18
  %address.local312 = getelementptr [2 x i8], ptr addrspace(1) %qr.n583, i64 %address.local311, !dbg !18
  %qr.n587 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %qr.n588 = ptrtoint ptr addrspace(1) %address.local312 to i64, !dbg !18
  %qr.n589 = sub i64 %qr.n588, %qr.n587, !dbg !18
  %dma.buffer.offset886 = trunc i64 %qr.n589 to i32, !dbg !18
  %qr.n590 = trunc i64 %qr.n587 to i32, !dbg !18
  %qr.n591 = insertelement <4 x i32> poison, i32 %qr.n590, i64 0, !dbg !18
  %qr.n592 = lshr i64 %qr.n587, 32, !dbg !18
  %qr.n593 = trunc i64 %qr.n592 to i32, !dbg !18
  %qr.n594 = insertelement <4 x i32> %qr.n591, i32 %qr.n593, i64 1, !dbg !18
  %qr.n595 = insertelement <4 x i32> %qr.n594, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc887 = insertelement <4 x i32> %qr.n595, i32 131072, i64 3, !dbg !18
  %qr.n596 = ptrtoint ptr addrspace(3) %qr.n586 to i32, !dbg !18
  %dma.buffer.lds888 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n596), !dbg !18
  %qr.n597 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds888, i32 %dma.buffer.offset886, <4 x i32> %dma.buffer.rsrc887) #10, !dbg !18
  %qr.n598 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local145), !dbg !18
  %qr.n599 = zext i32 %qr.n598 to i64, !dbg !18
  %qr.n600 = inttoptr i64 %qr.n599 to ptr addrspace(3), !dbg !18
  %address.local313 = shl nuw nsw i64 %address.local140, 5, !dbg !18
  %address.local314 = and i64 %address.local313, 16256, !dbg !18
  %address.local315 = shl nuw nsw i64 %address.local, 3, !dbg !18
  %address.local316 = and i64 %address.local315, 24, !dbg !18
  %address.local317 = or disjoint i64 %address.local314, %address.local316, !dbg !18
  %address.local318 = getelementptr [2 x i8], ptr addrspace(1) %qr.n583, i64 %address.local317, !dbg !18
  %qr.n601 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %qr.n602 = ptrtoint ptr addrspace(1) %address.local318 to i64, !dbg !18
  %qr.n603 = sub i64 %qr.n602, %qr.n601, !dbg !18
  %dma.buffer.offset883 = trunc i64 %qr.n603 to i32, !dbg !18
  %qr.n604 = trunc i64 %qr.n601 to i32, !dbg !18
  %qr.n605 = insertelement <4 x i32> poison, i32 %qr.n604, i64 0, !dbg !18
  %qr.n606 = lshr i64 %qr.n601, 32, !dbg !18
  %qr.n607 = trunc i64 %qr.n606 to i32, !dbg !18
  %qr.n608 = insertelement <4 x i32> %qr.n605, i32 %qr.n607, i64 1, !dbg !18
  %qr.n609 = insertelement <4 x i32> %qr.n608, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc884 = insertelement <4 x i32> %qr.n609, i32 131072, i64 3, !dbg !18
  %qr.n610 = ptrtoint ptr addrspace(3) %qr.n600 to i32, !dbg !18
  %dma.buffer.lds885 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n610), !dbg !18
  %qr.n611 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds885, i32 %dma.buffer.offset883, <4 x i32> %dma.buffer.rsrc884) #10, !dbg !18
  %qr.n612 = load <4 x half>, ptr addrspace(3) %ksw.read0.pointer, align 8, !dbg !17
  %qr.n613 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n574, <4 x half> %qr.n612, <4 x float> %qr.n432) #9, !dbg !17
  %qr.n614 = load <4 x half>, ptr addrspace(3) %ksw.read1.pointer, align 8, !dbg !17
  %qr.n615 = load <4 x half>, ptr addrspace(3) %ksw.read2.pointer, align 8, !dbg !17
  %qr.n616 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n576, <4 x half> %qr.n614, <4 x float> %qr.n613) #9, !dbg !17
  %qr.n617 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n574, <4 x half> %qr.n615, <4 x float> %qr.n436) #9, !dbg !17
  %qr.n618 = load <4 x half>, ptr addrspace(3) %ksw.read3.pointer, align 8, !dbg !17
  %qr.n619 = load <4 x half>, ptr addrspace(3) %ksw.read4.pointer, align 8, !dbg !17
  %qr.n620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n576, <4 x half> %qr.n618, <4 x float> %qr.n617) #9, !dbg !17
  %qr.n621 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n574, <4 x half> %qr.n619, <4 x float> %qr.n440) #9, !dbg !17
  %qr.n622 = load <4 x half>, ptr addrspace(3) %ksw.read5.pointer, align 8, !dbg !17
  %qr.n623 = load <4 x half>, ptr addrspace(3) %ksw.read6.pointer, align 8, !dbg !17
  %qr.n624 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n576, <4 x half> %qr.n622, <4 x float> %qr.n621) #9, !dbg !17
  %qr.n625 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n574, <4 x half> %qr.n623, <4 x float> %qr.n444) #9, !dbg !17
  %qr.n626 = load <4 x half>, ptr addrspace(3) %ksw.read7.pointer, align 8, !dbg !17
  %qr.n627 = load <4 x half>, ptr addrspace(3) %ksw.read8.pointer, align 8, !dbg !17
  %qr.n628 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n576, <4 x half> %qr.n626, <4 x float> %qr.n625) #9, !dbg !17
  %qr.n629 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n574, <4 x half> %qr.n627, <4 x float> %qr.n448) #9, !dbg !17
  %qr.n630 = load <4 x half>, ptr addrspace(3) %ksw.read9.pointer, align 8, !dbg !17
  %qr.n631 = load <4 x half>, ptr addrspace(3) %ksw.read10.pointer, align 8, !dbg !17
  %qr.n632 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n576, <4 x half> %qr.n630, <4 x float> %qr.n629) #9, !dbg !17
  %qr.n633 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n574, <4 x half> %qr.n631, <4 x float> %qr.n452) #9, !dbg !17
  %qr.n634 = load <4 x half>, ptr addrspace(3) %ksw.read11.pointer, align 8, !dbg !17
  %qr.n635 = load <4 x half>, ptr addrspace(3) %ksw.read12.pointer, align 8, !dbg !17
  %qr.n636 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n576, <4 x half> %qr.n634, <4 x float> %qr.n633) #9, !dbg !17
  %qr.n637 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n574, <4 x half> %qr.n635, <4 x float> %qr.n456) #9, !dbg !17
  %qr.n638 = load <4 x half>, ptr addrspace(3) %ksw.read13.pointer, align 8, !dbg !17
  %qr.n639 = load <4 x half>, ptr addrspace(3) %ksw.read14.pointer, align 8, !dbg !17
  %qr.n640 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n576, <4 x half> %qr.n638, <4 x float> %qr.n637) #9, !dbg !17
  %qr.n641 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n574, <4 x half> %qr.n639, <4 x float> %qr.n460) #9, !dbg !17
  %qr.n642 = load <4 x half>, ptr addrspace(3) %ksw.read15.pointer, align 8, !dbg !17
  %qr.n643 = load <4 x half>, ptr addrspace(3) %ksw.read0.pointer, align 8, !dbg !17
  %qr.n644 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n576, <4 x half> %qr.n642, <4 x float> %qr.n641) #9, !dbg !17
  %qr.n645 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n577, <4 x half> %qr.n643, <4 x float> %qr.n464) #9, !dbg !17
  %qr.n646 = load <4 x half>, ptr addrspace(3) %ksw.read1.pointer, align 8, !dbg !17
  %qr.n647 = load <4 x half>, ptr addrspace(3) %ksw.read2.pointer, align 8, !dbg !17
  %qr.n648 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n578, <4 x half> %qr.n646, <4 x float> %qr.n645) #9, !dbg !17
  %qr.n649 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n577, <4 x half> %qr.n647, <4 x float> %qr.n468) #9, !dbg !17
  %qr.n650 = load <4 x half>, ptr addrspace(3) %ksw.read3.pointer, align 8, !dbg !17
  %qr.n651 = load <4 x half>, ptr addrspace(3) %ksw.read4.pointer, align 8, !dbg !17
  %qr.n652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n578, <4 x half> %qr.n650, <4 x float> %qr.n649) #9, !dbg !17
  %qr.n653 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n577, <4 x half> %qr.n651, <4 x float> %qr.n472) #9, !dbg !17
  %qr.n654 = load <4 x half>, ptr addrspace(3) %ksw.read5.pointer, align 8, !dbg !17
  %qr.n655 = load <4 x half>, ptr addrspace(3) %ksw.read6.pointer, align 8, !dbg !17
  %qr.n656 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n578, <4 x half> %qr.n654, <4 x float> %qr.n653) #9, !dbg !17
  %qr.n657 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n577, <4 x half> %qr.n655, <4 x float> %qr.n476) #9, !dbg !17
  %qr.n658 = load <4 x half>, ptr addrspace(3) %ksw.read7.pointer, align 8, !dbg !17
  %qr.n659 = load <4 x half>, ptr addrspace(3) %ksw.read8.pointer, align 8, !dbg !17
  %qr.n660 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n578, <4 x half> %qr.n658, <4 x float> %qr.n657) #9, !dbg !17
  %qr.n661 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n577, <4 x half> %qr.n659, <4 x float> %qr.n480) #9, !dbg !17
  %qr.n662 = load <4 x half>, ptr addrspace(3) %ksw.read9.pointer, align 8, !dbg !17
  %qr.n663 = load <4 x half>, ptr addrspace(3) %ksw.read10.pointer, align 8, !dbg !17
  %qr.n664 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n578, <4 x half> %qr.n662, <4 x float> %qr.n661) #9, !dbg !17
  %qr.n665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n577, <4 x half> %qr.n663, <4 x float> %qr.n484) #9, !dbg !17
  %qr.n666 = load <4 x half>, ptr addrspace(3) %ksw.read11.pointer, align 8, !dbg !17
  %qr.n667 = load <4 x half>, ptr addrspace(3) %ksw.read12.pointer, align 8, !dbg !17
  %qr.n668 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n578, <4 x half> %qr.n666, <4 x float> %qr.n665) #9, !dbg !17
  %qr.n669 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n577, <4 x half> %qr.n667, <4 x float> %qr.n488) #9, !dbg !17
  %qr.n670 = load <4 x half>, ptr addrspace(3) %ksw.read13.pointer, align 8, !dbg !17
  %qr.n671 = load <4 x half>, ptr addrspace(3) %ksw.read14.pointer, align 8, !dbg !17
  %qr.n672 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n578, <4 x half> %qr.n670, <4 x float> %qr.n669) #9, !dbg !17
  %qr.n673 = load <4 x half>, ptr addrspace(3) %ksw.read15.pointer, align 8, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !17
  %qr.n674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n577, <4 x half> %qr.n671, <4 x float> %qr.n491) #9, !dbg !17
  call void @llvm.amdgcn.sched.barrier(i32 0) #11, !dbg !13
  %qr.n675 = or disjoint i64 %qr.n48, 127, !dbg !13
  %qr.n676 = icmp uge i64 %qr.n11, %qr.n675, !dbg !13
  %qr.n677 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %qr.n578, <4 x half> %qr.n673, <4 x float> %qr.n674) #9, !dbg !17
  %qr.n678 = and i1 %qr.n29, %qr.n676, !dbg !13
  br i1 %qr.n678, label %qr.n679, label %qr.n680, !dbg !13

qr.n679:                                              ; preds = %qr.n47
  br label %qr.n824, !dbg !13

qr.n680:                                              ; preds = %qr.n47
  %qr.n681 = add nsw i64 %qr.n11, %qr.n25, !dbg !13
  %qr.n682 = shl nuw nsw i64 %qr.n26, 4, !dbg !13
  %qr.n683 = add nsw i64 %qr.n681, %qr.n682, !dbg !13
  %qr.n684 = and i32 %qr.n24, 240, !dbg !13
  %index.compact = zext nneg i32 %qr.n684 to i64, !dbg !13
  %qr.n685 = sub nsw i64 %qr.n683, %index.compact, !dbg !13
  %qr.n686 = and i64 %qr.n25, 48, !dbg !13
  %qr.n687 = or disjoint i64 %qr.n48, %qr.n686, !dbg !13
  %qr.n688 = add nsw i64 %qr.n685, 1, !dbg !23
  %qr.n689 = lshr i64 %qr.n688, 32, !dbg !24
  %compare.high = trunc i64 %qr.n689 to i32, !dbg !24
  %qr.n690 = lshr i64 %qr.n687, 32, !dbg !24
  %compare.high612 = trunc i64 %qr.n690 to i32, !dbg !24
  %compare.high.equal = icmp eq i32 %compare.high, %compare.high612, !dbg !24
  %compare.high.order = icmp ugt i32 %compare.high, %compare.high612, !dbg !24
  %compare.low.base = trunc i64 %qr.n688 to i32, !dbg !24
  %compare.low.base613 = trunc i64 %qr.n687 to i32, !dbg !24
  %compare.low = icmp ugt i32 %compare.low.base, %compare.low.base613, !dbg !24
  %compare.result = select i1 %compare.high.equal, i1 %compare.low, i1 %compare.high.order, !dbg !24
  %qr.n691 = select i1 %compare.result, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 1) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local = or i32 %compare.local.base, 1, !dbg !24
  %compare.low614 = icmp ugt i32 %compare.low.base, %compare.local, !dbg !24
  %compare.result615 = select i1 %compare.high.equal, i1 %compare.low614, i1 %compare.high.order, !dbg !24
  %qr.n692 = select i1 %compare.result615, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base616 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 2) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local617 = or i32 %compare.local.base616, 2, !dbg !24
  %compare.low618 = icmp ugt i32 %compare.low.base, %compare.local617, !dbg !24
  %compare.result619 = select i1 %compare.high.equal, i1 %compare.low618, i1 %compare.high.order, !dbg !24
  %qr.n693 = select i1 %compare.result619, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base620 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 3) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local621 = or i32 %compare.local.base620, 3, !dbg !24
  %compare.low622 = icmp ugt i32 %compare.low.base, %compare.local621, !dbg !24
  %compare.result623 = select i1 %compare.high.equal, i1 %compare.low622, i1 %compare.high.order, !dbg !24
  %qr.n694 = select i1 %compare.result623, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base624 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 4) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local625 = or i32 %compare.local.base624, 4, !dbg !24
  %compare.low626 = icmp ugt i32 %compare.low.base, %compare.local625, !dbg !24
  %compare.result627 = select i1 %compare.high.equal, i1 %compare.low626, i1 %compare.high.order, !dbg !24
  %qr.n695 = select i1 %compare.result627, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base628 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 5) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local629 = or i32 %compare.local.base628, 5, !dbg !24
  %compare.low630 = icmp ugt i32 %compare.low.base, %compare.local629, !dbg !24
  %compare.result631 = select i1 %compare.high.equal, i1 %compare.low630, i1 %compare.high.order, !dbg !24
  %qr.n696 = select i1 %compare.result631, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base632 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 6) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local633 = or i32 %compare.local.base632, 6, !dbg !24
  %compare.low634 = icmp ugt i32 %compare.low.base, %compare.local633, !dbg !24
  %compare.result635 = select i1 %compare.high.equal, i1 %compare.low634, i1 %compare.high.order, !dbg !24
  %qr.n697 = select i1 %compare.result635, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base636 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 7) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local637 = or i32 %compare.local.base636, 7, !dbg !24
  %compare.low638 = icmp ugt i32 %compare.low.base, %compare.local637, !dbg !24
  %compare.result639 = select i1 %compare.high.equal, i1 %compare.low638, i1 %compare.high.order, !dbg !24
  %qr.n698 = select i1 %compare.result639, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base640 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 8) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local641 = or i32 %compare.local.base640, 8, !dbg !24
  %compare.low642 = icmp ugt i32 %compare.low.base, %compare.local641, !dbg !24
  %compare.result643 = select i1 %compare.high.equal, i1 %compare.low642, i1 %compare.high.order, !dbg !24
  %qr.n699 = select i1 %compare.result643, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base644 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 9) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local645 = or i32 %compare.local.base644, 9, !dbg !24
  %compare.low646 = icmp ugt i32 %compare.low.base, %compare.local645, !dbg !24
  %compare.result647 = select i1 %compare.high.equal, i1 %compare.low646, i1 %compare.high.order, !dbg !24
  %qr.n700 = select i1 %compare.result647, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base648 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 10) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local649 = or i32 %compare.local.base648, 10, !dbg !24
  %compare.low650 = icmp ugt i32 %compare.low.base, %compare.local649, !dbg !24
  %compare.result651 = select i1 %compare.high.equal, i1 %compare.low650, i1 %compare.high.order, !dbg !24
  %qr.n701 = select i1 %compare.result651, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base652 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 11) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local653 = or i32 %compare.local.base652, 11, !dbg !24
  %compare.low654 = icmp ugt i32 %compare.low.base, %compare.local653, !dbg !24
  %compare.result655 = select i1 %compare.high.equal, i1 %compare.low654, i1 %compare.high.order, !dbg !24
  %qr.n702 = select i1 %compare.result655, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base656 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 12) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local657 = or i32 %compare.local.base656, 12, !dbg !24
  %compare.low658 = icmp ugt i32 %compare.low.base, %compare.local657, !dbg !24
  %compare.result659 = select i1 %compare.high.equal, i1 %compare.low658, i1 %compare.high.order, !dbg !24
  %qr.n703 = select i1 %compare.result659, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base660 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 13) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local661 = or i32 %compare.local.base660, 13, !dbg !24
  %compare.low662 = icmp ugt i32 %compare.low.base, %compare.local661, !dbg !24
  %compare.result663 = select i1 %compare.high.equal, i1 %compare.low662, i1 %compare.high.order, !dbg !24
  %qr.n704 = select i1 %compare.result663, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base664 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 14) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local665 = or i32 %compare.local.base664, 14, !dbg !24
  %compare.low666 = icmp ugt i32 %compare.low.base, %compare.local665, !dbg !24
  %compare.result667 = select i1 %compare.high.equal, i1 %compare.low666, i1 %compare.high.order, !dbg !24
  %qr.n705 = select i1 %compare.result667, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base668 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 15) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local669 = or i32 %compare.local.base668, 15, !dbg !24
  %compare.low670 = icmp ugt i32 %compare.low.base, %compare.local669, !dbg !24
  %compare.result671 = select i1 %compare.high.equal, i1 %compare.low670, i1 %compare.high.order, !dbg !24
  %qr.n706 = select i1 %compare.result671, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base672 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 16) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local673 = or i32 %compare.local.base672, 64, !dbg !24
  %compare.low674 = icmp ugt i32 %compare.low.base, %compare.local673, !dbg !24
  %compare.result675 = select i1 %compare.high.equal, i1 %compare.low674, i1 %compare.high.order, !dbg !24
  %qr.n707 = select i1 %compare.result675, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base676 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 17) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local677 = or i32 %compare.local.base676, 65, !dbg !24
  %compare.low678 = icmp ugt i32 %compare.low.base, %compare.local677, !dbg !24
  %compare.result679 = select i1 %compare.high.equal, i1 %compare.low678, i1 %compare.high.order, !dbg !24
  %qr.n708 = select i1 %compare.result679, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base680 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 18) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local681 = or i32 %compare.local.base680, 66, !dbg !24
  %compare.low682 = icmp ugt i32 %compare.low.base, %compare.local681, !dbg !24
  %compare.result683 = select i1 %compare.high.equal, i1 %compare.low682, i1 %compare.high.order, !dbg !24
  %qr.n709 = select i1 %compare.result683, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base684 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 19) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local685 = or i32 %compare.local.base684, 67, !dbg !24
  %compare.low686 = icmp ugt i32 %compare.low.base, %compare.local685, !dbg !24
  %compare.result687 = select i1 %compare.high.equal, i1 %compare.low686, i1 %compare.high.order, !dbg !24
  %qr.n710 = select i1 %compare.result687, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base688 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 20) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local689 = or i32 %compare.local.base688, 68, !dbg !24
  %compare.low690 = icmp ugt i32 %compare.low.base, %compare.local689, !dbg !24
  %compare.result691 = select i1 %compare.high.equal, i1 %compare.low690, i1 %compare.high.order, !dbg !24
  %qr.n711 = select i1 %compare.result691, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base692 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 21) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local693 = or i32 %compare.local.base692, 69, !dbg !24
  %compare.low694 = icmp ugt i32 %compare.low.base, %compare.local693, !dbg !24
  %compare.result695 = select i1 %compare.high.equal, i1 %compare.low694, i1 %compare.high.order, !dbg !24
  %qr.n712 = select i1 %compare.result695, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base696 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 22) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local697 = or i32 %compare.local.base696, 70, !dbg !24
  %compare.low698 = icmp ugt i32 %compare.low.base, %compare.local697, !dbg !24
  %compare.result699 = select i1 %compare.high.equal, i1 %compare.low698, i1 %compare.high.order, !dbg !24
  %qr.n713 = select i1 %compare.result699, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base700 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 23) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local701 = or i32 %compare.local.base700, 71, !dbg !24
  %compare.low702 = icmp ugt i32 %compare.low.base, %compare.local701, !dbg !24
  %compare.result703 = select i1 %compare.high.equal, i1 %compare.low702, i1 %compare.high.order, !dbg !24
  %qr.n714 = select i1 %compare.result703, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base704 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 24) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local705 = or i32 %compare.local.base704, 72, !dbg !24
  %compare.low706 = icmp ugt i32 %compare.low.base, %compare.local705, !dbg !24
  %compare.result707 = select i1 %compare.high.equal, i1 %compare.low706, i1 %compare.high.order, !dbg !24
  %qr.n715 = select i1 %compare.result707, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base708 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 25) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local709 = or i32 %compare.local.base708, 73, !dbg !24
  %compare.low710 = icmp ugt i32 %compare.low.base, %compare.local709, !dbg !24
  %compare.result711 = select i1 %compare.high.equal, i1 %compare.low710, i1 %compare.high.order, !dbg !24
  %qr.n716 = select i1 %compare.result711, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base712 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 26) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local713 = or i32 %compare.local.base712, 74, !dbg !24
  %compare.low714 = icmp ugt i32 %compare.low.base, %compare.local713, !dbg !24
  %compare.result715 = select i1 %compare.high.equal, i1 %compare.low714, i1 %compare.high.order, !dbg !24
  %qr.n717 = select i1 %compare.result715, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base716 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 27) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local717 = or i32 %compare.local.base716, 75, !dbg !24
  %compare.low718 = icmp ugt i32 %compare.low.base, %compare.local717, !dbg !24
  %compare.result719 = select i1 %compare.high.equal, i1 %compare.low718, i1 %compare.high.order, !dbg !24
  %qr.n718 = select i1 %compare.result719, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base720 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 28) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local721 = or i32 %compare.local.base720, 76, !dbg !24
  %compare.low722 = icmp ugt i32 %compare.low.base, %compare.local721, !dbg !24
  %compare.result723 = select i1 %compare.high.equal, i1 %compare.low722, i1 %compare.high.order, !dbg !24
  %qr.n719 = select i1 %compare.result723, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base724 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 29) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local725 = or i32 %compare.local.base724, 77, !dbg !24
  %compare.low726 = icmp ugt i32 %compare.low.base, %compare.local725, !dbg !24
  %compare.result727 = select i1 %compare.high.equal, i1 %compare.low726, i1 %compare.high.order, !dbg !24
  %qr.n720 = select i1 %compare.result727, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base728 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 30) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local729 = or i32 %compare.local.base728, 78, !dbg !24
  %compare.low730 = icmp ugt i32 %compare.low.base, %compare.local729, !dbg !24
  %compare.result731 = select i1 %compare.high.equal, i1 %compare.low730, i1 %compare.high.order, !dbg !24
  %qr.n721 = select i1 %compare.result731, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base732 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base613, i32 31) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local733 = or i32 %compare.local.base732, 79, !dbg !24
  %compare.low734 = icmp ugt i32 %compare.low.base, %compare.local733, !dbg !24
  %compare.result735 = select i1 %compare.high.equal, i1 %compare.low734, i1 %compare.high.order, !dbg !24
  %qr.n722 = select i1 %compare.result735, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %qr.n723 = add nsw i64 %qr.n685, 65, !dbg !23
  %qr.n724 = lshr i64 %qr.n723, 32, !dbg !24
  %compare.high736 = trunc i64 %qr.n724 to i32, !dbg !24
  %qr.n725 = lshr i64 %qr.n687, 32, !dbg !24
  %compare.high737 = trunc i64 %qr.n725 to i32, !dbg !24
  %compare.high.equal738 = icmp eq i32 %compare.high736, %compare.high737, !dbg !24
  %compare.high.order739 = icmp ugt i32 %compare.high736, %compare.high737, !dbg !24
  %compare.low.base740 = trunc i64 %qr.n723 to i32, !dbg !24
  %compare.low.base741 = trunc i64 %qr.n687 to i32, !dbg !24
  %compare.low742 = icmp ugt i32 %compare.low.base740, %compare.low.base741, !dbg !24
  %compare.result743 = select i1 %compare.high.equal738, i1 %compare.low742, i1 %compare.high.order739, !dbg !24
  %qr.n726 = select i1 %compare.result743, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base744 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 32) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local745 = or i32 %compare.local.base744, 1, !dbg !24
  %compare.low746 = icmp ugt i32 %compare.low.base740, %compare.local745, !dbg !24
  %compare.result747 = select i1 %compare.high.equal738, i1 %compare.low746, i1 %compare.high.order739, !dbg !24
  %qr.n727 = select i1 %compare.result747, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base748 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 33) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local749 = or i32 %compare.local.base748, 2, !dbg !24
  %compare.low750 = icmp ugt i32 %compare.low.base740, %compare.local749, !dbg !24
  %compare.result751 = select i1 %compare.high.equal738, i1 %compare.low750, i1 %compare.high.order739, !dbg !24
  %qr.n728 = select i1 %compare.result751, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base752 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 34) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local753 = or i32 %compare.local.base752, 3, !dbg !24
  %compare.low754 = icmp ugt i32 %compare.low.base740, %compare.local753, !dbg !24
  %compare.result755 = select i1 %compare.high.equal738, i1 %compare.low754, i1 %compare.high.order739, !dbg !24
  %qr.n729 = select i1 %compare.result755, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base756 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 35) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local757 = or i32 %compare.local.base756, 4, !dbg !24
  %compare.low758 = icmp ugt i32 %compare.low.base740, %compare.local757, !dbg !24
  %compare.result759 = select i1 %compare.high.equal738, i1 %compare.low758, i1 %compare.high.order739, !dbg !24
  %qr.n730 = select i1 %compare.result759, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base760 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 36) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local761 = or i32 %compare.local.base760, 5, !dbg !24
  %compare.low762 = icmp ugt i32 %compare.low.base740, %compare.local761, !dbg !24
  %compare.result763 = select i1 %compare.high.equal738, i1 %compare.low762, i1 %compare.high.order739, !dbg !24
  %qr.n731 = select i1 %compare.result763, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base764 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 37) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local765 = or i32 %compare.local.base764, 6, !dbg !24
  %compare.low766 = icmp ugt i32 %compare.low.base740, %compare.local765, !dbg !24
  %compare.result767 = select i1 %compare.high.equal738, i1 %compare.low766, i1 %compare.high.order739, !dbg !24
  %qr.n732 = select i1 %compare.result767, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base768 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 38) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local769 = or i32 %compare.local.base768, 7, !dbg !24
  %compare.low770 = icmp ugt i32 %compare.low.base740, %compare.local769, !dbg !24
  %compare.result771 = select i1 %compare.high.equal738, i1 %compare.low770, i1 %compare.high.order739, !dbg !24
  %qr.n733 = select i1 %compare.result771, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base772 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 39) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local773 = or i32 %compare.local.base772, 8, !dbg !24
  %compare.low774 = icmp ugt i32 %compare.low.base740, %compare.local773, !dbg !24
  %compare.result775 = select i1 %compare.high.equal738, i1 %compare.low774, i1 %compare.high.order739, !dbg !24
  %qr.n734 = select i1 %compare.result775, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base776 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 40) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local777 = or i32 %compare.local.base776, 9, !dbg !24
  %compare.low778 = icmp ugt i32 %compare.low.base740, %compare.local777, !dbg !24
  %compare.result779 = select i1 %compare.high.equal738, i1 %compare.low778, i1 %compare.high.order739, !dbg !24
  %qr.n735 = select i1 %compare.result779, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base780 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 41) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local781 = or i32 %compare.local.base780, 10, !dbg !24
  %compare.low782 = icmp ugt i32 %compare.low.base740, %compare.local781, !dbg !24
  %compare.result783 = select i1 %compare.high.equal738, i1 %compare.low782, i1 %compare.high.order739, !dbg !24
  %qr.n736 = select i1 %compare.result783, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base784 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 42) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local785 = or i32 %compare.local.base784, 11, !dbg !24
  %compare.low786 = icmp ugt i32 %compare.low.base740, %compare.local785, !dbg !24
  %compare.result787 = select i1 %compare.high.equal738, i1 %compare.low786, i1 %compare.high.order739, !dbg !24
  %qr.n737 = select i1 %compare.result787, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base788 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 43) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local789 = or i32 %compare.local.base788, 12, !dbg !24
  %compare.low790 = icmp ugt i32 %compare.low.base740, %compare.local789, !dbg !24
  %compare.result791 = select i1 %compare.high.equal738, i1 %compare.low790, i1 %compare.high.order739, !dbg !24
  %qr.n738 = select i1 %compare.result791, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base792 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 44) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local793 = or i32 %compare.local.base792, 13, !dbg !24
  %compare.low794 = icmp ugt i32 %compare.low.base740, %compare.local793, !dbg !24
  %compare.result795 = select i1 %compare.high.equal738, i1 %compare.low794, i1 %compare.high.order739, !dbg !24
  %qr.n739 = select i1 %compare.result795, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base796 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 45) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local797 = or i32 %compare.local.base796, 14, !dbg !24
  %compare.low798 = icmp ugt i32 %compare.low.base740, %compare.local797, !dbg !24
  %compare.result799 = select i1 %compare.high.equal738, i1 %compare.low798, i1 %compare.high.order739, !dbg !24
  %qr.n740 = select i1 %compare.result799, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base800 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 46) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local801 = or i32 %compare.local.base800, 15, !dbg !24
  %compare.low802 = icmp ugt i32 %compare.low.base740, %compare.local801, !dbg !24
  %compare.result803 = select i1 %compare.high.equal738, i1 %compare.low802, i1 %compare.high.order739, !dbg !24
  %qr.n741 = select i1 %compare.result803, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base804 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 47) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local805 = or i32 %compare.local.base804, 64, !dbg !24
  %compare.low806 = icmp ugt i32 %compare.low.base740, %compare.local805, !dbg !24
  %compare.result807 = select i1 %compare.high.equal738, i1 %compare.low806, i1 %compare.high.order739, !dbg !24
  %qr.n742 = select i1 %compare.result807, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base808 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 48) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local809 = or i32 %compare.local.base808, 65, !dbg !24
  %compare.low810 = icmp ugt i32 %compare.low.base740, %compare.local809, !dbg !24
  %compare.result811 = select i1 %compare.high.equal738, i1 %compare.low810, i1 %compare.high.order739, !dbg !24
  %qr.n743 = select i1 %compare.result811, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base812 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 49) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local813 = or i32 %compare.local.base812, 66, !dbg !24
  %compare.low814 = icmp ugt i32 %compare.low.base740, %compare.local813, !dbg !24
  %compare.result815 = select i1 %compare.high.equal738, i1 %compare.low814, i1 %compare.high.order739, !dbg !24
  %qr.n744 = select i1 %compare.result815, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base816 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 50) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local817 = or i32 %compare.local.base816, 67, !dbg !24
  %compare.low818 = icmp ugt i32 %compare.low.base740, %compare.local817, !dbg !24
  %compare.result819 = select i1 %compare.high.equal738, i1 %compare.low818, i1 %compare.high.order739, !dbg !24
  %qr.n745 = select i1 %compare.result819, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base820 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 51) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local821 = or i32 %compare.local.base820, 68, !dbg !24
  %compare.low822 = icmp ugt i32 %compare.low.base740, %compare.local821, !dbg !24
  %compare.result823 = select i1 %compare.high.equal738, i1 %compare.low822, i1 %compare.high.order739, !dbg !24
  %qr.n746 = select i1 %compare.result823, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base824 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 52) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local825 = or i32 %compare.local.base824, 69, !dbg !24
  %compare.low826 = icmp ugt i32 %compare.low.base740, %compare.local825, !dbg !24
  %compare.result827 = select i1 %compare.high.equal738, i1 %compare.low826, i1 %compare.high.order739, !dbg !24
  %qr.n747 = select i1 %compare.result827, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base828 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 53) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local829 = or i32 %compare.local.base828, 70, !dbg !24
  %compare.low830 = icmp ugt i32 %compare.low.base740, %compare.local829, !dbg !24
  %compare.result831 = select i1 %compare.high.equal738, i1 %compare.low830, i1 %compare.high.order739, !dbg !24
  %qr.n748 = select i1 %compare.result831, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base832 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 54) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local833 = or i32 %compare.local.base832, 71, !dbg !24
  %compare.low834 = icmp ugt i32 %compare.low.base740, %compare.local833, !dbg !24
  %compare.result835 = select i1 %compare.high.equal738, i1 %compare.low834, i1 %compare.high.order739, !dbg !24
  %qr.n749 = select i1 %compare.result835, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base836 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 55) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local837 = or i32 %compare.local.base836, 72, !dbg !24
  %compare.low838 = icmp ugt i32 %compare.low.base740, %compare.local837, !dbg !24
  %compare.result839 = select i1 %compare.high.equal738, i1 %compare.low838, i1 %compare.high.order739, !dbg !24
  %qr.n750 = select i1 %compare.result839, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base840 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 56) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local841 = or i32 %compare.local.base840, 73, !dbg !24
  %compare.low842 = icmp ugt i32 %compare.low.base740, %compare.local841, !dbg !24
  %compare.result843 = select i1 %compare.high.equal738, i1 %compare.low842, i1 %compare.high.order739, !dbg !24
  %qr.n751 = select i1 %compare.result843, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base844 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 57) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local845 = or i32 %compare.local.base844, 74, !dbg !24
  %compare.low846 = icmp ugt i32 %compare.low.base740, %compare.local845, !dbg !24
  %compare.result847 = select i1 %compare.high.equal738, i1 %compare.low846, i1 %compare.high.order739, !dbg !24
  %qr.n752 = select i1 %compare.result847, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base848 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 58) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local849 = or i32 %compare.local.base848, 75, !dbg !24
  %compare.low850 = icmp ugt i32 %compare.low.base740, %compare.local849, !dbg !24
  %compare.result851 = select i1 %compare.high.equal738, i1 %compare.low850, i1 %compare.high.order739, !dbg !24
  %qr.n753 = select i1 %compare.result851, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base852 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 59) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local853 = or i32 %compare.local.base852, 76, !dbg !24
  %compare.low854 = icmp ugt i32 %compare.low.base740, %compare.local853, !dbg !24
  %compare.result855 = select i1 %compare.high.equal738, i1 %compare.low854, i1 %compare.high.order739, !dbg !24
  %qr.n754 = select i1 %compare.result855, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base856 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 60) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local857 = or i32 %compare.local.base856, 77, !dbg !24
  %compare.low858 = icmp ugt i32 %compare.low.base740, %compare.local857, !dbg !24
  %compare.result859 = select i1 %compare.high.equal738, i1 %compare.low858, i1 %compare.high.order739, !dbg !24
  %qr.n755 = select i1 %compare.result859, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base860 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 61) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local861 = or i32 %compare.local.base860, 78, !dbg !24
  %compare.low862 = icmp ugt i32 %compare.low.base740, %compare.local861, !dbg !24
  %compare.result863 = select i1 %compare.high.equal738, i1 %compare.low862, i1 %compare.high.order739, !dbg !24
  %qr.n756 = select i1 %compare.result863, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %compare.local.base864 = call i32 asm sideeffect "", "=v,0,i"(i32 %compare.low.base741, i32 62) #9, !dbg !24, !frisk.register_lifetimes !1
  %compare.local865 = or i32 %compare.local.base864, 79, !dbg !24
  %compare.low866 = icmp ugt i32 %compare.low.base740, %compare.local865, !dbg !24
  %compare.result867 = select i1 %compare.high.equal738, i1 %compare.low866, i1 %compare.high.order739, !dbg !24
  %qr.n757 = select i1 %compare.result867, float 0x0000000000000000, float 0xFFF0000000000000, !dbg !25
  %qr.n758 = insertelement <32 x float> poison, float %qr.n691, i64 0, !dbg !13
  %qr.n759 = insertelement <32 x float> %qr.n758, float %qr.n692, i64 1, !dbg !13
  %qr.n760 = insertelement <32 x float> %qr.n759, float %qr.n693, i64 2, !dbg !13
  %qr.n761 = insertelement <32 x float> %qr.n760, float %qr.n694, i64 3, !dbg !13
  %qr.n762 = insertelement <32 x float> %qr.n761, float %qr.n695, i64 4, !dbg !13
  %qr.n763 = insertelement <32 x float> %qr.n762, float %qr.n696, i64 5, !dbg !13
  %qr.n764 = insertelement <32 x float> %qr.n763, float %qr.n697, i64 6, !dbg !13
  %qr.n765 = insertelement <32 x float> %qr.n764, float %qr.n698, i64 7, !dbg !13
  %qr.n766 = insertelement <32 x float> %qr.n765, float %qr.n699, i64 8, !dbg !13
  %qr.n767 = insertelement <32 x float> %qr.n766, float %qr.n700, i64 9, !dbg !13
  %qr.n768 = insertelement <32 x float> %qr.n767, float %qr.n701, i64 10, !dbg !13
  %qr.n769 = insertelement <32 x float> %qr.n768, float %qr.n702, i64 11, !dbg !13
  %qr.n770 = insertelement <32 x float> %qr.n769, float %qr.n703, i64 12, !dbg !13
  %qr.n771 = insertelement <32 x float> %qr.n770, float %qr.n704, i64 13, !dbg !13
  %qr.n772 = insertelement <32 x float> %qr.n771, float %qr.n705, i64 14, !dbg !13
  %qr.n773 = insertelement <32 x float> %qr.n772, float %qr.n706, i64 15, !dbg !13
  %qr.n774 = insertelement <32 x float> %qr.n773, float %qr.n707, i64 16, !dbg !13
  %qr.n775 = insertelement <32 x float> %qr.n774, float %qr.n708, i64 17, !dbg !13
  %qr.n776 = insertelement <32 x float> %qr.n775, float %qr.n709, i64 18, !dbg !13
  %qr.n777 = insertelement <32 x float> %qr.n776, float %qr.n710, i64 19, !dbg !13
  %qr.n778 = insertelement <32 x float> %qr.n777, float %qr.n711, i64 20, !dbg !13
  %qr.n779 = insertelement <32 x float> %qr.n778, float %qr.n712, i64 21, !dbg !13
  %qr.n780 = insertelement <32 x float> %qr.n779, float %qr.n713, i64 22, !dbg !13
  %qr.n781 = insertelement <32 x float> %qr.n780, float %qr.n714, i64 23, !dbg !13
  %qr.n782 = insertelement <32 x float> %qr.n781, float %qr.n715, i64 24, !dbg !13
  %qr.n783 = insertelement <32 x float> %qr.n782, float %qr.n716, i64 25, !dbg !13
  %qr.n784 = insertelement <32 x float> %qr.n783, float %qr.n717, i64 26, !dbg !13
  %qr.n785 = insertelement <32 x float> %qr.n784, float %qr.n718, i64 27, !dbg !13
  %qr.n786 = insertelement <32 x float> %qr.n785, float %qr.n719, i64 28, !dbg !13
  %qr.n787 = insertelement <32 x float> %qr.n786, float %qr.n720, i64 29, !dbg !13
  %qr.n788 = insertelement <32 x float> %qr.n787, float %qr.n721, i64 30, !dbg !13
  %qr.n789 = insertelement <32 x float> %qr.n788, float %qr.n722, i64 31, !dbg !13
  %qr.n790 = insertvalue [2 x <32 x float>] poison, <32 x float> %qr.n789, 0, !dbg !13
  %qr.n791 = insertelement <32 x float> poison, float %qr.n726, i64 0, !dbg !13
  %qr.n792 = insertelement <32 x float> %qr.n791, float %qr.n727, i64 1, !dbg !13
  %qr.n793 = insertelement <32 x float> %qr.n792, float %qr.n728, i64 2, !dbg !13
  %qr.n794 = insertelement <32 x float> %qr.n793, float %qr.n729, i64 3, !dbg !13
  %qr.n795 = insertelement <32 x float> %qr.n794, float %qr.n730, i64 4, !dbg !13
  %qr.n796 = insertelement <32 x float> %qr.n795, float %qr.n731, i64 5, !dbg !13
  %qr.n797 = insertelement <32 x float> %qr.n796, float %qr.n732, i64 6, !dbg !13
  %qr.n798 = insertelement <32 x float> %qr.n797, float %qr.n733, i64 7, !dbg !13
  %qr.n799 = insertelement <32 x float> %qr.n798, float %qr.n734, i64 8, !dbg !13
  %qr.n800 = insertelement <32 x float> %qr.n799, float %qr.n735, i64 9, !dbg !13
  %qr.n801 = insertelement <32 x float> %qr.n800, float %qr.n736, i64 10, !dbg !13
  %qr.n802 = insertelement <32 x float> %qr.n801, float %qr.n737, i64 11, !dbg !13
  %qr.n803 = insertelement <32 x float> %qr.n802, float %qr.n738, i64 12, !dbg !13
  %qr.n804 = insertelement <32 x float> %qr.n803, float %qr.n739, i64 13, !dbg !13
  %qr.n805 = insertelement <32 x float> %qr.n804, float %qr.n740, i64 14, !dbg !13
  %qr.n806 = insertelement <32 x float> %qr.n805, float %qr.n741, i64 15, !dbg !13
  %qr.n807 = insertelement <32 x float> %qr.n806, float %qr.n742, i64 16, !dbg !13
  %qr.n808 = insertelement <32 x float> %qr.n807, float %qr.n743, i64 17, !dbg !13
  %qr.n809 = insertelement <32 x float> %qr.n808, float %qr.n744, i64 18, !dbg !13
  %qr.n810 = insertelement <32 x float> %qr.n809, float %qr.n745, i64 19, !dbg !13
  %qr.n811 = insertelement <32 x float> %qr.n810, float %qr.n746, i64 20, !dbg !13
  %qr.n812 = insertelement <32 x float> %qr.n811, float %qr.n747, i64 21, !dbg !13
  %qr.n813 = insertelement <32 x float> %qr.n812, float %qr.n748, i64 22, !dbg !13
  %qr.n814 = insertelement <32 x float> %qr.n813, float %qr.n749, i64 23, !dbg !13
  %qr.n815 = insertelement <32 x float> %qr.n814, float %qr.n750, i64 24, !dbg !13
  %qr.n816 = insertelement <32 x float> %qr.n815, float %qr.n751, i64 25, !dbg !13
  %qr.n817 = insertelement <32 x float> %qr.n816, float %qr.n752, i64 26, !dbg !13
  %qr.n818 = insertelement <32 x float> %qr.n817, float %qr.n753, i64 27, !dbg !13
  %qr.n819 = insertelement <32 x float> %qr.n818, float %qr.n754, i64 28, !dbg !13
  %qr.n820 = insertelement <32 x float> %qr.n819, float %qr.n755, i64 29, !dbg !13
  %qr.n821 = insertelement <32 x float> %qr.n820, float %qr.n756, i64 30, !dbg !13
  %qr.n822 = insertelement <32 x float> %qr.n821, float %qr.n757, i64 31, !dbg !13
  %qr.n823 = insertvalue [2 x <32 x float>] %qr.n790, <32 x float> %qr.n822, 1, !dbg !13
  br label %qr.n824, !dbg !13

qr.n824:                                              ; preds = %qr.n679, %qr.n680
  %qr.n825 = phi [2 x <32 x float>] [ %qr.n823, %qr.n680 ], [ zeroinitializer, %qr.n679 ], !dbg !13
  br label %qr.n826, !dbg !13

qr.n826:                                              ; preds = %qr.n824
  %qr.n827 = extractvalue [2 x <32 x float>] %qr.n825, 0, !dbg !26
  %qr.n828 = shufflevector <32 x float> %qr.n827, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %qr.n829 = fadd <4 x float> %qr.n616, %qr.n828, !dbg !26
  %qr.n830 = extractelement <4 x float> %qr.n829, i64 0, !dbg !27
  %qr.n831 = call float @llvm.amdgcn.exp2.f32(float %qr.n830), !dbg !27
  %qr.n832 = insertelement <4 x float> poison, float %qr.n831, i64 0, !dbg !29
  %qr.n833 = extractelement <4 x float> %qr.n829, i64 1, !dbg !27
  %qr.n834 = call float @llvm.amdgcn.exp2.f32(float %qr.n833), !dbg !27
  %qr.n835 = insertelement <4 x float> %qr.n832, float %qr.n834, i64 1, !dbg !29
  %qr.n836 = fadd float %qr.n831, %qr.n834, !dbg !28
  %qr.n837 = extractelement <4 x float> %qr.n829, i64 2, !dbg !27
  %qr.n838 = call float @llvm.amdgcn.exp2.f32(float %qr.n837), !dbg !27
  %qr.n839 = insertelement <4 x float> %qr.n835, float %qr.n838, i64 2, !dbg !29
  %qr.n840 = fadd float %qr.n836, %qr.n838, !dbg !28
  %qr.n841 = extractelement <4 x float> %qr.n829, i64 3, !dbg !27
  %qr.n842 = call float @llvm.amdgcn.exp2.f32(float %qr.n841), !dbg !27
  %qr.n843 = insertelement <4 x float> %qr.n839, float %qr.n842, i64 3, !dbg !29
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n843) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n844 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !29
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n844) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n845 = fadd float %qr.n840, %qr.n842, !dbg !28
  %qr.n846 = shufflevector <32 x float> %qr.n827, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %qr.n847 = fadd <4 x float> %qr.n620, %qr.n846, !dbg !26
  %qr.n848 = extractelement <4 x float> %qr.n847, i64 0, !dbg !27
  %qr.n849 = call float @llvm.amdgcn.exp2.f32(float %qr.n848), !dbg !27
  %qr.n850 = insertelement <4 x float> poison, float %qr.n849, i64 0, !dbg !29
  %qr.n851 = fadd float %qr.n845, %qr.n849, !dbg !28
  %qr.n852 = extractelement <4 x float> %qr.n847, i64 1, !dbg !27
  %qr.n853 = call float @llvm.amdgcn.exp2.f32(float %qr.n852), !dbg !27
  %qr.n854 = insertelement <4 x float> %qr.n850, float %qr.n853, i64 1, !dbg !29
  %qr.n855 = fadd float %qr.n851, %qr.n853, !dbg !28
  %qr.n856 = extractelement <4 x float> %qr.n847, i64 2, !dbg !27
  %qr.n857 = call float @llvm.amdgcn.exp2.f32(float %qr.n856), !dbg !27
  %qr.n858 = insertelement <4 x float> %qr.n854, float %qr.n857, i64 2, !dbg !29
  %qr.n859 = fadd float %qr.n855, %qr.n857, !dbg !28
  %qr.n860 = extractelement <4 x float> %qr.n847, i64 3, !dbg !27
  %qr.n861 = call float @llvm.amdgcn.exp2.f32(float %qr.n860), !dbg !27
  %qr.n862 = insertelement <4 x float> %qr.n858, float %qr.n861, i64 3, !dbg !29
  %math.result99 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n862) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n863 = fptrunc <4 x float> %math.result99 to <4 x half>, !dbg !29
  %math.packed100 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n863) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n864 = fadd float %qr.n859, %qr.n861, !dbg !28
  %qr.n865 = shufflevector <32 x float> %qr.n827, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %qr.n866 = fadd <4 x float> %qr.n624, %qr.n865, !dbg !26
  %qr.n867 = extractelement <4 x float> %qr.n866, i64 0, !dbg !27
  %qr.n868 = call float @llvm.amdgcn.exp2.f32(float %qr.n867), !dbg !27
  %qr.n869 = insertelement <4 x float> poison, float %qr.n868, i64 0, !dbg !29
  %qr.n870 = fadd float %qr.n864, %qr.n868, !dbg !28
  %qr.n871 = extractelement <4 x float> %qr.n866, i64 1, !dbg !27
  %qr.n872 = call float @llvm.amdgcn.exp2.f32(float %qr.n871), !dbg !27
  %qr.n873 = insertelement <4 x float> %qr.n869, float %qr.n872, i64 1, !dbg !29
  %qr.n874 = fadd float %qr.n870, %qr.n872, !dbg !28
  %qr.n875 = extractelement <4 x float> %qr.n866, i64 2, !dbg !27
  %qr.n876 = call float @llvm.amdgcn.exp2.f32(float %qr.n875), !dbg !27
  %qr.n877 = insertelement <4 x float> %qr.n873, float %qr.n876, i64 2, !dbg !29
  %qr.n878 = fadd float %qr.n874, %qr.n876, !dbg !28
  %qr.n879 = extractelement <4 x float> %qr.n866, i64 3, !dbg !27
  %qr.n880 = call float @llvm.amdgcn.exp2.f32(float %qr.n879), !dbg !27
  %qr.n881 = insertelement <4 x float> %qr.n877, float %qr.n880, i64 3, !dbg !29
  %math.result101 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n881) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n882 = fptrunc <4 x float> %math.result101 to <4 x half>, !dbg !29
  %math.packed102 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n882) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n883 = fadd float %qr.n878, %qr.n880, !dbg !28
  %qr.n884 = shufflevector <32 x float> %qr.n827, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %qr.n885 = fadd <4 x float> %qr.n628, %qr.n884, !dbg !26
  %qr.n886 = extractelement <4 x float> %qr.n885, i64 0, !dbg !27
  %qr.n887 = call float @llvm.amdgcn.exp2.f32(float %qr.n886), !dbg !27
  %qr.n888 = insertelement <4 x float> poison, float %qr.n887, i64 0, !dbg !29
  %qr.n889 = fadd float %qr.n883, %qr.n887, !dbg !28
  %qr.n890 = extractelement <4 x float> %qr.n885, i64 1, !dbg !27
  %qr.n891 = call float @llvm.amdgcn.exp2.f32(float %qr.n890), !dbg !27
  %qr.n892 = insertelement <4 x float> %qr.n888, float %qr.n891, i64 1, !dbg !29
  %qr.n893 = fadd float %qr.n889, %qr.n891, !dbg !28
  %qr.n894 = extractelement <4 x float> %qr.n885, i64 2, !dbg !27
  %qr.n895 = call float @llvm.amdgcn.exp2.f32(float %qr.n894), !dbg !27
  %qr.n896 = insertelement <4 x float> %qr.n892, float %qr.n895, i64 2, !dbg !29
  %qr.n897 = fadd float %qr.n893, %qr.n895, !dbg !28
  %qr.n898 = extractelement <4 x float> %qr.n885, i64 3, !dbg !27
  %qr.n899 = call float @llvm.amdgcn.exp2.f32(float %qr.n898), !dbg !27
  %qr.n900 = insertelement <4 x float> %qr.n896, float %qr.n899, i64 3, !dbg !29
  %math.result103 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n900) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n901 = fptrunc <4 x float> %math.result103 to <4 x half>, !dbg !29
  %math.packed104 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n901) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n902 = fadd float %qr.n897, %qr.n899, !dbg !28
  %qr.n903 = shufflevector <32 x float> %qr.n827, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %qr.n904 = fadd <4 x float> %qr.n632, %qr.n903, !dbg !26
  %qr.n905 = extractelement <4 x float> %qr.n904, i64 0, !dbg !27
  %qr.n906 = call float @llvm.amdgcn.exp2.f32(float %qr.n905), !dbg !27
  %qr.n907 = insertelement <4 x float> poison, float %qr.n906, i64 0, !dbg !29
  %qr.n908 = fadd float %qr.n902, %qr.n906, !dbg !28
  %qr.n909 = extractelement <4 x float> %qr.n904, i64 1, !dbg !27
  %qr.n910 = call float @llvm.amdgcn.exp2.f32(float %qr.n909), !dbg !27
  %qr.n911 = insertelement <4 x float> %qr.n907, float %qr.n910, i64 1, !dbg !29
  %qr.n912 = fadd float %qr.n908, %qr.n910, !dbg !28
  %qr.n913 = extractelement <4 x float> %qr.n904, i64 2, !dbg !27
  %qr.n914 = call float @llvm.amdgcn.exp2.f32(float %qr.n913), !dbg !27
  %qr.n915 = insertelement <4 x float> %qr.n911, float %qr.n914, i64 2, !dbg !29
  %qr.n916 = fadd float %qr.n912, %qr.n914, !dbg !28
  %qr.n917 = extractelement <4 x float> %qr.n904, i64 3, !dbg !27
  %qr.n918 = call float @llvm.amdgcn.exp2.f32(float %qr.n917), !dbg !27
  %qr.n919 = insertelement <4 x float> %qr.n915, float %qr.n918, i64 3, !dbg !29
  %math.result105 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n919) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n920 = fptrunc <4 x float> %math.result105 to <4 x half>, !dbg !29
  %math.packed106 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n920) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n921 = fadd float %qr.n916, %qr.n918, !dbg !28
  %qr.n922 = shufflevector <32 x float> %qr.n827, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %qr.n923 = fadd <4 x float> %qr.n636, %qr.n922, !dbg !26
  %qr.n924 = extractelement <4 x float> %qr.n923, i64 0, !dbg !27
  %qr.n925 = call float @llvm.amdgcn.exp2.f32(float %qr.n924), !dbg !27
  %qr.n926 = insertelement <4 x float> poison, float %qr.n925, i64 0, !dbg !29
  %qr.n927 = fadd float %qr.n921, %qr.n925, !dbg !28
  %qr.n928 = extractelement <4 x float> %qr.n923, i64 1, !dbg !27
  %qr.n929 = call float @llvm.amdgcn.exp2.f32(float %qr.n928), !dbg !27
  %qr.n930 = insertelement <4 x float> %qr.n926, float %qr.n929, i64 1, !dbg !29
  %qr.n931 = fadd float %qr.n927, %qr.n929, !dbg !28
  %qr.n932 = extractelement <4 x float> %qr.n923, i64 2, !dbg !27
  %qr.n933 = call float @llvm.amdgcn.exp2.f32(float %qr.n932), !dbg !27
  %qr.n934 = insertelement <4 x float> %qr.n930, float %qr.n933, i64 2, !dbg !29
  %qr.n935 = fadd float %qr.n931, %qr.n933, !dbg !28
  %qr.n936 = extractelement <4 x float> %qr.n923, i64 3, !dbg !27
  %qr.n937 = call float @llvm.amdgcn.exp2.f32(float %qr.n936), !dbg !27
  %qr.n938 = insertelement <4 x float> %qr.n934, float %qr.n937, i64 3, !dbg !29
  %math.result107 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n938) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n939 = fptrunc <4 x float> %math.result107 to <4 x half>, !dbg !29
  %math.packed108 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n939) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n940 = fadd float %qr.n935, %qr.n937, !dbg !28
  %qr.n941 = shufflevector <32 x float> %qr.n827, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %qr.n942 = fadd <4 x float> %qr.n640, %qr.n941, !dbg !26
  %qr.n943 = extractelement <4 x float> %qr.n942, i64 0, !dbg !27
  %qr.n944 = call float @llvm.amdgcn.exp2.f32(float %qr.n943), !dbg !27
  %qr.n945 = insertelement <4 x float> poison, float %qr.n944, i64 0, !dbg !29
  %qr.n946 = fadd float %qr.n940, %qr.n944, !dbg !28
  %qr.n947 = extractelement <4 x float> %qr.n942, i64 1, !dbg !27
  %qr.n948 = call float @llvm.amdgcn.exp2.f32(float %qr.n947), !dbg !27
  %qr.n949 = insertelement <4 x float> %qr.n945, float %qr.n948, i64 1, !dbg !29
  %qr.n950 = fadd float %qr.n946, %qr.n948, !dbg !28
  %qr.n951 = extractelement <4 x float> %qr.n942, i64 2, !dbg !27
  %qr.n952 = call float @llvm.amdgcn.exp2.f32(float %qr.n951), !dbg !27
  %qr.n953 = insertelement <4 x float> %qr.n949, float %qr.n952, i64 2, !dbg !29
  %qr.n954 = fadd float %qr.n950, %qr.n952, !dbg !28
  %qr.n955 = extractelement <4 x float> %qr.n942, i64 3, !dbg !27
  %qr.n956 = call float @llvm.amdgcn.exp2.f32(float %qr.n955), !dbg !27
  %qr.n957 = insertelement <4 x float> %qr.n953, float %qr.n956, i64 3, !dbg !29
  %math.result109 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n957) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n958 = fptrunc <4 x float> %math.result109 to <4 x half>, !dbg !29
  %math.packed110 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n958) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n959 = fadd float %qr.n954, %qr.n956, !dbg !28
  %qr.n960 = shufflevector <32 x float> %qr.n827, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %qr.n961 = fadd <4 x float> %qr.n644, %qr.n960, !dbg !26
  %qr.n962 = extractelement <4 x float> %qr.n961, i64 0, !dbg !27
  %qr.n963 = call float @llvm.amdgcn.exp2.f32(float %qr.n962), !dbg !27
  %qr.n964 = insertelement <4 x float> poison, float %qr.n963, i64 0, !dbg !29
  %qr.n965 = fadd float %qr.n959, %qr.n963, !dbg !28
  %qr.n966 = extractelement <4 x float> %qr.n961, i64 1, !dbg !27
  %qr.n967 = call float @llvm.amdgcn.exp2.f32(float %qr.n966), !dbg !27
  %qr.n968 = insertelement <4 x float> %qr.n964, float %qr.n967, i64 1, !dbg !29
  %qr.n969 = fadd float %qr.n965, %qr.n967, !dbg !28
  %qr.n970 = extractelement <4 x float> %qr.n961, i64 2, !dbg !27
  %qr.n971 = call float @llvm.amdgcn.exp2.f32(float %qr.n970), !dbg !27
  %qr.n972 = insertelement <4 x float> %qr.n968, float %qr.n971, i64 2, !dbg !29
  %qr.n973 = fadd float %qr.n969, %qr.n971, !dbg !28
  %qr.n974 = extractelement <4 x float> %qr.n961, i64 3, !dbg !27
  %qr.n975 = call float @llvm.amdgcn.exp2.f32(float %qr.n974), !dbg !27
  %qr.n976 = insertelement <4 x float> %qr.n972, float %qr.n975, i64 3, !dbg !29
  %math.result111 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n976) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n977 = fptrunc <4 x float> %math.result111 to <4 x half>, !dbg !29
  %math.packed112 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n977) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n978 = fadd float %qr.n973, %qr.n975, !dbg !28
  %qr.n979 = extractvalue [2 x <32 x float>] %qr.n825, 1, !dbg !26
  %qr.n980 = shufflevector <32 x float> %qr.n979, <32 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !26
  %qr.n981 = fadd <4 x float> %qr.n648, %qr.n980, !dbg !26
  %qr.n982 = extractelement <4 x float> %qr.n981, i64 0, !dbg !27
  %qr.n983 = call float @llvm.amdgcn.exp2.f32(float %qr.n982), !dbg !27
  %qr.n984 = insertelement <4 x float> poison, float %qr.n983, i64 0, !dbg !29
  %qr.n985 = extractelement <4 x float> %qr.n981, i64 1, !dbg !27
  %qr.n986 = call float @llvm.amdgcn.exp2.f32(float %qr.n985), !dbg !27
  %qr.n987 = insertelement <4 x float> %qr.n984, float %qr.n986, i64 1, !dbg !29
  %qr.n988 = fadd float %qr.n983, %qr.n986, !dbg !28
  %qr.n989 = extractelement <4 x float> %qr.n981, i64 2, !dbg !27
  %qr.n990 = call float @llvm.amdgcn.exp2.f32(float %qr.n989), !dbg !27
  %qr.n991 = insertelement <4 x float> %qr.n987, float %qr.n990, i64 2, !dbg !29
  %qr.n992 = fadd float %qr.n988, %qr.n990, !dbg !28
  %qr.n993 = extractelement <4 x float> %qr.n981, i64 3, !dbg !27
  %qr.n994 = call float @llvm.amdgcn.exp2.f32(float %qr.n993), !dbg !27
  %qr.n995 = insertelement <4 x float> %qr.n991, float %qr.n994, i64 3, !dbg !29
  %math.result113 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n995) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n996 = fptrunc <4 x float> %math.result113 to <4 x half>, !dbg !29
  %math.packed114 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n996) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n997 = fadd float %qr.n992, %qr.n994, !dbg !28
  %qr.n998 = shufflevector <32 x float> %qr.n979, <32 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !26
  %qr.n999 = fadd <4 x float> %qr.n652, %qr.n998, !dbg !26
  %qr.n1000 = extractelement <4 x float> %qr.n999, i64 0, !dbg !27
  %qr.n1001 = call float @llvm.amdgcn.exp2.f32(float %qr.n1000), !dbg !27
  %qr.n1002 = insertelement <4 x float> poison, float %qr.n1001, i64 0, !dbg !29
  %qr.n1003 = fadd float %qr.n997, %qr.n1001, !dbg !28
  %qr.n1004 = extractelement <4 x float> %qr.n999, i64 1, !dbg !27
  %qr.n1005 = call float @llvm.amdgcn.exp2.f32(float %qr.n1004), !dbg !27
  %qr.n1006 = insertelement <4 x float> %qr.n1002, float %qr.n1005, i64 1, !dbg !29
  %qr.n1007 = fadd float %qr.n1003, %qr.n1005, !dbg !28
  %qr.n1008 = extractelement <4 x float> %qr.n999, i64 2, !dbg !27
  %qr.n1009 = call float @llvm.amdgcn.exp2.f32(float %qr.n1008), !dbg !27
  %qr.n1010 = insertelement <4 x float> %qr.n1006, float %qr.n1009, i64 2, !dbg !29
  %qr.n1011 = fadd float %qr.n1007, %qr.n1009, !dbg !28
  %qr.n1012 = extractelement <4 x float> %qr.n999, i64 3, !dbg !27
  %qr.n1013 = call float @llvm.amdgcn.exp2.f32(float %qr.n1012), !dbg !27
  %qr.n1014 = insertelement <4 x float> %qr.n1010, float %qr.n1013, i64 3, !dbg !29
  %math.result115 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n1014) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n1015 = fptrunc <4 x float> %math.result115 to <4 x half>, !dbg !29
  %math.packed116 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n1015) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n1016 = fadd float %qr.n1011, %qr.n1013, !dbg !28
  %qr.n1017 = shufflevector <32 x float> %qr.n979, <32 x float> poison, <4 x i32> <i32 8, i32 9, i32 10, i32 11>, !dbg !26
  %qr.n1018 = fadd <4 x float> %qr.n656, %qr.n1017, !dbg !26
  %qr.n1019 = extractelement <4 x float> %qr.n1018, i64 0, !dbg !27
  %qr.n1020 = call float @llvm.amdgcn.exp2.f32(float %qr.n1019), !dbg !27
  %qr.n1021 = insertelement <4 x float> poison, float %qr.n1020, i64 0, !dbg !29
  %qr.n1022 = fadd float %qr.n1016, %qr.n1020, !dbg !28
  %qr.n1023 = extractelement <4 x float> %qr.n1018, i64 1, !dbg !27
  %qr.n1024 = call float @llvm.amdgcn.exp2.f32(float %qr.n1023), !dbg !27
  %qr.n1025 = insertelement <4 x float> %qr.n1021, float %qr.n1024, i64 1, !dbg !29
  %qr.n1026 = fadd float %qr.n1022, %qr.n1024, !dbg !28
  %qr.n1027 = extractelement <4 x float> %qr.n1018, i64 2, !dbg !27
  %qr.n1028 = call float @llvm.amdgcn.exp2.f32(float %qr.n1027), !dbg !27
  %qr.n1029 = insertelement <4 x float> %qr.n1025, float %qr.n1028, i64 2, !dbg !29
  %qr.n1030 = fadd float %qr.n1026, %qr.n1028, !dbg !28
  %qr.n1031 = extractelement <4 x float> %qr.n1018, i64 3, !dbg !27
  %qr.n1032 = call float @llvm.amdgcn.exp2.f32(float %qr.n1031), !dbg !27
  %qr.n1033 = insertelement <4 x float> %qr.n1029, float %qr.n1032, i64 3, !dbg !29
  %math.result117 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n1033) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n1034 = fptrunc <4 x float> %math.result117 to <4 x half>, !dbg !29
  %math.packed118 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n1034) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n1035 = fadd float %qr.n1030, %qr.n1032, !dbg !28
  %qr.n1036 = shufflevector <32 x float> %qr.n979, <32 x float> poison, <4 x i32> <i32 12, i32 13, i32 14, i32 15>, !dbg !26
  %qr.n1037 = fadd <4 x float> %qr.n660, %qr.n1036, !dbg !26
  %qr.n1038 = extractelement <4 x float> %qr.n1037, i64 0, !dbg !27
  %qr.n1039 = call float @llvm.amdgcn.exp2.f32(float %qr.n1038), !dbg !27
  %qr.n1040 = insertelement <4 x float> poison, float %qr.n1039, i64 0, !dbg !29
  %qr.n1041 = fadd float %qr.n1035, %qr.n1039, !dbg !28
  %qr.n1042 = extractelement <4 x float> %qr.n1037, i64 1, !dbg !27
  %qr.n1043 = call float @llvm.amdgcn.exp2.f32(float %qr.n1042), !dbg !27
  %qr.n1044 = insertelement <4 x float> %qr.n1040, float %qr.n1043, i64 1, !dbg !29
  %qr.n1045 = fadd float %qr.n1041, %qr.n1043, !dbg !28
  %qr.n1046 = extractelement <4 x float> %qr.n1037, i64 2, !dbg !27
  %qr.n1047 = call float @llvm.amdgcn.exp2.f32(float %qr.n1046), !dbg !27
  %qr.n1048 = insertelement <4 x float> %qr.n1044, float %qr.n1047, i64 2, !dbg !29
  %qr.n1049 = fadd float %qr.n1045, %qr.n1047, !dbg !28
  %qr.n1050 = extractelement <4 x float> %qr.n1037, i64 3, !dbg !27
  %qr.n1051 = call float @llvm.amdgcn.exp2.f32(float %qr.n1050), !dbg !27
  %qr.n1052 = insertelement <4 x float> %qr.n1048, float %qr.n1051, i64 3, !dbg !29
  %math.result119 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n1052) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n1053 = fptrunc <4 x float> %math.result119 to <4 x half>, !dbg !29
  %math.packed120 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n1053) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n1054 = fadd float %qr.n1049, %qr.n1051, !dbg !28
  %qr.n1055 = shufflevector <32 x float> %qr.n979, <32 x float> poison, <4 x i32> <i32 16, i32 17, i32 18, i32 19>, !dbg !26
  %qr.n1056 = fadd <4 x float> %qr.n664, %qr.n1055, !dbg !26
  %qr.n1057 = extractelement <4 x float> %qr.n1056, i64 0, !dbg !27
  %qr.n1058 = call float @llvm.amdgcn.exp2.f32(float %qr.n1057), !dbg !27
  %qr.n1059 = insertelement <4 x float> poison, float %qr.n1058, i64 0, !dbg !29
  %qr.n1060 = fadd float %qr.n1054, %qr.n1058, !dbg !28
  %qr.n1061 = extractelement <4 x float> %qr.n1056, i64 1, !dbg !27
  %qr.n1062 = call float @llvm.amdgcn.exp2.f32(float %qr.n1061), !dbg !27
  %qr.n1063 = insertelement <4 x float> %qr.n1059, float %qr.n1062, i64 1, !dbg !29
  %qr.n1064 = fadd float %qr.n1060, %qr.n1062, !dbg !28
  %qr.n1065 = extractelement <4 x float> %qr.n1056, i64 2, !dbg !27
  %qr.n1066 = call float @llvm.amdgcn.exp2.f32(float %qr.n1065), !dbg !27
  %qr.n1067 = insertelement <4 x float> %qr.n1063, float %qr.n1066, i64 2, !dbg !29
  %qr.n1068 = fadd float %qr.n1064, %qr.n1066, !dbg !28
  %qr.n1069 = extractelement <4 x float> %qr.n1056, i64 3, !dbg !27
  %qr.n1070 = call float @llvm.amdgcn.exp2.f32(float %qr.n1069), !dbg !27
  %qr.n1071 = insertelement <4 x float> %qr.n1067, float %qr.n1070, i64 3, !dbg !29
  %math.result121 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n1071) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n1072 = fptrunc <4 x float> %math.result121 to <4 x half>, !dbg !29
  %math.packed122 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n1072) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n1073 = fadd float %qr.n1068, %qr.n1070, !dbg !28
  %qr.n1074 = shufflevector <32 x float> %qr.n979, <32 x float> poison, <4 x i32> <i32 20, i32 21, i32 22, i32 23>, !dbg !26
  %qr.n1075 = fadd <4 x float> %qr.n668, %qr.n1074, !dbg !26
  %qr.n1076 = extractelement <4 x float> %qr.n1075, i64 0, !dbg !27
  %qr.n1077 = call float @llvm.amdgcn.exp2.f32(float %qr.n1076), !dbg !27
  %qr.n1078 = insertelement <4 x float> poison, float %qr.n1077, i64 0, !dbg !29
  %qr.n1079 = fadd float %qr.n1073, %qr.n1077, !dbg !28
  %qr.n1080 = extractelement <4 x float> %qr.n1075, i64 1, !dbg !27
  %qr.n1081 = call float @llvm.amdgcn.exp2.f32(float %qr.n1080), !dbg !27
  %qr.n1082 = insertelement <4 x float> %qr.n1078, float %qr.n1081, i64 1, !dbg !29
  %qr.n1083 = fadd float %qr.n1079, %qr.n1081, !dbg !28
  %qr.n1084 = extractelement <4 x float> %qr.n1075, i64 2, !dbg !27
  %qr.n1085 = call float @llvm.amdgcn.exp2.f32(float %qr.n1084), !dbg !27
  %qr.n1086 = insertelement <4 x float> %qr.n1082, float %qr.n1085, i64 2, !dbg !29
  %qr.n1087 = fadd float %qr.n1083, %qr.n1085, !dbg !28
  %qr.n1088 = extractelement <4 x float> %qr.n1075, i64 3, !dbg !27
  %qr.n1089 = call float @llvm.amdgcn.exp2.f32(float %qr.n1088), !dbg !27
  %qr.n1090 = insertelement <4 x float> %qr.n1086, float %qr.n1089, i64 3, !dbg !29
  %math.result123 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n1090) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n1091 = fptrunc <4 x float> %math.result123 to <4 x half>, !dbg !29
  %math.packed124 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n1091) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n1092 = fadd float %qr.n1087, %qr.n1089, !dbg !28
  %qr.n1093 = shufflevector <32 x float> %qr.n979, <32 x float> poison, <4 x i32> <i32 24, i32 25, i32 26, i32 27>, !dbg !26
  %qr.n1094 = fadd <4 x float> %qr.n672, %qr.n1093, !dbg !26
  %qr.n1095 = extractelement <4 x float> %qr.n1094, i64 0, !dbg !27
  %qr.n1096 = call float @llvm.amdgcn.exp2.f32(float %qr.n1095), !dbg !27
  %qr.n1097 = insertelement <4 x float> poison, float %qr.n1096, i64 0, !dbg !29
  %qr.n1098 = fadd float %qr.n1092, %qr.n1096, !dbg !28
  %qr.n1099 = extractelement <4 x float> %qr.n1094, i64 1, !dbg !27
  %qr.n1100 = call float @llvm.amdgcn.exp2.f32(float %qr.n1099), !dbg !27
  %qr.n1101 = insertelement <4 x float> %qr.n1097, float %qr.n1100, i64 1, !dbg !29
  %qr.n1102 = fadd float %qr.n1098, %qr.n1100, !dbg !28
  %qr.n1103 = extractelement <4 x float> %qr.n1094, i64 2, !dbg !27
  %qr.n1104 = call float @llvm.amdgcn.exp2.f32(float %qr.n1103), !dbg !27
  %qr.n1105 = insertelement <4 x float> %qr.n1101, float %qr.n1104, i64 2, !dbg !29
  %qr.n1106 = fadd float %qr.n1102, %qr.n1104, !dbg !28
  %qr.n1107 = extractelement <4 x float> %qr.n1094, i64 3, !dbg !27
  %qr.n1108 = call float @llvm.amdgcn.exp2.f32(float %qr.n1107), !dbg !27
  %qr.n1109 = insertelement <4 x float> %qr.n1105, float %qr.n1108, i64 3, !dbg !29
  %math.result125 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n1109) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n1110 = fptrunc <4 x float> %math.result125 to <4 x half>, !dbg !29
  %math.packed126 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n1110) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n1111 = fadd float %qr.n1106, %qr.n1108, !dbg !28
  %qr.n1112 = shufflevector <32 x float> %qr.n979, <32 x float> poison, <4 x i32> <i32 28, i32 29, i32 30, i32 31>, !dbg !26
  %qr.n1113 = fadd <4 x float> %qr.n677, %qr.n1112, !dbg !26
  %qr.n1114 = extractelement <4 x float> %qr.n1113, i64 0, !dbg !27
  %qr.n1115 = call float @llvm.amdgcn.exp2.f32(float %qr.n1114), !dbg !27
  %qr.n1116 = insertelement <4 x float> poison, float %qr.n1115, i64 0, !dbg !29
  %qr.n1117 = fadd float %qr.n1111, %qr.n1115, !dbg !28
  %qr.n1118 = extractelement <4 x float> %qr.n1113, i64 1, !dbg !27
  %qr.n1119 = call float @llvm.amdgcn.exp2.f32(float %qr.n1118), !dbg !27
  %qr.n1120 = insertelement <4 x float> %qr.n1116, float %qr.n1119, i64 1, !dbg !29
  %qr.n1121 = fadd float %qr.n1117, %qr.n1119, !dbg !28
  %qr.n1122 = extractelement <4 x float> %qr.n1113, i64 2, !dbg !27
  %qr.n1123 = call float @llvm.amdgcn.exp2.f32(float %qr.n1122), !dbg !27
  %qr.n1124 = insertelement <4 x float> %qr.n1120, float %qr.n1123, i64 2, !dbg !29
  %qr.n1125 = fadd float %qr.n1121, %qr.n1123, !dbg !28
  %qr.n1126 = extractelement <4 x float> %qr.n1113, i64 3, !dbg !27
  %qr.n1127 = call float @llvm.amdgcn.exp2.f32(float %qr.n1126), !dbg !27
  %qr.n1128 = insertelement <4 x float> %qr.n1124, float %qr.n1127, i64 3, !dbg !29
  %math.result127 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %qr.n1128) #9, !dbg !29, !frisk.register_lifetimes !1
  %qr.n1129 = fptrunc <4 x float> %math.result127 to <4 x half>, !dbg !29
  %math.packed128 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %qr.n1129) #9, !dbg !28, !frisk.register_lifetimes !1
  %qr.n1130 = fadd float %qr.n1125, %qr.n1127, !dbg !28
  %qr.n1131 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !28
  %qr.n1132 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %qr.n1131), !dbg !28
  %qr.n1133 = add nuw nsw i32 %qr.n1132, 64, !dbg !28
  %qr.n1134 = xor i32 %qr.n1132, 16, !dbg !28
  %qr.n1135 = and i32 %qr.n1133, 192, !dbg !28
  %qr.n1136 = icmp samesign ult i32 %qr.n1134, %qr.n1135, !dbg !28
  %qr.n1137 = select i1 %qr.n1136, i32 %qr.n1134, i32 %qr.n1132, !dbg !28
  %qr.n1138 = shl nuw nsw i32 %qr.n1137, 2, !dbg !28
  %qr.n1139 = bitcast float %qr.n978 to i32, !dbg !28
  %qr.n1140 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1138, i32 %qr.n1139), !dbg !28
  %qr.n1141 = bitcast i32 %qr.n1140 to float, !dbg !28
  %qr.n1142 = fadd float %qr.n978, %qr.n1141, !dbg !28
  %qr.n1143 = xor i32 %qr.n1132, 32, !dbg !28
  %qr.n1144 = icmp samesign ult i32 %qr.n1143, %qr.n1135, !dbg !28
  %qr.n1145 = select i1 %qr.n1144, i32 %qr.n1143, i32 %qr.n1132, !dbg !28
  %qr.n1146 = shl nuw nsw i32 %qr.n1145, 2, !dbg !28
  %qr.n1147 = bitcast float %qr.n1142 to i32, !dbg !28
  %qr.n1148 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1146, i32 %qr.n1147), !dbg !28
  %qr.n1149 = bitcast i32 %qr.n1148 to float, !dbg !28
  %qr.n1150 = fadd float %qr.n1142, %qr.n1149, !dbg !28
  %qr.n1151 = extractvalue [2 x <1 x float>] %qr.n43, 0, !dbg !28
  %qr.n1152 = extractelement <1 x float> %qr.n1151, i64 0, !dbg !28
  %qr.n1153 = fadd float %qr.n1150, %qr.n1152, !dbg !28
  %qr.n1154 = insertelement <1 x float> poison, float %qr.n1153, i64 0, !dbg !15
  %qr.n1155 = insertvalue [2 x <1 x float>] poison, <1 x float> %qr.n1154, 0, !dbg !15
  %qr.n1156 = bitcast float %qr.n1130 to i32, !dbg !28
  %qr.n1157 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1138, i32 %qr.n1156), !dbg !28
  %qr.n1158 = bitcast i32 %qr.n1157 to float, !dbg !28
  %qr.n1159 = fadd float %qr.n1130, %qr.n1158, !dbg !28
  %qr.n1160 = bitcast float %qr.n1159 to i32, !dbg !28
  %qr.n1161 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1146, i32 %qr.n1160), !dbg !28
  %qr.n1162 = bitcast i32 %qr.n1161 to float, !dbg !28
  %qr.n1163 = fadd float %qr.n1159, %qr.n1162, !dbg !28
  %qr.n1164 = extractvalue [2 x <1 x float>] %qr.n43, 1, !dbg !28
  %qr.n1165 = extractelement <1 x float> %qr.n1164, i64 0, !dbg !28
  %qr.n1166 = fadd float %qr.n1163, %qr.n1165, !dbg !28
  %qr.n1167 = insertelement <1 x float> poison, float %qr.n1166, i64 0, !dbg !15
  %qr.n1168 = insertvalue [2 x <1 x float>] %qr.n1155, <1 x float> %qr.n1167, 1, !dbg !15
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %address.thread319 = call i32 asm sideeffect "", "=v,0"(i32 %qr.n24) #9, !dbg !14, !range !33, !frisk.register_lifetimes !1
  %address.local320 = shl nuw nsw i32 %address.thread319, 3, !dbg !14
  %address.local321 = and i32 %address.local320, 120, !dbg !14
  %address.local322 = shl nuw nsw i32 %address.thread319, 5, !dbg !14
  %address.local323 = and i32 %address.local322, 1536, !dbg !14
  %address.local324 = or disjoint i32 %address.local321, %address.local323, !dbg !14
  %address.local325 = zext nneg i32 %address.local324 to i64, !dbg !14
  %address.local326 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local325, !dbg !14
  %qr.n1169 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %qr.n1170 = bitcast <4 x i32> %qr.n1169 to <8 x half>, !dbg !14
  %address.local327 = zext nneg i32 %address.thread319 to i64, !dbg !12
  %address.local328 = and i64 %address.local327, 60, !dbg !14
  %address.local329 = add nuw nsw i64 %address.local328, %address.local327, !dbg !14
  %address.local330 = shl nuw nsw i64 %address.local329, 3, !dbg !14
  %address.local331 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local332 = and i64 %address.local327, 63, !dbg !12
  %address.local333 = lshr i64 %address.local332, 4, !dbg !14
  %address.local334 = mul nuw nsw i64 %address.local333, 384, !dbg !14
  %address.local335 = getelementptr [2 x i8], ptr addrspace(3) %address.local331, i64 %address.local334, !dbg !14
  %address.local336 = lshr i64 %address.local327, 2, !dbg !17
  %address.local337 = mul nsw i64 %address.local336, -32, !dbg !14
  %address.local338 = getelementptr [2 x i8], ptr addrspace(3) %address.local335, i64 %address.local337, !dbg !14
  %address.local339 = getelementptr i8, ptr addrspace(3) %address.local338, i64 256, !dbg !14
  %qr.n1171 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #0, !dbg !14
  %qr.n1172 = bitcast <4 x i32> %qr.n1171 to <8 x half>, !dbg !14
  %address.local340 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local341 = getelementptr [2 x i8], ptr addrspace(3) %address.local340, i64 %address.local334, !dbg !14
  %address.local342 = getelementptr [2 x i8], ptr addrspace(3) %address.local341, i64 %address.local337, !dbg !14
  %address.local343 = getelementptr i8, ptr addrspace(3) %address.local342, i64 512, !dbg !14
  %qr.n1173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !14
  %qr.n1174 = bitcast <4 x i32> %qr.n1173 to <8 x half>, !dbg !14
  %address.local344 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local345 = getelementptr [2 x i8], ptr addrspace(3) %address.local344, i64 %address.local334, !dbg !14
  %address.local346 = getelementptr [2 x i8], ptr addrspace(3) %address.local345, i64 %address.local337, !dbg !14
  %address.local347 = getelementptr i8, ptr addrspace(3) %address.local346, i64 768, !dbg !14
  %qr.n1175 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #0, !dbg !14
  %qr.n1176 = bitcast <4 x i32> %qr.n1175 to <8 x half>, !dbg !14
  %address.local348 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local325, !dbg !14
  %address.local349 = getelementptr i8, ptr addrspace(3) %address.local348, i64 4096, !dbg !14
  %qr.n1177 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !14
  %qr.n1178 = bitcast <4 x i32> %qr.n1177 to <8 x half>, !dbg !14
  %address.local350 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local351 = getelementptr [2 x i8], ptr addrspace(3) %address.local350, i64 %address.local334, !dbg !14
  %address.local352 = getelementptr [2 x i8], ptr addrspace(3) %address.local351, i64 %address.local337, !dbg !14
  %address.local353 = getelementptr i8, ptr addrspace(3) %address.local352, i64 4352, !dbg !14
  %qr.n1179 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #0, !dbg !14
  %qr.n1180 = bitcast <4 x i32> %qr.n1179 to <8 x half>, !dbg !14
  %address.local354 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local355 = getelementptr [2 x i8], ptr addrspace(3) %address.local354, i64 %address.local334, !dbg !14
  %address.local356 = getelementptr [2 x i8], ptr addrspace(3) %address.local355, i64 %address.local337, !dbg !14
  %address.local357 = getelementptr i8, ptr addrspace(3) %address.local356, i64 4608, !dbg !14
  %qr.n1181 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #0, !dbg !14
  %qr.n1182 = bitcast <4 x i32> %qr.n1181 to <8 x half>, !dbg !14
  %address.local358 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local330, !dbg !14
  %address.local359 = getelementptr [2 x i8], ptr addrspace(3) %address.local358, i64 %address.local334, !dbg !14
  %address.local360 = getelementptr [2 x i8], ptr addrspace(3) %address.local359, i64 %address.local337, !dbg !14
  %address.local361 = getelementptr i8, ptr addrspace(3) %address.local360, i64 4864, !dbg !14
  %qr.n1183 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %qr.n1184 = bitcast <4 x i32> %qr.n1183 to <8 x half>, !dbg !14
  %qr.n1185 = add nsw i64 %qr.n79, %qr.n10, !dbg !18
  %qr.n1186 = shl i64 %qr.n1185, 19, !dbg !18
  %qr.n1187 = and i64 %qr.n1186, 16252928, !dbg !18
  %qr.n1188 = add i64 %qr.n1187, %qr.n85, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %qr.n1189 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %qr.n1188, !dbg !18
  %qr.n1190 = getelementptr i8, ptr addrspace(1) %qr.n1189, i64 64, !dbg !18
  %address.local362 = shl nuw nsw i64 %address.local327, 3, !dbg !11
  %address.local363 = and i64 %address.local362, 1536, !dbg !11
  %address.local364 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local363, !dbg !11
  %address.local365 = ptrtoint ptr addrspace(3) %address.local364 to i64, !dbg !11
  %address.local366 = trunc i64 %address.local365 to i32, !dbg !11
  %qr.n1191 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local366), !dbg !18
  %qr.n1192 = zext i32 %qr.n1191 to i64, !dbg !18
  %qr.n1193 = inttoptr i64 %qr.n1192 to ptr addrspace(3), !dbg !18
  %address.local367 = shl nuw nsw i64 %address.local327, 5, !dbg !18
  %address.local368 = and i64 %address.local367, 8064, !dbg !18
  %address.local369 = shl nuw nsw i64 %address.local327, 3, !dbg !18
  %address.local370 = and i64 %address.local369, 24, !dbg !18
  %address.local371 = or disjoint i64 %address.local368, %address.local370, !dbg !18
  %address.local372 = getelementptr [2 x i8], ptr addrspace(1) %qr.n1190, i64 %address.local371, !dbg !18
  %qr.n1194 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %qr.n1195 = ptrtoint ptr addrspace(1) %address.local372 to i64, !dbg !18
  %qr.n1196 = sub i64 %qr.n1195, %qr.n1194, !dbg !18
  %dma.buffer.offset880 = trunc i64 %qr.n1196 to i32, !dbg !18
  %qr.n1197 = trunc i64 %qr.n1194 to i32, !dbg !18
  %qr.n1198 = insertelement <4 x i32> poison, i32 %qr.n1197, i64 0, !dbg !18
  %qr.n1199 = lshr i64 %qr.n1194, 32, !dbg !18
  %qr.n1200 = trunc i64 %qr.n1199 to i32, !dbg !18
  %qr.n1201 = insertelement <4 x i32> %qr.n1198, i32 %qr.n1200, i64 1, !dbg !18
  %qr.n1202 = insertelement <4 x i32> %qr.n1201, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc881 = insertelement <4 x i32> %qr.n1202, i32 131072, i64 3, !dbg !18
  %qr.n1203 = ptrtoint ptr addrspace(3) %qr.n1193 to i32, !dbg !18
  %dma.buffer.lds882 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n1203), !dbg !18
  %qr.n1204 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds882, i32 %dma.buffer.offset880, <4 x i32> %dma.buffer.rsrc881) #10, !dbg !18
  %address.local373 = or disjoint i64 %address.local327, 256, !dbg !11
  %address.local374 = shl nuw nsw i64 %address.local373, 3, !dbg !11
  %address.local375 = and i64 %address.local374, 3584, !dbg !11
  %address.local376 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local375, !dbg !11
  %address.local377 = ptrtoint ptr addrspace(3) %address.local376 to i64, !dbg !11
  %address.local378 = trunc i64 %address.local377 to i32, !dbg !11
  %qr.n1205 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local378), !dbg !18
  %qr.n1206 = zext i32 %qr.n1205 to i64, !dbg !18
  %qr.n1207 = inttoptr i64 %qr.n1206 to ptr addrspace(3), !dbg !18
  %address.local379 = shl nuw nsw i64 %address.local373, 5, !dbg !18
  %address.local380 = and i64 %address.local379, 16256, !dbg !18
  %address.local381 = shl nuw nsw i64 %address.local327, 3, !dbg !18
  %address.local382 = and i64 %address.local381, 24, !dbg !18
  %address.local383 = or disjoint i64 %address.local380, %address.local382, !dbg !18
  %address.local384 = getelementptr [2 x i8], ptr addrspace(1) %qr.n1190, i64 %address.local383, !dbg !18
  %qr.n1208 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %qr.n1209 = ptrtoint ptr addrspace(1) %address.local384 to i64, !dbg !18
  %qr.n1210 = sub i64 %qr.n1209, %qr.n1208, !dbg !18
  %dma.buffer.offset877 = trunc i64 %qr.n1210 to i32, !dbg !18
  %qr.n1211 = trunc i64 %qr.n1208 to i32, !dbg !18
  %qr.n1212 = insertelement <4 x i32> poison, i32 %qr.n1211, i64 0, !dbg !18
  %qr.n1213 = lshr i64 %qr.n1208, 32, !dbg !18
  %qr.n1214 = trunc i64 %qr.n1213 to i32, !dbg !18
  %qr.n1215 = insertelement <4 x i32> %qr.n1212, i32 %qr.n1214, i64 1, !dbg !18
  %qr.n1216 = insertelement <4 x i32> %qr.n1215, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc878 = insertelement <4 x i32> %qr.n1216, i32 131072, i64 3, !dbg !18
  %qr.n1217 = ptrtoint ptr addrspace(3) %qr.n1207 to i32, !dbg !18
  %dma.buffer.lds879 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n1217), !dbg !18
  %qr.n1218 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds879, i32 %dma.buffer.offset877, <4 x i32> %dma.buffer.rsrc878) #10, !dbg !18
  %qr.n1219 = shufflevector <8 x half> %qr.n1170, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1220 = extractvalue [2 x <8 x float>] %qr.n42, 0, !dbg !30
  %qr.n1221 = shufflevector <8 x float> %qr.n1220, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %qr.n1222 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %qr.n1219, <4 x float> %qr.n1221) #9, !dbg !14
  %qr.n1223 = shufflevector <8 x half> %qr.n1172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %qr.n1223, <4 x float> %qr.n1222) #9, !dbg !14
  %qr.n1225 = shufflevector <8 x half> %qr.n1174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %qr.n1225, <4 x float> %qr.n1224) #9, !dbg !14
  %qr.n1227 = shufflevector <8 x half> %qr.n1176, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1228 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %qr.n1227, <4 x float> %qr.n1226) #9, !dbg !14
  %qr.n1229 = shufflevector <8 x half> %qr.n1178, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %qr.n1229, <4 x float> %qr.n1228) #9, !dbg !14
  %qr.n1231 = shufflevector <8 x half> %qr.n1180, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %qr.n1231, <4 x float> %qr.n1230) #9, !dbg !14
  %qr.n1233 = shufflevector <8 x half> %qr.n1182, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1234 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %qr.n1233, <4 x float> %qr.n1232) #9, !dbg !14
  %qr.n1235 = shufflevector <8 x half> %qr.n1184, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %qr.n1235, <4 x float> %qr.n1234) #9, !dbg !14
  %qr.n1237 = shufflevector <8 x half> %qr.n1170, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1238 = shufflevector <8 x float> %qr.n1220, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %qr.n1239 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %qr.n1237, <4 x float> %qr.n1238) #9, !dbg !14
  %qr.n1240 = shufflevector <8 x half> %qr.n1172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1241 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %qr.n1240, <4 x float> %qr.n1239) #9, !dbg !14
  %qr.n1242 = shufflevector <8 x half> %qr.n1174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %qr.n1242, <4 x float> %qr.n1241) #9, !dbg !14
  %qr.n1244 = shufflevector <8 x half> %qr.n1176, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %qr.n1244, <4 x float> %qr.n1243) #9, !dbg !14
  %qr.n1246 = shufflevector <8 x half> %qr.n1178, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %qr.n1246, <4 x float> %qr.n1245) #9, !dbg !14
  %qr.n1248 = shufflevector <8 x half> %qr.n1180, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %qr.n1248, <4 x float> %qr.n1247) #9, !dbg !14
  %qr.n1250 = shufflevector <8 x half> %qr.n1182, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1251 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %qr.n1250, <4 x float> %qr.n1249) #9, !dbg !14
  %qr.n1252 = shufflevector <8 x half> %qr.n1184, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1253 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %qr.n1252, <4 x float> %qr.n1251) #9, !dbg !14
  %qr.n1254 = extractvalue [2 x <8 x float>] %qr.n42, 1, !dbg !30
  %qr.n1255 = shufflevector <8 x float> %qr.n1254, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %qr.n1256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %qr.n1219, <4 x float> %qr.n1255) #9, !dbg !14
  %qr.n1257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %qr.n1223, <4 x float> %qr.n1256) #9, !dbg !14
  %qr.n1258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %qr.n1225, <4 x float> %qr.n1257) #9, !dbg !14
  %qr.n1259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %qr.n1227, <4 x float> %qr.n1258) #9, !dbg !14
  %qr.n1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %qr.n1229, <4 x float> %qr.n1259) #9, !dbg !14
  %qr.n1261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %qr.n1231, <4 x float> %qr.n1260) #9, !dbg !14
  %qr.n1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %qr.n1233, <4 x float> %qr.n1261) #9, !dbg !14
  %qr.n1263 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %qr.n1235, <4 x float> %qr.n1262) #9, !dbg !14
  %qr.n1264 = shufflevector <8 x float> %qr.n1254, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %qr.n1265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %qr.n1237, <4 x float> %qr.n1264) #9, !dbg !14
  %qr.n1266 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %qr.n1240, <4 x float> %qr.n1265) #9, !dbg !14
  %qr.n1267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %qr.n1242, <4 x float> %qr.n1266) #9, !dbg !14
  %qr.n1268 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %qr.n1244, <4 x float> %qr.n1267) #9, !dbg !14
  %qr.n1269 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %qr.n1246, <4 x float> %qr.n1268) #9, !dbg !14
  %qr.n1270 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %qr.n1248, <4 x float> %qr.n1269) #9, !dbg !14
  %qr.n1271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %qr.n1250, <4 x float> %qr.n1270) #9, !dbg !14
  %qr.n1272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %qr.n1252, <4 x float> %qr.n1271) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %qr.n1273 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %qr.n1274 = bitcast <4 x i32> %qr.n1273 to <8 x half>, !dbg !14
  %qr.n1275 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #0, !dbg !14
  %qr.n1276 = bitcast <4 x i32> %qr.n1275 to <8 x half>, !dbg !14
  %qr.n1277 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !14
  %qr.n1278 = bitcast <4 x i32> %qr.n1277 to <8 x half>, !dbg !14
  %qr.n1279 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #0, !dbg !14
  %qr.n1280 = bitcast <4 x i32> %qr.n1279 to <8 x half>, !dbg !14
  %qr.n1281 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !14
  %qr.n1282 = bitcast <4 x i32> %qr.n1281 to <8 x half>, !dbg !14
  %qr.n1283 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #0, !dbg !14
  %qr.n1284 = bitcast <4 x i32> %qr.n1283 to <8 x half>, !dbg !14
  %qr.n1285 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #0, !dbg !14
  %qr.n1286 = bitcast <4 x i32> %qr.n1285 to <8 x half>, !dbg !14
  %qr.n1287 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %qr.n1288 = bitcast <4 x i32> %qr.n1287 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %qr.n1289 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %qr.n1188, !dbg !18
  %qr.n1290 = getelementptr i8, ptr addrspace(1) %qr.n1289, i64 128, !dbg !18
  %qr.n1291 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local366), !dbg !18
  %qr.n1292 = zext i32 %qr.n1291 to i64, !dbg !18
  %qr.n1293 = inttoptr i64 %qr.n1292 to ptr addrspace(3), !dbg !18
  %address.local385 = getelementptr [2 x i8], ptr addrspace(1) %qr.n1290, i64 %address.local371, !dbg !18
  %qr.n1294 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %qr.n1295 = ptrtoint ptr addrspace(1) %address.local385 to i64, !dbg !18
  %qr.n1296 = sub i64 %qr.n1295, %qr.n1294, !dbg !18
  %dma.buffer.offset874 = trunc i64 %qr.n1296 to i32, !dbg !18
  %qr.n1297 = trunc i64 %qr.n1294 to i32, !dbg !18
  %qr.n1298 = insertelement <4 x i32> poison, i32 %qr.n1297, i64 0, !dbg !18
  %qr.n1299 = lshr i64 %qr.n1294, 32, !dbg !18
  %qr.n1300 = trunc i64 %qr.n1299 to i32, !dbg !18
  %qr.n1301 = insertelement <4 x i32> %qr.n1298, i32 %qr.n1300, i64 1, !dbg !18
  %qr.n1302 = insertelement <4 x i32> %qr.n1301, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc875 = insertelement <4 x i32> %qr.n1302, i32 131072, i64 3, !dbg !18
  %qr.n1303 = ptrtoint ptr addrspace(3) %qr.n1293 to i32, !dbg !18
  %dma.buffer.lds876 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n1303), !dbg !18
  %qr.n1304 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds876, i32 %dma.buffer.offset874, <4 x i32> %dma.buffer.rsrc875) #10, !dbg !18
  %qr.n1305 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local378), !dbg !18
  %qr.n1306 = zext i32 %qr.n1305 to i64, !dbg !18
  %qr.n1307 = inttoptr i64 %qr.n1306 to ptr addrspace(3), !dbg !18
  %address.local386 = getelementptr [2 x i8], ptr addrspace(1) %qr.n1290, i64 %address.local383, !dbg !18
  %qr.n1308 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %qr.n1309 = ptrtoint ptr addrspace(1) %address.local386 to i64, !dbg !18
  %qr.n1310 = sub i64 %qr.n1309, %qr.n1308, !dbg !18
  %dma.buffer.offset871 = trunc i64 %qr.n1310 to i32, !dbg !18
  %qr.n1311 = trunc i64 %qr.n1308 to i32, !dbg !18
  %qr.n1312 = insertelement <4 x i32> poison, i32 %qr.n1311, i64 0, !dbg !18
  %qr.n1313 = lshr i64 %qr.n1308, 32, !dbg !18
  %qr.n1314 = trunc i64 %qr.n1313 to i32, !dbg !18
  %qr.n1315 = insertelement <4 x i32> %qr.n1312, i32 %qr.n1314, i64 1, !dbg !18
  %qr.n1316 = insertelement <4 x i32> %qr.n1315, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc872 = insertelement <4 x i32> %qr.n1316, i32 131072, i64 3, !dbg !18
  %qr.n1317 = ptrtoint ptr addrspace(3) %qr.n1307 to i32, !dbg !18
  %dma.buffer.lds873 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n1317), !dbg !18
  %qr.n1318 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds873, i32 %dma.buffer.offset871, <4 x i32> %dma.buffer.rsrc872) #10, !dbg !18
  %qr.n1319 = shufflevector <8 x half> %qr.n1274, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1320 = extractvalue [2 x <8 x float>] %qr.n44, 0, !dbg !30
  %qr.n1321 = shufflevector <8 x float> %qr.n1320, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %qr.n1322 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %qr.n1319, <4 x float> %qr.n1321) #9, !dbg !14
  %qr.n1323 = shufflevector <8 x half> %qr.n1276, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %qr.n1323, <4 x float> %qr.n1322) #9, !dbg !14
  %qr.n1325 = shufflevector <8 x half> %qr.n1278, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %qr.n1325, <4 x float> %qr.n1324) #9, !dbg !14
  %qr.n1327 = shufflevector <8 x half> %qr.n1280, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %qr.n1327, <4 x float> %qr.n1326) #9, !dbg !14
  %qr.n1329 = shufflevector <8 x half> %qr.n1282, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %qr.n1329, <4 x float> %qr.n1328) #9, !dbg !14
  %qr.n1331 = shufflevector <8 x half> %qr.n1284, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %qr.n1331, <4 x float> %qr.n1330) #9, !dbg !14
  %qr.n1333 = shufflevector <8 x half> %qr.n1286, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %qr.n1333, <4 x float> %qr.n1332) #9, !dbg !14
  %qr.n1335 = shufflevector <8 x half> %qr.n1288, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %qr.n1335, <4 x float> %qr.n1334) #9, !dbg !14
  %qr.n1337 = shufflevector <8 x half> %qr.n1274, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1338 = shufflevector <8 x float> %qr.n1320, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %qr.n1339 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %qr.n1337, <4 x float> %qr.n1338) #9, !dbg !14
  %qr.n1340 = shufflevector <8 x half> %qr.n1276, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1341 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %qr.n1340, <4 x float> %qr.n1339) #9, !dbg !14
  %qr.n1342 = shufflevector <8 x half> %qr.n1278, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1343 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %qr.n1342, <4 x float> %qr.n1341) #9, !dbg !14
  %qr.n1344 = shufflevector <8 x half> %qr.n1280, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1345 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %qr.n1344, <4 x float> %qr.n1343) #9, !dbg !14
  %qr.n1346 = shufflevector <8 x half> %qr.n1282, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %qr.n1346, <4 x float> %qr.n1345) #9, !dbg !14
  %qr.n1348 = shufflevector <8 x half> %qr.n1284, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %qr.n1348, <4 x float> %qr.n1347) #9, !dbg !14
  %qr.n1350 = shufflevector <8 x half> %qr.n1286, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %qr.n1350, <4 x float> %qr.n1349) #9, !dbg !14
  %qr.n1352 = shufflevector <8 x half> %qr.n1288, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %qr.n1352, <4 x float> %qr.n1351) #9, !dbg !14
  %qr.n1354 = extractvalue [2 x <8 x float>] %qr.n44, 1, !dbg !30
  %qr.n1355 = shufflevector <8 x float> %qr.n1354, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %qr.n1356 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %qr.n1319, <4 x float> %qr.n1355) #9, !dbg !14
  %qr.n1357 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %qr.n1323, <4 x float> %qr.n1356) #9, !dbg !14
  %qr.n1358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %qr.n1325, <4 x float> %qr.n1357) #9, !dbg !14
  %qr.n1359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %qr.n1327, <4 x float> %qr.n1358) #9, !dbg !14
  %qr.n1360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %qr.n1329, <4 x float> %qr.n1359) #9, !dbg !14
  %qr.n1361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %qr.n1331, <4 x float> %qr.n1360) #9, !dbg !14
  %qr.n1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %qr.n1333, <4 x float> %qr.n1361) #9, !dbg !14
  %qr.n1363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %qr.n1335, <4 x float> %qr.n1362) #9, !dbg !14
  %qr.n1364 = shufflevector <8 x float> %qr.n1354, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %qr.n1365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %qr.n1337, <4 x float> %qr.n1364) #9, !dbg !14
  %qr.n1366 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %qr.n1340, <4 x float> %qr.n1365) #9, !dbg !14
  %qr.n1367 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %qr.n1342, <4 x float> %qr.n1366) #9, !dbg !14
  %qr.n1368 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %qr.n1344, <4 x float> %qr.n1367) #9, !dbg !14
  %qr.n1369 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %qr.n1346, <4 x float> %qr.n1368) #9, !dbg !14
  %qr.n1370 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %qr.n1348, <4 x float> %qr.n1369) #9, !dbg !14
  %qr.n1371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %qr.n1350, <4 x float> %qr.n1370) #9, !dbg !14
  %qr.n1372 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %qr.n1352, <4 x float> %qr.n1371) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %qr.n1373 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %qr.n1374 = bitcast <4 x i32> %qr.n1373 to <8 x half>, !dbg !14
  %qr.n1375 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #0, !dbg !14
  %qr.n1376 = bitcast <4 x i32> %qr.n1375 to <8 x half>, !dbg !14
  %qr.n1377 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !14
  %qr.n1378 = bitcast <4 x i32> %qr.n1377 to <8 x half>, !dbg !14
  %qr.n1379 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #0, !dbg !14
  %qr.n1380 = bitcast <4 x i32> %qr.n1379 to <8 x half>, !dbg !14
  %qr.n1381 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !14
  %qr.n1382 = bitcast <4 x i32> %qr.n1381 to <8 x half>, !dbg !14
  %qr.n1383 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #0, !dbg !14
  %qr.n1384 = bitcast <4 x i32> %qr.n1383 to <8 x half>, !dbg !14
  %qr.n1385 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #0, !dbg !14
  %qr.n1386 = bitcast <4 x i32> %qr.n1385 to <8 x half>, !dbg !14
  %qr.n1387 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %qr.n1388 = bitcast <4 x i32> %qr.n1387 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %qr.n1389 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %qr.n1188, !dbg !18
  %qr.n1390 = getelementptr i8, ptr addrspace(1) %qr.n1389, i64 192, !dbg !18
  %qr.n1391 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local366), !dbg !18
  %qr.n1392 = zext i32 %qr.n1391 to i64, !dbg !18
  %qr.n1393 = inttoptr i64 %qr.n1392 to ptr addrspace(3), !dbg !18
  %address.local387 = getelementptr [2 x i8], ptr addrspace(1) %qr.n1390, i64 %address.local371, !dbg !18
  %qr.n1394 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %qr.n1395 = ptrtoint ptr addrspace(1) %address.local387 to i64, !dbg !18
  %qr.n1396 = sub i64 %qr.n1395, %qr.n1394, !dbg !18
  %dma.buffer.offset868 = trunc i64 %qr.n1396 to i32, !dbg !18
  %qr.n1397 = trunc i64 %qr.n1394 to i32, !dbg !18
  %qr.n1398 = insertelement <4 x i32> poison, i32 %qr.n1397, i64 0, !dbg !18
  %qr.n1399 = lshr i64 %qr.n1394, 32, !dbg !18
  %qr.n1400 = trunc i64 %qr.n1399 to i32, !dbg !18
  %qr.n1401 = insertelement <4 x i32> %qr.n1398, i32 %qr.n1400, i64 1, !dbg !18
  %qr.n1402 = insertelement <4 x i32> %qr.n1401, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc869 = insertelement <4 x i32> %qr.n1402, i32 131072, i64 3, !dbg !18
  %qr.n1403 = ptrtoint ptr addrspace(3) %qr.n1393 to i32, !dbg !18
  %dma.buffer.lds870 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n1403), !dbg !18
  %qr.n1404 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds870, i32 %dma.buffer.offset868, <4 x i32> %dma.buffer.rsrc869) #10, !dbg !18
  %qr.n1405 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local378), !dbg !18
  %qr.n1406 = zext i32 %qr.n1405 to i64, !dbg !18
  %qr.n1407 = inttoptr i64 %qr.n1406 to ptr addrspace(3), !dbg !18
  %address.local388 = getelementptr [2 x i8], ptr addrspace(1) %qr.n1390, i64 %address.local383, !dbg !18
  %qr.n1408 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !18
  %qr.n1409 = ptrtoint ptr addrspace(1) %address.local388 to i64, !dbg !18
  %qr.n1410 = sub i64 %qr.n1409, %qr.n1408, !dbg !18
  %dma.buffer.offset = trunc i64 %qr.n1410 to i32, !dbg !18
  %qr.n1411 = trunc i64 %qr.n1408 to i32, !dbg !18
  %qr.n1412 = insertelement <4 x i32> poison, i32 %qr.n1411, i64 0, !dbg !18
  %qr.n1413 = lshr i64 %qr.n1408, 32, !dbg !18
  %qr.n1414 = trunc i64 %qr.n1413 to i32, !dbg !18
  %qr.n1415 = insertelement <4 x i32> %qr.n1412, i32 %qr.n1414, i64 1, !dbg !18
  %qr.n1416 = insertelement <4 x i32> %qr.n1415, i32 33554432, i64 2, !dbg !18
  %dma.buffer.rsrc = insertelement <4 x i32> %qr.n1416, i32 131072, i64 3, !dbg !18
  %qr.n1417 = ptrtoint ptr addrspace(3) %qr.n1407 to i32, !dbg !18
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %qr.n1417), !dbg !18
  %qr.n1418 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #10, !dbg !18
  %qr.n1419 = shufflevector <8 x half> %qr.n1374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1420 = extractvalue [2 x <8 x float>] %qr.n45, 0, !dbg !30
  %qr.n1421 = shufflevector <8 x float> %qr.n1420, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %qr.n1422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %qr.n1419, <4 x float> %qr.n1421) #9, !dbg !14
  %qr.n1423 = shufflevector <8 x half> %qr.n1376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1424 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %qr.n1423, <4 x float> %qr.n1422) #9, !dbg !14
  %qr.n1425 = shufflevector <8 x half> %qr.n1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %qr.n1425, <4 x float> %qr.n1424) #9, !dbg !14
  %qr.n1427 = shufflevector <8 x half> %qr.n1380, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %qr.n1427, <4 x float> %qr.n1426) #9, !dbg !14
  %qr.n1429 = shufflevector <8 x half> %qr.n1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %qr.n1429, <4 x float> %qr.n1428) #9, !dbg !14
  %qr.n1431 = shufflevector <8 x half> %qr.n1384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %qr.n1431, <4 x float> %qr.n1430) #9, !dbg !14
  %qr.n1433 = shufflevector <8 x half> %qr.n1386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %qr.n1433, <4 x float> %qr.n1432) #9, !dbg !14
  %qr.n1435 = shufflevector <8 x half> %qr.n1388, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %qr.n1435, <4 x float> %qr.n1434) #9, !dbg !14
  %qr.n1437 = shufflevector <8 x half> %qr.n1374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1438 = shufflevector <8 x float> %qr.n1420, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %qr.n1439 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %qr.n1437, <4 x float> %qr.n1438) #9, !dbg !14
  %qr.n1440 = shufflevector <8 x half> %qr.n1376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %qr.n1440, <4 x float> %qr.n1439) #9, !dbg !14
  %qr.n1442 = shufflevector <8 x half> %qr.n1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %qr.n1442, <4 x float> %qr.n1441) #9, !dbg !14
  %qr.n1444 = shufflevector <8 x half> %qr.n1380, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %qr.n1444, <4 x float> %qr.n1443) #9, !dbg !14
  %qr.n1446 = shufflevector <8 x half> %qr.n1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %qr.n1446, <4 x float> %qr.n1445) #9, !dbg !14
  %qr.n1448 = shufflevector <8 x half> %qr.n1384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1449 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %qr.n1448, <4 x float> %qr.n1447) #9, !dbg !14
  %qr.n1450 = shufflevector <8 x half> %qr.n1386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1451 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %qr.n1450, <4 x float> %qr.n1449) #9, !dbg !14
  %qr.n1452 = shufflevector <8 x half> %qr.n1388, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1453 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %qr.n1452, <4 x float> %qr.n1451) #9, !dbg !14
  %qr.n1454 = extractvalue [2 x <8 x float>] %qr.n45, 1, !dbg !30
  %qr.n1455 = shufflevector <8 x float> %qr.n1454, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %qr.n1456 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %qr.n1419, <4 x float> %qr.n1455) #9, !dbg !14
  %qr.n1457 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %qr.n1423, <4 x float> %qr.n1456) #9, !dbg !14
  %qr.n1458 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %qr.n1425, <4 x float> %qr.n1457) #9, !dbg !14
  %qr.n1459 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %qr.n1427, <4 x float> %qr.n1458) #9, !dbg !14
  %qr.n1460 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %qr.n1429, <4 x float> %qr.n1459) #9, !dbg !14
  %qr.n1461 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %qr.n1431, <4 x float> %qr.n1460) #9, !dbg !14
  %qr.n1462 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %qr.n1433, <4 x float> %qr.n1461) #9, !dbg !14
  %qr.n1463 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %qr.n1435, <4 x float> %qr.n1462) #9, !dbg !14
  %qr.n1464 = shufflevector <8 x float> %qr.n1454, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %qr.n1465 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %qr.n1437, <4 x float> %qr.n1464) #9, !dbg !14
  %qr.n1466 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %qr.n1440, <4 x float> %qr.n1465) #9, !dbg !14
  %qr.n1467 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %qr.n1442, <4 x float> %qr.n1466) #9, !dbg !14
  %qr.n1468 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %qr.n1444, <4 x float> %qr.n1467) #9, !dbg !14
  %qr.n1469 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %qr.n1446, <4 x float> %qr.n1468) #9, !dbg !14
  %qr.n1470 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %qr.n1448, <4 x float> %qr.n1469) #9, !dbg !14
  %qr.n1471 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %qr.n1450, <4 x float> %qr.n1470) #9, !dbg !14
  %qr.n1472 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %qr.n1452, <4 x float> %qr.n1471) #9, !dbg !14
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #10, !dbg !18
  fence syncscope("workgroup") release, !dbg !18
  call void @llvm.amdgcn.s.barrier(), !dbg !18
  fence syncscope("workgroup") acquire, !dbg !18
  %qr.n1473 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local326, i32 0) #0, !dbg !14
  %qr.n1474 = bitcast <4 x i32> %qr.n1473 to <8 x half>, !dbg !14
  %qr.n1475 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local339, i32 0) #0, !dbg !14
  %qr.n1476 = bitcast <4 x i32> %qr.n1475 to <8 x half>, !dbg !14
  %qr.n1477 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local343, i32 0) #0, !dbg !14
  %qr.n1478 = bitcast <4 x i32> %qr.n1477 to <8 x half>, !dbg !14
  %qr.n1479 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local347, i32 0) #0, !dbg !14
  %qr.n1480 = bitcast <4 x i32> %qr.n1479 to <8 x half>, !dbg !14
  %qr.n1481 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local349, i32 0) #0, !dbg !14
  %qr.n1482 = bitcast <4 x i32> %qr.n1481 to <8 x half>, !dbg !14
  %qr.n1483 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local353, i32 0) #0, !dbg !14
  %qr.n1484 = bitcast <4 x i32> %qr.n1483 to <8 x half>, !dbg !14
  %qr.n1485 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local357, i32 0) #0, !dbg !14
  %qr.n1486 = bitcast <4 x i32> %qr.n1485 to <8 x half>, !dbg !14
  %qr.n1487 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local361, i32 0) #0, !dbg !14
  %qr.n1488 = bitcast <4 x i32> %qr.n1487 to <8 x half>, !dbg !14
  %qr.n1489 = shufflevector <8 x half> %qr.n1474, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1490 = extractvalue [2 x <8 x float>] %qr.n46, 0, !dbg !30
  %qr.n1491 = shufflevector <8 x float> %qr.n1490, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %qr.n1492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %qr.n1489, <4 x float> %qr.n1491) #9, !dbg !14
  %qr.n1493 = shufflevector <8 x half> %qr.n1476, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1494 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %qr.n1493, <4 x float> %qr.n1492) #9, !dbg !14
  %qr.n1495 = shufflevector <8 x half> %qr.n1478, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1496 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %qr.n1495, <4 x float> %qr.n1494) #9, !dbg !14
  %qr.n1497 = shufflevector <8 x half> %qr.n1480, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1498 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %qr.n1497, <4 x float> %qr.n1496) #9, !dbg !14
  %qr.n1499 = shufflevector <8 x half> %qr.n1482, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1500 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %qr.n1499, <4 x float> %qr.n1498) #9, !dbg !14
  %qr.n1501 = shufflevector <8 x half> %qr.n1484, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1502 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %qr.n1501, <4 x float> %qr.n1500) #9, !dbg !14
  %qr.n1503 = shufflevector <8 x half> %qr.n1486, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1504 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %qr.n1503, <4 x float> %qr.n1502) #9, !dbg !14
  %qr.n1505 = shufflevector <8 x half> %qr.n1488, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %qr.n1506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %qr.n1505, <4 x float> %qr.n1504) #9, !dbg !14
  %qr.n1507 = shufflevector <8 x half> %qr.n1474, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1508 = shufflevector <8 x float> %qr.n1490, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %qr.n1509 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %qr.n1507, <4 x float> %qr.n1508) #9, !dbg !14
  %qr.n1510 = shufflevector <8 x half> %qr.n1476, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1511 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed100, <4 x half> %qr.n1510, <4 x float> %qr.n1509) #9, !dbg !14
  %qr.n1512 = shufflevector <8 x half> %qr.n1478, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1513 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed102, <4 x half> %qr.n1512, <4 x float> %qr.n1511) #9, !dbg !14
  %qr.n1514 = shufflevector <8 x half> %qr.n1480, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1515 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed104, <4 x half> %qr.n1514, <4 x float> %qr.n1513) #9, !dbg !14
  %qr.n1516 = shufflevector <8 x half> %qr.n1482, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1517 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed106, <4 x half> %qr.n1516, <4 x float> %qr.n1515) #9, !dbg !14
  %qr.n1518 = shufflevector <8 x half> %qr.n1484, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1519 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed108, <4 x half> %qr.n1518, <4 x float> %qr.n1517) #9, !dbg !14
  %qr.n1520 = shufflevector <8 x half> %qr.n1486, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1521 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed110, <4 x half> %qr.n1520, <4 x float> %qr.n1519) #9, !dbg !14
  %qr.n1522 = shufflevector <8 x half> %qr.n1488, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %qr.n1523 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed112, <4 x half> %qr.n1522, <4 x float> %qr.n1521) #9, !dbg !14
  %qr.n1524 = extractvalue [2 x <8 x float>] %qr.n46, 1, !dbg !30
  %qr.n1525 = shufflevector <8 x float> %qr.n1524, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !30
  %qr.n1526 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %qr.n1489, <4 x float> %qr.n1525) #9, !dbg !14
  %qr.n1527 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %qr.n1493, <4 x float> %qr.n1526) #9, !dbg !14
  %qr.n1528 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %qr.n1495, <4 x float> %qr.n1527) #9, !dbg !14
  %qr.n1529 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %qr.n1497, <4 x float> %qr.n1528) #9, !dbg !14
  %qr.n1530 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %qr.n1499, <4 x float> %qr.n1529) #9, !dbg !14
  %qr.n1531 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %qr.n1501, <4 x float> %qr.n1530) #9, !dbg !14
  %qr.n1532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %qr.n1503, <4 x float> %qr.n1531) #9, !dbg !14
  %qr.n1533 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %qr.n1505, <4 x float> %qr.n1532) #9, !dbg !14
  %qr.n1534 = shufflevector <8 x float> %qr.n1524, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !30
  %qr.n1535 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed114, <4 x half> %qr.n1507, <4 x float> %qr.n1534) #9, !dbg !14
  %qr.n1536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed116, <4 x half> %qr.n1510, <4 x float> %qr.n1535) #9, !dbg !14
  %qr.n1537 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed118, <4 x half> %qr.n1512, <4 x float> %qr.n1536) #9, !dbg !14
  %qr.n1538 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed120, <4 x half> %qr.n1514, <4 x float> %qr.n1537) #9, !dbg !14
  %qr.n1539 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed122, <4 x half> %qr.n1516, <4 x float> %qr.n1538) #9, !dbg !14
  %qr.n1540 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed124, <4 x half> %qr.n1518, <4 x float> %qr.n1539) #9, !dbg !14
  %qr.n1541 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed126, <4 x half> %qr.n1520, <4 x float> %qr.n1540) #9, !dbg !14
  %qr.n1542 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed128, <4 x half> %qr.n1522, <4 x float> %qr.n1541) #9, !dbg !14
  %qr.n1543 = shufflevector <4 x float> %qr.n1236, <4 x float> %qr.n1253, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %qr.n1544 = insertvalue [2 x <8 x float>] poison, <8 x float> %qr.n1543, 0, !dbg !15
  %qr.n1545 = shufflevector <4 x float> %qr.n1263, <4 x float> %qr.n1272, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %qr.n1546 = insertvalue [2 x <8 x float>] %qr.n1544, <8 x float> %qr.n1545, 1, !dbg !15
  %qr.n1547 = shufflevector <4 x float> %qr.n1336, <4 x float> %qr.n1353, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %qr.n1548 = insertvalue [2 x <8 x float>] poison, <8 x float> %qr.n1547, 0, !dbg !15
  %qr.n1549 = shufflevector <4 x float> %qr.n1363, <4 x float> %qr.n1372, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %qr.n1550 = insertvalue [2 x <8 x float>] %qr.n1548, <8 x float> %qr.n1549, 1, !dbg !15
  %qr.n1551 = shufflevector <4 x float> %qr.n1436, <4 x float> %qr.n1453, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %qr.n1552 = insertvalue [2 x <8 x float>] poison, <8 x float> %qr.n1551, 0, !dbg !15
  %qr.n1553 = shufflevector <4 x float> %qr.n1463, <4 x float> %qr.n1472, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %qr.n1554 = insertvalue [2 x <8 x float>] %qr.n1552, <8 x float> %qr.n1553, 1, !dbg !15
  %qr.n1555 = shufflevector <4 x float> %qr.n1506, <4 x float> %qr.n1523, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %qr.n1556 = insertvalue [2 x <8 x float>] poison, <8 x float> %qr.n1555, 0, !dbg !15
  %qr.n1557 = shufflevector <4 x float> %qr.n1533, <4 x float> %qr.n1542, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !15
  %qr.n1558 = insertvalue [2 x <8 x float>] %qr.n1556, <8 x float> %qr.n1557, 1, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %qr.n1559 = add nuw nsw i64 %qr.n41, 1, !dbg !15
  br label %qr.n40, !dbg !15

qr.n1560:                                             ; preds = %qr.n40
  %qr.n1561 = and i32 %qr.n24, 16, !dbg !15
  %.not2 = icmp eq i32 %qr.n1561, 0, !dbg !15
  %qr.n1562 = and i32 %qr.n24, 32, !dbg !15
  %.not3 = icmp eq i32 %qr.n1562, 0, !dbg !15
  %qr.n1563 = extractvalue [2 x <8 x float>] %qr.n46, 0, !dbg !15
  %qr.n1564 = extractelement <8 x float> %qr.n1563, i64 0, !dbg !15
  %qr.n1565 = extractelement <8 x float> %qr.n1563, i64 1, !dbg !15
  %qr.n1566 = extractelement <8 x float> %qr.n1563, i64 2, !dbg !15
  %qr.n1567 = extractelement <8 x float> %qr.n1563, i64 3, !dbg !15
  %qr.n1568 = select i1 %.not2, float %qr.n1565, float %qr.n1564, !dbg !15
  %qr.n1569 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !15
  %qr.n1570 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %qr.n1569), !dbg !15
  %qr.n1571 = add nuw nsw i32 %qr.n1570, 64, !dbg !15
  %qr.n1572 = xor i32 %qr.n1570, 16, !dbg !15
  %qr.n1573 = and i32 %qr.n1571, 192, !dbg !15
  %qr.n1574 = icmp samesign ult i32 %qr.n1572, %qr.n1573, !dbg !15
  %qr.n1575 = select i1 %qr.n1574, i32 %qr.n1572, i32 %qr.n1570, !dbg !15
  %qr.n1576 = shl nuw nsw i32 %qr.n1575, 2, !dbg !15
  %qr.n1577 = bitcast float %qr.n1568 to i32, !dbg !15
  %qr.n1578 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1577), !dbg !15
  %qr.n1579 = bitcast i32 %qr.n1578 to float, !dbg !15
  %qr.n1580 = select i1 %.not2, float %qr.n1564, float %qr.n1579, !dbg !15
  %qr.n1581 = select i1 %.not2, float %qr.n1579, float %qr.n1565, !dbg !15
  %qr.n1582 = select i1 %.not2, float %qr.n1567, float %qr.n1566, !dbg !15
  %qr.n1583 = bitcast float %qr.n1582 to i32, !dbg !15
  %qr.n1584 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1583), !dbg !15
  %qr.n1585 = bitcast i32 %qr.n1584 to float, !dbg !15
  %qr.n1586 = select i1 %.not2, float %qr.n1566, float %qr.n1585, !dbg !15
  %qr.n1587 = select i1 %.not2, float %qr.n1585, float %qr.n1567, !dbg !15
  %qr.n1588 = select i1 %.not3, float %qr.n1586, float %qr.n1580, !dbg !15
  %qr.n1589 = xor i32 %qr.n1570, 32, !dbg !15
  %qr.n1590 = icmp samesign ult i32 %qr.n1589, %qr.n1573, !dbg !15
  %qr.n1591 = select i1 %qr.n1590, i32 %qr.n1589, i32 %qr.n1570, !dbg !15
  %qr.n1592 = shl nuw nsw i32 %qr.n1591, 2, !dbg !15
  %qr.n1593 = bitcast float %qr.n1588 to i32, !dbg !15
  %qr.n1594 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1593), !dbg !15
  %qr.n1595 = bitcast i32 %qr.n1594 to float, !dbg !15
  %qr.n1596 = select i1 %.not3, float %qr.n1580, float %qr.n1595, !dbg !15
  %qr.n1597 = select i1 %.not3, float %qr.n1595, float %qr.n1586, !dbg !15
  %qr.n1598 = select i1 %.not3, float %qr.n1587, float %qr.n1581, !dbg !15
  %qr.n1599 = bitcast float %qr.n1598 to i32, !dbg !15
  %qr.n1600 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1599), !dbg !15
  %qr.n1601 = bitcast i32 %qr.n1600 to float, !dbg !15
  %qr.n1602 = select i1 %.not3, float %qr.n1581, float %qr.n1601, !dbg !15
  %qr.n1603 = select i1 %.not3, float %qr.n1601, float %qr.n1587, !dbg !15
  %qr.n1604 = extractvalue [2 x <8 x float>] %qr.n45, 0, !dbg !15
  %qr.n1605 = extractelement <8 x float> %qr.n1604, i64 0, !dbg !15
  %qr.n1606 = extractelement <8 x float> %qr.n1604, i64 1, !dbg !15
  %qr.n1607 = extractelement <8 x float> %qr.n1604, i64 2, !dbg !15
  %qr.n1608 = extractelement <8 x float> %qr.n1604, i64 3, !dbg !15
  %qr.n1609 = select i1 %.not2, float %qr.n1606, float %qr.n1605, !dbg !15
  %qr.n1610 = bitcast float %qr.n1609 to i32, !dbg !15
  %qr.n1611 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1610), !dbg !15
  %qr.n1612 = bitcast i32 %qr.n1611 to float, !dbg !15
  %qr.n1613 = select i1 %.not2, float %qr.n1605, float %qr.n1612, !dbg !15
  %qr.n1614 = select i1 %.not2, float %qr.n1612, float %qr.n1606, !dbg !15
  %qr.n1615 = select i1 %.not2, float %qr.n1608, float %qr.n1607, !dbg !15
  %qr.n1616 = bitcast float %qr.n1615 to i32, !dbg !15
  %qr.n1617 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1616), !dbg !15
  %qr.n1618 = bitcast i32 %qr.n1617 to float, !dbg !15
  %qr.n1619 = select i1 %.not2, float %qr.n1607, float %qr.n1618, !dbg !15
  %qr.n1620 = select i1 %.not2, float %qr.n1618, float %qr.n1608, !dbg !15
  %qr.n1621 = select i1 %.not3, float %qr.n1619, float %qr.n1613, !dbg !15
  %qr.n1622 = bitcast float %qr.n1621 to i32, !dbg !15
  %qr.n1623 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1622), !dbg !15
  %qr.n1624 = bitcast i32 %qr.n1623 to float, !dbg !15
  %qr.n1625 = select i1 %.not3, float %qr.n1613, float %qr.n1624, !dbg !15
  %qr.n1626 = select i1 %.not3, float %qr.n1624, float %qr.n1619, !dbg !15
  %qr.n1627 = select i1 %.not3, float %qr.n1620, float %qr.n1614, !dbg !15
  %qr.n1628 = bitcast float %qr.n1627 to i32, !dbg !15
  %qr.n1629 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1628), !dbg !15
  %qr.n1630 = bitcast i32 %qr.n1629 to float, !dbg !15
  %qr.n1631 = select i1 %.not3, float %qr.n1614, float %qr.n1630, !dbg !15
  %qr.n1632 = select i1 %.not3, float %qr.n1630, float %qr.n1620, !dbg !15
  %qr.n1633 = extractvalue [2 x <8 x float>] %qr.n44, 0, !dbg !15
  %qr.n1634 = extractelement <8 x float> %qr.n1633, i64 0, !dbg !15
  %qr.n1635 = extractelement <8 x float> %qr.n1633, i64 1, !dbg !15
  %qr.n1636 = extractelement <8 x float> %qr.n1633, i64 2, !dbg !15
  %qr.n1637 = extractelement <8 x float> %qr.n1633, i64 3, !dbg !15
  %qr.n1638 = select i1 %.not2, float %qr.n1635, float %qr.n1634, !dbg !15
  %qr.n1639 = bitcast float %qr.n1638 to i32, !dbg !15
  %qr.n1640 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1639), !dbg !15
  %qr.n1641 = bitcast i32 %qr.n1640 to float, !dbg !15
  %qr.n1642 = select i1 %.not2, float %qr.n1634, float %qr.n1641, !dbg !15
  %qr.n1643 = select i1 %.not2, float %qr.n1641, float %qr.n1635, !dbg !15
  %qr.n1644 = select i1 %.not2, float %qr.n1637, float %qr.n1636, !dbg !15
  %qr.n1645 = bitcast float %qr.n1644 to i32, !dbg !15
  %qr.n1646 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1645), !dbg !15
  %qr.n1647 = bitcast i32 %qr.n1646 to float, !dbg !15
  %qr.n1648 = select i1 %.not2, float %qr.n1636, float %qr.n1647, !dbg !15
  %qr.n1649 = select i1 %.not2, float %qr.n1647, float %qr.n1637, !dbg !15
  %qr.n1650 = select i1 %.not3, float %qr.n1648, float %qr.n1642, !dbg !15
  %qr.n1651 = bitcast float %qr.n1650 to i32, !dbg !15
  %qr.n1652 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1651), !dbg !15
  %qr.n1653 = bitcast i32 %qr.n1652 to float, !dbg !15
  %qr.n1654 = select i1 %.not3, float %qr.n1642, float %qr.n1653, !dbg !15
  %qr.n1655 = select i1 %.not3, float %qr.n1653, float %qr.n1648, !dbg !15
  %qr.n1656 = select i1 %.not3, float %qr.n1649, float %qr.n1643, !dbg !15
  %qr.n1657 = bitcast float %qr.n1656 to i32, !dbg !15
  %qr.n1658 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1657), !dbg !15
  %qr.n1659 = bitcast i32 %qr.n1658 to float, !dbg !15
  %qr.n1660 = select i1 %.not3, float %qr.n1643, float %qr.n1659, !dbg !15
  %qr.n1661 = select i1 %.not3, float %qr.n1659, float %qr.n1649, !dbg !15
  %qr.n1662 = extractvalue [2 x <8 x float>] %qr.n42, 0, !dbg !15
  %qr.n1663 = extractelement <8 x float> %qr.n1662, i64 0, !dbg !15
  %qr.n1664 = extractelement <8 x float> %qr.n1662, i64 1, !dbg !15
  %qr.n1665 = extractelement <8 x float> %qr.n1662, i64 2, !dbg !15
  %qr.n1666 = extractelement <8 x float> %qr.n1662, i64 3, !dbg !15
  %qr.n1667 = select i1 %.not2, float %qr.n1664, float %qr.n1663, !dbg !15
  %qr.n1668 = bitcast float %qr.n1667 to i32, !dbg !15
  %qr.n1669 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1668), !dbg !15
  %qr.n1670 = bitcast i32 %qr.n1669 to float, !dbg !15
  %qr.n1671 = select i1 %.not2, float %qr.n1663, float %qr.n1670, !dbg !15
  %qr.n1672 = select i1 %.not2, float %qr.n1670, float %qr.n1664, !dbg !15
  %qr.n1673 = select i1 %.not2, float %qr.n1666, float %qr.n1665, !dbg !15
  %qr.n1674 = bitcast float %qr.n1673 to i32, !dbg !15
  %qr.n1675 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1674), !dbg !15
  %qr.n1676 = bitcast i32 %qr.n1675 to float, !dbg !15
  %qr.n1677 = select i1 %.not2, float %qr.n1665, float %qr.n1676, !dbg !15
  %qr.n1678 = select i1 %.not2, float %qr.n1676, float %qr.n1666, !dbg !15
  %qr.n1679 = select i1 %.not3, float %qr.n1677, float %qr.n1671, !dbg !15
  %qr.n1680 = bitcast float %qr.n1679 to i32, !dbg !15
  %qr.n1681 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1680), !dbg !15
  %qr.n1682 = select i1 %.not3, float %qr.n1678, float %qr.n1672, !dbg !15
  %qr.n1683 = bitcast float %qr.n1682 to i32, !dbg !15
  %qr.n1684 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1683), !dbg !15
  %qr.n1685 = extractvalue [2 x <1 x float>] %qr.n43, 0, !dbg !19
  %qr.n1686 = shufflevector <1 x float> %qr.n1685, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %qr.n1687 = extractelement <8 x float> %qr.n1563, i64 4, !dbg !15
  %qr.n1688 = extractelement <8 x float> %qr.n1563, i64 5, !dbg !15
  %qr.n1689 = extractelement <8 x float> %qr.n1563, i64 6, !dbg !15
  %qr.n1690 = extractelement <8 x float> %qr.n1563, i64 7, !dbg !15
  %qr.n1691 = select i1 %.not2, float %qr.n1688, float %qr.n1687, !dbg !15
  %qr.n1692 = bitcast float %qr.n1691 to i32, !dbg !15
  %qr.n1693 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1692), !dbg !15
  %qr.n1694 = bitcast i32 %qr.n1693 to float, !dbg !15
  %qr.n1695 = select i1 %.not2, float %qr.n1687, float %qr.n1694, !dbg !15
  %qr.n1696 = select i1 %.not2, float %qr.n1694, float %qr.n1688, !dbg !15
  %qr.n1697 = select i1 %.not2, float %qr.n1690, float %qr.n1689, !dbg !15
  %qr.n1698 = bitcast float %qr.n1697 to i32, !dbg !15
  %qr.n1699 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1698), !dbg !15
  %qr.n1700 = bitcast i32 %qr.n1699 to float, !dbg !15
  %qr.n1701 = select i1 %.not2, float %qr.n1689, float %qr.n1700, !dbg !15
  %qr.n1702 = select i1 %.not2, float %qr.n1700, float %qr.n1690, !dbg !15
  %qr.n1703 = select i1 %.not3, float %qr.n1701, float %qr.n1695, !dbg !15
  %qr.n1704 = bitcast float %qr.n1703 to i32, !dbg !15
  %qr.n1705 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1704), !dbg !15
  %qr.n1706 = bitcast i32 %qr.n1705 to float, !dbg !15
  %qr.n1707 = select i1 %.not3, float %qr.n1695, float %qr.n1706, !dbg !15
  %qr.n1708 = select i1 %.not3, float %qr.n1706, float %qr.n1701, !dbg !15
  %qr.n1709 = select i1 %.not3, float %qr.n1702, float %qr.n1696, !dbg !15
  %qr.n1710 = bitcast float %qr.n1709 to i32, !dbg !15
  %qr.n1711 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1710), !dbg !15
  %qr.n1712 = bitcast i32 %qr.n1711 to float, !dbg !15
  %qr.n1713 = select i1 %.not3, float %qr.n1696, float %qr.n1712, !dbg !15
  %qr.n1714 = select i1 %.not3, float %qr.n1712, float %qr.n1702, !dbg !15
  %qr.n1715 = extractelement <8 x float> %qr.n1604, i64 4, !dbg !15
  %qr.n1716 = extractelement <8 x float> %qr.n1604, i64 5, !dbg !15
  %qr.n1717 = extractelement <8 x float> %qr.n1604, i64 6, !dbg !15
  %qr.n1718 = extractelement <8 x float> %qr.n1604, i64 7, !dbg !15
  %qr.n1719 = select i1 %.not2, float %qr.n1716, float %qr.n1715, !dbg !15
  %qr.n1720 = bitcast float %qr.n1719 to i32, !dbg !15
  %qr.n1721 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1720), !dbg !15
  %qr.n1722 = bitcast i32 %qr.n1721 to float, !dbg !15
  %qr.n1723 = select i1 %.not2, float %qr.n1715, float %qr.n1722, !dbg !15
  %qr.n1724 = select i1 %.not2, float %qr.n1722, float %qr.n1716, !dbg !15
  %qr.n1725 = select i1 %.not2, float %qr.n1718, float %qr.n1717, !dbg !15
  %qr.n1726 = bitcast float %qr.n1725 to i32, !dbg !15
  %qr.n1727 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1726), !dbg !15
  %qr.n1728 = bitcast i32 %qr.n1727 to float, !dbg !15
  %qr.n1729 = select i1 %.not2, float %qr.n1717, float %qr.n1728, !dbg !15
  %qr.n1730 = select i1 %.not2, float %qr.n1728, float %qr.n1718, !dbg !15
  %qr.n1731 = select i1 %.not3, float %qr.n1729, float %qr.n1723, !dbg !15
  %qr.n1732 = bitcast float %qr.n1731 to i32, !dbg !15
  %qr.n1733 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1732), !dbg !15
  %qr.n1734 = bitcast i32 %qr.n1733 to float, !dbg !15
  %qr.n1735 = select i1 %.not3, float %qr.n1723, float %qr.n1734, !dbg !15
  %qr.n1736 = select i1 %.not3, float %qr.n1734, float %qr.n1729, !dbg !15
  %qr.n1737 = select i1 %.not3, float %qr.n1730, float %qr.n1724, !dbg !15
  %qr.n1738 = bitcast float %qr.n1737 to i32, !dbg !15
  %qr.n1739 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1738), !dbg !15
  %qr.n1740 = bitcast i32 %qr.n1739 to float, !dbg !15
  %qr.n1741 = select i1 %.not3, float %qr.n1724, float %qr.n1740, !dbg !15
  %qr.n1742 = select i1 %.not3, float %qr.n1740, float %qr.n1730, !dbg !15
  %qr.n1743 = extractelement <8 x float> %qr.n1633, i64 4, !dbg !15
  %qr.n1744 = extractelement <8 x float> %qr.n1633, i64 5, !dbg !15
  %qr.n1745 = extractelement <8 x float> %qr.n1633, i64 6, !dbg !15
  %qr.n1746 = extractelement <8 x float> %qr.n1633, i64 7, !dbg !15
  %qr.n1747 = select i1 %.not2, float %qr.n1744, float %qr.n1743, !dbg !15
  %qr.n1748 = bitcast float %qr.n1747 to i32, !dbg !15
  %qr.n1749 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1748), !dbg !15
  %qr.n1750 = bitcast i32 %qr.n1749 to float, !dbg !15
  %qr.n1751 = select i1 %.not2, float %qr.n1743, float %qr.n1750, !dbg !15
  %qr.n1752 = select i1 %.not2, float %qr.n1750, float %qr.n1744, !dbg !15
  %qr.n1753 = select i1 %.not2, float %qr.n1746, float %qr.n1745, !dbg !15
  %qr.n1754 = bitcast float %qr.n1753 to i32, !dbg !15
  %qr.n1755 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1754), !dbg !15
  %qr.n1756 = bitcast i32 %qr.n1755 to float, !dbg !15
  %qr.n1757 = select i1 %.not2, float %qr.n1745, float %qr.n1756, !dbg !15
  %qr.n1758 = select i1 %.not2, float %qr.n1756, float %qr.n1746, !dbg !15
  %qr.n1759 = select i1 %.not3, float %qr.n1757, float %qr.n1751, !dbg !15
  %qr.n1760 = bitcast float %qr.n1759 to i32, !dbg !15
  %qr.n1761 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1760), !dbg !15
  %qr.n1762 = bitcast i32 %qr.n1761 to float, !dbg !15
  %qr.n1763 = select i1 %.not3, float %qr.n1751, float %qr.n1762, !dbg !15
  %qr.n1764 = select i1 %.not3, float %qr.n1762, float %qr.n1757, !dbg !15
  %qr.n1765 = select i1 %.not3, float %qr.n1758, float %qr.n1752, !dbg !15
  %qr.n1766 = bitcast float %qr.n1765 to i32, !dbg !15
  %qr.n1767 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1766), !dbg !15
  %qr.n1768 = bitcast i32 %qr.n1767 to float, !dbg !15
  %qr.n1769 = select i1 %.not3, float %qr.n1752, float %qr.n1768, !dbg !15
  %qr.n1770 = select i1 %.not3, float %qr.n1768, float %qr.n1758, !dbg !15
  %qr.n1771 = extractelement <8 x float> %qr.n1662, i64 4, !dbg !15
  %qr.n1772 = extractelement <8 x float> %qr.n1662, i64 5, !dbg !15
  %qr.n1773 = extractelement <8 x float> %qr.n1662, i64 6, !dbg !15
  %qr.n1774 = extractelement <8 x float> %qr.n1662, i64 7, !dbg !15
  %qr.n1775 = select i1 %.not2, float %qr.n1772, float %qr.n1771, !dbg !15
  %qr.n1776 = bitcast float %qr.n1775 to i32, !dbg !15
  %qr.n1777 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1776), !dbg !15
  %qr.n1778 = bitcast i32 %qr.n1777 to float, !dbg !15
  %qr.n1779 = select i1 %.not2, float %qr.n1771, float %qr.n1778, !dbg !15
  %qr.n1780 = select i1 %.not2, float %qr.n1778, float %qr.n1772, !dbg !15
  %qr.n1781 = select i1 %.not2, float %qr.n1774, float %qr.n1773, !dbg !15
  %qr.n1782 = bitcast float %qr.n1781 to i32, !dbg !15
  %qr.n1783 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1782), !dbg !15
  %qr.n1784 = bitcast i32 %qr.n1783 to float, !dbg !15
  %qr.n1785 = select i1 %.not2, float %qr.n1773, float %qr.n1784, !dbg !15
  %qr.n1786 = select i1 %.not2, float %qr.n1784, float %qr.n1774, !dbg !15
  %qr.n1787 = select i1 %.not3, float %qr.n1785, float %qr.n1779, !dbg !15
  %qr.n1788 = bitcast float %qr.n1787 to i32, !dbg !15
  %qr.n1789 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1788), !dbg !15
  %qr.n1790 = select i1 %.not3, float %qr.n1786, float %qr.n1780, !dbg !15
  %qr.n1791 = bitcast float %qr.n1790 to i32, !dbg !15
  %qr.n1792 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1791), !dbg !15
  %qr.n1793 = extractvalue [2 x <8 x float>] %qr.n46, 1, !dbg !15
  %qr.n1794 = extractelement <8 x float> %qr.n1793, i64 0, !dbg !15
  %qr.n1795 = extractelement <8 x float> %qr.n1793, i64 1, !dbg !15
  %qr.n1796 = extractelement <8 x float> %qr.n1793, i64 2, !dbg !15
  %qr.n1797 = extractelement <8 x float> %qr.n1793, i64 3, !dbg !15
  %qr.n1798 = select i1 %.not2, float %qr.n1795, float %qr.n1794, !dbg !15
  %qr.n1799 = bitcast float %qr.n1798 to i32, !dbg !15
  %qr.n1800 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1799), !dbg !15
  %qr.n1801 = bitcast i32 %qr.n1800 to float, !dbg !15
  %qr.n1802 = select i1 %.not2, float %qr.n1794, float %qr.n1801, !dbg !15
  %qr.n1803 = select i1 %.not2, float %qr.n1801, float %qr.n1795, !dbg !15
  %qr.n1804 = select i1 %.not2, float %qr.n1797, float %qr.n1796, !dbg !15
  %qr.n1805 = bitcast float %qr.n1804 to i32, !dbg !15
  %qr.n1806 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1805), !dbg !15
  %qr.n1807 = bitcast i32 %qr.n1806 to float, !dbg !15
  %qr.n1808 = select i1 %.not2, float %qr.n1796, float %qr.n1807, !dbg !15
  %qr.n1809 = select i1 %.not2, float %qr.n1807, float %qr.n1797, !dbg !15
  %qr.n1810 = select i1 %.not3, float %qr.n1808, float %qr.n1802, !dbg !15
  %qr.n1811 = bitcast float %qr.n1810 to i32, !dbg !15
  %qr.n1812 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1811), !dbg !15
  %qr.n1813 = bitcast i32 %qr.n1812 to float, !dbg !15
  %qr.n1814 = select i1 %.not3, float %qr.n1802, float %qr.n1813, !dbg !15
  %qr.n1815 = select i1 %.not3, float %qr.n1813, float %qr.n1808, !dbg !15
  %qr.n1816 = select i1 %.not3, float %qr.n1809, float %qr.n1803, !dbg !15
  %qr.n1817 = bitcast float %qr.n1816 to i32, !dbg !15
  %qr.n1818 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1817), !dbg !15
  %qr.n1819 = bitcast i32 %qr.n1818 to float, !dbg !15
  %qr.n1820 = select i1 %.not3, float %qr.n1803, float %qr.n1819, !dbg !15
  %qr.n1821 = select i1 %.not3, float %qr.n1819, float %qr.n1809, !dbg !15
  %qr.n1822 = extractvalue [2 x <8 x float>] %qr.n45, 1, !dbg !15
  %qr.n1823 = extractelement <8 x float> %qr.n1822, i64 0, !dbg !15
  %qr.n1824 = extractelement <8 x float> %qr.n1822, i64 1, !dbg !15
  %qr.n1825 = extractelement <8 x float> %qr.n1822, i64 2, !dbg !15
  %qr.n1826 = extractelement <8 x float> %qr.n1822, i64 3, !dbg !15
  %qr.n1827 = select i1 %.not2, float %qr.n1824, float %qr.n1823, !dbg !15
  %qr.n1828 = bitcast float %qr.n1827 to i32, !dbg !15
  %qr.n1829 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1828), !dbg !15
  %qr.n1830 = bitcast i32 %qr.n1829 to float, !dbg !15
  %qr.n1831 = select i1 %.not2, float %qr.n1823, float %qr.n1830, !dbg !15
  %qr.n1832 = select i1 %.not2, float %qr.n1830, float %qr.n1824, !dbg !15
  %qr.n1833 = select i1 %.not2, float %qr.n1826, float %qr.n1825, !dbg !15
  %qr.n1834 = bitcast float %qr.n1833 to i32, !dbg !15
  %qr.n1835 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1834), !dbg !15
  %qr.n1836 = bitcast i32 %qr.n1835 to float, !dbg !15
  %qr.n1837 = select i1 %.not2, float %qr.n1825, float %qr.n1836, !dbg !15
  %qr.n1838 = select i1 %.not2, float %qr.n1836, float %qr.n1826, !dbg !15
  %qr.n1839 = select i1 %.not3, float %qr.n1837, float %qr.n1831, !dbg !15
  %qr.n1840 = bitcast float %qr.n1839 to i32, !dbg !15
  %qr.n1841 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1840), !dbg !15
  %qr.n1842 = bitcast i32 %qr.n1841 to float, !dbg !15
  %qr.n1843 = select i1 %.not3, float %qr.n1831, float %qr.n1842, !dbg !15
  %qr.n1844 = select i1 %.not3, float %qr.n1842, float %qr.n1837, !dbg !15
  %qr.n1845 = select i1 %.not3, float %qr.n1838, float %qr.n1832, !dbg !15
  %qr.n1846 = bitcast float %qr.n1845 to i32, !dbg !15
  %qr.n1847 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1846), !dbg !15
  %qr.n1848 = bitcast i32 %qr.n1847 to float, !dbg !15
  %qr.n1849 = select i1 %.not3, float %qr.n1832, float %qr.n1848, !dbg !15
  %qr.n1850 = select i1 %.not3, float %qr.n1848, float %qr.n1838, !dbg !15
  %qr.n1851 = extractvalue [2 x <8 x float>] %qr.n44, 1, !dbg !15
  %qr.n1852 = extractelement <8 x float> %qr.n1851, i64 0, !dbg !15
  %qr.n1853 = extractelement <8 x float> %qr.n1851, i64 1, !dbg !15
  %qr.n1854 = extractelement <8 x float> %qr.n1851, i64 2, !dbg !15
  %qr.n1855 = extractelement <8 x float> %qr.n1851, i64 3, !dbg !15
  %qr.n1856 = select i1 %.not2, float %qr.n1853, float %qr.n1852, !dbg !15
  %qr.n1857 = bitcast float %qr.n1856 to i32, !dbg !15
  %qr.n1858 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1857), !dbg !15
  %qr.n1859 = bitcast i32 %qr.n1858 to float, !dbg !15
  %qr.n1860 = select i1 %.not2, float %qr.n1852, float %qr.n1859, !dbg !15
  %qr.n1861 = select i1 %.not2, float %qr.n1859, float %qr.n1853, !dbg !15
  %qr.n1862 = select i1 %.not2, float %qr.n1855, float %qr.n1854, !dbg !15
  %qr.n1863 = bitcast float %qr.n1862 to i32, !dbg !15
  %qr.n1864 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1863), !dbg !15
  %qr.n1865 = bitcast i32 %qr.n1864 to float, !dbg !15
  %qr.n1866 = select i1 %.not2, float %qr.n1854, float %qr.n1865, !dbg !15
  %qr.n1867 = select i1 %.not2, float %qr.n1865, float %qr.n1855, !dbg !15
  %qr.n1868 = select i1 %.not3, float %qr.n1866, float %qr.n1860, !dbg !15
  %qr.n1869 = bitcast float %qr.n1868 to i32, !dbg !15
  %qr.n1870 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1869), !dbg !15
  %qr.n1871 = bitcast i32 %qr.n1870 to float, !dbg !15
  %qr.n1872 = select i1 %.not3, float %qr.n1860, float %qr.n1871, !dbg !15
  %qr.n1873 = select i1 %.not3, float %qr.n1871, float %qr.n1866, !dbg !15
  %qr.n1874 = select i1 %.not3, float %qr.n1867, float %qr.n1861, !dbg !15
  %qr.n1875 = bitcast float %qr.n1874 to i32, !dbg !15
  %qr.n1876 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1875), !dbg !15
  %qr.n1877 = bitcast i32 %qr.n1876 to float, !dbg !15
  %qr.n1878 = select i1 %.not3, float %qr.n1861, float %qr.n1877, !dbg !15
  %qr.n1879 = select i1 %.not3, float %qr.n1877, float %qr.n1867, !dbg !15
  %qr.n1880 = extractvalue [2 x <8 x float>] %qr.n42, 1, !dbg !15
  %qr.n1881 = extractelement <8 x float> %qr.n1880, i64 0, !dbg !15
  %qr.n1882 = extractelement <8 x float> %qr.n1880, i64 1, !dbg !15
  %qr.n1883 = extractelement <8 x float> %qr.n1880, i64 2, !dbg !15
  %qr.n1884 = extractelement <8 x float> %qr.n1880, i64 3, !dbg !15
  %qr.n1885 = select i1 %.not2, float %qr.n1882, float %qr.n1881, !dbg !15
  %qr.n1886 = bitcast float %qr.n1885 to i32, !dbg !15
  %qr.n1887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1886), !dbg !15
  %qr.n1888 = bitcast i32 %qr.n1887 to float, !dbg !15
  %qr.n1889 = select i1 %.not2, float %qr.n1881, float %qr.n1888, !dbg !15
  %qr.n1890 = select i1 %.not2, float %qr.n1888, float %qr.n1882, !dbg !15
  %qr.n1891 = select i1 %.not2, float %qr.n1884, float %qr.n1883, !dbg !15
  %qr.n1892 = bitcast float %qr.n1891 to i32, !dbg !15
  %qr.n1893 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1892), !dbg !15
  %qr.n1894 = bitcast i32 %qr.n1893 to float, !dbg !15
  %qr.n1895 = select i1 %.not2, float %qr.n1883, float %qr.n1894, !dbg !15
  %qr.n1896 = select i1 %.not2, float %qr.n1894, float %qr.n1884, !dbg !15
  %qr.n1897 = select i1 %.not3, float %qr.n1895, float %qr.n1889, !dbg !15
  %qr.n1898 = bitcast float %qr.n1897 to i32, !dbg !15
  %qr.n1899 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1898), !dbg !15
  %qr.n1900 = select i1 %.not3, float %qr.n1896, float %qr.n1890, !dbg !15
  %qr.n1901 = bitcast float %qr.n1900 to i32, !dbg !15
  %qr.n1902 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1901), !dbg !15
  %qr.n1903 = extractvalue [2 x <1 x float>] %qr.n43, 1, !dbg !19
  %qr.n1904 = shufflevector <1 x float> %qr.n1903, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !19
  %qr.n1905 = extractelement <8 x float> %qr.n1793, i64 4, !dbg !15
  %qr.n1906 = extractelement <8 x float> %qr.n1793, i64 5, !dbg !15
  %qr.n1907 = extractelement <8 x float> %qr.n1793, i64 6, !dbg !15
  %qr.n1908 = extractelement <8 x float> %qr.n1793, i64 7, !dbg !15
  %qr.n1909 = select i1 %.not2, float %qr.n1906, float %qr.n1905, !dbg !15
  %qr.n1910 = bitcast float %qr.n1909 to i32, !dbg !15
  %qr.n1911 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1910), !dbg !15
  %qr.n1912 = bitcast i32 %qr.n1911 to float, !dbg !15
  %qr.n1913 = select i1 %.not2, float %qr.n1905, float %qr.n1912, !dbg !15
  %qr.n1914 = select i1 %.not2, float %qr.n1912, float %qr.n1906, !dbg !15
  %qr.n1915 = select i1 %.not2, float %qr.n1908, float %qr.n1907, !dbg !15
  %qr.n1916 = bitcast float %qr.n1915 to i32, !dbg !15
  %qr.n1917 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1916), !dbg !15
  %qr.n1918 = bitcast i32 %qr.n1917 to float, !dbg !15
  %qr.n1919 = select i1 %.not2, float %qr.n1907, float %qr.n1918, !dbg !15
  %qr.n1920 = select i1 %.not2, float %qr.n1918, float %qr.n1908, !dbg !15
  %qr.n1921 = select i1 %.not3, float %qr.n1919, float %qr.n1913, !dbg !15
  %qr.n1922 = bitcast float %qr.n1921 to i32, !dbg !15
  %qr.n1923 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1922), !dbg !15
  %qr.n1924 = bitcast i32 %qr.n1923 to float, !dbg !15
  %qr.n1925 = select i1 %.not3, float %qr.n1913, float %qr.n1924, !dbg !15
  %qr.n1926 = select i1 %.not3, float %qr.n1924, float %qr.n1919, !dbg !15
  %qr.n1927 = select i1 %.not3, float %qr.n1920, float %qr.n1914, !dbg !15
  %qr.n1928 = bitcast float %qr.n1927 to i32, !dbg !15
  %qr.n1929 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1928), !dbg !15
  %qr.n1930 = bitcast i32 %qr.n1929 to float, !dbg !15
  %qr.n1931 = select i1 %.not3, float %qr.n1914, float %qr.n1930, !dbg !15
  %qr.n1932 = select i1 %.not3, float %qr.n1930, float %qr.n1920, !dbg !15
  %qr.n1933 = extractelement <8 x float> %qr.n1822, i64 4, !dbg !15
  %qr.n1934 = extractelement <8 x float> %qr.n1822, i64 5, !dbg !15
  %qr.n1935 = extractelement <8 x float> %qr.n1822, i64 6, !dbg !15
  %qr.n1936 = extractelement <8 x float> %qr.n1822, i64 7, !dbg !15
  %qr.n1937 = select i1 %.not2, float %qr.n1934, float %qr.n1933, !dbg !15
  %qr.n1938 = bitcast float %qr.n1937 to i32, !dbg !15
  %qr.n1939 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1938), !dbg !15
  %qr.n1940 = bitcast i32 %qr.n1939 to float, !dbg !15
  %qr.n1941 = select i1 %.not2, float %qr.n1933, float %qr.n1940, !dbg !15
  %qr.n1942 = select i1 %.not2, float %qr.n1940, float %qr.n1934, !dbg !15
  %qr.n1943 = select i1 %.not2, float %qr.n1936, float %qr.n1935, !dbg !15
  %qr.n1944 = bitcast float %qr.n1943 to i32, !dbg !15
  %qr.n1945 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1944), !dbg !15
  %qr.n1946 = bitcast i32 %qr.n1945 to float, !dbg !15
  %qr.n1947 = select i1 %.not2, float %qr.n1935, float %qr.n1946, !dbg !15
  %qr.n1948 = select i1 %.not2, float %qr.n1946, float %qr.n1936, !dbg !15
  %qr.n1949 = select i1 %.not3, float %qr.n1947, float %qr.n1941, !dbg !15
  %qr.n1950 = bitcast float %qr.n1949 to i32, !dbg !15
  %qr.n1951 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1950), !dbg !15
  %qr.n1952 = bitcast i32 %qr.n1951 to float, !dbg !15
  %qr.n1953 = select i1 %.not3, float %qr.n1941, float %qr.n1952, !dbg !15
  %qr.n1954 = select i1 %.not3, float %qr.n1952, float %qr.n1947, !dbg !15
  %qr.n1955 = select i1 %.not3, float %qr.n1948, float %qr.n1942, !dbg !15
  %qr.n1956 = bitcast float %qr.n1955 to i32, !dbg !15
  %qr.n1957 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1956), !dbg !15
  %qr.n1958 = bitcast i32 %qr.n1957 to float, !dbg !15
  %qr.n1959 = select i1 %.not3, float %qr.n1942, float %qr.n1958, !dbg !15
  %qr.n1960 = select i1 %.not3, float %qr.n1958, float %qr.n1948, !dbg !15
  %qr.n1961 = extractelement <8 x float> %qr.n1851, i64 4, !dbg !15
  %qr.n1962 = extractelement <8 x float> %qr.n1851, i64 5, !dbg !15
  %qr.n1963 = extractelement <8 x float> %qr.n1851, i64 6, !dbg !15
  %qr.n1964 = extractelement <8 x float> %qr.n1851, i64 7, !dbg !15
  %qr.n1965 = select i1 %.not2, float %qr.n1962, float %qr.n1961, !dbg !15
  %qr.n1966 = bitcast float %qr.n1965 to i32, !dbg !15
  %qr.n1967 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1966), !dbg !15
  %qr.n1968 = bitcast i32 %qr.n1967 to float, !dbg !15
  %qr.n1969 = select i1 %.not2, float %qr.n1961, float %qr.n1968, !dbg !15
  %qr.n1970 = select i1 %.not2, float %qr.n1968, float %qr.n1962, !dbg !15
  %qr.n1971 = select i1 %.not2, float %qr.n1964, float %qr.n1963, !dbg !15
  %qr.n1972 = bitcast float %qr.n1971 to i32, !dbg !15
  %qr.n1973 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1972), !dbg !15
  %qr.n1974 = bitcast i32 %qr.n1973 to float, !dbg !15
  %qr.n1975 = select i1 %.not2, float %qr.n1963, float %qr.n1974, !dbg !15
  %qr.n1976 = select i1 %.not2, float %qr.n1974, float %qr.n1964, !dbg !15
  %qr.n1977 = select i1 %.not3, float %qr.n1975, float %qr.n1969, !dbg !15
  %qr.n1978 = bitcast float %qr.n1977 to i32, !dbg !15
  %qr.n1979 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1978), !dbg !15
  %qr.n1980 = bitcast i32 %qr.n1979 to float, !dbg !15
  %qr.n1981 = select i1 %.not3, float %qr.n1969, float %qr.n1980, !dbg !15
  %qr.n1982 = select i1 %.not3, float %qr.n1980, float %qr.n1975, !dbg !15
  %qr.n1983 = select i1 %.not3, float %qr.n1976, float %qr.n1970, !dbg !15
  %qr.n1984 = bitcast float %qr.n1983 to i32, !dbg !15
  %qr.n1985 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n1984), !dbg !15
  %qr.n1986 = bitcast i32 %qr.n1985 to float, !dbg !15
  %qr.n1987 = select i1 %.not3, float %qr.n1970, float %qr.n1986, !dbg !15
  %qr.n1988 = select i1 %.not3, float %qr.n1986, float %qr.n1976, !dbg !15
  %qr.n1989 = extractelement <8 x float> %qr.n1880, i64 4, !dbg !15
  %qr.n1990 = extractelement <8 x float> %qr.n1880, i64 5, !dbg !15
  %qr.n1991 = extractelement <8 x float> %qr.n1880, i64 6, !dbg !15
  %qr.n1992 = extractelement <8 x float> %qr.n1880, i64 7, !dbg !15
  %qr.n1993 = select i1 %.not2, float %qr.n1990, float %qr.n1989, !dbg !15
  %qr.n1994 = bitcast float %qr.n1993 to i32, !dbg !15
  %qr.n1995 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n1994), !dbg !15
  %qr.n1996 = bitcast i32 %qr.n1995 to float, !dbg !15
  %qr.n1997 = select i1 %.not2, float %qr.n1989, float %qr.n1996, !dbg !15
  %qr.n1998 = select i1 %.not2, float %qr.n1996, float %qr.n1990, !dbg !15
  %qr.n1999 = select i1 %.not2, float %qr.n1992, float %qr.n1991, !dbg !15
  %qr.n2000 = bitcast float %qr.n1999 to i32, !dbg !15
  %qr.n2001 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1576, i32 %qr.n2000), !dbg !15
  %qr.n2002 = bitcast i32 %qr.n2001 to float, !dbg !15
  %qr.n2003 = select i1 %.not2, float %qr.n1991, float %qr.n2002, !dbg !15
  %qr.n2004 = select i1 %.not2, float %qr.n2002, float %qr.n1992, !dbg !15
  %qr.n2005 = select i1 %.not3, float %qr.n2003, float %qr.n1997, !dbg !15
  %qr.n2006 = bitcast float %qr.n2005 to i32, !dbg !15
  %qr.n2007 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n2006), !dbg !15
  %qr.n2008 = select i1 %.not3, float %qr.n2004, float %qr.n1998, !dbg !15
  %qr.n2009 = bitcast float %qr.n2008 to i32, !dbg !15
  %qr.n2010 = call i32 @llvm.amdgcn.ds.bpermute(i32 %qr.n1592, i32 %qr.n2009), !dbg !15
  br i1 true, label %qr.n2011, label %qr.n2075, !dbg !21

qr.n2011:                                             ; preds = %qr.n1560
  %qr.n2012 = bitcast i32 %qr.n2007 to float, !dbg !15
  %qr.n2013 = select i1 %.not3, float %qr.n1997, float %qr.n2012, !dbg !15
  %qr.n2014 = insertelement <4 x float> poison, float %qr.n2013, i64 0, !dbg !19
  %qr.n2015 = bitcast i32 %qr.n2010 to float, !dbg !15
  %qr.n2016 = select i1 %.not3, float %qr.n1998, float %qr.n2015, !dbg !15
  %qr.n2017 = insertelement <4 x float> %qr.n2014, float %qr.n2016, i64 1, !dbg !19
  %qr.n2018 = select i1 %.not3, float %qr.n2012, float %qr.n2003, !dbg !15
  %qr.n2019 = insertelement <4 x float> %qr.n2017, float %qr.n2018, i64 2, !dbg !19
  %qr.n2020 = select i1 %.not3, float %qr.n2015, float %qr.n2004, !dbg !15
  %qr.n2021 = insertelement <4 x float> %qr.n2019, float %qr.n2020, i64 3, !dbg !19
  %row.reciprocal = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %qr.n1904, !dbg !19
  %normalized = fmul <4 x float> %qr.n2021, %row.reciprocal, !dbg !19
  %qr.n2022 = fptrunc <4 x float> %normalized to <4 x half>, !dbg !20
  %qr.n2023 = extractelement <4 x half> %qr.n2022, i64 3, !dbg !20
  %qr.n2024 = extractelement <4 x half> %qr.n2022, i64 2, !dbg !20
  %qr.n2025 = extractelement <4 x half> %qr.n2022, i64 1, !dbg !20
  %qr.n2026 = extractelement <4 x half> %qr.n2022, i64 0, !dbg !20
  %qr.n2027 = bitcast i32 %qr.n1899 to float, !dbg !15
  %qr.n2028 = select i1 %.not3, float %qr.n1889, float %qr.n2027, !dbg !15
  %qr.n2029 = insertelement <4 x float> poison, float %qr.n2028, i64 0, !dbg !19
  %qr.n2030 = bitcast i32 %qr.n1902 to float, !dbg !15
  %qr.n2031 = select i1 %.not3, float %qr.n1890, float %qr.n2030, !dbg !15
  %qr.n2032 = insertelement <4 x float> %qr.n2029, float %qr.n2031, i64 1, !dbg !19
  %qr.n2033 = select i1 %.not3, float %qr.n2027, float %qr.n1895, !dbg !15
  %qr.n2034 = insertelement <4 x float> %qr.n2032, float %qr.n2033, i64 2, !dbg !19
  %qr.n2035 = select i1 %.not3, float %qr.n2030, float %qr.n1896, !dbg !15
  %qr.n2036 = insertelement <4 x float> %qr.n2034, float %qr.n2035, i64 3, !dbg !19
  %normalized71 = fmul <4 x float> %qr.n2036, %row.reciprocal, !dbg !19
  %qr.n2037 = fptrunc <4 x float> %normalized71 to <4 x half>, !dbg !20
  %qr.n2038 = extractelement <4 x half> %qr.n2037, i64 3, !dbg !20
  %qr.n2039 = extractelement <4 x half> %qr.n2037, i64 2, !dbg !20
  %qr.n2040 = extractelement <4 x half> %qr.n2037, i64 1, !dbg !20
  %qr.n2041 = extractelement <4 x half> %qr.n2037, i64 0, !dbg !20
  %qr.n2042 = bitcast i32 %qr.n1789 to float, !dbg !15
  %qr.n2043 = select i1 %.not3, float %qr.n1779, float %qr.n2042, !dbg !15
  %qr.n2044 = insertelement <4 x float> poison, float %qr.n2043, i64 0, !dbg !19
  %qr.n2045 = bitcast i32 %qr.n1792 to float, !dbg !15
  %qr.n2046 = select i1 %.not3, float %qr.n1780, float %qr.n2045, !dbg !15
  %qr.n2047 = insertelement <4 x float> %qr.n2044, float %qr.n2046, i64 1, !dbg !19
  %qr.n2048 = select i1 %.not3, float %qr.n2042, float %qr.n1785, !dbg !15
  %qr.n2049 = insertelement <4 x float> %qr.n2047, float %qr.n2048, i64 2, !dbg !19
  %qr.n2050 = select i1 %.not3, float %qr.n2045, float %qr.n1786, !dbg !15
  %qr.n2051 = insertelement <4 x float> %qr.n2049, float %qr.n2050, i64 3, !dbg !19
  %row.reciprocal72 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %qr.n1686, !dbg !19
  %normalized73 = fmul <4 x float> %qr.n2051, %row.reciprocal72, !dbg !19
  %qr.n2052 = fptrunc <4 x float> %normalized73 to <4 x half>, !dbg !20
  %qr.n2053 = extractelement <4 x half> %qr.n2052, i64 3, !dbg !20
  %qr.n2054 = extractelement <4 x half> %qr.n2052, i64 2, !dbg !20
  %qr.n2055 = extractelement <4 x half> %qr.n2052, i64 1, !dbg !20
  %qr.n2056 = extractelement <4 x half> %qr.n2052, i64 0, !dbg !20
  %qr.n2057 = bitcast i32 %qr.n1681 to float, !dbg !15
  %qr.n2058 = select i1 %.not3, float %qr.n1671, float %qr.n2057, !dbg !15
  %qr.n2059 = insertelement <4 x float> poison, float %qr.n2058, i64 0, !dbg !19
  %qr.n2060 = bitcast i32 %qr.n1684 to float, !dbg !15
  %qr.n2061 = select i1 %.not3, float %qr.n1672, float %qr.n2060, !dbg !15
  %qr.n2062 = insertelement <4 x float> %qr.n2059, float %qr.n2061, i64 1, !dbg !19
  %qr.n2063 = select i1 %.not3, float %qr.n2057, float %qr.n1677, !dbg !15
  %qr.n2064 = insertelement <4 x float> %qr.n2062, float %qr.n2063, i64 2, !dbg !19
  %qr.n2065 = select i1 %.not3, float %qr.n2060, float %qr.n1678, !dbg !15
  %qr.n2066 = insertelement <4 x float> %qr.n2064, float %qr.n2065, i64 3, !dbg !19
  %normalized74 = fmul <4 x float> %qr.n2066, %row.reciprocal72, !dbg !19
  %qr.n2067 = fptrunc <4 x float> %normalized74 to <4 x half>, !dbg !20
  %qr.n2068 = extractelement <4 x half> %qr.n2067, i64 3, !dbg !20
  %qr.n2069 = extractelement <4 x half> %qr.n2067, i64 2, !dbg !20
  %qr.n2070 = extractelement <4 x half> %qr.n2067, i64 1, !dbg !20
  %qr.n2071 = extractelement <4 x half> %qr.n2067, i64 0, !dbg !20
  %qr.n2072 = add nsw i64 %qr.n10, %qr.n15, !dbg !21
  %qr.n2073 = shl nsw i64 %qr.n2072, 19, !dbg !21
  %qr.n2074 = and i64 %qr.n2073, 16252928, !dbg !21
  %address.thread389 = call i32 asm sideeffect "", "=v,0"(i32 %qr.n24) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local390 = zext nneg i32 %address.thread389 to i64, !dbg !12
  %address.local391 = add nsw i64 %qr.n18, %address.local390, !dbg !21
  %address.local392 = lshr i64 %address.local390, 6, !dbg !12
  %address.local393 = shl nuw nsw i64 %address.local392, 4, !dbg !21
  %address.local394 = add nsw i64 %address.local391, %address.local393, !dbg !21
  %address.local395 = and i64 %address.local390, 240, !dbg !21
  %address.local396 = sub nsw i64 %address.local394, %address.local395, !dbg !21
  %address.local397 = add nsw i64 %address.local396, %qr.n19, !dbg !21
  %address.local398 = shl nuw nsw i64 %address.local397, 7, !dbg !21
  %address.local399 = add nuw nsw i64 %qr.n2074, %address.local398, !dbg !21
  %address.local400 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local401 = lshr i64 %address.local390, 2, !dbg !12
  %address.local402 = and i64 %address.local401, 12, !dbg !12
  %address.local403 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local400, i64 %address.local402, !dbg !21
  store half %qr.n2071, ptr addrspace(1) %address.local403, align 2, !dbg !21
  %address.local404 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local405 = or disjoint i64 %address.local402, 1, !dbg !21
  %address.local406 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local404, i64 %address.local405, !dbg !21
  store half %qr.n2070, ptr addrspace(1) %address.local406, align 2, !dbg !21
  %address.local407 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local408 = or disjoint i64 %address.local402, 2, !dbg !21
  %address.local409 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local407, i64 %address.local408, !dbg !21
  store half %qr.n2069, ptr addrspace(1) %address.local409, align 2, !dbg !21
  %address.local410 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local411 = or disjoint i64 %address.local402, 3, !dbg !21
  %address.local412 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local410, i64 %address.local411, !dbg !21
  store half %qr.n2068, ptr addrspace(1) %address.local412, align 2, !dbg !21
  %address.local413 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local414 = or disjoint i64 %address.local402, 16, !dbg !21
  %address.local415 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local413, i64 %address.local414, !dbg !21
  store half %qr.n2056, ptr addrspace(1) %address.local415, align 2, !dbg !21
  %address.local416 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local417 = or disjoint i64 %address.local402, 17, !dbg !21
  %address.local418 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local416, i64 %address.local417, !dbg !21
  store half %qr.n2055, ptr addrspace(1) %address.local418, align 2, !dbg !21
  %address.local419 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local420 = or disjoint i64 %address.local402, 18, !dbg !21
  %address.local421 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local419, i64 %address.local420, !dbg !21
  store half %qr.n2054, ptr addrspace(1) %address.local421, align 2, !dbg !21
  %address.local422 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local399, !dbg !21
  %address.local423 = or disjoint i64 %address.local402, 19, !dbg !21
  %address.local424 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local422, i64 %address.local423, !dbg !21
  store half %qr.n2053, ptr addrspace(1) %address.local424, align 2, !dbg !21
  %address.local425 = shl nsw i64 %address.local397, 7, !dbg !21
  %address.local426 = add i64 %address.local425, 8192, !dbg !21
  %address.local427 = add nuw nsw i64 %qr.n2074, %address.local426, !dbg !21
  %address.local428 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local429 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local428, i64 %address.local402, !dbg !21
  store half %qr.n2041, ptr addrspace(1) %address.local429, align 2, !dbg !21
  %address.local430 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local431 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local430, i64 %address.local405, !dbg !21
  store half %qr.n2040, ptr addrspace(1) %address.local431, align 2, !dbg !21
  %address.local432 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local433 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local432, i64 %address.local408, !dbg !21
  store half %qr.n2039, ptr addrspace(1) %address.local433, align 2, !dbg !21
  %address.local434 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local435 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local434, i64 %address.local411, !dbg !21
  store half %qr.n2038, ptr addrspace(1) %address.local435, align 2, !dbg !21
  %address.local436 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local437 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local436, i64 %address.local414, !dbg !21
  store half %qr.n2026, ptr addrspace(1) %address.local437, align 2, !dbg !21
  %address.local438 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local439 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local438, i64 %address.local417, !dbg !21
  store half %qr.n2025, ptr addrspace(1) %address.local439, align 2, !dbg !21
  %address.local440 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local441 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local440, i64 %address.local420, !dbg !21
  store half %qr.n2024, ptr addrspace(1) %address.local441, align 2, !dbg !21
  %address.local442 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local427, !dbg !21
  %address.local443 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local442, i64 %address.local423, !dbg !21
  store half %qr.n2023, ptr addrspace(1) %address.local443, align 2, !dbg !21
  br label %qr.n2075, !dbg !21

qr.n2075:                                             ; preds = %qr.n2011, %qr.n1560
  br i1 true, label %qr.n2076, label %qr.n2114, !dbg !21

qr.n2076:                                             ; preds = %qr.n2075
  %qr.n2077 = insertelement <4 x float> poison, float %qr.n1981, i64 0, !dbg !19
  %qr.n2078 = insertelement <4 x float> %qr.n2077, float %qr.n1987, i64 1, !dbg !19
  %qr.n2079 = insertelement <4 x float> %qr.n2078, float %qr.n1982, i64 2, !dbg !19
  %qr.n2080 = insertelement <4 x float> %qr.n2079, float %qr.n1988, i64 3, !dbg !19
  %row.reciprocal75 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %qr.n1904, !dbg !19
  %normalized76 = fmul <4 x float> %qr.n2080, %row.reciprocal75, !dbg !19
  %qr.n2081 = fptrunc <4 x float> %normalized76 to <4 x half>, !dbg !20
  %qr.n2082 = extractelement <4 x half> %qr.n2081, i64 3, !dbg !20
  %qr.n2083 = extractelement <4 x half> %qr.n2081, i64 2, !dbg !20
  %qr.n2084 = extractelement <4 x half> %qr.n2081, i64 1, !dbg !20
  %qr.n2085 = extractelement <4 x half> %qr.n2081, i64 0, !dbg !20
  %qr.n2086 = insertelement <4 x float> poison, float %qr.n1872, i64 0, !dbg !19
  %qr.n2087 = insertelement <4 x float> %qr.n2086, float %qr.n1878, i64 1, !dbg !19
  %qr.n2088 = insertelement <4 x float> %qr.n2087, float %qr.n1873, i64 2, !dbg !19
  %qr.n2089 = insertelement <4 x float> %qr.n2088, float %qr.n1879, i64 3, !dbg !19
  %normalized77 = fmul <4 x float> %qr.n2089, %row.reciprocal75, !dbg !19
  %qr.n2090 = fptrunc <4 x float> %normalized77 to <4 x half>, !dbg !20
  %qr.n2091 = extractelement <4 x half> %qr.n2090, i64 3, !dbg !20
  %qr.n2092 = extractelement <4 x half> %qr.n2090, i64 2, !dbg !20
  %qr.n2093 = extractelement <4 x half> %qr.n2090, i64 1, !dbg !20
  %qr.n2094 = extractelement <4 x half> %qr.n2090, i64 0, !dbg !20
  %qr.n2095 = insertelement <4 x float> poison, float %qr.n1763, i64 0, !dbg !19
  %qr.n2096 = insertelement <4 x float> %qr.n2095, float %qr.n1769, i64 1, !dbg !19
  %qr.n2097 = insertelement <4 x float> %qr.n2096, float %qr.n1764, i64 2, !dbg !19
  %qr.n2098 = insertelement <4 x float> %qr.n2097, float %qr.n1770, i64 3, !dbg !19
  %row.reciprocal78 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %qr.n1686, !dbg !19
  %normalized79 = fmul <4 x float> %qr.n2098, %row.reciprocal78, !dbg !19
  %qr.n2099 = fptrunc <4 x float> %normalized79 to <4 x half>, !dbg !20
  %qr.n2100 = extractelement <4 x half> %qr.n2099, i64 3, !dbg !20
  %qr.n2101 = extractelement <4 x half> %qr.n2099, i64 2, !dbg !20
  %qr.n2102 = extractelement <4 x half> %qr.n2099, i64 1, !dbg !20
  %qr.n2103 = extractelement <4 x half> %qr.n2099, i64 0, !dbg !20
  %qr.n2104 = insertelement <4 x float> poison, float %qr.n1654, i64 0, !dbg !19
  %qr.n2105 = insertelement <4 x float> %qr.n2104, float %qr.n1660, i64 1, !dbg !19
  %qr.n2106 = insertelement <4 x float> %qr.n2105, float %qr.n1655, i64 2, !dbg !19
  %qr.n2107 = insertelement <4 x float> %qr.n2106, float %qr.n1661, i64 3, !dbg !19
  %normalized80 = fmul <4 x float> %qr.n2107, %row.reciprocal78, !dbg !19
  %qr.n2108 = fptrunc <4 x float> %normalized80 to <4 x half>, !dbg !20
  %qr.n2109 = extractelement <4 x half> %qr.n2108, i64 3, !dbg !20
  %qr.n2110 = extractelement <4 x half> %qr.n2108, i64 2, !dbg !20
  %qr.n2111 = extractelement <4 x half> %qr.n2108, i64 1, !dbg !20
  %qr.n2112 = extractelement <4 x half> %qr.n2108, i64 0, !dbg !20
  %qr.n2113 = shl nuw nsw i64 %qr.n17, 19, !dbg !21
  %address.thread444 = call i32 asm sideeffect "", "=v,0"(i32 %qr.n24) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local445 = zext nneg i32 %address.thread444 to i64, !dbg !12
  %address.local446 = add nsw i64 %qr.n18, %address.local445, !dbg !21
  %address.local447 = lshr i64 %address.local445, 6, !dbg !12
  %address.local448 = shl nuw nsw i64 %address.local447, 4, !dbg !21
  %address.local449 = add nsw i64 %address.local446, %address.local448, !dbg !21
  %address.local450 = and i64 %address.local445, 240, !dbg !21
  %address.local451 = sub nsw i64 %address.local449, %address.local450, !dbg !21
  %address.local452 = add nsw i64 %address.local451, %qr.n19, !dbg !21
  %address.local453 = shl nuw nsw i64 %address.local452, 7, !dbg !21
  %address.local454 = add nuw nsw i64 %qr.n2113, %address.local453, !dbg !21
  %address.local455 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local456 = lshr i64 %address.local445, 2, !dbg !12
  %address.local457 = and i64 %address.local456, 12, !dbg !12
  %address.local458 = or disjoint i64 %address.local457, 32, !dbg !21
  %address.local459 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local455, i64 %address.local458, !dbg !21
  store half %qr.n2112, ptr addrspace(1) %address.local459, align 2, !dbg !21
  %address.local460 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local461 = or disjoint i64 %address.local457, 33, !dbg !21
  %address.local462 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local460, i64 %address.local461, !dbg !21
  store half %qr.n2111, ptr addrspace(1) %address.local462, align 2, !dbg !21
  %address.local463 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local464 = or disjoint i64 %address.local457, 34, !dbg !21
  %address.local465 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local463, i64 %address.local464, !dbg !21
  store half %qr.n2110, ptr addrspace(1) %address.local465, align 2, !dbg !21
  %address.local466 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local467 = or disjoint i64 %address.local457, 35, !dbg !21
  %address.local468 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local466, i64 %address.local467, !dbg !21
  store half %qr.n2109, ptr addrspace(1) %address.local468, align 2, !dbg !21
  %address.local469 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local470 = or disjoint i64 %address.local457, 48, !dbg !21
  %address.local471 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local469, i64 %address.local470, !dbg !21
  store half %qr.n2103, ptr addrspace(1) %address.local471, align 2, !dbg !21
  %address.local472 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local473 = or disjoint i64 %address.local457, 49, !dbg !21
  %address.local474 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local472, i64 %address.local473, !dbg !21
  store half %qr.n2102, ptr addrspace(1) %address.local474, align 2, !dbg !21
  %address.local475 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local476 = or disjoint i64 %address.local457, 50, !dbg !21
  %address.local477 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local475, i64 %address.local476, !dbg !21
  store half %qr.n2101, ptr addrspace(1) %address.local477, align 2, !dbg !21
  %address.local478 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local454, !dbg !21
  %address.local479 = or i64 %address.local456, 51, !dbg !21
  %address.local480 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local478, i64 %address.local479, !dbg !21
  store half %qr.n2100, ptr addrspace(1) %address.local480, align 2, !dbg !21
  %address.local481 = shl nsw i64 %address.local452, 7, !dbg !21
  %address.local482 = add i64 %address.local481, 8192, !dbg !21
  %address.local483 = add nuw nsw i64 %qr.n2113, %address.local482, !dbg !21
  %address.local484 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local485 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local484, i64 %address.local458, !dbg !21
  store half %qr.n2094, ptr addrspace(1) %address.local485, align 2, !dbg !21
  %address.local486 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local487 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local486, i64 %address.local461, !dbg !21
  store half %qr.n2093, ptr addrspace(1) %address.local487, align 2, !dbg !21
  %address.local488 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local489 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local488, i64 %address.local464, !dbg !21
  store half %qr.n2092, ptr addrspace(1) %address.local489, align 2, !dbg !21
  %address.local490 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local491 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local490, i64 %address.local467, !dbg !21
  store half %qr.n2091, ptr addrspace(1) %address.local491, align 2, !dbg !21
  %address.local492 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local493 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local492, i64 %address.local470, !dbg !21
  store half %qr.n2085, ptr addrspace(1) %address.local493, align 2, !dbg !21
  %address.local494 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local495 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local494, i64 %address.local473, !dbg !21
  store half %qr.n2084, ptr addrspace(1) %address.local495, align 2, !dbg !21
  %address.local496 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local497 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local496, i64 %address.local476, !dbg !21
  store half %qr.n2083, ptr addrspace(1) %address.local497, align 2, !dbg !21
  %address.local498 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local483, !dbg !21
  %address.local499 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local498, i64 %address.local479, !dbg !21
  store half %qr.n2082, ptr addrspace(1) %address.local499, align 2, !dbg !21
  br label %qr.n2114, !dbg !21

qr.n2114:                                             ; preds = %qr.n2076, %qr.n2075
  br i1 true, label %qr.n2115, label %qr.n2153, !dbg !21

qr.n2115:                                             ; preds = %qr.n2114
  %qr.n2116 = insertelement <4 x float> poison, float %qr.n1953, i64 0, !dbg !19
  %qr.n2117 = insertelement <4 x float> %qr.n2116, float %qr.n1959, i64 1, !dbg !19
  %qr.n2118 = insertelement <4 x float> %qr.n2117, float %qr.n1954, i64 2, !dbg !19
  %qr.n2119 = insertelement <4 x float> %qr.n2118, float %qr.n1960, i64 3, !dbg !19
  %row.reciprocal81 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %qr.n1904, !dbg !19
  %normalized82 = fmul <4 x float> %qr.n2119, %row.reciprocal81, !dbg !19
  %qr.n2120 = fptrunc <4 x float> %normalized82 to <4 x half>, !dbg !20
  %qr.n2121 = extractelement <4 x half> %qr.n2120, i64 3, !dbg !20
  %qr.n2122 = extractelement <4 x half> %qr.n2120, i64 2, !dbg !20
  %qr.n2123 = extractelement <4 x half> %qr.n2120, i64 1, !dbg !20
  %qr.n2124 = extractelement <4 x half> %qr.n2120, i64 0, !dbg !20
  %qr.n2125 = insertelement <4 x float> poison, float %qr.n1843, i64 0, !dbg !19
  %qr.n2126 = insertelement <4 x float> %qr.n2125, float %qr.n1849, i64 1, !dbg !19
  %qr.n2127 = insertelement <4 x float> %qr.n2126, float %qr.n1844, i64 2, !dbg !19
  %qr.n2128 = insertelement <4 x float> %qr.n2127, float %qr.n1850, i64 3, !dbg !19
  %normalized83 = fmul <4 x float> %qr.n2128, %row.reciprocal81, !dbg !19
  %qr.n2129 = fptrunc <4 x float> %normalized83 to <4 x half>, !dbg !20
  %qr.n2130 = extractelement <4 x half> %qr.n2129, i64 3, !dbg !20
  %qr.n2131 = extractelement <4 x half> %qr.n2129, i64 2, !dbg !20
  %qr.n2132 = extractelement <4 x half> %qr.n2129, i64 1, !dbg !20
  %qr.n2133 = extractelement <4 x half> %qr.n2129, i64 0, !dbg !20
  %qr.n2134 = insertelement <4 x float> poison, float %qr.n1735, i64 0, !dbg !19
  %qr.n2135 = insertelement <4 x float> %qr.n2134, float %qr.n1741, i64 1, !dbg !19
  %qr.n2136 = insertelement <4 x float> %qr.n2135, float %qr.n1736, i64 2, !dbg !19
  %qr.n2137 = insertelement <4 x float> %qr.n2136, float %qr.n1742, i64 3, !dbg !19
  %row.reciprocal84 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %qr.n1686, !dbg !19
  %normalized85 = fmul <4 x float> %qr.n2137, %row.reciprocal84, !dbg !19
  %qr.n2138 = fptrunc <4 x float> %normalized85 to <4 x half>, !dbg !20
  %qr.n2139 = extractelement <4 x half> %qr.n2138, i64 3, !dbg !20
  %qr.n2140 = extractelement <4 x half> %qr.n2138, i64 2, !dbg !20
  %qr.n2141 = extractelement <4 x half> %qr.n2138, i64 1, !dbg !20
  %qr.n2142 = extractelement <4 x half> %qr.n2138, i64 0, !dbg !20
  %qr.n2143 = insertelement <4 x float> poison, float %qr.n1625, i64 0, !dbg !19
  %qr.n2144 = insertelement <4 x float> %qr.n2143, float %qr.n1631, i64 1, !dbg !19
  %qr.n2145 = insertelement <4 x float> %qr.n2144, float %qr.n1626, i64 2, !dbg !19
  %qr.n2146 = insertelement <4 x float> %qr.n2145, float %qr.n1632, i64 3, !dbg !19
  %normalized86 = fmul <4 x float> %qr.n2146, %row.reciprocal84, !dbg !19
  %qr.n2147 = fptrunc <4 x float> %normalized86 to <4 x half>, !dbg !20
  %qr.n2148 = extractelement <4 x half> %qr.n2147, i64 3, !dbg !20
  %qr.n2149 = extractelement <4 x half> %qr.n2147, i64 2, !dbg !20
  %qr.n2150 = extractelement <4 x half> %qr.n2147, i64 1, !dbg !20
  %qr.n2151 = extractelement <4 x half> %qr.n2147, i64 0, !dbg !20
  %qr.n2152 = shl nuw nsw i64 %qr.n17, 19, !dbg !21
  %address.thread500 = call i32 asm sideeffect "", "=v,0"(i32 %qr.n24) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local501 = zext nneg i32 %address.thread500 to i64, !dbg !12
  %address.local502 = add nsw i64 %qr.n18, %address.local501, !dbg !21
  %address.local503 = lshr i64 %address.local501, 6, !dbg !12
  %address.local504 = shl nuw nsw i64 %address.local503, 4, !dbg !21
  %address.local505 = add nsw i64 %address.local502, %address.local504, !dbg !21
  %address.local506 = and i64 %address.local501, 240, !dbg !21
  %address.local507 = sub nsw i64 %address.local505, %address.local506, !dbg !21
  %address.local508 = add nsw i64 %address.local507, %qr.n19, !dbg !21
  %address.local509 = shl nuw nsw i64 %address.local508, 7, !dbg !21
  %address.local510 = add nuw nsw i64 %qr.n2152, %address.local509, !dbg !21
  %address.local511 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local512 = lshr i64 %address.local501, 2, !dbg !12
  %address.local513 = and i64 %address.local512, 12, !dbg !12
  %address.local514 = or disjoint i64 %address.local513, 64, !dbg !21
  %address.local515 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local511, i64 %address.local514, !dbg !21
  store half %qr.n2151, ptr addrspace(1) %address.local515, align 2, !dbg !21
  %address.local516 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local517 = or disjoint i64 %address.local513, 65, !dbg !21
  %address.local518 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local516, i64 %address.local517, !dbg !21
  store half %qr.n2150, ptr addrspace(1) %address.local518, align 2, !dbg !21
  %address.local519 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local520 = or disjoint i64 %address.local513, 66, !dbg !21
  %address.local521 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local519, i64 %address.local520, !dbg !21
  store half %qr.n2149, ptr addrspace(1) %address.local521, align 2, !dbg !21
  %address.local522 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local523 = or disjoint i64 %address.local513, 67, !dbg !21
  %address.local524 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local522, i64 %address.local523, !dbg !21
  store half %qr.n2148, ptr addrspace(1) %address.local524, align 2, !dbg !21
  %address.local525 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local526 = or disjoint i64 %address.local513, 80, !dbg !21
  %address.local527 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local525, i64 %address.local526, !dbg !21
  store half %qr.n2142, ptr addrspace(1) %address.local527, align 2, !dbg !21
  %address.local528 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local529 = or disjoint i64 %address.local513, 81, !dbg !21
  %address.local530 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local528, i64 %address.local529, !dbg !21
  store half %qr.n2141, ptr addrspace(1) %address.local530, align 2, !dbg !21
  %address.local531 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local532 = or disjoint i64 %address.local513, 82, !dbg !21
  %address.local533 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local531, i64 %address.local532, !dbg !21
  store half %qr.n2140, ptr addrspace(1) %address.local533, align 2, !dbg !21
  %address.local534 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local510, !dbg !21
  %address.local535 = or disjoint i64 %address.local513, 83, !dbg !21
  %address.local536 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local534, i64 %address.local535, !dbg !21
  store half %qr.n2139, ptr addrspace(1) %address.local536, align 2, !dbg !21
  %address.local537 = shl nsw i64 %address.local508, 7, !dbg !21
  %address.local538 = add i64 %address.local537, 8192, !dbg !21
  %address.local539 = add nuw nsw i64 %qr.n2152, %address.local538, !dbg !21
  %address.local540 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local541 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local540, i64 %address.local514, !dbg !21
  store half %qr.n2133, ptr addrspace(1) %address.local541, align 2, !dbg !21
  %address.local542 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local543 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local542, i64 %address.local517, !dbg !21
  store half %qr.n2132, ptr addrspace(1) %address.local543, align 2, !dbg !21
  %address.local544 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local545 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local544, i64 %address.local520, !dbg !21
  store half %qr.n2131, ptr addrspace(1) %address.local545, align 2, !dbg !21
  %address.local546 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local547 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local546, i64 %address.local523, !dbg !21
  store half %qr.n2130, ptr addrspace(1) %address.local547, align 2, !dbg !21
  %address.local548 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local549 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local548, i64 %address.local526, !dbg !21
  store half %qr.n2124, ptr addrspace(1) %address.local549, align 2, !dbg !21
  %address.local550 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local551 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local550, i64 %address.local529, !dbg !21
  store half %qr.n2123, ptr addrspace(1) %address.local551, align 2, !dbg !21
  %address.local552 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local553 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local552, i64 %address.local532, !dbg !21
  store half %qr.n2122, ptr addrspace(1) %address.local553, align 2, !dbg !21
  %address.local554 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local539, !dbg !21
  %address.local555 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local554, i64 %address.local535, !dbg !21
  store half %qr.n2121, ptr addrspace(1) %address.local555, align 2, !dbg !21
  br label %qr.n2153, !dbg !21

qr.n2153:                                             ; preds = %qr.n2115, %qr.n2114
  br i1 true, label %qr.n2154, label %qr.n2192, !dbg !21

qr.n2154:                                             ; preds = %qr.n2153
  %qr.n2155 = insertelement <4 x float> poison, float %qr.n1925, i64 0, !dbg !19
  %qr.n2156 = insertelement <4 x float> %qr.n2155, float %qr.n1931, i64 1, !dbg !19
  %qr.n2157 = insertelement <4 x float> %qr.n2156, float %qr.n1926, i64 2, !dbg !19
  %qr.n2158 = insertelement <4 x float> %qr.n2157, float %qr.n1932, i64 3, !dbg !19
  %row.reciprocal87 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %qr.n1904, !dbg !19
  %normalized88 = fmul <4 x float> %qr.n2158, %row.reciprocal87, !dbg !19
  %qr.n2159 = fptrunc <4 x float> %normalized88 to <4 x half>, !dbg !20
  %qr.n2160 = extractelement <4 x half> %qr.n2159, i64 3, !dbg !20
  %qr.n2161 = extractelement <4 x half> %qr.n2159, i64 2, !dbg !20
  %qr.n2162 = extractelement <4 x half> %qr.n2159, i64 1, !dbg !20
  %qr.n2163 = extractelement <4 x half> %qr.n2159, i64 0, !dbg !20
  %qr.n2164 = insertelement <4 x float> poison, float %qr.n1814, i64 0, !dbg !19
  %qr.n2165 = insertelement <4 x float> %qr.n2164, float %qr.n1820, i64 1, !dbg !19
  %qr.n2166 = insertelement <4 x float> %qr.n2165, float %qr.n1815, i64 2, !dbg !19
  %qr.n2167 = insertelement <4 x float> %qr.n2166, float %qr.n1821, i64 3, !dbg !19
  %normalized89 = fmul <4 x float> %qr.n2167, %row.reciprocal87, !dbg !19
  %qr.n2168 = fptrunc <4 x float> %normalized89 to <4 x half>, !dbg !20
  %qr.n2169 = extractelement <4 x half> %qr.n2168, i64 3, !dbg !20
  %qr.n2170 = extractelement <4 x half> %qr.n2168, i64 2, !dbg !20
  %qr.n2171 = extractelement <4 x half> %qr.n2168, i64 1, !dbg !20
  %qr.n2172 = extractelement <4 x half> %qr.n2168, i64 0, !dbg !20
  %qr.n2173 = insertelement <4 x float> poison, float %qr.n1707, i64 0, !dbg !19
  %qr.n2174 = insertelement <4 x float> %qr.n2173, float %qr.n1713, i64 1, !dbg !19
  %qr.n2175 = insertelement <4 x float> %qr.n2174, float %qr.n1708, i64 2, !dbg !19
  %qr.n2176 = insertelement <4 x float> %qr.n2175, float %qr.n1714, i64 3, !dbg !19
  %row.reciprocal90 = fdiv arcp <4 x float> splat (float 0x3FF0000000000000), %qr.n1686, !dbg !19
  %normalized91 = fmul <4 x float> %qr.n2176, %row.reciprocal90, !dbg !19
  %qr.n2177 = fptrunc <4 x float> %normalized91 to <4 x half>, !dbg !20
  %qr.n2178 = extractelement <4 x half> %qr.n2177, i64 3, !dbg !20
  %qr.n2179 = extractelement <4 x half> %qr.n2177, i64 2, !dbg !20
  %qr.n2180 = extractelement <4 x half> %qr.n2177, i64 1, !dbg !20
  %qr.n2181 = extractelement <4 x half> %qr.n2177, i64 0, !dbg !20
  %qr.n2182 = insertelement <4 x float> poison, float %qr.n1596, i64 0, !dbg !19
  %qr.n2183 = insertelement <4 x float> %qr.n2182, float %qr.n1602, i64 1, !dbg !19
  %qr.n2184 = insertelement <4 x float> %qr.n2183, float %qr.n1597, i64 2, !dbg !19
  %qr.n2185 = insertelement <4 x float> %qr.n2184, float %qr.n1603, i64 3, !dbg !19
  %normalized92 = fmul <4 x float> %qr.n2185, %row.reciprocal90, !dbg !19
  %qr.n2186 = fptrunc <4 x float> %normalized92 to <4 x half>, !dbg !20
  %qr.n2187 = extractelement <4 x half> %qr.n2186, i64 3, !dbg !20
  %qr.n2188 = extractelement <4 x half> %qr.n2186, i64 2, !dbg !20
  %qr.n2189 = extractelement <4 x half> %qr.n2186, i64 1, !dbg !20
  %qr.n2190 = extractelement <4 x half> %qr.n2186, i64 0, !dbg !20
  %qr.n2191 = shl nuw nsw i64 %qr.n17, 19, !dbg !21
  %address.thread556 = call i32 asm sideeffect "", "=v,0"(i32 %qr.n24) #9, !dbg !21, !range !33, !frisk.register_lifetimes !1
  %address.local557 = zext nneg i32 %address.thread556 to i64, !dbg !12
  %address.local558 = add nsw i64 %qr.n18, %address.local557, !dbg !21
  %address.local559 = lshr i64 %address.local557, 6, !dbg !12
  %address.local560 = shl nuw nsw i64 %address.local559, 4, !dbg !21
  %address.local561 = add nsw i64 %address.local558, %address.local560, !dbg !21
  %address.local562 = and i64 %address.local557, 240, !dbg !21
  %address.local563 = sub nsw i64 %address.local561, %address.local562, !dbg !21
  %address.local564 = add nsw i64 %address.local563, %qr.n19, !dbg !21
  %address.local565 = shl nuw nsw i64 %address.local564, 7, !dbg !21
  %address.local566 = add nuw nsw i64 %qr.n2191, %address.local565, !dbg !21
  %address.local567 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local568 = lshr i64 %address.local557, 2, !dbg !12
  %address.local569 = and i64 %address.local568, 12, !dbg !12
  %address.local570 = or disjoint i64 %address.local569, 96, !dbg !21
  %address.local571 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local567, i64 %address.local570, !dbg !21
  store half %qr.n2190, ptr addrspace(1) %address.local571, align 2, !dbg !21
  %address.local572 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local573 = or disjoint i64 %address.local569, 97, !dbg !21
  %address.local574 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local572, i64 %address.local573, !dbg !21
  store half %qr.n2189, ptr addrspace(1) %address.local574, align 2, !dbg !21
  %address.local575 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local576 = or disjoint i64 %address.local569, 98, !dbg !21
  %address.local577 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local575, i64 %address.local576, !dbg !21
  store half %qr.n2188, ptr addrspace(1) %address.local577, align 2, !dbg !21
  %address.local578 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local579 = or disjoint i64 %address.local569, 99, !dbg !21
  %address.local580 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local578, i64 %address.local579, !dbg !21
  store half %qr.n2187, ptr addrspace(1) %address.local580, align 2, !dbg !21
  %address.local581 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local582 = or disjoint i64 %address.local569, 112, !dbg !21
  %address.local583 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local581, i64 %address.local582, !dbg !21
  store half %qr.n2181, ptr addrspace(1) %address.local583, align 2, !dbg !21
  %address.local584 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local585 = or disjoint i64 %address.local569, 113, !dbg !21
  %address.local586 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local584, i64 %address.local585, !dbg !21
  store half %qr.n2180, ptr addrspace(1) %address.local586, align 2, !dbg !21
  %address.local587 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local588 = or disjoint i64 %address.local569, 114, !dbg !21
  %address.local589 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local587, i64 %address.local588, !dbg !21
  store half %qr.n2179, ptr addrspace(1) %address.local589, align 2, !dbg !21
  %address.local590 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local566, !dbg !21
  %address.local591 = or i64 %address.local568, 115, !dbg !21
  %address.local592 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local590, i64 %address.local591, !dbg !21
  store half %qr.n2178, ptr addrspace(1) %address.local592, align 2, !dbg !21
  %address.local593 = shl nsw i64 %address.local564, 7, !dbg !21
  %address.local594 = add i64 %address.local593, 8192, !dbg !21
  %address.local595 = add nuw nsw i64 %qr.n2191, %address.local594, !dbg !21
  %address.local596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local597 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local596, i64 %address.local570, !dbg !21
  store half %qr.n2172, ptr addrspace(1) %address.local597, align 2, !dbg !21
  %address.local598 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local599 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local598, i64 %address.local573, !dbg !21
  store half %qr.n2171, ptr addrspace(1) %address.local599, align 2, !dbg !21
  %address.local600 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local600, i64 %address.local576, !dbg !21
  store half %qr.n2170, ptr addrspace(1) %address.local601, align 2, !dbg !21
  %address.local602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local603 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local602, i64 %address.local579, !dbg !21
  store half %qr.n2169, ptr addrspace(1) %address.local603, align 2, !dbg !21
  %address.local604 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local605 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local604, i64 %address.local582, !dbg !21
  store half %qr.n2163, ptr addrspace(1) %address.local605, align 2, !dbg !21
  %address.local606 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local606, i64 %address.local585, !dbg !21
  store half %qr.n2162, ptr addrspace(1) %address.local607, align 2, !dbg !21
  %address.local608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local609 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local608, i64 %address.local588, !dbg !21
  store half %qr.n2161, ptr addrspace(1) %address.local609, align 2, !dbg !21
  %address.local610 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local595, !dbg !21
  %address.local611 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local610, i64 %address.local591, !dbg !21
  store half %qr.n2160, ptr addrspace(1) %address.local611, align 2, !dbg !21
  br label %qr.n2192, !dbg !21

qr.n2192:                                             ; preds = %qr.n2154, %qr.n2153
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
!llvm.module.flags = !{!32}

!0 = !DIFile(filename: "nohint_unrolled.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/pipeline_cleanup")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 22, type: !2, scopeLine: 22, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 22, column: 3, scope: !4)
!9 = !DILocation(line: 46, column: 17, scope: !4)
!11 = !DILocation(line: 66, column: 22, scope: !4)
!12 = !DILocation(line: 68, column: 24, scope: !4)
!13 = !DILocation(line: 73, column: 19, scope: !4)
!14 = !DILocation(line: 93, column: 23, scope: !4)
!15 = !DILocation(line: 56, column: 24, scope: !4)
!16 = !DILocation(line: 67, column: 22, scope: !4)
!17 = !DILocation(line: 69, column: 26, scope: !4)
!18 = !DILocation(line: 92, column: 20, scope: !4)
!19 = !DILocation(line: 97, column: 23, scope: !4)
!20 = !DILocation(line: 98, column: 19, scope: !4)
!21 = !DILocation(line: 99, column: 9, scope: !4)
!23 = !DILocation(line: 75, column: 24, scope: !4)
!24 = !DILocation(line: 76, column: 23, scope: !4)
!25 = !DILocation(line: 77, column: 26, scope: !4)
!26 = !DILocation(line: 84, column: 21, scope: !4)
!27 = !DILocation(line: 85, column: 22, scope: !4)
!28 = !DILocation(line: 86, column: 20, scope: !4)
!29 = !DILocation(line: 87, column: 16, scope: !4)
!30 = !DILocation(line: 94, column: 20, scope: !4)
!31 = !{i32 256, i32 1, i32 1}
!32 = !{i32 2, !"Debug Info Version", i32 3}
!33 = !{i32 0, i32 256}
