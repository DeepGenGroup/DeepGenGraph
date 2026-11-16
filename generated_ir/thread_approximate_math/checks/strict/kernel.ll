; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/input.mlir'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math/input.mlir"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: convergent willreturn memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #0

declare float @__ocml_exp2_f32(float)

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #1 !dbg !4 !reqd_work_group_size !30 {
  %5 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0), !dbg !8
  %6 = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %5), !dbg !8
  %7 = add nuw nsw i32 %6, 64, !dbg !8
  %8 = xor i32 %6, 16, !dbg !8
  %9 = and i32 %7, 192, !dbg !8
  %10 = icmp samesign ult i32 %8, %9, !dbg !8
  %11 = select i1 %10, i32 %8, i32 %6, !dbg !8
  %12 = shl nuw nsw i32 %11, 2, !dbg !8
  %13 = xor i32 %6, 32, !dbg !8
  %14 = icmp samesign ult i32 %13, %9, !dbg !8
  %15 = select i1 %14, i32 %13, i32 %6, !dbg !8
  %16 = shl nuw nsw i32 %15, 2, !dbg !8
  %17 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5
  %18 = sext i32 %17 to i64, !dbg !5
  %19 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5
  %20 = sext i32 %19 to i64, !dbg !5
  %.lobit = ashr i32 %19, 31, !dbg !11
  %21 = xor i32 %19, %.lobit, !dbg !11
  %22 = sdiv i32 %21, 32, !dbg !11
  %.lobit610 = ashr i32 %19, 31, !dbg !11
  %23 = xor i32 %22, %.lobit610, !dbg !11
  %24 = sext i32 %23 to i64, !dbg !11
  %25 = add nsw i64 %18, %24, !dbg !11
  %26 = and i64 %25, 31, !dbg !11
  %27 = shl nsw i64 %20, 7, !dbg !11
  %28 = mul nsw i64 %24, -4096, !dbg !11
  %29 = add nsw i64 %27, %28, !dbg !11
  %30 = shl nsw i64 %29, 7, !dbg !11
  %31 = call i32 @llvm.amdgcn.workitem.id.x(), !dbg !13, !range !32
  %32 = zext nneg i32 %31 to i64, !dbg !13
  %33 = add nsw i64 %18, %24, !dbg !11
  %34 = and i64 %33, 31, !dbg !11
  %35 = shl nuw nsw i64 %34, 19, !dbg !11
  %36 = add nsw i64 %35, %30, !dbg !11
  %address.thread = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !11, !range !32, !frisk.register_lifetimes !1
  %address.local = zext nneg i32 %address.thread to i64, !dbg !13
  %address.local704 = lshr i64 %address.local, 6, !dbg !11
  %address.local705 = shl nuw nsw i64 %address.local704, 7, !dbg !11
  %address.local706 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local705, !dbg !11
  %address.local707 = ptrtoint ptr addrspace(3) %address.local706 to i64, !dbg !11
  %address.local708 = trunc i64 %address.local707 to i32, !dbg !11
  %37 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local708), !dbg !11
  %38 = zext i32 %37 to i64, !dbg !11
  %39 = inttoptr i64 %38 to ptr addrspace(3), !dbg !11
  %address.local709 = lshr i64 %address.local, 6, !dbg !11
  %address.local710 = or disjoint i64 %address.local709, 4, !dbg !11
  %address.local711 = shl nuw nsw i64 %address.local710, 7, !dbg !11
  %address.local712 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local711, !dbg !11
  %address.local713 = ptrtoint ptr addrspace(3) %address.local712 to i64, !dbg !11
  %address.local714 = trunc i64 %address.local713 to i32, !dbg !11
  %40 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local714), !dbg !11
  %41 = zext i32 %40 to i64, !dbg !11
  %42 = inttoptr i64 %41 to ptr addrspace(3), !dbg !11
  %address.local715 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local716 = and i64 %address.local715, 384, !dbg !11
  %address.local717 = or disjoint i64 %address.local716, 1024, !dbg !11
  %address.local718 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local717, !dbg !11
  %address.local719 = ptrtoint ptr addrspace(3) %address.local718 to i64, !dbg !11
  %address.local720 = trunc i64 %address.local719 to i32, !dbg !11
  %43 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local720), !dbg !11
  %44 = zext i32 %43 to i64, !dbg !11
  %45 = inttoptr i64 %44 to ptr addrspace(3), !dbg !11
  %address.local721 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local722 = and i64 %address.local721, 384, !dbg !11
  %address.local723 = or disjoint i64 %address.local722, 1536, !dbg !11
  %address.local724 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local723, !dbg !11
  %address.local725 = ptrtoint ptr addrspace(3) %address.local724 to i64, !dbg !11
  %address.local726 = trunc i64 %address.local725 to i32, !dbg !11
  %46 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local726), !dbg !11
  %47 = zext i32 %46 to i64, !dbg !11
  %48 = inttoptr i64 %47 to ptr addrspace(3), !dbg !11
  %address.local727 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local728 = and i64 %address.local727, 384, !dbg !11
  %address.local729 = or disjoint i64 %address.local728, 2048, !dbg !11
  %address.local730 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local729, !dbg !11
  %address.local731 = ptrtoint ptr addrspace(3) %address.local730 to i64, !dbg !11
  %address.local732 = trunc i64 %address.local731 to i32, !dbg !11
  %49 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local732), !dbg !11
  %50 = zext i32 %49 to i64, !dbg !11
  %51 = inttoptr i64 %50 to ptr addrspace(3), !dbg !11
  %address.local733 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local734 = and i64 %address.local733, 384, !dbg !11
  %address.local735 = or disjoint i64 %address.local734, 2560, !dbg !11
  %address.local736 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local735, !dbg !11
  %address.local737 = ptrtoint ptr addrspace(3) %address.local736 to i64, !dbg !11
  %address.local738 = trunc i64 %address.local737 to i32, !dbg !11
  %52 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local738), !dbg !11
  %53 = zext i32 %52 to i64, !dbg !11
  %54 = inttoptr i64 %53 to ptr addrspace(3), !dbg !11
  %address.local739 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local740 = and i64 %address.local739, 384, !dbg !11
  %address.local741 = or disjoint i64 %address.local740, 3072, !dbg !11
  %address.local742 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local741, !dbg !11
  %address.local743 = ptrtoint ptr addrspace(3) %address.local742 to i64, !dbg !11
  %address.local744 = trunc i64 %address.local743 to i32, !dbg !11
  %55 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local744), !dbg !11
  %56 = zext i32 %55 to i64, !dbg !11
  %57 = inttoptr i64 %56 to ptr addrspace(3), !dbg !11
  %address.local745 = shl nuw nsw i64 %address.local, 1, !dbg !11
  %address.local746 = and i64 %address.local745, 384, !dbg !11
  %address.local747 = or disjoint i64 %address.local746, 3584, !dbg !11
  %address.local748 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local747, !dbg !11
  %address.local749 = ptrtoint ptr addrspace(3) %address.local748 to i64, !dbg !11
  %address.local750 = trunc i64 %address.local749 to i32, !dbg !11
  %58 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local750), !dbg !11
  %59 = zext i32 %58 to i64, !dbg !11
  %60 = inttoptr i64 %59 to ptr addrspace(3), !dbg !11
  %address.local751 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local705, !dbg !12
  %address.local752 = ptrtoint ptr addrspace(3) %address.local751 to i64, !dbg !12
  %address.local753 = trunc i64 %address.local752 to i32, !dbg !12
  %61 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local753), !dbg !12
  %62 = zext i32 %61 to i64, !dbg !12
  %63 = inttoptr i64 %62 to ptr addrspace(3), !dbg !12
  %address.local754 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local711, !dbg !12
  %address.local755 = ptrtoint ptr addrspace(3) %address.local754 to i64, !dbg !12
  %address.local756 = trunc i64 %address.local755 to i32, !dbg !12
  %64 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local756), !dbg !12
  %65 = zext i32 %64 to i64, !dbg !12
  %66 = inttoptr i64 %65 to ptr addrspace(3), !dbg !12
  %address.local757 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local717, !dbg !12
  %address.local758 = ptrtoint ptr addrspace(3) %address.local757 to i64, !dbg !12
  %address.local759 = trunc i64 %address.local758 to i32, !dbg !12
  %67 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local759), !dbg !12
  %68 = zext i32 %67 to i64, !dbg !12
  %69 = inttoptr i64 %68 to ptr addrspace(3), !dbg !12
  %address.local760 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local723, !dbg !12
  %address.local761 = ptrtoint ptr addrspace(3) %address.local760 to i64, !dbg !12
  %address.local762 = trunc i64 %address.local761 to i32, !dbg !12
  %70 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local762), !dbg !12
  %71 = zext i32 %70 to i64, !dbg !12
  %72 = inttoptr i64 %71 to ptr addrspace(3), !dbg !12
  %address.local763 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local729, !dbg !12
  %address.local764 = ptrtoint ptr addrspace(3) %address.local763 to i64, !dbg !12
  %address.local765 = trunc i64 %address.local764 to i32, !dbg !12
  %73 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local765), !dbg !12
  %74 = zext i32 %73 to i64, !dbg !12
  %75 = inttoptr i64 %74 to ptr addrspace(3), !dbg !12
  %address.local766 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local735, !dbg !12
  %address.local767 = ptrtoint ptr addrspace(3) %address.local766 to i64, !dbg !12
  %address.local768 = trunc i64 %address.local767 to i32, !dbg !12
  %76 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local768), !dbg !12
  %77 = zext i32 %76 to i64, !dbg !12
  %78 = inttoptr i64 %77 to ptr addrspace(3), !dbg !12
  %address.local769 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local741, !dbg !12
  %address.local770 = ptrtoint ptr addrspace(3) %address.local769 to i64, !dbg !12
  %address.local771 = trunc i64 %address.local770 to i32, !dbg !12
  %79 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local771), !dbg !12
  %80 = zext i32 %79 to i64, !dbg !12
  %81 = inttoptr i64 %80 to ptr addrspace(3), !dbg !12
  %address.local772 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local747, !dbg !12
  %address.local773 = ptrtoint ptr addrspace(3) %address.local772 to i64, !dbg !12
  %address.local774 = trunc i64 %address.local773 to i32, !dbg !12
  %82 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local774), !dbg !12
  %83 = zext i32 %82 to i64, !dbg !12
  %84 = inttoptr i64 %83 to ptr addrspace(3), !dbg !12
  %address.local775 = shl nuw nsw i64 %address.local704, 10, !dbg !15
  %address.local776 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local775, !dbg !15
  %address.local777 = ptrtoint ptr addrspace(3) %address.local776 to i64, !dbg !15
  %address.local778 = trunc i64 %address.local777 to i32, !dbg !15
  %85 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local778), !dbg !15
  %86 = zext i32 %85 to i64, !dbg !15
  %87 = inttoptr i64 %86 to ptr addrspace(3), !dbg !15
  %address.local779 = shl nuw nsw i64 %address.local710, 10, !dbg !15
  %address.local780 = getelementptr i8, ptr addrspace(3) @shm_0, i64 %address.local779, !dbg !15
  %address.local781 = ptrtoint ptr addrspace(3) %address.local780 to i64, !dbg !15
  %address.local782 = trunc i64 %address.local781 to i32, !dbg !15
  %88 = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %address.local782), !dbg !15
  %89 = zext i32 %88 to i64, !dbg !15
  %90 = inttoptr i64 %89 to ptr addrspace(3), !dbg !15
  %.idx612 = shl nuw nsw i64 %26, 20, !dbg !11
  %91 = getelementptr i8, ptr addrspace(1) %0, i64 %.idx612, !dbg !11
  %92 = getelementptr [2 x i8], ptr addrspace(1) %91, i64 %30, !dbg !11
  %93 = lshr i32 %31, 2, !dbg !16
  %index.compact242 = zext nneg i32 %93 to i64, !dbg !16
  %94 = lshr i32 %31, 2, !dbg !16
  %95 = and i32 %94, 3, !dbg !16
  %index.compact237 = zext nneg i32 %95 to i64, !dbg !16
  %96 = lshr i32 %31, 1, !dbg !16
  %index.compact234 = zext nneg i32 %96 to i64, !dbg !16
  %97 = lshr i32 %31, 5, !dbg !16
  %98 = and i32 %97, 1, !dbg !16
  %index.compact227 = zext nneg i32 %98 to i64, !dbg !16
  %99 = lshr i32 %31, 1, !dbg !16
  %100 = and i32 %99, 126, !dbg !16
  %101 = shl nuw nsw i32 %31, 2, !dbg !16
  %102 = and i32 %101, 12, !dbg !16
  %103 = sub nsw i32 %102, %100, !dbg !16
  %index.compact225 = sext i32 %103 to i64, !dbg !16
  %104 = add nsw i64 %index.compact225, %index.compact234, !dbg !16
  %105 = add nsw i64 %104, %index.compact227, !dbg !16
  %.neg627 = sdiv i64 %105, -2, !dbg !16
  %106 = mul nuw nsw i64 %32, 18, !dbg !16
  %107 = mul nsw i64 %index.compact242, -72, !dbg !16
  %108 = add nsw i64 %106, %107, !dbg !16
  %109 = add nsw i64 %108, %index.compact237, !dbg !16
  %.neg628 = sdiv i64 %109, -4, !dbg !16
  %110 = lshr i32 %31, 1, !dbg !16
  %111 = and i32 %110, 126, !dbg !16
  %112 = shl nuw nsw i32 %31, 2, !dbg !16
  %113 = and i32 %112, 12, !dbg !16
  %114 = sub nsw i32 %113, %111, !dbg !16
  %index.compact197 = sext i32 %114 to i64, !dbg !16
  %115 = add nsw i64 %index.compact197, %index.compact234, !dbg !16
  %116 = add nsw i64 %115, %index.compact227, !dbg !16
  %.neg638 = sdiv i64 %116, -2, !dbg !16
  %117 = add nsw i64 %106, %107, !dbg !16
  %118 = add nsw i64 %117, %index.compact237, !dbg !16
  %119 = add nsw i64 %118, 1, !dbg !16
  %.neg639 = sdiv i64 %119, -4, !dbg !16
  %120 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %36, !dbg !11
  %121 = getelementptr i8, ptr addrspace(1) %120, i64 64, !dbg !11
  %122 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %36, !dbg !11
  %123 = getelementptr i8, ptr addrspace(1) %122, i64 128, !dbg !11
  %124 = getelementptr [2 x i8], ptr addrspace(1) %0, i64 %36, !dbg !11
  %125 = getelementptr i8, ptr addrspace(1) %124, i64 192, !dbg !11
  br label %126, !dbg !6

126:                                              ; preds = %135, %4
  %127 = phi i64 [ %1455, %135 ], [ 0, %4 ], !dbg !6
  %128 = phi [2 x <8 x float>] [ %1164, %135 ], [ zeroinitializer, %4 ], !dbg !6
  %129 = phi [2 x <1 x float>] [ %1061, %135 ], [ zeroinitializer, %4 ], !dbg !6
  %130 = phi [2 x <1 x float>] [ %1454, %135 ], [ [<1 x float> splat (float 0xFFF0000000000000), <1 x float> splat (float 0xFFF0000000000000)], %4 ], !dbg !6
  %131 = phi [2 x <8 x float>] [ %1267, %135 ], [ zeroinitializer, %4 ], !dbg !6
  %132 = phi [2 x <8 x float>] [ %1370, %135 ], [ zeroinitializer, %4 ], !dbg !6
  %133 = phi [2 x <8 x float>] [ %1452, %135 ], [ zeroinitializer, %4 ], !dbg !6
  %134 = icmp samesign ult i64 %127, 32, !dbg !6
  br i1 %134, label %135, label %1456, !dbg !6

135:                                              ; preds = %126
  %address.thread783 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !11, !range !32, !frisk.register_lifetimes !1
  %address.local784 = shl nuw nsw i32 %address.thread783, 1, !dbg !11
  %address.local785 = and i32 %address.local784, 2, !dbg !11
  %address.local786 = shl nuw nsw i32 %address.thread783, 6, !dbg !11
  %address.local787 = and i32 %address.local786, 16256, !dbg !11
  %address.local788 = or disjoint i32 %address.local785, %address.local787, !dbg !11
  %address.local789 = zext nneg i32 %address.local788 to i64, !dbg !11
  %address.local790 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %address.local789, !dbg !11
  %dma.source.bits = ptrtoint ptr addrspace(1) %address.local790 to i64, !dbg !11
  %dma.source.opaque = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #10, !dbg !11
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %39, i32 4, i32 0, i32 0), !dbg !11
  %address.local791 = shl nuw nsw i32 %address.thread783, 1, !dbg !11
  %address.local792 = and i32 %address.local791, 2, !dbg !11
  %address.local793 = shl nuw nsw i32 %address.thread783, 6, !dbg !11
  %address.local794 = and i32 %address.local793, 16256, !dbg !11
  %address.local795 = or disjoint i32 %address.local792, %address.local794, !dbg !11
  %address.local796 = or disjoint i32 %address.local795, 4, !dbg !11
  %address.local797 = zext nneg i32 %address.local796 to i64, !dbg !11
  %address.local798 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %address.local797, !dbg !11
  %dma.source.bits1912 = ptrtoint ptr addrspace(1) %address.local798 to i64, !dbg !11
  %dma.source.opaque1913 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1912) #10, !dbg !11
  %dma.source.ptr1914 = inttoptr i64 %dma.source.opaque1913 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1914, ptr addrspace(3) %42, i32 4, i32 0, i32 0), !dbg !11
  %address.local799 = shl nuw nsw i32 %address.thread783, 1, !dbg !11
  %address.local800 = and i32 %address.local799, 2, !dbg !11
  %address.local801 = shl nuw nsw i32 %address.thread783, 6, !dbg !11
  %address.local802 = and i32 %address.local801, 16256, !dbg !11
  %address.local803 = or disjoint i32 %address.local800, %address.local802, !dbg !11
  %address.local804 = or disjoint i32 %address.local803, 8, !dbg !11
  %address.local805 = zext nneg i32 %address.local804 to i64, !dbg !11
  %address.local806 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %address.local805, !dbg !11
  %dma.source.bits1915 = ptrtoint ptr addrspace(1) %address.local806 to i64, !dbg !11
  %dma.source.opaque1916 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1915) #10, !dbg !11
  %dma.source.ptr1917 = inttoptr i64 %dma.source.opaque1916 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1917, ptr addrspace(3) %45, i32 4, i32 0, i32 0), !dbg !11
  %address.local807 = shl nuw nsw i32 %address.thread783, 1, !dbg !11
  %address.local808 = and i32 %address.local807, 2, !dbg !11
  %address.local809 = shl nuw nsw i32 %address.thread783, 6, !dbg !11
  %address.local810 = and i32 %address.local809, 16256, !dbg !11
  %address.local811 = or disjoint i32 %address.local808, %address.local810, !dbg !11
  %address.local812 = or disjoint i32 %address.local811, 12, !dbg !11
  %address.local813 = zext nneg i32 %address.local812 to i64, !dbg !11
  %address.local814 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %address.local813, !dbg !11
  %dma.source.bits1918 = ptrtoint ptr addrspace(1) %address.local814 to i64, !dbg !11
  %dma.source.opaque1919 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1918) #10, !dbg !11
  %dma.source.ptr1920 = inttoptr i64 %dma.source.opaque1919 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1920, ptr addrspace(3) %48, i32 4, i32 0, i32 0), !dbg !11
  %address.local815 = shl nuw nsw i32 %address.thread783, 1, !dbg !11
  %address.local816 = and i32 %address.local815, 2, !dbg !11
  %address.local817 = shl nuw nsw i32 %address.thread783, 6, !dbg !11
  %address.local818 = and i32 %address.local817, 16256, !dbg !11
  %address.local819 = or disjoint i32 %address.local816, %address.local818, !dbg !11
  %address.local820 = or disjoint i32 %address.local819, 16, !dbg !11
  %address.local821 = zext nneg i32 %address.local820 to i64, !dbg !11
  %address.local822 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %address.local821, !dbg !11
  %dma.source.bits1921 = ptrtoint ptr addrspace(1) %address.local822 to i64, !dbg !11
  %dma.source.opaque1922 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1921) #10, !dbg !11
  %dma.source.ptr1923 = inttoptr i64 %dma.source.opaque1922 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1923, ptr addrspace(3) %51, i32 4, i32 0, i32 0), !dbg !11
  %address.local823 = shl nuw nsw i32 %address.thread783, 1, !dbg !11
  %address.local824 = and i32 %address.local823, 2, !dbg !11
  %address.local825 = shl nuw nsw i32 %address.thread783, 6, !dbg !11
  %address.local826 = and i32 %address.local825, 16256, !dbg !11
  %address.local827 = or disjoint i32 %address.local824, %address.local826, !dbg !11
  %address.local828 = or disjoint i32 %address.local827, 20, !dbg !11
  %address.local829 = zext nneg i32 %address.local828 to i64, !dbg !11
  %address.local830 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %address.local829, !dbg !11
  %dma.source.bits1924 = ptrtoint ptr addrspace(1) %address.local830 to i64, !dbg !11
  %dma.source.opaque1925 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1924) #10, !dbg !11
  %dma.source.ptr1926 = inttoptr i64 %dma.source.opaque1925 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1926, ptr addrspace(3) %54, i32 4, i32 0, i32 0), !dbg !11
  %address.local831 = shl nuw nsw i32 %address.thread783, 1, !dbg !11
  %address.local832 = and i32 %address.local831, 2, !dbg !11
  %address.local833 = shl nuw nsw i32 %address.thread783, 6, !dbg !11
  %address.local834 = and i32 %address.local833, 16256, !dbg !11
  %address.local835 = or disjoint i32 %address.local832, %address.local834, !dbg !11
  %address.local836 = or disjoint i32 %address.local835, 24, !dbg !11
  %address.local837 = zext nneg i32 %address.local836 to i64, !dbg !11
  %address.local838 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %address.local837, !dbg !11
  %dma.source.bits1927 = ptrtoint ptr addrspace(1) %address.local838 to i64, !dbg !11
  %dma.source.opaque1928 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1927) #10, !dbg !11
  %dma.source.ptr1929 = inttoptr i64 %dma.source.opaque1928 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1929, ptr addrspace(3) %57, i32 4, i32 0, i32 0), !dbg !11
  %address.local839 = shl nuw nsw i32 %address.thread783, 1, !dbg !11
  %address.local840 = and i32 %address.local839, 2, !dbg !11
  %address.local841 = shl nuw nsw i32 %address.thread783, 6, !dbg !11
  %address.local842 = and i32 %address.local841, 16256, !dbg !11
  %address.local843 = or disjoint i32 %address.local840, %address.local842, !dbg !11
  %address.local844 = or disjoint i32 %address.local843, 28, !dbg !11
  %address.local845 = zext nneg i32 %address.local844 to i64, !dbg !11
  %address.local846 = getelementptr [2 x i8], ptr addrspace(1) %92, i64 %address.local845, !dbg !11
  %dma.source.bits1930 = ptrtoint ptr addrspace(1) %address.local846 to i64, !dbg !11
  %dma.source.opaque1931 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1930) #10, !dbg !11
  %dma.source.ptr1932 = inttoptr i64 %dma.source.opaque1931 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1932, ptr addrspace(3) %60, i32 4, i32 0, i32 0), !dbg !11
  %136 = shl nsw i64 %18, 19, !dbg !12
  %137 = and i64 %136, 16252928, !dbg !12
  %138 = shl nuw nsw i64 %127, 14, !dbg !12
  %139 = or disjoint i64 %137, %138, !dbg !12
  %140 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %139, !dbg !12
  %address.local847 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local848 = zext nneg i32 %address.local847 to i64, !dbg !12
  %address.local849 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local850 = zext nneg i32 %address.local849 to i64, !dbg !12
  %address.local851 = sub nsw i64 %address.local848, %address.local850, !dbg !12
  %address.local852 = sdiv i64 %address.local851, -4, !dbg !12
  %address.local853 = shl i64 %address.local852, 9, !dbg !12
  %address.local854 = lshr i32 %address.thread783, 4, !dbg !12
  %address.local855 = zext nneg i32 %address.local854 to i64, !dbg !12
  %address.local856 = lshr i32 %address.thread783, 7, !dbg !12
  %address.local857 = zext nneg i32 %address.local856 to i64, !dbg !12
  %address.local858 = sub nsw i64 %address.local855, %address.local857, !dbg !12
  %address.local859 = sdiv i64 %address.local858, -2, !dbg !12
  %address.local860 = shl i64 %address.local859, 10, !dbg !12
  %address.local861 = mul nuw nsw i64 %address.local850, 7, !dbg !12
  %address.local862 = add nuw nsw i64 %address.local861, %address.local848, !dbg !12
  %address.local863 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local864 = and i32 %address.local863, 60, !dbg !12
  %address.local865 = zext nneg i32 %address.local864 to i64, !dbg !12
  %address.local866 = add nuw nsw i64 %address.local862, %address.local865, !dbg !12
  %address.local867 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local868 = and i32 %address.local867, 4, !dbg !12
  %address.local869 = sub nsw i32 0, %address.local868, !dbg !12
  %address.local870 = sext i32 %address.local869 to i64, !dbg !12
  %address.local871 = add nsw i64 %address.local866, %address.local870, !dbg !12
  %address.local872 = shl nsw i64 %address.local871, 7, !dbg !12
  %address.local873 = add i64 %address.local860, %address.local872, !dbg !12
  %address.local874 = add i64 %address.local853, %address.local873, !dbg !12
  %address.local875 = shl nuw nsw i32 %address.thread783, 1, !dbg !12
  %address.local876 = and i32 %address.local875, 2, !dbg !12
  %address.local877 = shl nuw nsw i32 %address.thread783, 3, !dbg !12
  %address.local878 = and i32 %address.local877, 16, !dbg !12
  %address.local879 = or disjoint i32 %address.local876, %address.local878, !dbg !12
  %address.local880 = zext nneg i32 %address.local879 to i64, !dbg !12
  %address.local881 = or disjoint i64 %address.local874, %address.local880, !dbg !12
  %address.local882 = getelementptr [2 x i8], ptr addrspace(1) %140, i64 %address.local881, !dbg !12
  %dma.source.bits1933 = ptrtoint ptr addrspace(1) %address.local882 to i64, !dbg !12
  %dma.source.opaque1934 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1933) #10, !dbg !12
  %dma.source.ptr1935 = inttoptr i64 %dma.source.opaque1934 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1935, ptr addrspace(3) %63, i32 4, i32 0, i32 0), !dbg !12
  %address.local883 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local884 = or disjoint i32 %address.local883, 8, !dbg !12
  %address.local885 = zext nneg i32 %address.local884 to i64, !dbg !12
  %address.local886 = mul nuw nsw i64 %address.local885, 7, !dbg !12
  %address.local887 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local888 = or disjoint i32 %address.local887, 64, !dbg !12
  %address.local889 = zext nneg i32 %address.local888 to i64, !dbg !12
  %address.local890 = add nuw nsw i64 %address.local886, %address.local889, !dbg !12
  %address.local891 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local892 = and i32 %address.local891, 60, !dbg !12
  %address.local893 = or disjoint i32 %address.local892, 64, !dbg !12
  %address.local894 = zext nneg i32 %address.local893 to i64, !dbg !12
  %address.local895 = add nuw nsw i64 %address.local890, %address.local894, !dbg !12
  %address.local896 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local897 = and i32 %address.local896, 4, !dbg !12
  %address.local898 = sub nuw nsw i32 -8, %address.local897, !dbg !12
  %address.local899 = sext i32 %address.local898 to i64, !dbg !12
  %address.local900 = add nsw i64 %address.local895, %address.local899, !dbg !12
  %address.local901 = sub nuw nsw i64 %address.local889, %address.local885, !dbg !12
  %address.local902 = and i64 %address.local901, 124, !dbg !12
  %address.local903 = lshr i32 %address.thread783, 4, !dbg !12
  %address.local904 = or disjoint i32 %address.local903, 16, !dbg !12
  %address.local905 = lshr i32 %address.thread783, 7, !dbg !12
  %address.local906 = or disjoint i32 %address.local905, 2, !dbg !12
  %address.local907 = sub nuw nsw i32 %address.local904, %address.local906, !dbg !12
  %address.local908 = shl nuw nsw i32 %address.local907, 2, !dbg !12
  %address.local909 = and i32 %address.local908, 120, !dbg !12
  %address.local910 = zext nneg i32 %address.local909 to i64, !dbg !12
  %address.local911 = add nuw nsw i64 %address.local902, %address.local910, !dbg !12
  %address.local912 = sub nsw i64 %address.local900, %address.local911, !dbg !12
  %address.local913 = shl nsw i64 %address.local912, 7, !dbg !12
  %address.local914 = shl nuw nsw i32 %address.thread783, 1, !dbg !12
  %address.local915 = and i32 %address.local914, 2, !dbg !12
  %address.local916 = shl nuw nsw i32 %address.thread783, 3, !dbg !12
  %address.local917 = and i32 %address.local916, 16, !dbg !12
  %address.local918 = or disjoint i32 %address.local915, %address.local917, !dbg !12
  %address.local919 = zext nneg i32 %address.local918 to i64, !dbg !12
  %address.local920 = or disjoint i64 %address.local913, %address.local919, !dbg !12
  %address.local921 = getelementptr [2 x i8], ptr addrspace(1) %140, i64 %address.local920, !dbg !12
  %dma.source.bits1936 = ptrtoint ptr addrspace(1) %address.local921 to i64, !dbg !12
  %dma.source.opaque1937 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1936) #10, !dbg !12
  %dma.source.ptr1938 = inttoptr i64 %dma.source.opaque1937 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1938, ptr addrspace(3) %66, i32 4, i32 0, i32 0), !dbg !12
  %address.local922 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local923 = zext nneg i32 %address.local922 to i64, !dbg !12
  %address.local924 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local925 = zext nneg i32 %address.local924 to i64, !dbg !12
  %address.local926 = sub nsw i64 %address.local923, %address.local925, !dbg !12
  %address.local927 = sdiv i64 %address.local926, -4, !dbg !12
  %address.local928 = shl i64 %address.local927, 9, !dbg !12
  %address.local929 = lshr i32 %address.thread783, 4, !dbg !12
  %address.local930 = zext nneg i32 %address.local929 to i64, !dbg !12
  %address.local931 = lshr i32 %address.thread783, 7, !dbg !12
  %address.local932 = zext nneg i32 %address.local931 to i64, !dbg !12
  %address.local933 = sub nsw i64 %address.local930, %address.local932, !dbg !12
  %address.local934 = sdiv i64 %address.local933, -2, !dbg !12
  %address.local935 = shl i64 %address.local934, 10, !dbg !12
  %address.local936 = mul nuw nsw i64 %address.local925, 7, !dbg !12
  %address.local937 = add nuw nsw i64 %address.local936, %address.local923, !dbg !12
  %address.local938 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local939 = and i32 %address.local938, 60, !dbg !12
  %address.local940 = zext nneg i32 %address.local939 to i64, !dbg !12
  %address.local941 = add nuw nsw i64 %address.local937, %address.local940, !dbg !12
  %address.local942 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local943 = and i32 %address.local942, 4, !dbg !12
  %address.local944 = sub nsw i32 0, %address.local943, !dbg !12
  %address.local945 = sext i32 %address.local944 to i64, !dbg !12
  %address.local946 = add nsw i64 %address.local941, %address.local945, !dbg !12
  %address.local947 = shl nsw i64 %address.local946, 7, !dbg !12
  %address.local948 = add i64 %address.local935, %address.local947, !dbg !12
  %address.local949 = add i64 %address.local928, %address.local948, !dbg !12
  %address.local950 = shl nuw nsw i32 %address.thread783, 1, !dbg !12
  %address.local951 = and i32 %address.local950, 2, !dbg !12
  %address.local952 = shl nuw nsw i32 %address.thread783, 3, !dbg !12
  %address.local953 = and i32 %address.local952, 16, !dbg !12
  %address.local954 = or disjoint i32 %address.local951, %address.local953, !dbg !12
  %address.local955 = or disjoint i32 %address.local954, 4, !dbg !12
  %address.local956 = zext nneg i32 %address.local955 to i64, !dbg !12
  %address.local957 = or disjoint i64 %address.local949, %address.local956, !dbg !12
  %address.local958 = getelementptr [2 x i8], ptr addrspace(1) %140, i64 %address.local957, !dbg !12
  %dma.source.bits1939 = ptrtoint ptr addrspace(1) %address.local958 to i64, !dbg !12
  %dma.source.opaque1940 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1939) #10, !dbg !12
  %dma.source.ptr1941 = inttoptr i64 %dma.source.opaque1940 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1941, ptr addrspace(3) %69, i32 4, i32 0, i32 0), !dbg !12
  %address.local959 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local960 = or disjoint i32 %address.local959, 8, !dbg !12
  %address.local961 = zext nneg i32 %address.local960 to i64, !dbg !12
  %address.local962 = mul nuw nsw i64 %address.local961, 7, !dbg !12
  %address.local963 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local964 = or disjoint i32 %address.local963, 64, !dbg !12
  %address.local965 = zext nneg i32 %address.local964 to i64, !dbg !12
  %address.local966 = add nuw nsw i64 %address.local962, %address.local965, !dbg !12
  %address.local967 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local968 = and i32 %address.local967, 60, !dbg !12
  %address.local969 = or disjoint i32 %address.local968, 64, !dbg !12
  %address.local970 = zext nneg i32 %address.local969 to i64, !dbg !12
  %address.local971 = add nuw nsw i64 %address.local966, %address.local970, !dbg !12
  %address.local972 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local973 = and i32 %address.local972, 4, !dbg !12
  %address.local974 = sub nuw nsw i32 -8, %address.local973, !dbg !12
  %address.local975 = sext i32 %address.local974 to i64, !dbg !12
  %address.local976 = add nsw i64 %address.local971, %address.local975, !dbg !12
  %address.local977 = sub nuw nsw i64 %address.local965, %address.local961, !dbg !12
  %address.local978 = and i64 %address.local977, 124, !dbg !12
  %address.local979 = lshr i32 %address.thread783, 4, !dbg !12
  %address.local980 = or disjoint i32 %address.local979, 16, !dbg !12
  %address.local981 = lshr i32 %address.thread783, 7, !dbg !12
  %address.local982 = or disjoint i32 %address.local981, 2, !dbg !12
  %address.local983 = sub nuw nsw i32 %address.local980, %address.local982, !dbg !12
  %address.local984 = shl nuw nsw i32 %address.local983, 2, !dbg !12
  %address.local985 = and i32 %address.local984, 120, !dbg !12
  %address.local986 = zext nneg i32 %address.local985 to i64, !dbg !12
  %address.local987 = add nuw nsw i64 %address.local978, %address.local986, !dbg !12
  %address.local988 = sub nsw i64 %address.local976, %address.local987, !dbg !12
  %address.local989 = shl nsw i64 %address.local988, 7, !dbg !12
  %address.local990 = shl nuw nsw i32 %address.thread783, 1, !dbg !12
  %address.local991 = and i32 %address.local990, 2, !dbg !12
  %address.local992 = shl nuw nsw i32 %address.thread783, 3, !dbg !12
  %address.local993 = and i32 %address.local992, 16, !dbg !12
  %address.local994 = or disjoint i32 %address.local991, %address.local993, !dbg !12
  %address.local995 = or disjoint i32 %address.local994, 4, !dbg !12
  %address.local996 = zext nneg i32 %address.local995 to i64, !dbg !12
  %address.local997 = or disjoint i64 %address.local989, %address.local996, !dbg !12
  %address.local998 = getelementptr [2 x i8], ptr addrspace(1) %140, i64 %address.local997, !dbg !12
  %dma.source.bits1942 = ptrtoint ptr addrspace(1) %address.local998 to i64, !dbg !12
  %dma.source.opaque1943 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1942) #10, !dbg !12
  %dma.source.ptr1944 = inttoptr i64 %dma.source.opaque1943 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1944, ptr addrspace(3) %72, i32 4, i32 0, i32 0), !dbg !12
  %address.local999 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local1000 = zext nneg i32 %address.local999 to i64, !dbg !12
  %address.local1001 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local1002 = zext nneg i32 %address.local1001 to i64, !dbg !12
  %address.local1003 = sub nsw i64 %address.local1000, %address.local1002, !dbg !12
  %address.local1004 = sdiv i64 %address.local1003, -4, !dbg !12
  %address.local1005 = shl i64 %address.local1004, 9, !dbg !12
  %address.local1006 = mul nuw nsw i64 %address.local1002, 7, !dbg !12
  %address.local1007 = add nuw nsw i64 %address.local1006, %address.local1000, !dbg !12
  %address.local1008 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local1009 = and i32 %address.local1008, 60, !dbg !12
  %address.local1010 = zext nneg i32 %address.local1009 to i64, !dbg !12
  %address.local1011 = add nuw nsw i64 %address.local1007, %address.local1010, !dbg !12
  %address.local1012 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local1013 = and i32 %address.local1012, 4, !dbg !12
  %address.local1014 = sub nsw i32 0, %address.local1013, !dbg !12
  %address.local1015 = sext i32 %address.local1014 to i64, !dbg !12
  %address.local1016 = add nsw i64 %address.local1011, %address.local1015, !dbg !12
  %address.local1017 = shl nsw i64 %address.local1016, 7, !dbg !12
  %address.local1018 = add nsw i64 %address.local1017, -512, !dbg !12
  %address.local1019 = lshr i32 %address.thread783, 7, !dbg !12
  %address.local1020 = xor i32 %address.local1019, -1, !dbg !12
  %address.local1021 = sext i32 %address.local1020 to i64, !dbg !12
  %address.local1022 = lshr i32 %address.thread783, 4, !dbg !12
  %address.local1023 = zext nneg i32 %address.local1022 to i64, !dbg !12
  %address.local1024 = add nsw i64 %address.local1021, %address.local1023, !dbg !12
  %address.local1025 = ashr i64 %address.local1024, 63, !dbg !12
  %address.local1026 = xor i64 %address.local1024, %address.local1025, !dbg !12
  %address.local1027 = sdiv i64 %address.local1026, 2, !dbg !12
  %address.local1028 = ashr i64 %address.local1024, 63, !dbg !12
  %address.local1029 = xor i64 %address.local1027, %address.local1028, !dbg !12
  %address.local1030 = shl nsw i64 %address.local1029, 10, !dbg !12
  %address.local1031 = sub i64 %address.local1018, %address.local1030, !dbg !12
  %address.local1032 = add i64 %address.local1005, %address.local1031, !dbg !12
  %address.local1033 = shl nuw nsw i32 %address.thread783, 1, !dbg !12
  %address.local1034 = and i32 %address.local1033, 2, !dbg !12
  %address.local1035 = shl nuw nsw i32 %address.thread783, 3, !dbg !12
  %address.local1036 = and i32 %address.local1035, 16, !dbg !12
  %address.local1037 = or disjoint i32 %address.local1034, %address.local1036, !dbg !12
  %address.local1038 = or disjoint i32 %address.local1037, 8, !dbg !12
  %address.local1039 = zext nneg i32 %address.local1038 to i64, !dbg !12
  %address.local1040 = or disjoint i64 %address.local1032, %address.local1039, !dbg !12
  %address.local1041 = getelementptr [2 x i8], ptr addrspace(1) %140, i64 %address.local1040, !dbg !12
  %dma.source.bits1945 = ptrtoint ptr addrspace(1) %address.local1041 to i64, !dbg !12
  %dma.source.opaque1946 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1945) #10, !dbg !12
  %dma.source.ptr1947 = inttoptr i64 %dma.source.opaque1946 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1947, ptr addrspace(3) %75, i32 4, i32 0, i32 0), !dbg !12
  %address.local1042 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local1043 = or disjoint i32 %address.local1042, 8, !dbg !12
  %address.local1044 = zext nneg i32 %address.local1043 to i64, !dbg !12
  %address.local1045 = mul nuw nsw i64 %address.local1044, 7, !dbg !12
  %address.local1046 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local1047 = or disjoint i32 %address.local1046, 64, !dbg !12
  %address.local1048 = zext nneg i32 %address.local1047 to i64, !dbg !12
  %address.local1049 = add nuw nsw i64 %address.local1045, %address.local1048, !dbg !12
  %address.local1050 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local1051 = and i32 %address.local1050, 60, !dbg !12
  %address.local1052 = or disjoint i32 %address.local1051, 64, !dbg !12
  %address.local1053 = zext nneg i32 %address.local1052 to i64, !dbg !12
  %address.local1054 = add nuw nsw i64 %address.local1049, %address.local1053, !dbg !12
  %address.local1055 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local1056 = and i32 %address.local1055, 4, !dbg !12
  %address.local1057 = sub nuw nsw i32 -8, %address.local1056, !dbg !12
  %address.local1058 = sext i32 %address.local1057 to i64, !dbg !12
  %address.local1059 = add nsw i64 %address.local1054, %address.local1058, !dbg !12
  %address.local1060 = add nsw i64 %address.local1059, 144115188075855868, !dbg !12
  %address.local1061 = lshr i32 %address.thread783, 4, !dbg !12
  %address.local1062 = or disjoint i32 %address.local1061, 16, !dbg !12
  %address.local1063 = lshr i32 %address.thread783, 7, !dbg !12
  %address.local1064 = or disjoint i32 %address.local1063, 2, !dbg !12
  %address.local1065 = sub nuw nsw i32 %address.local1062, %address.local1064, !dbg !12
  %address.local1066 = zext nneg i32 %address.local1065 to i64, !dbg !12
  %address.local1067 = add nsw i64 %address.local1066, -1, !dbg !12
  %address.local1068 = sdiv i64 %address.local1067, -2, !dbg !12
  %address.local1069 = shl nsw i64 %address.local1068, 3, !dbg !12
  %address.local1070 = add i64 %address.local1060, %address.local1069, !dbg !12
  %address.local1071 = sub nuw nsw i64 %address.local1048, %address.local1044, !dbg !12
  %address.local1072 = and i64 %address.local1071, 124, !dbg !12
  %address.local1073 = sub i64 %address.local1070, %address.local1072, !dbg !12
  %address.local1074 = shl i64 %address.local1073, 7, !dbg !12
  %address.local1075 = shl nuw nsw i32 %address.thread783, 1, !dbg !12
  %address.local1076 = and i32 %address.local1075, 2, !dbg !12
  %address.local1077 = shl nuw nsw i32 %address.thread783, 3, !dbg !12
  %address.local1078 = and i32 %address.local1077, 16, !dbg !12
  %address.local1079 = or disjoint i32 %address.local1076, %address.local1078, !dbg !12
  %address.local1080 = or disjoint i32 %address.local1079, 8, !dbg !12
  %address.local1081 = zext nneg i32 %address.local1080 to i64, !dbg !12
  %address.local1082 = or disjoint i64 %address.local1074, %address.local1081, !dbg !12
  %address.local1083 = getelementptr [2 x i8], ptr addrspace(1) %140, i64 %address.local1082, !dbg !12
  %dma.source.bits1948 = ptrtoint ptr addrspace(1) %address.local1083 to i64, !dbg !12
  %dma.source.opaque1949 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1948) #10, !dbg !12
  %dma.source.ptr1950 = inttoptr i64 %dma.source.opaque1949 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1950, ptr addrspace(3) %78, i32 4, i32 0, i32 0), !dbg !12
  %address.local1084 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local1085 = zext nneg i32 %address.local1084 to i64, !dbg !12
  %address.local1086 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local1087 = zext nneg i32 %address.local1086 to i64, !dbg !12
  %address.local1088 = sub nsw i64 %address.local1085, %address.local1087, !dbg !12
  %address.local1089 = sdiv i64 %address.local1088, -4, !dbg !12
  %address.local1090 = shl i64 %address.local1089, 9, !dbg !12
  %address.local1091 = mul nuw nsw i64 %address.local1087, 7, !dbg !12
  %address.local1092 = add nuw nsw i64 %address.local1091, %address.local1085, !dbg !12
  %address.local1093 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local1094 = and i32 %address.local1093, 60, !dbg !12
  %address.local1095 = zext nneg i32 %address.local1094 to i64, !dbg !12
  %address.local1096 = add nuw nsw i64 %address.local1092, %address.local1095, !dbg !12
  %address.local1097 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local1098 = and i32 %address.local1097, 4, !dbg !12
  %address.local1099 = sub nsw i32 0, %address.local1098, !dbg !12
  %address.local1100 = sext i32 %address.local1099 to i64, !dbg !12
  %address.local1101 = add nsw i64 %address.local1096, %address.local1100, !dbg !12
  %address.local1102 = shl nsw i64 %address.local1101, 7, !dbg !12
  %address.local1103 = add nsw i64 %address.local1102, -512, !dbg !12
  %address.local1104 = lshr i32 %address.thread783, 7, !dbg !12
  %address.local1105 = xor i32 %address.local1104, -1, !dbg !12
  %address.local1106 = sext i32 %address.local1105 to i64, !dbg !12
  %address.local1107 = lshr i32 %address.thread783, 4, !dbg !12
  %address.local1108 = zext nneg i32 %address.local1107 to i64, !dbg !12
  %address.local1109 = add nsw i64 %address.local1106, %address.local1108, !dbg !12
  %address.local1110 = ashr i64 %address.local1109, 63, !dbg !12
  %address.local1111 = xor i64 %address.local1109, %address.local1110, !dbg !12
  %address.local1112 = sdiv i64 %address.local1111, 2, !dbg !12
  %address.local1113 = ashr i64 %address.local1109, 63, !dbg !12
  %address.local1114 = xor i64 %address.local1112, %address.local1113, !dbg !12
  %address.local1115 = shl nsw i64 %address.local1114, 10, !dbg !12
  %address.local1116 = sub i64 %address.local1103, %address.local1115, !dbg !12
  %address.local1117 = add i64 %address.local1090, %address.local1116, !dbg !12
  %address.local1118 = shl nuw nsw i32 %address.thread783, 1, !dbg !12
  %address.local1119 = and i32 %address.local1118, 2, !dbg !12
  %address.local1120 = shl nuw nsw i32 %address.thread783, 3, !dbg !12
  %address.local1121 = and i32 %address.local1120, 16, !dbg !12
  %address.local1122 = or disjoint i32 %address.local1119, %address.local1121, !dbg !12
  %address.local1123 = or disjoint i32 %address.local1122, 12, !dbg !12
  %address.local1124 = zext nneg i32 %address.local1123 to i64, !dbg !12
  %address.local1125 = or disjoint i64 %address.local1117, %address.local1124, !dbg !12
  %address.local1126 = getelementptr [2 x i8], ptr addrspace(1) %140, i64 %address.local1125, !dbg !12
  %dma.source.bits1951 = ptrtoint ptr addrspace(1) %address.local1126 to i64, !dbg !12
  %dma.source.opaque1952 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1951) #10, !dbg !12
  %dma.source.ptr1953 = inttoptr i64 %dma.source.opaque1952 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1953, ptr addrspace(3) %81, i32 4, i32 0, i32 0), !dbg !12
  %address.local1127 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local1128 = or disjoint i32 %address.local1127, 8, !dbg !12
  %address.local1129 = zext nneg i32 %address.local1128 to i64, !dbg !12
  %address.local1130 = mul nuw nsw i64 %address.local1129, 7, !dbg !12
  %address.local1131 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local1132 = or disjoint i32 %address.local1131, 64, !dbg !12
  %address.local1133 = zext nneg i32 %address.local1132 to i64, !dbg !12
  %address.local1134 = add nuw nsw i64 %address.local1130, %address.local1133, !dbg !12
  %address.local1135 = lshr i32 %address.thread783, 2, !dbg !12
  %address.local1136 = and i32 %address.local1135, 60, !dbg !12
  %address.local1137 = or disjoint i32 %address.local1136, 64, !dbg !12
  %address.local1138 = zext nneg i32 %address.local1137 to i64, !dbg !12
  %address.local1139 = add nuw nsw i64 %address.local1134, %address.local1138, !dbg !12
  %address.local1140 = lshr i32 %address.thread783, 5, !dbg !12
  %address.local1141 = and i32 %address.local1140, 4, !dbg !12
  %address.local1142 = sub nuw nsw i32 -8, %address.local1141, !dbg !12
  %address.local1143 = sext i32 %address.local1142 to i64, !dbg !12
  %address.local1144 = add nsw i64 %address.local1139, %address.local1143, !dbg !12
  %address.local1145 = add nsw i64 %address.local1144, 144115188075855868, !dbg !12
  %address.local1146 = lshr i32 %address.thread783, 4, !dbg !12
  %address.local1147 = or disjoint i32 %address.local1146, 16, !dbg !12
  %address.local1148 = lshr i32 %address.thread783, 7, !dbg !12
  %address.local1149 = or disjoint i32 %address.local1148, 2, !dbg !12
  %address.local1150 = sub nuw nsw i32 %address.local1147, %address.local1149, !dbg !12
  %address.local1151 = zext nneg i32 %address.local1150 to i64, !dbg !12
  %address.local1152 = add nsw i64 %address.local1151, -1, !dbg !12
  %address.local1153 = sdiv i64 %address.local1152, -2, !dbg !12
  %address.local1154 = shl nsw i64 %address.local1153, 3, !dbg !12
  %address.local1155 = add i64 %address.local1145, %address.local1154, !dbg !12
  %address.local1156 = sub nuw nsw i64 %address.local1133, %address.local1129, !dbg !12
  %address.local1157 = and i64 %address.local1156, 124, !dbg !12
  %address.local1158 = sub i64 %address.local1155, %address.local1157, !dbg !12
  %address.local1159 = shl i64 %address.local1158, 7, !dbg !12
  %address.local1160 = shl nuw nsw i32 %address.thread783, 1, !dbg !12
  %address.local1161 = and i32 %address.local1160, 2, !dbg !12
  %address.local1162 = shl nuw nsw i32 %address.thread783, 3, !dbg !12
  %address.local1163 = and i32 %address.local1162, 16, !dbg !12
  %address.local1164 = or disjoint i32 %address.local1161, %address.local1163, !dbg !12
  %address.local1165 = or disjoint i32 %address.local1164, 12, !dbg !12
  %address.local1166 = zext nneg i32 %address.local1165 to i64, !dbg !12
  %address.local1167 = or disjoint i64 %address.local1159, %address.local1166, !dbg !12
  %address.local1168 = getelementptr [2 x i8], ptr addrspace(1) %140, i64 %address.local1167, !dbg !12
  %dma.source.bits1954 = ptrtoint ptr addrspace(1) %address.local1168 to i64, !dbg !12
  %dma.source.opaque1955 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1954) #10, !dbg !12
  %dma.source.ptr1956 = inttoptr i64 %dma.source.opaque1955 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1956, ptr addrspace(3) %84, i32 4, i32 0, i32 0), !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %address.local1169 = and i32 %address.thread783, 192, !dbg !13
  %address.local1170 = shl nuw nsw i32 %address.thread783, 2, !dbg !13
  %address.local1171 = and i32 %address.local1170, 60, !dbg !13
  %address.local1172 = or disjoint i32 %address.local1169, %address.local1171, !dbg !13
  %address.local1173 = shl nuw nsw i32 %address.thread783, 5, !dbg !13
  %address.local1174 = and i32 %address.local1173, 1536, !dbg !13
  %address.local1175 = or disjoint i32 %address.local1172, %address.local1174, !dbg !13
  %address.local1176 = zext nneg i32 %address.local1175 to i64, !dbg !13
  %address.local1177 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1176, !dbg !13
  %141 = load <4 x half>, ptr addrspace(3) %address.local1177, align 2, !dbg !13
  %address.local1178 = and i32 %address.thread783, 192, !dbg !13
  %address.local1179 = shl nuw nsw i32 %address.thread783, 2, !dbg !13
  %address.local1180 = and i32 %address.local1179, 60, !dbg !13
  %address.local1181 = or disjoint i32 %address.local1178, %address.local1180, !dbg !13
  %address.local1182 = shl nuw nsw i32 %address.thread783, 5, !dbg !13
  %address.local1183 = and i32 %address.local1182, 1536, !dbg !13
  %address.local1184 = or disjoint i32 %address.local1181, %address.local1183, !dbg !13
  %address.local1185 = zext nneg i32 %address.local1184 to i64, !dbg !13
  %address.local1186 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1185, !dbg !13
  %address.local1187 = getelementptr i8, ptr addrspace(3) %address.local1186, i64 4096, !dbg !13
  %142 = load <4 x half>, ptr addrspace(3) %address.local1187, align 2, !dbg !13
  %address.local1188 = and i32 %address.thread783, 192, !dbg !13
  %address.local1189 = shl nuw nsw i32 %address.thread783, 2, !dbg !13
  %address.local1190 = and i32 %address.local1189, 60, !dbg !13
  %address.local1191 = or disjoint i32 %address.local1188, %address.local1190, !dbg !13
  %address.local1192 = shl nuw nsw i32 %address.thread783, 5, !dbg !13
  %address.local1193 = and i32 %address.local1192, 1536, !dbg !13
  %address.local1194 = or disjoint i32 %address.local1191, %address.local1193, !dbg !13
  %address.local1195 = zext nneg i32 %address.local1194 to i64, !dbg !13
  %address.local1196 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1195, !dbg !13
  %address.local1197 = getelementptr i8, ptr addrspace(3) %address.local1196, i64 512, !dbg !13
  %143 = load <4 x half>, ptr addrspace(3) %address.local1197, align 2, !dbg !13
  %144 = fmul <4 x half> %141, splat (half 0xH3015), !dbg !13
  %address.local1198 = and i32 %address.thread783, 192, !dbg !13
  %address.local1199 = shl nuw nsw i32 %address.thread783, 2, !dbg !13
  %address.local1200 = and i32 %address.local1199, 60, !dbg !13
  %address.local1201 = or disjoint i32 %address.local1198, %address.local1200, !dbg !13
  %address.local1202 = shl nuw nsw i32 %address.thread783, 5, !dbg !13
  %address.local1203 = and i32 %address.local1202, 1536, !dbg !13
  %address.local1204 = or disjoint i32 %address.local1201, %address.local1203, !dbg !13
  %address.local1205 = zext nneg i32 %address.local1204 to i64, !dbg !13
  %address.local1206 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1205, !dbg !13
  %address.local1207 = getelementptr i8, ptr addrspace(3) %address.local1206, i64 4608, !dbg !13
  %145 = load <4 x half>, ptr addrspace(3) %address.local1207, align 2, !dbg !13
  %146 = fmul <4 x half> %142, splat (half 0xH3015), !dbg !13
  %147 = fmul <4 x half> %143, splat (half 0xH3015), !dbg !13
  %148 = fmul <4 x half> %145, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %address.local1208 = zext nneg i32 %address.thread783 to i64, !dbg !13
  %address.local1209 = mul nuw nsw i64 %address.local1208, 400, !dbg !16
  %address.local1210 = lshr i32 %address.thread783, 2, !dbg !16
  %address.local1211 = zext nneg i32 %address.local1210 to i64, !dbg !16
  %address.local1212 = mul nsw i64 %address.local1211, -1664, !dbg !16
  %address.local1213 = add nsw i64 %address.local1209, %address.local1212, !dbg !16
  %address.local1214 = shl nuw nsw i32 %address.thread783, 1, !dbg !16
  %address.local1215 = and i32 %address.local1214, 24, !dbg !16
  %address.local1216 = zext nneg i32 %address.local1215 to i64, !dbg !16
  %address.local1217 = add nsw i64 %address.local1213, %address.local1216, !dbg !16
  %address.local1218 = shl nuw nsw i32 %address.thread783, 4, !dbg !16
  %address.local1219 = and i32 %address.local1218, 4064, !dbg !16
  %address.local1220 = zext nneg i32 %address.local1219 to i64, !dbg !16
  %address.local1221 = add nsw i64 %address.local1217, %address.local1220, !dbg !16
  %address.local1222 = shl nuw nsw i32 %address.thread783, 6, !dbg !16
  %address.local1223 = and i32 %address.local1222, 3072, !dbg !16
  %address.local1224 = zext nneg i32 %address.local1223 to i64, !dbg !16
  %address.local1225 = add nsw i64 %address.local1221, %address.local1224, !dbg !16
  %address.local1226 = and i32 %address.thread783, 32, !dbg !16
  %address.local1227 = zext nneg i32 %address.local1226 to i64, !dbg !16
  %address.local1228 = add nsw i64 %address.local1225, %address.local1227, !dbg !16
  %address.base = call i64 asm sideeffect "", "=v,0"(i64 %.neg627) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local1229 = shl nsw i64 %address.base, 6, !dbg !16
  %address.local1230 = add i64 %address.local1228, %address.local1229, !dbg !16
  %address.base1231 = call i64 asm sideeffect "", "=v,0"(i64 %.neg628) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local1232 = shl nsw i64 %address.base1231, 5, !dbg !16
  %address.local1233 = add i64 %address.local1230, %address.local1232, !dbg !16
  %address.local1234 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1233, !dbg !16
  %149 = load <8 x half>, ptr addrspace(3) %address.local1234, align 2, !dbg !16
  %150 = shufflevector <8 x half> %149, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %151 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %150, <4 x float> zeroinitializer) #9, !dbg !16
  %152 = shufflevector <8 x half> %149, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %152, <4 x float> %151) #9, !dbg !16
  %154 = fadd <4 x float> %153, zeroinitializer, !dbg !20
  %address.local1235 = shl nuw nsw i32 %address.thread783, 1, !dbg !16
  %address.local1236 = and i32 %address.local1235, 6, !dbg !16
  %address.local1237 = mul nuw nsw i32 %address.local1236, 72, !dbg !16
  %address.local1238 = shl nuw nsw i32 %address.thread783, 1, !dbg !16
  %address.local1239 = and i32 %address.local1238, 24, !dbg !16
  %address.local1240 = shl nuw nsw i32 %address.thread783, 8, !dbg !16
  %address.local1241 = and i32 %address.local1240, 768, !dbg !16
  %address.local1242 = or disjoint i32 %address.local1239, %address.local1241, !dbg !16
  %address.local1243 = add nuw nsw i32 %address.local1237, %address.local1242, !dbg !16
  %address.local1244 = shl nuw nsw i32 %address.thread783, 4, !dbg !16
  %address.local1245 = and i32 %address.local1244, 32, !dbg !16
  %address.local1246 = add nuw nsw i32 %address.local1243, %address.local1245, !dbg !16
  %address.local1247 = add nuw nsw i32 %address.local1246, %address.local1223, !dbg !16
  %address.local1248 = add nuw nsw i32 %address.local1247, %address.local1226, !dbg !16
  %address.local1249 = zext nneg i32 %address.local1248 to i64, !dbg !16
  %address.local1250 = shl nuw nsw i32 %address.thread783, 2, !dbg !16
  %address.local1251 = and i32 %address.local1250, 12, !dbg !16
  %address.local1252 = lshr i32 %address.thread783, 1, !dbg !16
  %address.local1253 = and i32 %address.local1252, 1, !dbg !16
  %address.local1254 = or disjoint i32 %address.local1251, %address.local1253, !dbg !16
  %address.local1255 = lshr i32 %address.thread783, 5, !dbg !16
  %address.local1256 = and i32 %address.local1255, 1, !dbg !16
  %address.local1257 = add nuw nsw i32 %address.local1254, %address.local1256, !dbg !16
  %address.local1258 = shl nuw nsw i32 %address.local1257, 5, !dbg !16
  %address.local1259 = add nuw nsw i32 %address.local1258, 32, !dbg !16
  %address.local1260 = and i32 %address.local1259, 960, !dbg !16
  %address.local1261 = lshr i32 %address.thread783, 2, !dbg !16
  %address.local1262 = and i32 %address.local1261, 2, !dbg !16
  %address.local1263 = shl nuw nsw i32 %address.thread783, 4, !dbg !16
  %address.local1264 = and i32 %address.local1263, 48, !dbg !16
  %address.local1265 = or disjoint i32 %address.local1262, %address.local1264, !dbg !16
  %address.local1266 = add nuw nsw i32 %address.local1265, %address.local1236, !dbg !16
  %address.local1267 = shl nuw nsw i32 %address.local1266, 3, !dbg !16
  %address.local1268 = and i32 %address.local1267, 480, !dbg !16
  %address.local1269 = add nuw nsw i32 %address.local1260, %address.local1268, !dbg !16
  %address.local1270 = zext nneg i32 %address.local1269 to i64, !dbg !16
  %address.local1271 = sub nsw i64 %address.local1249, %address.local1270, !dbg !16
  %address.local1272 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1271, !dbg !16
  %address.local1273 = getelementptr i8, ptr addrspace(3) %address.local1272, i64 64, !dbg !16
  %155 = load <8 x half>, ptr addrspace(3) %address.local1273, align 2, !dbg !16
  %address.local1274 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1209, !dbg !16
  %address.local1275 = getelementptr [2 x i8], ptr addrspace(3) %address.local1274, i64 %address.local1212, !dbg !16
  %address.local1276 = getelementptr [2 x i8], ptr addrspace(3) %address.local1275, i64 %address.local1216, !dbg !16
  %address.local1277 = getelementptr [2 x i8], ptr addrspace(3) %address.local1276, i64 %address.local1220, !dbg !16
  %address.local1278 = getelementptr [2 x i8], ptr addrspace(3) %address.local1277, i64 %address.local1224, !dbg !16
  %address.local1279 = getelementptr [2 x i8], ptr addrspace(3) %address.local1278, i64 %address.local1227, !dbg !16
  %address.base1280 = call i64 asm sideeffect "", "=v,0"(i64 %.neg638) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local1281 = shl i64 %address.base1280, 7, !dbg !16
  %address.local1282 = getelementptr i8, ptr addrspace(3) %address.local1279, i64 %address.local1281, !dbg !16
  %address.base1283 = call i64 asm sideeffect "", "=v,0"(i64 %.neg639) #9, !dbg !16, !frisk.register_lifetimes !1
  %address.local1284 = shl i64 %address.base1283, 6, !dbg !16
  %address.local1285 = getelementptr i8, ptr addrspace(3) %address.local1282, i64 %address.local1284, !dbg !16
  %address.local1286 = getelementptr i8, ptr addrspace(3) %address.local1285, i64 144, !dbg !16
  %156 = load <8 x half>, ptr addrspace(3) %address.local1286, align 2, !dbg !16
  %address.local1287 = shl nuw nsw i32 %address.thread783, 1, !dbg !16
  %address.local1288 = and i32 %address.local1287, 6, !dbg !16
  %address.local1289 = or disjoint i32 %address.local1288, 1, !dbg !16
  %address.local1290 = mul nuw nsw i32 %address.local1289, 72, !dbg !16
  %address.local1291 = shl nuw nsw i32 %address.thread783, 1, !dbg !16
  %address.local1292 = and i32 %address.local1291, 24, !dbg !16
  %address.local1293 = shl nuw nsw i32 %address.thread783, 8, !dbg !16
  %address.local1294 = and i32 %address.local1293, 768, !dbg !16
  %address.local1295 = or disjoint i32 %address.local1292, %address.local1294, !dbg !16
  %address.local1296 = add nuw nsw i32 %address.local1290, %address.local1295, !dbg !16
  %address.local1297 = shl nuw nsw i32 %address.thread783, 4, !dbg !16
  %address.local1298 = and i32 %address.local1297, 32, !dbg !16
  %address.local1299 = add nuw nsw i32 %address.local1296, %address.local1298, !dbg !16
  %address.local1300 = add nuw nsw i32 %address.local1299, %address.local1223, !dbg !16
  %address.local1301 = add nuw nsw i32 %address.local1300, %address.local1226, !dbg !16
  %address.local1302 = zext nneg i32 %address.local1301 to i64, !dbg !16
  %address.local1303 = shl nuw nsw i32 %address.thread783, 2, !dbg !16
  %address.local1304 = and i32 %address.local1303, 12, !dbg !16
  %address.local1305 = lshr i32 %address.thread783, 1, !dbg !16
  %address.local1306 = and i32 %address.local1305, 1, !dbg !16
  %address.local1307 = or disjoint i32 %address.local1304, %address.local1306, !dbg !16
  %address.local1308 = add nuw nsw i32 %address.local1307, %address.local1256, !dbg !16
  %address.local1309 = shl nuw nsw i32 %address.local1308, 5, !dbg !16
  %address.local1310 = add nuw nsw i32 %address.local1309, 96, !dbg !16
  %address.local1311 = and i32 %address.local1310, 960, !dbg !16
  %address.local1312 = lshr i32 %address.thread783, 2, !dbg !16
  %address.local1313 = and i32 %address.local1312, 3, !dbg !16
  %address.local1314 = shl nuw nsw i32 %address.thread783, 4, !dbg !16
  %address.local1315 = and i32 %address.local1314, 48, !dbg !16
  %address.local1316 = or disjoint i32 %address.local1313, %address.local1315, !dbg !16
  %address.local1317 = add nuw nsw i32 %address.local1316, %address.local1289, !dbg !16
  %address.local1318 = shl nuw nsw i32 %address.local1317, 3, !dbg !16
  %address.local1319 = and i32 %address.local1318, 480, !dbg !16
  %address.local1320 = add nuw nsw i32 %address.local1311, %address.local1319, !dbg !16
  %address.local1321 = zext nneg i32 %address.local1320 to i64, !dbg !16
  %address.local1322 = sub nsw i64 %address.local1302, %address.local1321, !dbg !16
  %address.local1323 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1322, !dbg !16
  %address.local1324 = getelementptr i8, ptr addrspace(3) %address.local1323, i64 192, !dbg !16
  %157 = load <8 x half>, ptr addrspace(3) %address.local1324, align 2, !dbg !16
  %158 = shufflevector <8 x half> %155, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %158, <4 x float> zeroinitializer) #9, !dbg !16
  %160 = shufflevector <8 x half> %155, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %161 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %160, <4 x float> %159) #9, !dbg !16
  %162 = shufflevector <8 x half> %156, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %163 = shufflevector <8 x half> %156, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %164 = fadd <4 x float> %161, zeroinitializer, !dbg !20
  %165 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %162, <4 x float> zeroinitializer) #9, !dbg !16
  %166 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %163, <4 x float> %165) #9, !dbg !16
  %167 = shufflevector <8 x half> %157, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %168 = shufflevector <8 x half> %157, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %169 = fadd <4 x float> %166, zeroinitializer, !dbg !20
  %170 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %167, <4 x float> zeroinitializer) #9, !dbg !16
  %address.local1325 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1233, !dbg !16
  %address.local1326 = getelementptr i8, ptr addrspace(3) %address.local1325, i64 1024, !dbg !16
  %171 = load <8 x half>, ptr addrspace(3) %address.local1326, align 2, !dbg !16
  %172 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %168, <4 x float> %170) #9, !dbg !16
  %address.local1327 = shl nuw nsw i32 %address.thread783, 1, !dbg !16
  %address.local1328 = and i32 %address.local1327, 6, !dbg !16
  %address.local1329 = or disjoint i32 %address.local1328, 8, !dbg !16
  %address.local1330 = mul nuw nsw i32 %address.local1329, 72, !dbg !16
  %address.local1331 = shl nuw nsw i32 %address.thread783, 1, !dbg !16
  %address.local1332 = and i32 %address.local1331, 24, !dbg !16
  %address.local1333 = shl nuw nsw i32 %address.thread783, 8, !dbg !16
  %address.local1334 = and i32 %address.local1333, 768, !dbg !16
  %address.local1335 = or disjoint i32 %address.local1332, %address.local1334, !dbg !16
  %address.local1336 = or disjoint i32 %address.local1335, 1024, !dbg !16
  %address.local1337 = add nuw nsw i32 %address.local1330, %address.local1336, !dbg !16
  %address.local1338 = shl nuw nsw i32 %address.thread783, 4, !dbg !16
  %address.local1339 = and i32 %address.local1338, 32, !dbg !16
  %address.local1340 = or disjoint i32 %address.local1339, 64, !dbg !16
  %address.local1341 = add nuw nsw i32 %address.local1337, %address.local1340, !dbg !16
  %address.local1342 = add nuw nsw i32 %address.local1341, %address.local1223, !dbg !16
  %address.local1343 = add nuw nsw i32 %address.local1342, %address.local1226, !dbg !16
  %address.local1344 = zext i32 %address.local1343 to i64, !dbg !16
  %address.local1345 = shl nuw nsw i32 %address.thread783, 2, !dbg !16
  %address.local1346 = and i32 %address.local1345, 12, !dbg !16
  %address.local1347 = lshr i32 %address.thread783, 1, !dbg !16
  %address.local1348 = and i32 %address.local1347, 1, !dbg !16
  %address.local1349 = or disjoint i32 %address.local1346, %address.local1348, !dbg !16
  %address.local1350 = or disjoint i32 %address.local1349, 18, !dbg !16
  %address.local1351 = add nuw nsw i32 %address.local1350, %address.local1256, !dbg !16
  %address.local1352 = shl nuw nsw i32 %address.local1351, 5, !dbg !16
  %address.local1353 = add nuw nsw i32 %address.local1352, 32, !dbg !16
  %address.local1354 = and i32 %address.local1353, 2147483584, !dbg !16
  %address.local1355 = lshr i32 %address.thread783, 2, !dbg !16
  %address.local1356 = and i32 %address.local1355, 2, !dbg !16
  %address.local1357 = shl nuw nsw i32 %address.thread783, 4, !dbg !16
  %address.local1358 = and i32 %address.local1357, 48, !dbg !16
  %address.local1359 = or disjoint i32 %address.local1356, %address.local1358, !dbg !16
  %address.local1360 = or disjoint i32 %address.local1359, 64, !dbg !16
  %address.local1361 = add nuw nsw i32 %address.local1360, %address.local1329, !dbg !16
  %address.local1362 = shl nuw nsw i32 %address.local1361, 3, !dbg !16
  %address.local1363 = and i32 %address.local1362, 2016, !dbg !16
  %address.local1364 = add nuw i32 %address.local1354, %address.local1363, !dbg !16
  %address.local1365 = zext i32 %address.local1364 to i64, !dbg !16
  %address.local1366 = sub nsw i64 %address.local1344, %address.local1365, !dbg !16
  %address.local1367 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1366, !dbg !16
  %address.local1368 = getelementptr i8, ptr addrspace(3) %address.local1367, i64 64, !dbg !16
  %173 = load <8 x half>, ptr addrspace(3) %address.local1368, align 2, !dbg !16
  %address.local1369 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1209, !dbg !16
  %address.local1370 = getelementptr [2 x i8], ptr addrspace(3) %address.local1369, i64 %address.local1212, !dbg !16
  %address.local1371 = getelementptr [2 x i8], ptr addrspace(3) %address.local1370, i64 %address.local1216, !dbg !16
  %address.local1372 = getelementptr [2 x i8], ptr addrspace(3) %address.local1371, i64 %address.local1220, !dbg !16
  %address.local1373 = getelementptr [2 x i8], ptr addrspace(3) %address.local1372, i64 %address.local1224, !dbg !16
  %address.local1374 = getelementptr [2 x i8], ptr addrspace(3) %address.local1373, i64 %address.local1227, !dbg !16
  %address.local1375 = shl nuw nsw i32 %address.thread783, 2, !dbg !16
  %address.local1376 = and i32 %address.local1375, 12, !dbg !16
  %address.local1377 = lshr i32 %address.thread783, 1, !dbg !16
  %address.local1378 = and i32 %address.local1377, 126, !dbg !16
  %address.local1379 = sub nsw i32 %address.local1376, %address.local1378, !dbg !16
  %address.local1380 = add nsw i32 %address.local1379, 18, !dbg !16
  %address.local1381 = sext i32 %address.local1380 to i64, !dbg !16
  %address.local1382 = lshr i32 %address.thread783, 1, !dbg !16
  %address.local1383 = zext nneg i32 %address.local1382 to i64, !dbg !16
  %address.local1384 = add nsw i64 %address.local1381, %address.local1383, !dbg !16
  %address.local1385 = zext nneg i32 %address.local1256 to i64, !dbg !16
  %address.local1386 = add nsw i64 %address.local1384, %address.local1385, !dbg !16
  %address.local1387 = sdiv i64 %address.local1386, -2, !dbg !16
  %address.local1388 = shl i64 %address.local1387, 7, !dbg !16
  %address.local1389 = getelementptr i8, ptr addrspace(3) %address.local1374, i64 %address.local1388, !dbg !16
  %address.local1390 = mul nuw nsw i64 %address.local1208, 18, !dbg !16
  %address.local1391 = add nuw nsw i64 %address.local1390, 72, !dbg !16
  %address.local1392 = mul nsw i64 %address.local1211, -72, !dbg !16
  %address.local1393 = add nsw i64 %address.local1391, %address.local1392, !dbg !16
  %address.local1394 = lshr i32 %address.thread783, 2, !dbg !16
  %address.local1395 = and i32 %address.local1394, 3, !dbg !16
  %address.local1396 = zext nneg i32 %address.local1395 to i64, !dbg !16
  %address.local1397 = add nsw i64 %address.local1393, %address.local1396, !dbg !16
  %address.local1398 = add nsw i64 %address.local1397, -15, !dbg !16
  %address.local1399 = sdiv i64 %address.local1398, -4, !dbg !16
  %address.local1400 = shl i64 %address.local1399, 6, !dbg !16
  %address.local1401 = getelementptr i8, ptr addrspace(3) %address.local1389, i64 %address.local1400, !dbg !16
  %address.local1402 = getelementptr i8, ptr addrspace(3) %address.local1401, i64 3216, !dbg !16
  %174 = load <8 x half>, ptr addrspace(3) %address.local1402, align 2, !dbg !16
  %175 = fadd <4 x float> %172, zeroinitializer, !dbg !20
  %176 = shufflevector <8 x half> %171, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %177 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %176, <4 x float> zeroinitializer) #9, !dbg !16
  %178 = shufflevector <8 x half> %171, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %179 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %178, <4 x float> %177) #9, !dbg !16
  %180 = shufflevector <8 x half> %173, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %181 = shufflevector <8 x half> %173, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %182 = fadd <4 x float> %179, zeroinitializer, !dbg !20
  %183 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %180, <4 x float> zeroinitializer) #9, !dbg !16
  %184 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %181, <4 x float> %183) #9, !dbg !16
  %185 = shufflevector <8 x half> %174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %186 = shufflevector <8 x half> %174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %187 = fadd <4 x float> %184, zeroinitializer, !dbg !20
  %188 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %185, <4 x float> zeroinitializer) #9, !dbg !16
  %189 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %186, <4 x float> %188) #9, !dbg !16
  %address.local1403 = shl nuw nsw i32 %address.thread783, 1, !dbg !16
  %address.local1404 = and i32 %address.local1403, 6, !dbg !16
  %address.local1405 = or disjoint i32 %address.local1404, 9, !dbg !16
  %address.local1406 = mul nuw nsw i32 %address.local1405, 72, !dbg !16
  %address.local1407 = shl nuw nsw i32 %address.thread783, 1, !dbg !16
  %address.local1408 = and i32 %address.local1407, 24, !dbg !16
  %address.local1409 = shl nuw nsw i32 %address.thread783, 8, !dbg !16
  %address.local1410 = and i32 %address.local1409, 768, !dbg !16
  %address.local1411 = or disjoint i32 %address.local1408, %address.local1410, !dbg !16
  %address.local1412 = or disjoint i32 %address.local1411, 1024, !dbg !16
  %address.local1413 = add nuw nsw i32 %address.local1406, %address.local1412, !dbg !16
  %address.local1414 = shl nuw nsw i32 %address.thread783, 4, !dbg !16
  %address.local1415 = and i32 %address.local1414, 32, !dbg !16
  %address.local1416 = or disjoint i32 %address.local1415, 64, !dbg !16
  %address.local1417 = add nuw nsw i32 %address.local1413, %address.local1416, !dbg !16
  %address.local1418 = add nuw nsw i32 %address.local1417, %address.local1223, !dbg !16
  %address.local1419 = add nuw nsw i32 %address.local1418, %address.local1226, !dbg !16
  %address.local1420 = zext i32 %address.local1419 to i64, !dbg !16
  %address.local1421 = shl nuw nsw i32 %address.thread783, 2, !dbg !16
  %address.local1422 = and i32 %address.local1421, 12, !dbg !16
  %address.local1423 = lshr i32 %address.thread783, 1, !dbg !16
  %address.local1424 = and i32 %address.local1423, 1, !dbg !16
  %address.local1425 = or disjoint i32 %address.local1422, %address.local1424, !dbg !16
  %address.local1426 = or disjoint i32 %address.local1425, 18, !dbg !16
  %address.local1427 = add nuw nsw i32 %address.local1426, %address.local1256, !dbg !16
  %address.local1428 = shl nuw nsw i32 %address.local1427, 5, !dbg !16
  %address.local1429 = add nuw nsw i32 %address.local1428, 96, !dbg !16
  %address.local1430 = and i32 %address.local1429, 2147483584, !dbg !16
  %address.local1431 = lshr i32 %address.thread783, 2, !dbg !16
  %address.local1432 = and i32 %address.local1431, 3, !dbg !16
  %address.local1433 = shl nuw nsw i32 %address.thread783, 4, !dbg !16
  %address.local1434 = and i32 %address.local1433, 48, !dbg !16
  %address.local1435 = or disjoint i32 %address.local1432, %address.local1434, !dbg !16
  %address.local1436 = or disjoint i32 %address.local1435, 64, !dbg !16
  %address.local1437 = add nuw nsw i32 %address.local1436, %address.local1405, !dbg !16
  %address.local1438 = shl nuw nsw i32 %address.local1437, 3, !dbg !16
  %address.local1439 = and i32 %address.local1438, 2016, !dbg !16
  %address.local1440 = add nuw i32 %address.local1430, %address.local1439, !dbg !16
  %address.local1441 = zext i32 %address.local1440 to i64, !dbg !16
  %address.local1442 = sub nsw i64 %address.local1420, %address.local1441, !dbg !16
  %address.local1443 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i64 8192), i64 %address.local1442, !dbg !16
  %address.local1444 = getelementptr i8, ptr addrspace(3) %address.local1443, i64 192, !dbg !16
  %190 = load <8 x half>, ptr addrspace(3) %address.local1444, align 2, !dbg !16
  %191 = fadd <4 x float> %189, zeroinitializer, !dbg !20
  %192 = load <8 x half>, ptr addrspace(3) %address.local1234, align 2, !dbg !16
  %193 = shufflevector <8 x half> %190, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %194 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %144, <4 x half> %193, <4 x float> zeroinitializer) #9, !dbg !16
  %195 = shufflevector <8 x half> %190, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %196 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %146, <4 x half> %195, <4 x float> %194) #9, !dbg !16
  %197 = shufflevector <8 x half> %192, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %198 = shufflevector <8 x half> %192, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %199 = fadd <4 x float> %196, zeroinitializer, !dbg !20
  %200 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %197, <4 x float> zeroinitializer) #9, !dbg !16
  %201 = load <8 x half>, ptr addrspace(3) %address.local1273, align 2, !dbg !16
  %202 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %198, <4 x float> %200) #9, !dbg !16
  %203 = load <8 x half>, ptr addrspace(3) %address.local1286, align 2, !dbg !16
  %204 = load <8 x half>, ptr addrspace(3) %address.local1324, align 2, !dbg !16
  %205 = fadd <4 x float> %202, zeroinitializer, !dbg !20
  %206 = shufflevector <8 x half> %201, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %206, <4 x float> zeroinitializer) #9, !dbg !16
  %208 = shufflevector <8 x half> %201, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %208, <4 x float> %207) #9, !dbg !16
  %210 = shufflevector <8 x half> %203, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %211 = shufflevector <8 x half> %203, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %212 = fadd <4 x float> %209, zeroinitializer, !dbg !20
  %213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %210, <4 x float> zeroinitializer) #9, !dbg !16
  %214 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %211, <4 x float> %213) #9, !dbg !16
  %215 = shufflevector <8 x half> %204, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %216 = shufflevector <8 x half> %204, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %217 = fadd <4 x float> %214, zeroinitializer, !dbg !20
  %218 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %215, <4 x float> zeroinitializer) #9, !dbg !16
  %219 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %216, <4 x float> %218) #9, !dbg !16
  %220 = load <8 x half>, ptr addrspace(3) %address.local1326, align 2, !dbg !16
  %221 = fadd <4 x float> %219, zeroinitializer, !dbg !20
  %222 = load <8 x half>, ptr addrspace(3) %address.local1368, align 2, !dbg !16
  %223 = shufflevector <8 x half> %220, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %224 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %223, <4 x float> zeroinitializer) #9, !dbg !16
  %225 = shufflevector <8 x half> %220, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %226 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %225, <4 x float> %224) #9, !dbg !16
  %227 = shufflevector <8 x half> %222, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %228 = shufflevector <8 x half> %222, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %229 = fadd <4 x float> %226, zeroinitializer, !dbg !20
  %230 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %227, <4 x float> zeroinitializer) #9, !dbg !16
  %231 = load <8 x half>, ptr addrspace(3) %address.local1402, align 2, !dbg !16
  %232 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %228, <4 x float> %230) #9, !dbg !16
  %233 = load <8 x half>, ptr addrspace(3) %address.local1444, align 2, !dbg !16
  %234 = fadd <4 x float> %232, zeroinitializer, !dbg !20
  %235 = shufflevector <8 x half> %231, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %236 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %235, <4 x float> zeroinitializer) #9, !dbg !16
  %237 = shufflevector <8 x half> %231, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %238 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %237, <4 x float> %236) #9, !dbg !16
  %239 = shufflevector <8 x half> %233, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %240 = shufflevector <8 x half> %233, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %241 = fadd <4 x float> %238, zeroinitializer, !dbg !20
  %242 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %147, <4 x half> %239, <4 x float> zeroinitializer) #9, !dbg !16
  %243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %148, <4 x half> %240, <4 x float> %242) #9, !dbg !16
  %244 = fadd <4 x float> %243, zeroinitializer, !dbg !20
  %address.local1445 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local789, !dbg !11
  %dma.source.bits1957 = ptrtoint ptr addrspace(1) %address.local1445 to i64, !dbg !11
  %dma.source.opaque1958 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1957) #10, !dbg !11
  %dma.source.ptr1959 = inttoptr i64 %dma.source.opaque1958 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1959, ptr addrspace(3) %39, i32 4, i32 0, i32 0), !dbg !11
  %address.local1446 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local797, !dbg !11
  %dma.source.bits1960 = ptrtoint ptr addrspace(1) %address.local1446 to i64, !dbg !11
  %dma.source.opaque1961 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1960) #10, !dbg !11
  %dma.source.ptr1962 = inttoptr i64 %dma.source.opaque1961 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1962, ptr addrspace(3) %42, i32 4, i32 0, i32 0), !dbg !11
  %address.local1447 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local805, !dbg !11
  %dma.source.bits1963 = ptrtoint ptr addrspace(1) %address.local1447 to i64, !dbg !11
  %dma.source.opaque1964 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1963) #10, !dbg !11
  %dma.source.ptr1965 = inttoptr i64 %dma.source.opaque1964 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1965, ptr addrspace(3) %45, i32 4, i32 0, i32 0), !dbg !11
  %address.local1448 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local813, !dbg !11
  %dma.source.bits1966 = ptrtoint ptr addrspace(1) %address.local1448 to i64, !dbg !11
  %dma.source.opaque1967 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1966) #10, !dbg !11
  %dma.source.ptr1968 = inttoptr i64 %dma.source.opaque1967 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1968, ptr addrspace(3) %48, i32 4, i32 0, i32 0), !dbg !11
  %address.local1449 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local821, !dbg !11
  %dma.source.bits1969 = ptrtoint ptr addrspace(1) %address.local1449 to i64, !dbg !11
  %dma.source.opaque1970 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1969) #10, !dbg !11
  %dma.source.ptr1971 = inttoptr i64 %dma.source.opaque1970 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1971, ptr addrspace(3) %51, i32 4, i32 0, i32 0), !dbg !11
  %address.local1450 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local829, !dbg !11
  %dma.source.bits1972 = ptrtoint ptr addrspace(1) %address.local1450 to i64, !dbg !11
  %dma.source.opaque1973 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1972) #10, !dbg !11
  %dma.source.ptr1974 = inttoptr i64 %dma.source.opaque1973 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1974, ptr addrspace(3) %54, i32 4, i32 0, i32 0), !dbg !11
  %address.local1451 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local837, !dbg !11
  %dma.source.bits1975 = ptrtoint ptr addrspace(1) %address.local1451 to i64, !dbg !11
  %dma.source.opaque1976 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1975) #10, !dbg !11
  %dma.source.ptr1977 = inttoptr i64 %dma.source.opaque1976 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1977, ptr addrspace(3) %57, i32 4, i32 0, i32 0), !dbg !11
  %address.local1452 = getelementptr [2 x i8], ptr addrspace(1) %121, i64 %address.local845, !dbg !11
  %dma.source.bits1978 = ptrtoint ptr addrspace(1) %address.local1452 to i64, !dbg !11
  %dma.source.opaque1979 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1978) #10, !dbg !11
  %dma.source.ptr1980 = inttoptr i64 %dma.source.opaque1979 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1980, ptr addrspace(3) %60, i32 4, i32 0, i32 0), !dbg !11
  %245 = or disjoint i64 %139, 32, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %246 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %245, !dbg !12
  %address.local1453 = getelementptr [2 x i8], ptr addrspace(1) %246, i64 %address.local881, !dbg !12
  %dma.source.bits1981 = ptrtoint ptr addrspace(1) %address.local1453 to i64, !dbg !12
  %dma.source.opaque1982 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1981) #10, !dbg !12
  %dma.source.ptr1983 = inttoptr i64 %dma.source.opaque1982 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1983, ptr addrspace(3) %63, i32 4, i32 0, i32 0), !dbg !12
  %address.local1454 = getelementptr [2 x i8], ptr addrspace(1) %246, i64 %address.local920, !dbg !12
  %dma.source.bits1984 = ptrtoint ptr addrspace(1) %address.local1454 to i64, !dbg !12
  %dma.source.opaque1985 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1984) #10, !dbg !12
  %dma.source.ptr1986 = inttoptr i64 %dma.source.opaque1985 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1986, ptr addrspace(3) %66, i32 4, i32 0, i32 0), !dbg !12
  %address.local1455 = getelementptr [2 x i8], ptr addrspace(1) %246, i64 %address.local957, !dbg !12
  %dma.source.bits1987 = ptrtoint ptr addrspace(1) %address.local1455 to i64, !dbg !12
  %dma.source.opaque1988 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1987) #10, !dbg !12
  %dma.source.ptr1989 = inttoptr i64 %dma.source.opaque1988 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1989, ptr addrspace(3) %69, i32 4, i32 0, i32 0), !dbg !12
  %address.local1456 = getelementptr [2 x i8], ptr addrspace(1) %246, i64 %address.local997, !dbg !12
  %dma.source.bits1990 = ptrtoint ptr addrspace(1) %address.local1456 to i64, !dbg !12
  %dma.source.opaque1991 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1990) #10, !dbg !12
  %dma.source.ptr1992 = inttoptr i64 %dma.source.opaque1991 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1992, ptr addrspace(3) %72, i32 4, i32 0, i32 0), !dbg !12
  %address.local1457 = getelementptr [2 x i8], ptr addrspace(1) %246, i64 %address.local1040, !dbg !12
  %dma.source.bits1993 = ptrtoint ptr addrspace(1) %address.local1457 to i64, !dbg !12
  %dma.source.opaque1994 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1993) #10, !dbg !12
  %dma.source.ptr1995 = inttoptr i64 %dma.source.opaque1994 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1995, ptr addrspace(3) %75, i32 4, i32 0, i32 0), !dbg !12
  %address.local1458 = getelementptr [2 x i8], ptr addrspace(1) %246, i64 %address.local1082, !dbg !12
  %dma.source.bits1996 = ptrtoint ptr addrspace(1) %address.local1458 to i64, !dbg !12
  %dma.source.opaque1997 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1996) #10, !dbg !12
  %dma.source.ptr1998 = inttoptr i64 %dma.source.opaque1997 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr1998, ptr addrspace(3) %78, i32 4, i32 0, i32 0), !dbg !12
  %address.local1459 = getelementptr [2 x i8], ptr addrspace(1) %246, i64 %address.local1125, !dbg !12
  %dma.source.bits1999 = ptrtoint ptr addrspace(1) %address.local1459 to i64, !dbg !12
  %dma.source.opaque2000 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1999) #10, !dbg !12
  %dma.source.ptr2001 = inttoptr i64 %dma.source.opaque2000 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2001, ptr addrspace(3) %81, i32 4, i32 0, i32 0), !dbg !12
  %address.local1460 = getelementptr [2 x i8], ptr addrspace(1) %246, i64 %address.local1167, !dbg !12
  %dma.source.bits2002 = ptrtoint ptr addrspace(1) %address.local1460 to i64, !dbg !12
  %dma.source.opaque2003 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2002) #10, !dbg !12
  %dma.source.ptr2004 = inttoptr i64 %dma.source.opaque2003 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2004, ptr addrspace(3) %84, i32 4, i32 0, i32 0), !dbg !12
  %247 = load <4 x half>, ptr addrspace(3) %address.local1177, align 2, !dbg !13
  %248 = load <4 x half>, ptr addrspace(3) %address.local1187, align 2, !dbg !13
  %249 = load <4 x half>, ptr addrspace(3) %address.local1197, align 2, !dbg !13
  %250 = fmul <4 x half> %247, splat (half 0xH3015), !dbg !13
  %251 = load <4 x half>, ptr addrspace(3) %address.local1207, align 2, !dbg !13
  %252 = fmul <4 x half> %248, splat (half 0xH3015), !dbg !13
  %253 = fmul <4 x half> %249, splat (half 0xH3015), !dbg !13
  %254 = fmul <4 x half> %251, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %255 = load <8 x half>, ptr addrspace(3) %address.local1234, align 2, !dbg !16
  %256 = shufflevector <8 x half> %255, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %250, <4 x half> %256, <4 x float> zeroinitializer) #9, !dbg !16
  %258 = shufflevector <8 x half> %255, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %252, <4 x half> %258, <4 x float> %257) #9, !dbg !16
  %260 = fadd <4 x float> %154, %259, !dbg !20
  %261 = load <8 x half>, ptr addrspace(3) %address.local1273, align 2, !dbg !16
  %262 = load <8 x half>, ptr addrspace(3) %address.local1286, align 2, !dbg !16
  %263 = load <8 x half>, ptr addrspace(3) %address.local1324, align 2, !dbg !16
  %264 = shufflevector <8 x half> %261, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %265 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %250, <4 x half> %264, <4 x float> zeroinitializer) #9, !dbg !16
  %266 = shufflevector <8 x half> %261, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %267 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %252, <4 x half> %266, <4 x float> %265) #9, !dbg !16
  %268 = shufflevector <8 x half> %262, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %269 = shufflevector <8 x half> %262, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %270 = fadd <4 x float> %164, %267, !dbg !20
  %271 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %250, <4 x half> %268, <4 x float> zeroinitializer) #9, !dbg !16
  %272 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %252, <4 x half> %269, <4 x float> %271) #9, !dbg !16
  %273 = shufflevector <8 x half> %263, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %274 = shufflevector <8 x half> %263, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %275 = fadd <4 x float> %169, %272, !dbg !20
  %276 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %250, <4 x half> %273, <4 x float> zeroinitializer) #9, !dbg !16
  %277 = load <8 x half>, ptr addrspace(3) %address.local1326, align 2, !dbg !16
  %278 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %252, <4 x half> %274, <4 x float> %276) #9, !dbg !16
  %279 = load <8 x half>, ptr addrspace(3) %address.local1368, align 2, !dbg !16
  %280 = load <8 x half>, ptr addrspace(3) %address.local1402, align 2, !dbg !16
  %281 = fadd <4 x float> %175, %278, !dbg !20
  %282 = shufflevector <8 x half> %277, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %283 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %250, <4 x half> %282, <4 x float> zeroinitializer) #9, !dbg !16
  %284 = shufflevector <8 x half> %277, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %285 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %252, <4 x half> %284, <4 x float> %283) #9, !dbg !16
  %286 = shufflevector <8 x half> %279, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %287 = shufflevector <8 x half> %279, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %288 = fadd <4 x float> %182, %285, !dbg !20
  %289 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %250, <4 x half> %286, <4 x float> zeroinitializer) #9, !dbg !16
  %290 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %252, <4 x half> %287, <4 x float> %289) #9, !dbg !16
  %291 = shufflevector <8 x half> %280, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %292 = shufflevector <8 x half> %280, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %293 = fadd <4 x float> %187, %290, !dbg !20
  %294 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %250, <4 x half> %291, <4 x float> zeroinitializer) #9, !dbg !16
  %295 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %252, <4 x half> %292, <4 x float> %294) #9, !dbg !16
  %296 = load <8 x half>, ptr addrspace(3) %address.local1444, align 2, !dbg !16
  %297 = fadd <4 x float> %191, %295, !dbg !20
  %298 = load <8 x half>, ptr addrspace(3) %address.local1234, align 2, !dbg !16
  %299 = shufflevector <8 x half> %296, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %300 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %250, <4 x half> %299, <4 x float> zeroinitializer) #9, !dbg !16
  %301 = shufflevector <8 x half> %296, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %302 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %252, <4 x half> %301, <4 x float> %300) #9, !dbg !16
  %303 = shufflevector <8 x half> %298, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %304 = shufflevector <8 x half> %298, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %305 = fadd <4 x float> %199, %302, !dbg !20
  %306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %253, <4 x half> %303, <4 x float> zeroinitializer) #9, !dbg !16
  %307 = load <8 x half>, ptr addrspace(3) %address.local1273, align 2, !dbg !16
  %308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %254, <4 x half> %304, <4 x float> %306) #9, !dbg !16
  %309 = load <8 x half>, ptr addrspace(3) %address.local1286, align 2, !dbg !16
  %310 = load <8 x half>, ptr addrspace(3) %address.local1324, align 2, !dbg !16
  %311 = fadd <4 x float> %205, %308, !dbg !20
  %312 = shufflevector <8 x half> %307, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %313 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %253, <4 x half> %312, <4 x float> zeroinitializer) #9, !dbg !16
  %314 = shufflevector <8 x half> %307, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %315 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %254, <4 x half> %314, <4 x float> %313) #9, !dbg !16
  %316 = shufflevector <8 x half> %309, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %317 = shufflevector <8 x half> %309, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %318 = fadd <4 x float> %212, %315, !dbg !20
  %319 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %253, <4 x half> %316, <4 x float> zeroinitializer) #9, !dbg !16
  %320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %254, <4 x half> %317, <4 x float> %319) #9, !dbg !16
  %321 = shufflevector <8 x half> %310, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %322 = shufflevector <8 x half> %310, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %323 = fadd <4 x float> %217, %320, !dbg !20
  %324 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %253, <4 x half> %321, <4 x float> zeroinitializer) #9, !dbg !16
  %325 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %254, <4 x half> %322, <4 x float> %324) #9, !dbg !16
  %326 = load <8 x half>, ptr addrspace(3) %address.local1326, align 2, !dbg !16
  %327 = fadd <4 x float> %221, %325, !dbg !20
  %328 = load <8 x half>, ptr addrspace(3) %address.local1368, align 2, !dbg !16
  %329 = shufflevector <8 x half> %326, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %253, <4 x half> %329, <4 x float> zeroinitializer) #9, !dbg !16
  %331 = shufflevector <8 x half> %326, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %254, <4 x half> %331, <4 x float> %330) #9, !dbg !16
  %333 = shufflevector <8 x half> %328, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %334 = shufflevector <8 x half> %328, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %335 = fadd <4 x float> %229, %332, !dbg !20
  %336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %253, <4 x half> %333, <4 x float> zeroinitializer) #9, !dbg !16
  %337 = load <8 x half>, ptr addrspace(3) %address.local1402, align 2, !dbg !16
  %338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %254, <4 x half> %334, <4 x float> %336) #9, !dbg !16
  %339 = load <8 x half>, ptr addrspace(3) %address.local1444, align 2, !dbg !16
  %340 = fadd <4 x float> %234, %338, !dbg !20
  %341 = shufflevector <8 x half> %337, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %342 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %253, <4 x half> %341, <4 x float> zeroinitializer) #9, !dbg !16
  %343 = shufflevector <8 x half> %337, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %344 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %254, <4 x half> %343, <4 x float> %342) #9, !dbg !16
  %345 = shufflevector <8 x half> %339, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %346 = shufflevector <8 x half> %339, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %347 = fadd <4 x float> %241, %344, !dbg !20
  %348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %253, <4 x half> %345, <4 x float> zeroinitializer) #9, !dbg !16
  %349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %254, <4 x half> %346, <4 x float> %348) #9, !dbg !16
  %350 = fadd <4 x float> %244, %349, !dbg !20
  %address.local1461 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local789, !dbg !11
  %dma.source.bits2005 = ptrtoint ptr addrspace(1) %address.local1461 to i64, !dbg !11
  %dma.source.opaque2006 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2005) #10, !dbg !11
  %dma.source.ptr2007 = inttoptr i64 %dma.source.opaque2006 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2007, ptr addrspace(3) %39, i32 4, i32 0, i32 0), !dbg !11
  %address.local1462 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local797, !dbg !11
  %dma.source.bits2008 = ptrtoint ptr addrspace(1) %address.local1462 to i64, !dbg !11
  %dma.source.opaque2009 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2008) #10, !dbg !11
  %dma.source.ptr2010 = inttoptr i64 %dma.source.opaque2009 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2010, ptr addrspace(3) %42, i32 4, i32 0, i32 0), !dbg !11
  %address.local1463 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local805, !dbg !11
  %dma.source.bits2011 = ptrtoint ptr addrspace(1) %address.local1463 to i64, !dbg !11
  %dma.source.opaque2012 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2011) #10, !dbg !11
  %dma.source.ptr2013 = inttoptr i64 %dma.source.opaque2012 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2013, ptr addrspace(3) %45, i32 4, i32 0, i32 0), !dbg !11
  %address.local1464 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local813, !dbg !11
  %dma.source.bits2014 = ptrtoint ptr addrspace(1) %address.local1464 to i64, !dbg !11
  %dma.source.opaque2015 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2014) #10, !dbg !11
  %dma.source.ptr2016 = inttoptr i64 %dma.source.opaque2015 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2016, ptr addrspace(3) %48, i32 4, i32 0, i32 0), !dbg !11
  %address.local1465 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local821, !dbg !11
  %dma.source.bits2017 = ptrtoint ptr addrspace(1) %address.local1465 to i64, !dbg !11
  %dma.source.opaque2018 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2017) #10, !dbg !11
  %dma.source.ptr2019 = inttoptr i64 %dma.source.opaque2018 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2019, ptr addrspace(3) %51, i32 4, i32 0, i32 0), !dbg !11
  %address.local1466 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local829, !dbg !11
  %dma.source.bits2020 = ptrtoint ptr addrspace(1) %address.local1466 to i64, !dbg !11
  %dma.source.opaque2021 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2020) #10, !dbg !11
  %dma.source.ptr2022 = inttoptr i64 %dma.source.opaque2021 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2022, ptr addrspace(3) %54, i32 4, i32 0, i32 0), !dbg !11
  %address.local1467 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local837, !dbg !11
  %dma.source.bits2023 = ptrtoint ptr addrspace(1) %address.local1467 to i64, !dbg !11
  %dma.source.opaque2024 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2023) #10, !dbg !11
  %dma.source.ptr2025 = inttoptr i64 %dma.source.opaque2024 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2025, ptr addrspace(3) %57, i32 4, i32 0, i32 0), !dbg !11
  %address.local1468 = getelementptr [2 x i8], ptr addrspace(1) %123, i64 %address.local845, !dbg !11
  %dma.source.bits2026 = ptrtoint ptr addrspace(1) %address.local1468 to i64, !dbg !11
  %dma.source.opaque2027 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2026) #10, !dbg !11
  %dma.source.ptr2028 = inttoptr i64 %dma.source.opaque2027 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2028, ptr addrspace(3) %60, i32 4, i32 0, i32 0), !dbg !11
  %351 = or disjoint i64 %139, 64, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %352 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %351, !dbg !12
  %address.local1469 = getelementptr [2 x i8], ptr addrspace(1) %352, i64 %address.local881, !dbg !12
  %dma.source.bits2029 = ptrtoint ptr addrspace(1) %address.local1469 to i64, !dbg !12
  %dma.source.opaque2030 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2029) #10, !dbg !12
  %dma.source.ptr2031 = inttoptr i64 %dma.source.opaque2030 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2031, ptr addrspace(3) %63, i32 4, i32 0, i32 0), !dbg !12
  %address.local1470 = getelementptr [2 x i8], ptr addrspace(1) %352, i64 %address.local920, !dbg !12
  %dma.source.bits2032 = ptrtoint ptr addrspace(1) %address.local1470 to i64, !dbg !12
  %dma.source.opaque2033 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2032) #10, !dbg !12
  %dma.source.ptr2034 = inttoptr i64 %dma.source.opaque2033 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2034, ptr addrspace(3) %66, i32 4, i32 0, i32 0), !dbg !12
  %address.local1471 = getelementptr [2 x i8], ptr addrspace(1) %352, i64 %address.local957, !dbg !12
  %dma.source.bits2035 = ptrtoint ptr addrspace(1) %address.local1471 to i64, !dbg !12
  %dma.source.opaque2036 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2035) #10, !dbg !12
  %dma.source.ptr2037 = inttoptr i64 %dma.source.opaque2036 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2037, ptr addrspace(3) %69, i32 4, i32 0, i32 0), !dbg !12
  %address.local1472 = getelementptr [2 x i8], ptr addrspace(1) %352, i64 %address.local997, !dbg !12
  %dma.source.bits2038 = ptrtoint ptr addrspace(1) %address.local1472 to i64, !dbg !12
  %dma.source.opaque2039 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2038) #10, !dbg !12
  %dma.source.ptr2040 = inttoptr i64 %dma.source.opaque2039 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2040, ptr addrspace(3) %72, i32 4, i32 0, i32 0), !dbg !12
  %address.local1473 = getelementptr [2 x i8], ptr addrspace(1) %352, i64 %address.local1040, !dbg !12
  %dma.source.bits2041 = ptrtoint ptr addrspace(1) %address.local1473 to i64, !dbg !12
  %dma.source.opaque2042 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2041) #10, !dbg !12
  %dma.source.ptr2043 = inttoptr i64 %dma.source.opaque2042 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2043, ptr addrspace(3) %75, i32 4, i32 0, i32 0), !dbg !12
  %address.local1474 = getelementptr [2 x i8], ptr addrspace(1) %352, i64 %address.local1082, !dbg !12
  %dma.source.bits2044 = ptrtoint ptr addrspace(1) %address.local1474 to i64, !dbg !12
  %dma.source.opaque2045 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2044) #10, !dbg !12
  %dma.source.ptr2046 = inttoptr i64 %dma.source.opaque2045 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2046, ptr addrspace(3) %78, i32 4, i32 0, i32 0), !dbg !12
  %address.local1475 = getelementptr [2 x i8], ptr addrspace(1) %352, i64 %address.local1125, !dbg !12
  %dma.source.bits2047 = ptrtoint ptr addrspace(1) %address.local1475 to i64, !dbg !12
  %dma.source.opaque2048 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2047) #10, !dbg !12
  %dma.source.ptr2049 = inttoptr i64 %dma.source.opaque2048 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2049, ptr addrspace(3) %81, i32 4, i32 0, i32 0), !dbg !12
  %address.local1476 = getelementptr [2 x i8], ptr addrspace(1) %352, i64 %address.local1167, !dbg !12
  %dma.source.bits2050 = ptrtoint ptr addrspace(1) %address.local1476 to i64, !dbg !12
  %dma.source.opaque2051 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2050) #10, !dbg !12
  %dma.source.ptr2052 = inttoptr i64 %dma.source.opaque2051 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2052, ptr addrspace(3) %84, i32 4, i32 0, i32 0), !dbg !12
  %353 = load <4 x half>, ptr addrspace(3) %address.local1177, align 2, !dbg !13
  %354 = load <4 x half>, ptr addrspace(3) %address.local1187, align 2, !dbg !13
  %355 = load <4 x half>, ptr addrspace(3) %address.local1197, align 2, !dbg !13
  %356 = fmul <4 x half> %353, splat (half 0xH3015), !dbg !13
  %357 = load <4 x half>, ptr addrspace(3) %address.local1207, align 2, !dbg !13
  %358 = fmul <4 x half> %354, splat (half 0xH3015), !dbg !13
  %359 = fmul <4 x half> %355, splat (half 0xH3015), !dbg !13
  %360 = fmul <4 x half> %357, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %361 = load <8 x half>, ptr addrspace(3) %address.local1234, align 2, !dbg !16
  %362 = shufflevector <8 x half> %361, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %356, <4 x half> %362, <4 x float> zeroinitializer) #9, !dbg !16
  %364 = shufflevector <8 x half> %361, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %364, <4 x float> %363) #9, !dbg !16
  %366 = fadd <4 x float> %260, %365, !dbg !20
  %367 = load <8 x half>, ptr addrspace(3) %address.local1273, align 2, !dbg !16
  %368 = load <8 x half>, ptr addrspace(3) %address.local1286, align 2, !dbg !16
  %369 = load <8 x half>, ptr addrspace(3) %address.local1324, align 2, !dbg !16
  %370 = shufflevector <8 x half> %367, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %371 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %356, <4 x half> %370, <4 x float> zeroinitializer) #9, !dbg !16
  %372 = shufflevector <8 x half> %367, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %373 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %372, <4 x float> %371) #9, !dbg !16
  %374 = shufflevector <8 x half> %368, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %375 = shufflevector <8 x half> %368, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %376 = fadd <4 x float> %270, %373, !dbg !20
  %377 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %356, <4 x half> %374, <4 x float> zeroinitializer) #9, !dbg !16
  %378 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %375, <4 x float> %377) #9, !dbg !16
  %379 = shufflevector <8 x half> %369, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %380 = shufflevector <8 x half> %369, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %381 = fadd <4 x float> %275, %378, !dbg !20
  %382 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %356, <4 x half> %379, <4 x float> zeroinitializer) #9, !dbg !16
  %383 = load <8 x half>, ptr addrspace(3) %address.local1326, align 2, !dbg !16
  %384 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %380, <4 x float> %382) #9, !dbg !16
  %385 = load <8 x half>, ptr addrspace(3) %address.local1368, align 2, !dbg !16
  %386 = load <8 x half>, ptr addrspace(3) %address.local1402, align 2, !dbg !16
  %387 = fadd <4 x float> %281, %384, !dbg !20
  %388 = shufflevector <8 x half> %383, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %389 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %356, <4 x half> %388, <4 x float> zeroinitializer) #9, !dbg !16
  %390 = shufflevector <8 x half> %383, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %391 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %390, <4 x float> %389) #9, !dbg !16
  %392 = shufflevector <8 x half> %385, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %393 = shufflevector <8 x half> %385, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %394 = fadd <4 x float> %288, %391, !dbg !20
  %395 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %356, <4 x half> %392, <4 x float> zeroinitializer) #9, !dbg !16
  %396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %393, <4 x float> %395) #9, !dbg !16
  %397 = shufflevector <8 x half> %386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %398 = shufflevector <8 x half> %386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %399 = fadd <4 x float> %293, %396, !dbg !20
  %400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %356, <4 x half> %397, <4 x float> zeroinitializer) #9, !dbg !16
  %401 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %398, <4 x float> %400) #9, !dbg !16
  %402 = load <8 x half>, ptr addrspace(3) %address.local1444, align 2, !dbg !16
  %403 = fadd <4 x float> %297, %401, !dbg !20
  %404 = load <8 x half>, ptr addrspace(3) %address.local1234, align 2, !dbg !16
  %405 = shufflevector <8 x half> %402, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %406 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %356, <4 x half> %405, <4 x float> zeroinitializer) #9, !dbg !16
  %407 = shufflevector <8 x half> %402, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %358, <4 x half> %407, <4 x float> %406) #9, !dbg !16
  %409 = shufflevector <8 x half> %404, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %410 = shufflevector <8 x half> %404, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %411 = fadd <4 x float> %305, %408, !dbg !20
  %412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %409, <4 x float> zeroinitializer) #9, !dbg !16
  %413 = load <8 x half>, ptr addrspace(3) %address.local1273, align 2, !dbg !16
  %414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %360, <4 x half> %410, <4 x float> %412) #9, !dbg !16
  %415 = load <8 x half>, ptr addrspace(3) %address.local1286, align 2, !dbg !16
  %416 = load <8 x half>, ptr addrspace(3) %address.local1324, align 2, !dbg !16
  %417 = fadd <4 x float> %311, %414, !dbg !20
  %418 = shufflevector <8 x half> %413, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %419 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %418, <4 x float> zeroinitializer) #9, !dbg !16
  %420 = shufflevector <8 x half> %413, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %421 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %360, <4 x half> %420, <4 x float> %419) #9, !dbg !16
  %422 = shufflevector <8 x half> %415, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %423 = shufflevector <8 x half> %415, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %424 = fadd <4 x float> %318, %421, !dbg !20
  %425 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %422, <4 x float> zeroinitializer) #9, !dbg !16
  %426 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %360, <4 x half> %423, <4 x float> %425) #9, !dbg !16
  %427 = shufflevector <8 x half> %416, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %428 = shufflevector <8 x half> %416, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %429 = fadd <4 x float> %323, %426, !dbg !20
  %430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %427, <4 x float> zeroinitializer) #9, !dbg !16
  %431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %360, <4 x half> %428, <4 x float> %430) #9, !dbg !16
  %432 = load <8 x half>, ptr addrspace(3) %address.local1326, align 2, !dbg !16
  %433 = fadd <4 x float> %327, %431, !dbg !20
  %434 = load <8 x half>, ptr addrspace(3) %address.local1368, align 2, !dbg !16
  %435 = shufflevector <8 x half> %432, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %436 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %435, <4 x float> zeroinitializer) #9, !dbg !16
  %437 = shufflevector <8 x half> %432, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %438 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %360, <4 x half> %437, <4 x float> %436) #9, !dbg !16
  %439 = shufflevector <8 x half> %434, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %440 = shufflevector <8 x half> %434, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %441 = fadd <4 x float> %335, %438, !dbg !20
  %442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %439, <4 x float> zeroinitializer) #9, !dbg !16
  %443 = load <8 x half>, ptr addrspace(3) %address.local1402, align 2, !dbg !16
  %444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %360, <4 x half> %440, <4 x float> %442) #9, !dbg !16
  %445 = load <8 x half>, ptr addrspace(3) %address.local1444, align 2, !dbg !16
  %446 = fadd <4 x float> %340, %444, !dbg !20
  %447 = shufflevector <8 x half> %443, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %448 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %447, <4 x float> zeroinitializer) #9, !dbg !16
  %449 = shufflevector <8 x half> %443, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %450 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %360, <4 x half> %449, <4 x float> %448) #9, !dbg !16
  %451 = shufflevector <8 x half> %445, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %452 = shufflevector <8 x half> %445, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %453 = fadd <4 x float> %347, %450, !dbg !20
  %454 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %359, <4 x half> %451, <4 x float> zeroinitializer) #9, !dbg !16
  %455 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %360, <4 x half> %452, <4 x float> %454) #9, !dbg !16
  %456 = fadd <4 x float> %350, %455, !dbg !20
  %address.local1477 = getelementptr [2 x i8], ptr addrspace(1) %125, i64 %address.local789, !dbg !11
  %dma.source.bits2053 = ptrtoint ptr addrspace(1) %address.local1477 to i64, !dbg !11
  %dma.source.opaque2054 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2053) #10, !dbg !11
  %dma.source.ptr2055 = inttoptr i64 %dma.source.opaque2054 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2055, ptr addrspace(3) %39, i32 4, i32 0, i32 0), !dbg !11
  %address.local1478 = getelementptr [2 x i8], ptr addrspace(1) %125, i64 %address.local797, !dbg !11
  %dma.source.bits2056 = ptrtoint ptr addrspace(1) %address.local1478 to i64, !dbg !11
  %dma.source.opaque2057 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2056) #10, !dbg !11
  %dma.source.ptr2058 = inttoptr i64 %dma.source.opaque2057 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2058, ptr addrspace(3) %42, i32 4, i32 0, i32 0), !dbg !11
  %address.local1479 = getelementptr [2 x i8], ptr addrspace(1) %125, i64 %address.local805, !dbg !11
  %dma.source.bits2059 = ptrtoint ptr addrspace(1) %address.local1479 to i64, !dbg !11
  %dma.source.opaque2060 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2059) #10, !dbg !11
  %dma.source.ptr2061 = inttoptr i64 %dma.source.opaque2060 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2061, ptr addrspace(3) %45, i32 4, i32 0, i32 0), !dbg !11
  %address.local1480 = getelementptr [2 x i8], ptr addrspace(1) %125, i64 %address.local813, !dbg !11
  %dma.source.bits2062 = ptrtoint ptr addrspace(1) %address.local1480 to i64, !dbg !11
  %dma.source.opaque2063 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2062) #10, !dbg !11
  %dma.source.ptr2064 = inttoptr i64 %dma.source.opaque2063 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2064, ptr addrspace(3) %48, i32 4, i32 0, i32 0), !dbg !11
  %address.local1481 = getelementptr [2 x i8], ptr addrspace(1) %125, i64 %address.local821, !dbg !11
  %dma.source.bits2065 = ptrtoint ptr addrspace(1) %address.local1481 to i64, !dbg !11
  %dma.source.opaque2066 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2065) #10, !dbg !11
  %dma.source.ptr2067 = inttoptr i64 %dma.source.opaque2066 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2067, ptr addrspace(3) %51, i32 4, i32 0, i32 0), !dbg !11
  %address.local1482 = getelementptr [2 x i8], ptr addrspace(1) %125, i64 %address.local829, !dbg !11
  %dma.source.bits2068 = ptrtoint ptr addrspace(1) %address.local1482 to i64, !dbg !11
  %dma.source.opaque2069 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2068) #10, !dbg !11
  %dma.source.ptr2070 = inttoptr i64 %dma.source.opaque2069 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2070, ptr addrspace(3) %54, i32 4, i32 0, i32 0), !dbg !11
  %address.local1483 = getelementptr [2 x i8], ptr addrspace(1) %125, i64 %address.local837, !dbg !11
  %dma.source.bits2071 = ptrtoint ptr addrspace(1) %address.local1483 to i64, !dbg !11
  %dma.source.opaque2072 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2071) #10, !dbg !11
  %dma.source.ptr2073 = inttoptr i64 %dma.source.opaque2072 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2073, ptr addrspace(3) %57, i32 4, i32 0, i32 0), !dbg !11
  %address.local1484 = getelementptr [2 x i8], ptr addrspace(1) %125, i64 %address.local845, !dbg !11
  %dma.source.bits2074 = ptrtoint ptr addrspace(1) %address.local1484 to i64, !dbg !11
  %dma.source.opaque2075 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2074) #10, !dbg !11
  %dma.source.ptr2076 = inttoptr i64 %dma.source.opaque2075 to ptr addrspace(1), !dbg !11
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2076, ptr addrspace(3) %60, i32 4, i32 0, i32 0), !dbg !11
  %457 = or disjoint i64 %139, 96, !dbg !12
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !11
  fence syncscope("workgroup") release, !dbg !11
  call void @llvm.amdgcn.s.barrier(), !dbg !11
  fence syncscope("workgroup") acquire, !dbg !11
  %458 = getelementptr [2 x i8], ptr addrspace(1) %1, i64 %457, !dbg !12
  %address.local1485 = getelementptr [2 x i8], ptr addrspace(1) %458, i64 %address.local881, !dbg !12
  %dma.source.bits2077 = ptrtoint ptr addrspace(1) %address.local1485 to i64, !dbg !12
  %dma.source.opaque2078 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2077) #10, !dbg !12
  %dma.source.ptr2079 = inttoptr i64 %dma.source.opaque2078 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2079, ptr addrspace(3) %63, i32 4, i32 0, i32 0), !dbg !12
  %address.local1486 = getelementptr [2 x i8], ptr addrspace(1) %458, i64 %address.local920, !dbg !12
  %dma.source.bits2080 = ptrtoint ptr addrspace(1) %address.local1486 to i64, !dbg !12
  %dma.source.opaque2081 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2080) #10, !dbg !12
  %dma.source.ptr2082 = inttoptr i64 %dma.source.opaque2081 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2082, ptr addrspace(3) %66, i32 4, i32 0, i32 0), !dbg !12
  %address.local1487 = getelementptr [2 x i8], ptr addrspace(1) %458, i64 %address.local957, !dbg !12
  %dma.source.bits2083 = ptrtoint ptr addrspace(1) %address.local1487 to i64, !dbg !12
  %dma.source.opaque2084 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2083) #10, !dbg !12
  %dma.source.ptr2085 = inttoptr i64 %dma.source.opaque2084 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2085, ptr addrspace(3) %69, i32 4, i32 0, i32 0), !dbg !12
  %address.local1488 = getelementptr [2 x i8], ptr addrspace(1) %458, i64 %address.local997, !dbg !12
  %dma.source.bits2086 = ptrtoint ptr addrspace(1) %address.local1488 to i64, !dbg !12
  %dma.source.opaque2087 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2086) #10, !dbg !12
  %dma.source.ptr2088 = inttoptr i64 %dma.source.opaque2087 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2088, ptr addrspace(3) %72, i32 4, i32 0, i32 0), !dbg !12
  %address.local1489 = getelementptr [2 x i8], ptr addrspace(1) %458, i64 %address.local1040, !dbg !12
  %dma.source.bits2089 = ptrtoint ptr addrspace(1) %address.local1489 to i64, !dbg !12
  %dma.source.opaque2090 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2089) #10, !dbg !12
  %dma.source.ptr2091 = inttoptr i64 %dma.source.opaque2090 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2091, ptr addrspace(3) %75, i32 4, i32 0, i32 0), !dbg !12
  %address.local1490 = getelementptr [2 x i8], ptr addrspace(1) %458, i64 %address.local1082, !dbg !12
  %dma.source.bits2092 = ptrtoint ptr addrspace(1) %address.local1490 to i64, !dbg !12
  %dma.source.opaque2093 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2092) #10, !dbg !12
  %dma.source.ptr2094 = inttoptr i64 %dma.source.opaque2093 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2094, ptr addrspace(3) %78, i32 4, i32 0, i32 0), !dbg !12
  %address.local1491 = getelementptr [2 x i8], ptr addrspace(1) %458, i64 %address.local1125, !dbg !12
  %dma.source.bits2095 = ptrtoint ptr addrspace(1) %address.local1491 to i64, !dbg !12
  %dma.source.opaque2096 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2095) #10, !dbg !12
  %dma.source.ptr2097 = inttoptr i64 %dma.source.opaque2096 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2097, ptr addrspace(3) %81, i32 4, i32 0, i32 0), !dbg !12
  %address.local1492 = getelementptr [2 x i8], ptr addrspace(1) %458, i64 %address.local1167, !dbg !12
  %dma.source.bits2098 = ptrtoint ptr addrspace(1) %address.local1492 to i64, !dbg !12
  %dma.source.opaque2099 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits2098) #10, !dbg !12
  %dma.source.ptr2100 = inttoptr i64 %dma.source.opaque2099 to ptr addrspace(1), !dbg !12
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr2100, ptr addrspace(3) %84, i32 4, i32 0, i32 0), !dbg !12
  %459 = load <4 x half>, ptr addrspace(3) %address.local1177, align 2, !dbg !13
  %460 = load <4 x half>, ptr addrspace(3) %address.local1187, align 2, !dbg !13
  %461 = load <4 x half>, ptr addrspace(3) %address.local1197, align 2, !dbg !13
  %462 = fmul <4 x half> %459, splat (half 0xH3015), !dbg !13
  %463 = load <4 x half>, ptr addrspace(3) %address.local1207, align 2, !dbg !13
  %464 = fmul <4 x half> %460, splat (half 0xH3015), !dbg !13
  %465 = fmul <4 x half> %461, splat (half 0xH3015), !dbg !13
  %466 = fmul <4 x half> %463, splat (half 0xH3015), !dbg !13
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !12
  fence syncscope("workgroup") release, !dbg !12
  call void @llvm.amdgcn.s.barrier(), !dbg !12
  fence syncscope("workgroup") acquire, !dbg !12
  %467 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %139, !dbg !15
  %address.local1493 = shl nuw nsw i32 %address.thread783, 3, !dbg !15
  %address.local1494 = and i32 %address.local1493, 24, !dbg !15
  %address.local1495 = shl nuw nsw i32 %address.thread783, 5, !dbg !15
  %address.local1496 = and i32 %address.local1495, 8064, !dbg !15
  %address.local1497 = or disjoint i32 %address.local1494, %address.local1496, !dbg !15
  %address.local1498 = zext nneg i32 %address.local1497 to i64, !dbg !15
  %address.local1499 = getelementptr [2 x i8], ptr addrspace(1) %467, i64 %address.local1498, !dbg !15
  %468 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %469 = ptrtoint ptr addrspace(1) %address.local1499 to i64, !dbg !15
  %470 = sub i64 %469, %468, !dbg !15
  %dma.buffer.offset1910 = trunc i64 %470 to i32, !dbg !15
  %471 = trunc i64 %468 to i32, !dbg !15
  %472 = insertelement <4 x i32> poison, i32 %471, i64 0, !dbg !15
  %473 = lshr i64 %468, 32, !dbg !15
  %474 = trunc i64 %473 to i32, !dbg !15
  %475 = insertelement <4 x i32> %472, i32 %474, i64 1, !dbg !15
  %476 = insertelement <4 x i32> %475, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1911 = insertelement <4 x i32> %476, i32 131072, i64 3, !dbg !15
  %477 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %85, i32 %dma.buffer.offset1910, <4 x i32> %dma.buffer.rsrc1911) #11, !dbg !15
  %address.local1500 = shl nuw nsw i32 %address.thread783, 3, !dbg !15
  %address.local1501 = and i32 %address.local1500, 24, !dbg !15
  %address.local1502 = shl nuw nsw i32 %address.thread783, 5, !dbg !15
  %address.local1503 = and i32 %address.local1502, 8064, !dbg !15
  %address.local1504 = or disjoint i32 %address.local1501, %address.local1503, !dbg !15
  %address.local1505 = or disjoint i32 %address.local1504, 8192, !dbg !15
  %address.local1506 = zext nneg i32 %address.local1505 to i64, !dbg !15
  %address.local1507 = getelementptr [2 x i8], ptr addrspace(1) %467, i64 %address.local1506, !dbg !15
  %478 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %479 = ptrtoint ptr addrspace(1) %address.local1507 to i64, !dbg !15
  %480 = sub i64 %479, %478, !dbg !15
  %dma.buffer.offset1908 = trunc i64 %480 to i32, !dbg !15
  %481 = trunc i64 %478 to i32, !dbg !15
  %482 = insertelement <4 x i32> poison, i32 %481, i64 0, !dbg !15
  %483 = lshr i64 %478, 32, !dbg !15
  %484 = trunc i64 %483 to i32, !dbg !15
  %485 = insertelement <4 x i32> %482, i32 %484, i64 1, !dbg !15
  %486 = insertelement <4 x i32> %485, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1909 = insertelement <4 x i32> %486, i32 131072, i64 3, !dbg !15
  %487 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %88, i32 %dma.buffer.offset1908, <4 x i32> %dma.buffer.rsrc1909) #11, !dbg !15
  %488 = load <8 x half>, ptr addrspace(3) %address.local1234, align 2, !dbg !16
  %489 = shufflevector <8 x half> %488, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %490 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %462, <4 x half> %489, <4 x float> zeroinitializer) #9, !dbg !16
  %491 = shufflevector <8 x half> %488, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %492 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %464, <4 x half> %491, <4 x float> %490) #9, !dbg !16
  %493 = fadd <4 x float> %366, %492, !dbg !20
  %494 = extractelement <4 x float> %493, i64 0, !dbg !20
  %495 = call float @llvm.maxnum.f32(float %494, float 0xFFF0000000000000), !dbg !8
  %496 = extractelement <4 x float> %493, i64 1, !dbg !20
  %497 = call float @llvm.maxnum.f32(float %495, float %496), !dbg !8
  %498 = extractelement <4 x float> %493, i64 2, !dbg !20
  %499 = call float @llvm.maxnum.f32(float %497, float %498), !dbg !8
  %500 = extractelement <4 x float> %493, i64 3, !dbg !20
  %501 = call float @llvm.maxnum.f32(float %499, float %500), !dbg !8
  %502 = load <8 x half>, ptr addrspace(3) %address.local1273, align 2, !dbg !16
  %503 = load <8 x half>, ptr addrspace(3) %address.local1286, align 2, !dbg !16
  %504 = load <8 x half>, ptr addrspace(3) %address.local1324, align 2, !dbg !16
  %505 = shufflevector <8 x half> %502, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %506 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %462, <4 x half> %505, <4 x float> zeroinitializer) #9, !dbg !16
  %507 = shufflevector <8 x half> %502, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %508 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %464, <4 x half> %507, <4 x float> %506) #9, !dbg !16
  %509 = shufflevector <8 x half> %503, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %510 = shufflevector <8 x half> %503, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %511 = fadd <4 x float> %376, %508, !dbg !20
  %512 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %462, <4 x half> %509, <4 x float> zeroinitializer) #9, !dbg !16
  %513 = extractelement <4 x float> %511, i64 0, !dbg !20
  %514 = call float @llvm.maxnum.f32(float %501, float %513), !dbg !8
  %515 = extractelement <4 x float> %511, i64 1, !dbg !20
  %516 = call float @llvm.maxnum.f32(float %514, float %515), !dbg !8
  %517 = extractelement <4 x float> %511, i64 2, !dbg !20
  %518 = call float @llvm.maxnum.f32(float %516, float %517), !dbg !8
  %519 = extractelement <4 x float> %511, i64 3, !dbg !20
  %520 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %464, <4 x half> %510, <4 x float> %512) #9, !dbg !16
  %521 = call float @llvm.maxnum.f32(float %518, float %519), !dbg !8
  %522 = fadd <4 x float> %381, %520, !dbg !20
  %523 = extractelement <4 x float> %522, i64 0, !dbg !20
  %524 = call float @llvm.maxnum.f32(float %521, float %523), !dbg !8
  %525 = extractelement <4 x float> %522, i64 1, !dbg !20
  %526 = call float @llvm.maxnum.f32(float %524, float %525), !dbg !8
  %527 = extractelement <4 x float> %522, i64 2, !dbg !20
  %528 = call float @llvm.maxnum.f32(float %526, float %527), !dbg !8
  %529 = extractelement <4 x float> %522, i64 3, !dbg !20
  %530 = call float @llvm.maxnum.f32(float %528, float %529), !dbg !8
  %531 = shufflevector <8 x half> %504, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %532 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %462, <4 x half> %531, <4 x float> zeroinitializer) #9, !dbg !16
  %533 = shufflevector <8 x half> %504, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %534 = load <8 x half>, ptr addrspace(3) %address.local1326, align 2, !dbg !16
  %535 = load <8 x half>, ptr addrspace(3) %address.local1368, align 2, !dbg !16
  %536 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %464, <4 x half> %533, <4 x float> %532) #9, !dbg !16
  %537 = load <8 x half>, ptr addrspace(3) %address.local1402, align 2, !dbg !16
  %538 = fadd <4 x float> %387, %536, !dbg !20
  %539 = extractelement <4 x float> %538, i64 0, !dbg !20
  %540 = call float @llvm.maxnum.f32(float %530, float %539), !dbg !8
  %541 = extractelement <4 x float> %538, i64 1, !dbg !20
  %542 = call float @llvm.maxnum.f32(float %540, float %541), !dbg !8
  %543 = extractelement <4 x float> %538, i64 2, !dbg !20
  %544 = call float @llvm.maxnum.f32(float %542, float %543), !dbg !8
  %545 = extractelement <4 x float> %538, i64 3, !dbg !20
  %546 = call float @llvm.maxnum.f32(float %544, float %545), !dbg !8
  %547 = shufflevector <8 x half> %534, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %548 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %462, <4 x half> %547, <4 x float> zeroinitializer) #9, !dbg !16
  %549 = shufflevector <8 x half> %534, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %550 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %464, <4 x half> %549, <4 x float> %548) #9, !dbg !16
  %551 = shufflevector <8 x half> %535, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %552 = shufflevector <8 x half> %535, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %553 = fadd <4 x float> %394, %550, !dbg !20
  %554 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %462, <4 x half> %551, <4 x float> zeroinitializer) #9, !dbg !16
  %555 = extractelement <4 x float> %553, i64 0, !dbg !20
  %556 = call float @llvm.maxnum.f32(float %546, float %555), !dbg !8
  %557 = extractelement <4 x float> %553, i64 1, !dbg !20
  %558 = call float @llvm.maxnum.f32(float %556, float %557), !dbg !8
  %559 = extractelement <4 x float> %553, i64 2, !dbg !20
  %560 = call float @llvm.maxnum.f32(float %558, float %559), !dbg !8
  %561 = extractelement <4 x float> %553, i64 3, !dbg !20
  %562 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %464, <4 x half> %552, <4 x float> %554) #9, !dbg !16
  %563 = call float @llvm.maxnum.f32(float %560, float %561), !dbg !8
  %564 = fadd <4 x float> %399, %562, !dbg !20
  %565 = extractelement <4 x float> %564, i64 0, !dbg !20
  %566 = call float @llvm.maxnum.f32(float %563, float %565), !dbg !8
  %567 = extractelement <4 x float> %564, i64 1, !dbg !20
  %568 = call float @llvm.maxnum.f32(float %566, float %567), !dbg !8
  %569 = extractelement <4 x float> %564, i64 2, !dbg !20
  %570 = call float @llvm.maxnum.f32(float %568, float %569), !dbg !8
  %571 = extractelement <4 x float> %564, i64 3, !dbg !20
  %572 = call float @llvm.maxnum.f32(float %570, float %571), !dbg !8
  %573 = shufflevector <8 x half> %537, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %574 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %462, <4 x half> %573, <4 x float> zeroinitializer) #9, !dbg !16
  %575 = shufflevector <8 x half> %537, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %576 = load <8 x half>, ptr addrspace(3) %address.local1444, align 2, !dbg !16
  %577 = load <8 x half>, ptr addrspace(3) %address.local1234, align 2, !dbg !16
  %578 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %464, <4 x half> %575, <4 x float> %574) #9, !dbg !16
  %579 = load <8 x half>, ptr addrspace(3) %address.local1273, align 2, !dbg !16
  %580 = fadd <4 x float> %403, %578, !dbg !20
  %581 = extractelement <4 x float> %580, i64 0, !dbg !20
  %582 = call float @llvm.maxnum.f32(float %572, float %581), !dbg !8
  %583 = extractelement <4 x float> %580, i64 1, !dbg !20
  %584 = call float @llvm.maxnum.f32(float %582, float %583), !dbg !8
  %585 = extractelement <4 x float> %580, i64 2, !dbg !20
  %586 = call float @llvm.maxnum.f32(float %584, float %585), !dbg !8
  %587 = extractelement <4 x float> %580, i64 3, !dbg !20
  %588 = call float @llvm.maxnum.f32(float %586, float %587), !dbg !8
  %589 = shufflevector <8 x half> %576, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %590 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %462, <4 x half> %589, <4 x float> zeroinitializer) #9, !dbg !16
  %591 = shufflevector <8 x half> %576, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %592 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %464, <4 x half> %591, <4 x float> %590) #9, !dbg !16
  %593 = shufflevector <8 x half> %577, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %594 = shufflevector <8 x half> %577, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %595 = fadd <4 x float> %411, %592, !dbg !20
  %596 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %465, <4 x half> %593, <4 x float> zeroinitializer) #9, !dbg !16
  %597 = extractelement <4 x float> %595, i64 0, !dbg !20
  %598 = call float @llvm.maxnum.f32(float %588, float %597), !dbg !8
  %599 = extractelement <4 x float> %595, i64 1, !dbg !20
  %600 = call float @llvm.maxnum.f32(float %598, float %599), !dbg !8
  %601 = extractelement <4 x float> %595, i64 2, !dbg !20
  %602 = call float @llvm.maxnum.f32(float %600, float %601), !dbg !8
  %603 = extractelement <4 x float> %595, i64 3, !dbg !20
  %604 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %466, <4 x half> %594, <4 x float> %596) #9, !dbg !16
  %605 = call float @llvm.maxnum.f32(float %602, float %603), !dbg !8
  %606 = fadd <4 x float> %417, %604, !dbg !20
  %607 = shufflevector <8 x half> %579, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %608 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %465, <4 x half> %607, <4 x float> zeroinitializer) #9, !dbg !16
  %609 = shufflevector <8 x half> %579, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %610 = extractelement <4 x float> %606, i64 0, !dbg !20
  %611 = extractelement <4 x float> %606, i64 1, !dbg !20
  %612 = extractelement <4 x float> %606, i64 2, !dbg !20
  %613 = extractelement <4 x float> %606, i64 3, !dbg !20
  %614 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %466, <4 x half> %609, <4 x float> %608) #9, !dbg !16
  %615 = load <8 x half>, ptr addrspace(3) %address.local1286, align 2, !dbg !16
  %616 = load <8 x half>, ptr addrspace(3) %address.local1324, align 2, !dbg !16
  %617 = load <8 x half>, ptr addrspace(3) %address.local1326, align 2, !dbg !16
  %618 = fadd <4 x float> %424, %614, !dbg !20
  %619 = shufflevector <8 x half> %615, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %620 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %465, <4 x half> %619, <4 x float> zeroinitializer) #9, !dbg !16
  %621 = shufflevector <8 x half> %615, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %622 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %466, <4 x half> %621, <4 x float> %620) #9, !dbg !16
  %623 = shufflevector <8 x half> %616, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %624 = shufflevector <8 x half> %616, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %625 = fadd <4 x float> %429, %622, !dbg !20
  %626 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %465, <4 x half> %623, <4 x float> zeroinitializer) #9, !dbg !16
  %627 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %466, <4 x half> %624, <4 x float> %626) #9, !dbg !16
  %628 = shufflevector <8 x half> %617, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %629 = shufflevector <8 x half> %617, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %630 = extractelement <4 x float> %618, i64 0, !dbg !20
  %631 = extractelement <4 x float> %618, i64 1, !dbg !20
  %632 = extractelement <4 x float> %618, i64 2, !dbg !20
  %633 = fadd <4 x float> %433, %627, !dbg !20
  %634 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %465, <4 x half> %628, <4 x float> zeroinitializer) #9, !dbg !16
  %635 = extractelement <4 x float> %618, i64 3, !dbg !20
  %636 = extractelement <4 x float> %625, i64 0, !dbg !20
  %637 = extractelement <4 x float> %625, i64 1, !dbg !20
  %638 = extractelement <4 x float> %625, i64 2, !dbg !20
  %639 = extractelement <4 x float> %625, i64 3, !dbg !20
  %640 = extractelement <4 x float> %633, i64 0, !dbg !20
  %641 = extractelement <4 x float> %633, i64 1, !dbg !20
  %642 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %466, <4 x half> %629, <4 x float> %634) #9, !dbg !16
  %643 = extractelement <4 x float> %633, i64 2, !dbg !20
  %644 = extractelement <4 x float> %633, i64 3, !dbg !20
  %645 = load <8 x half>, ptr addrspace(3) %address.local1368, align 2, !dbg !16
  %646 = fadd <4 x float> %441, %642, !dbg !20
  %647 = load <8 x half>, ptr addrspace(3) %address.local1402, align 2, !dbg !16
  %648 = extractelement <4 x float> %646, i64 0, !dbg !20
  %649 = extractelement <4 x float> %646, i64 1, !dbg !20
  %650 = extractelement <4 x float> %646, i64 2, !dbg !20
  %651 = shufflevector <8 x half> %645, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %652 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %465, <4 x half> %651, <4 x float> zeroinitializer) #9, !dbg !16
  %653 = shufflevector <8 x half> %645, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %654 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %466, <4 x half> %653, <4 x float> %652) #9, !dbg !16
  %655 = shufflevector <8 x half> %647, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %656 = shufflevector <8 x half> %647, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %657 = extractelement <4 x float> %646, i64 3, !dbg !20
  %658 = fadd <4 x float> %446, %654, !dbg !20
  %659 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %465, <4 x half> %655, <4 x float> zeroinitializer) #9, !dbg !16
  %660 = extractelement <4 x float> %658, i64 0, !dbg !20
  %661 = extractelement <4 x float> %658, i64 1, !dbg !20
  %662 = extractelement <4 x float> %658, i64 2, !dbg !20
  %663 = extractelement <4 x float> %658, i64 3, !dbg !20
  %664 = load <8 x half>, ptr addrspace(3) %address.local1444, align 2, !dbg !16
  %665 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %466, <4 x half> %656, <4 x float> %659) #9, !dbg !16
  %666 = fadd <4 x float> %453, %665, !dbg !20
  %667 = shufflevector <8 x half> %664, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !16
  %668 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %465, <4 x half> %667, <4 x float> zeroinitializer) #9, !dbg !16
  %669 = shufflevector <8 x half> %664, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !16
  %670 = extractelement <4 x float> %666, i64 0, !dbg !20
  %671 = extractelement <4 x float> %666, i64 1, !dbg !20
  %672 = extractelement <4 x float> %666, i64 2, !dbg !20
  %673 = extractelement <4 x float> %666, i64 3, !dbg !20
  %674 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %466, <4 x half> %669, <4 x float> %668) #9, !dbg !16
  %675 = fadd <4 x float> %456, %674, !dbg !20
  %676 = extractelement <4 x float> %675, i64 0, !dbg !20
  %677 = extractelement <4 x float> %675, i64 1, !dbg !20
  %678 = extractelement <4 x float> %675, i64 2, !dbg !20
  %679 = extractelement <4 x float> %675, i64 3, !dbg !20
  %680 = bitcast float %605 to i32, !dbg !8
  %681 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %680), !dbg !8
  %682 = bitcast i32 %681 to float, !dbg !8
  %683 = call float @llvm.maxnum.f32(float %605, float %682), !dbg !8
  %684 = bitcast float %683 to i32, !dbg !8
  %685 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %684), !dbg !8
  %686 = bitcast i32 %685 to float, !dbg !8
  %687 = call float @llvm.maxnum.f32(float %683, float %686), !dbg !8
  %688 = call float @llvm.maxnum.f32(float %610, float 0xFFF0000000000000), !dbg !8
  %689 = call float @llvm.maxnum.f32(float %688, float %611), !dbg !8
  %690 = call float @llvm.maxnum.f32(float %689, float %612), !dbg !8
  %691 = call float @llvm.maxnum.f32(float %690, float %613), !dbg !8
  %692 = call float @llvm.maxnum.f32(float %691, float %630), !dbg !8
  %693 = call float @llvm.maxnum.f32(float %692, float %631), !dbg !8
  %694 = call float @llvm.maxnum.f32(float %693, float %632), !dbg !8
  %695 = call float @llvm.maxnum.f32(float %694, float %635), !dbg !8
  %696 = call float @llvm.maxnum.f32(float %695, float %636), !dbg !8
  %697 = call float @llvm.maxnum.f32(float %696, float %637), !dbg !8
  %698 = call float @llvm.maxnum.f32(float %697, float %638), !dbg !8
  %699 = call float @llvm.maxnum.f32(float %698, float %639), !dbg !8
  %700 = call float @llvm.maxnum.f32(float %699, float %640), !dbg !8
  %701 = call float @llvm.maxnum.f32(float %700, float %641), !dbg !8
  %702 = call float @llvm.maxnum.f32(float %701, float %643), !dbg !8
  %703 = call float @llvm.maxnum.f32(float %702, float %644), !dbg !8
  %704 = call float @llvm.maxnum.f32(float %703, float %648), !dbg !8
  %705 = call float @llvm.maxnum.f32(float %704, float %649), !dbg !8
  %706 = call float @llvm.maxnum.f32(float %705, float %650), !dbg !8
  %707 = call float @llvm.maxnum.f32(float %706, float %657), !dbg !8
  %708 = call float @llvm.maxnum.f32(float %707, float %660), !dbg !8
  %709 = call float @llvm.maxnum.f32(float %708, float %661), !dbg !8
  %710 = call float @llvm.maxnum.f32(float %709, float %662), !dbg !8
  %711 = call float @llvm.maxnum.f32(float %710, float %663), !dbg !8
  %712 = call float @llvm.maxnum.f32(float %711, float %670), !dbg !8
  %713 = call float @llvm.maxnum.f32(float %712, float %671), !dbg !8
  %714 = call float @llvm.maxnum.f32(float %713, float %672), !dbg !8
  %715 = call float @llvm.maxnum.f32(float %714, float %673), !dbg !8
  %716 = call float @llvm.maxnum.f32(float %715, float %676), !dbg !8
  %717 = call float @llvm.maxnum.f32(float %716, float %677), !dbg !8
  %718 = call float @llvm.maxnum.f32(float %717, float %678), !dbg !8
  %719 = call float @llvm.maxnum.f32(float %718, float %679), !dbg !8
  %720 = bitcast float %719 to i32, !dbg !8
  %721 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %720), !dbg !8
  %722 = bitcast i32 %721 to float, !dbg !8
  %723 = call float @llvm.maxnum.f32(float %719, float %722), !dbg !8
  %724 = bitcast float %723 to i32, !dbg !8
  %725 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %724), !dbg !8
  %726 = bitcast i32 %725 to float, !dbg !8
  %727 = call float @llvm.maxnum.f32(float %723, float %726), !dbg !8
  %728 = extractvalue [2 x <1 x float>] %130, 0, !dbg !8
  %729 = insertelement <1 x float> poison, float %687, i64 0, !dbg !8
  %730 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %729, <1 x float> %728), !dbg !8
  %731 = extractvalue [2 x <1 x float>] %130, 1, !dbg !8
  %732 = insertelement <1 x float> poison, float %727, i64 0, !dbg !8
  %733 = call <1 x float> @llvm.maxnum.v1f32(<1 x float> %732, <1 x float> %731), !dbg !8
  %734 = shufflevector <1 x float> %730, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %735 = fsub <4 x float> %493, %734, !dbg !21
  %736 = fsub <4 x float> %511, %734, !dbg !21
  %737 = fsub <4 x float> %522, %734, !dbg !21
  %738 = fsub <4 x float> %538, %734, !dbg !21
  %739 = fsub <4 x float> %553, %734, !dbg !21
  %740 = fsub <4 x float> %564, %734, !dbg !21
  %741 = fsub <4 x float> %580, %734, !dbg !21
  %742 = fsub <4 x float> %595, %734, !dbg !21
  %743 = shufflevector <1 x float> %733, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !21
  %744 = fsub <4 x float> %606, %743, !dbg !21
  %745 = fsub <4 x float> %618, %743, !dbg !21
  %746 = fsub <4 x float> %625, %743, !dbg !21
  %747 = fsub <4 x float> %633, %743, !dbg !21
  %748 = fsub <4 x float> %646, %743, !dbg !21
  %749 = fsub <4 x float> %658, %743, !dbg !21
  %750 = fsub <4 x float> %666, %743, !dbg !21
  %751 = fsub <4 x float> %675, %743, !dbg !21
  %752 = fsub <1 x float> %728, %730, !dbg !22
  %753 = extractelement <1 x float> %752, i64 0, !dbg !23
  %754 = call float @__ocml_exp2_f32(float %753), !dbg !23
  %755 = insertelement <4 x float> poison, float %754, i64 0, !dbg !28
  %756 = shufflevector <4 x float> %755, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %757 = extractvalue [2 x <1 x float>] %129, 0, !dbg !24
  %758 = extractelement <1 x float> %757, i64 0, !dbg !24
  %759 = fmul float %758, %754, !dbg !24
  %760 = fsub <1 x float> %731, %733, !dbg !22
  %761 = extractelement <1 x float> %760, i64 0, !dbg !23
  %762 = call float @__ocml_exp2_f32(float %761), !dbg !23
  %763 = insertelement <4 x float> poison, float %762, i64 0, !dbg !28
  %764 = shufflevector <4 x float> %763, <4 x float> poison, <4 x i32> zeroinitializer, !dbg !28
  %765 = extractvalue [2 x <1 x float>] %129, 1, !dbg !24
  %766 = extractelement <1 x float> %765, i64 0, !dbg !24
  %767 = fmul float %766, %762, !dbg !24
  %768 = extractelement <4 x float> %735, i64 0, !dbg !25
  %769 = call float @__ocml_exp2_f32(float %768), !dbg !25
  %770 = insertelement <4 x float> poison, float %769, i64 0, !dbg !27
  %771 = fadd float %769, 0x0000000000000000, !dbg !26
  %772 = extractelement <4 x float> %735, i64 1, !dbg !25
  %773 = call float @__ocml_exp2_f32(float %772), !dbg !25
  %774 = insertelement <4 x float> %770, float %773, i64 1, !dbg !27
  %775 = fadd float %771, %773, !dbg !26
  %776 = extractelement <4 x float> %735, i64 2, !dbg !25
  %777 = call float @__ocml_exp2_f32(float %776), !dbg !25
  %778 = insertelement <4 x float> %774, float %777, i64 2, !dbg !27
  %779 = fadd float %775, %777, !dbg !26
  %780 = extractelement <4 x float> %735, i64 3, !dbg !25
  %781 = call float @__ocml_exp2_f32(float %780), !dbg !25
  %782 = insertelement <4 x float> %778, float %781, i64 3, !dbg !27
  %math.result = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %782) #9, !dbg !27, !frisk.register_lifetimes !1
  %783 = fptrunc <4 x float> %math.result to <4 x half>, !dbg !27
  %math.packed = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %783) #9, !dbg !26, !frisk.register_lifetimes !1
  %784 = fadd float %779, %781, !dbg !26
  %785 = extractelement <4 x float> %736, i64 0, !dbg !25
  %786 = call float @__ocml_exp2_f32(float %785), !dbg !25
  %787 = insertelement <4 x float> poison, float %786, i64 0, !dbg !27
  %788 = fadd float %784, %786, !dbg !26
  %789 = extractelement <4 x float> %736, i64 1, !dbg !25
  %790 = call float @__ocml_exp2_f32(float %789), !dbg !25
  %791 = insertelement <4 x float> %787, float %790, i64 1, !dbg !27
  %792 = fadd float %788, %790, !dbg !26
  %793 = extractelement <4 x float> %736, i64 2, !dbg !25
  %794 = call float @__ocml_exp2_f32(float %793), !dbg !25
  %795 = insertelement <4 x float> %791, float %794, i64 2, !dbg !27
  %796 = fadd float %792, %794, !dbg !26
  %797 = extractelement <4 x float> %736, i64 3, !dbg !25
  %798 = call float @__ocml_exp2_f32(float %797), !dbg !25
  %799 = insertelement <4 x float> %795, float %798, i64 3, !dbg !27
  %math.result674 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %799) #9, !dbg !27, !frisk.register_lifetimes !1
  %800 = fptrunc <4 x float> %math.result674 to <4 x half>, !dbg !27
  %math.packed675 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %800) #9, !dbg !26, !frisk.register_lifetimes !1
  %801 = fadd float %796, %798, !dbg !26
  %802 = extractelement <4 x float> %737, i64 0, !dbg !25
  %803 = call float @__ocml_exp2_f32(float %802), !dbg !25
  %804 = insertelement <4 x float> poison, float %803, i64 0, !dbg !27
  %805 = fadd float %801, %803, !dbg !26
  %806 = extractelement <4 x float> %737, i64 1, !dbg !25
  %807 = call float @__ocml_exp2_f32(float %806), !dbg !25
  %808 = insertelement <4 x float> %804, float %807, i64 1, !dbg !27
  %809 = fadd float %805, %807, !dbg !26
  %810 = extractelement <4 x float> %737, i64 2, !dbg !25
  %811 = call float @__ocml_exp2_f32(float %810), !dbg !25
  %812 = insertelement <4 x float> %808, float %811, i64 2, !dbg !27
  %813 = fadd float %809, %811, !dbg !26
  %814 = extractelement <4 x float> %737, i64 3, !dbg !25
  %815 = call float @__ocml_exp2_f32(float %814), !dbg !25
  %816 = insertelement <4 x float> %812, float %815, i64 3, !dbg !27
  %math.result676 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %816) #9, !dbg !27, !frisk.register_lifetimes !1
  %817 = fptrunc <4 x float> %math.result676 to <4 x half>, !dbg !27
  %math.packed677 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %817) #9, !dbg !26, !frisk.register_lifetimes !1
  %818 = fadd float %813, %815, !dbg !26
  %819 = extractelement <4 x float> %738, i64 0, !dbg !25
  %820 = call float @__ocml_exp2_f32(float %819), !dbg !25
  %821 = insertelement <4 x float> poison, float %820, i64 0, !dbg !27
  %822 = fadd float %818, %820, !dbg !26
  %823 = extractelement <4 x float> %738, i64 1, !dbg !25
  %824 = call float @__ocml_exp2_f32(float %823), !dbg !25
  %825 = insertelement <4 x float> %821, float %824, i64 1, !dbg !27
  %826 = fadd float %822, %824, !dbg !26
  %827 = extractelement <4 x float> %738, i64 2, !dbg !25
  %828 = call float @__ocml_exp2_f32(float %827), !dbg !25
  %829 = insertelement <4 x float> %825, float %828, i64 2, !dbg !27
  %830 = fadd float %826, %828, !dbg !26
  %831 = extractelement <4 x float> %738, i64 3, !dbg !25
  %832 = call float @__ocml_exp2_f32(float %831), !dbg !25
  %833 = insertelement <4 x float> %829, float %832, i64 3, !dbg !27
  %math.result678 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %833) #9, !dbg !27, !frisk.register_lifetimes !1
  %834 = fptrunc <4 x float> %math.result678 to <4 x half>, !dbg !27
  %math.packed679 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %834) #9, !dbg !26, !frisk.register_lifetimes !1
  %835 = fadd float %830, %832, !dbg !26
  %836 = extractelement <4 x float> %739, i64 0, !dbg !25
  %837 = call float @__ocml_exp2_f32(float %836), !dbg !25
  %838 = insertelement <4 x float> poison, float %837, i64 0, !dbg !27
  %839 = fadd float %835, %837, !dbg !26
  %840 = extractelement <4 x float> %739, i64 1, !dbg !25
  %841 = call float @__ocml_exp2_f32(float %840), !dbg !25
  %842 = insertelement <4 x float> %838, float %841, i64 1, !dbg !27
  %843 = fadd float %839, %841, !dbg !26
  %844 = extractelement <4 x float> %739, i64 2, !dbg !25
  %845 = call float @__ocml_exp2_f32(float %844), !dbg !25
  %846 = insertelement <4 x float> %842, float %845, i64 2, !dbg !27
  %847 = fadd float %843, %845, !dbg !26
  %848 = extractelement <4 x float> %739, i64 3, !dbg !25
  %849 = call float @__ocml_exp2_f32(float %848), !dbg !25
  %850 = insertelement <4 x float> %846, float %849, i64 3, !dbg !27
  %math.result680 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %850) #9, !dbg !27, !frisk.register_lifetimes !1
  %851 = fptrunc <4 x float> %math.result680 to <4 x half>, !dbg !27
  %math.packed681 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %851) #9, !dbg !26, !frisk.register_lifetimes !1
  %852 = fadd float %847, %849, !dbg !26
  %853 = extractelement <4 x float> %740, i64 0, !dbg !25
  %854 = call float @__ocml_exp2_f32(float %853), !dbg !25
  %855 = insertelement <4 x float> poison, float %854, i64 0, !dbg !27
  %856 = fadd float %852, %854, !dbg !26
  %857 = extractelement <4 x float> %740, i64 1, !dbg !25
  %858 = call float @__ocml_exp2_f32(float %857), !dbg !25
  %859 = insertelement <4 x float> %855, float %858, i64 1, !dbg !27
  %860 = fadd float %856, %858, !dbg !26
  %861 = extractelement <4 x float> %740, i64 2, !dbg !25
  %862 = call float @__ocml_exp2_f32(float %861), !dbg !25
  %863 = insertelement <4 x float> %859, float %862, i64 2, !dbg !27
  %864 = fadd float %860, %862, !dbg !26
  %865 = extractelement <4 x float> %740, i64 3, !dbg !25
  %866 = call float @__ocml_exp2_f32(float %865), !dbg !25
  %867 = insertelement <4 x float> %863, float %866, i64 3, !dbg !27
  %math.result682 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %867) #9, !dbg !27, !frisk.register_lifetimes !1
  %868 = fptrunc <4 x float> %math.result682 to <4 x half>, !dbg !27
  %math.packed683 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %868) #9, !dbg !26, !frisk.register_lifetimes !1
  %869 = fadd float %864, %866, !dbg !26
  %870 = extractelement <4 x float> %741, i64 0, !dbg !25
  %871 = call float @__ocml_exp2_f32(float %870), !dbg !25
  %872 = insertelement <4 x float> poison, float %871, i64 0, !dbg !27
  %873 = fadd float %869, %871, !dbg !26
  %874 = extractelement <4 x float> %741, i64 1, !dbg !25
  %875 = call float @__ocml_exp2_f32(float %874), !dbg !25
  %876 = insertelement <4 x float> %872, float %875, i64 1, !dbg !27
  %877 = fadd float %873, %875, !dbg !26
  %878 = extractelement <4 x float> %741, i64 2, !dbg !25
  %879 = call float @__ocml_exp2_f32(float %878), !dbg !25
  %880 = insertelement <4 x float> %876, float %879, i64 2, !dbg !27
  %881 = fadd float %877, %879, !dbg !26
  %882 = extractelement <4 x float> %741, i64 3, !dbg !25
  %883 = call float @__ocml_exp2_f32(float %882), !dbg !25
  %884 = insertelement <4 x float> %880, float %883, i64 3, !dbg !27
  %math.result684 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %884) #9, !dbg !27, !frisk.register_lifetimes !1
  %885 = fptrunc <4 x float> %math.result684 to <4 x half>, !dbg !27
  %math.packed685 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %885) #9, !dbg !26, !frisk.register_lifetimes !1
  %886 = fadd float %881, %883, !dbg !26
  %887 = extractelement <4 x float> %742, i64 0, !dbg !25
  %888 = call float @__ocml_exp2_f32(float %887), !dbg !25
  %889 = insertelement <4 x float> poison, float %888, i64 0, !dbg !27
  %890 = fadd float %886, %888, !dbg !26
  %891 = extractelement <4 x float> %742, i64 1, !dbg !25
  %892 = call float @__ocml_exp2_f32(float %891), !dbg !25
  %893 = insertelement <4 x float> %889, float %892, i64 1, !dbg !27
  %894 = fadd float %890, %892, !dbg !26
  %895 = extractelement <4 x float> %742, i64 2, !dbg !25
  %896 = call float @__ocml_exp2_f32(float %895), !dbg !25
  %897 = insertelement <4 x float> %893, float %896, i64 2, !dbg !27
  %898 = fadd float %894, %896, !dbg !26
  %899 = extractelement <4 x float> %742, i64 3, !dbg !25
  %900 = call float @__ocml_exp2_f32(float %899), !dbg !25
  %901 = insertelement <4 x float> %897, float %900, i64 3, !dbg !27
  %math.result686 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %901) #9, !dbg !27, !frisk.register_lifetimes !1
  %902 = fptrunc <4 x float> %math.result686 to <4 x half>, !dbg !27
  %math.packed687 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %902) #9, !dbg !26, !frisk.register_lifetimes !1
  %903 = fadd float %898, %900, !dbg !26
  %904 = extractelement <4 x float> %744, i64 0, !dbg !25
  %905 = call float @__ocml_exp2_f32(float %904), !dbg !25
  %906 = insertelement <4 x float> poison, float %905, i64 0, !dbg !27
  %907 = fadd float %905, 0x0000000000000000, !dbg !26
  %908 = extractelement <4 x float> %744, i64 1, !dbg !25
  %909 = call float @__ocml_exp2_f32(float %908), !dbg !25
  %910 = insertelement <4 x float> %906, float %909, i64 1, !dbg !27
  %911 = fadd float %907, %909, !dbg !26
  %912 = extractelement <4 x float> %744, i64 2, !dbg !25
  %913 = call float @__ocml_exp2_f32(float %912), !dbg !25
  %914 = insertelement <4 x float> %910, float %913, i64 2, !dbg !27
  %915 = fadd float %911, %913, !dbg !26
  %916 = extractelement <4 x float> %744, i64 3, !dbg !25
  %917 = call float @__ocml_exp2_f32(float %916), !dbg !25
  %918 = insertelement <4 x float> %914, float %917, i64 3, !dbg !27
  %math.result688 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %918) #9, !dbg !27, !frisk.register_lifetimes !1
  %919 = fptrunc <4 x float> %math.result688 to <4 x half>, !dbg !27
  %math.packed689 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %919) #9, !dbg !26, !frisk.register_lifetimes !1
  %920 = fadd float %915, %917, !dbg !26
  %921 = extractelement <4 x float> %745, i64 0, !dbg !25
  %922 = call float @__ocml_exp2_f32(float %921), !dbg !25
  %923 = insertelement <4 x float> poison, float %922, i64 0, !dbg !27
  %924 = fadd float %920, %922, !dbg !26
  %925 = extractelement <4 x float> %745, i64 1, !dbg !25
  %926 = call float @__ocml_exp2_f32(float %925), !dbg !25
  %927 = insertelement <4 x float> %923, float %926, i64 1, !dbg !27
  %928 = fadd float %924, %926, !dbg !26
  %929 = extractelement <4 x float> %745, i64 2, !dbg !25
  %930 = call float @__ocml_exp2_f32(float %929), !dbg !25
  %931 = insertelement <4 x float> %927, float %930, i64 2, !dbg !27
  %932 = fadd float %928, %930, !dbg !26
  %933 = extractelement <4 x float> %745, i64 3, !dbg !25
  %934 = call float @__ocml_exp2_f32(float %933), !dbg !25
  %935 = insertelement <4 x float> %931, float %934, i64 3, !dbg !27
  %math.result690 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %935) #9, !dbg !27, !frisk.register_lifetimes !1
  %936 = fptrunc <4 x float> %math.result690 to <4 x half>, !dbg !27
  %math.packed691 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %936) #9, !dbg !26, !frisk.register_lifetimes !1
  %937 = fadd float %932, %934, !dbg !26
  %938 = extractelement <4 x float> %746, i64 0, !dbg !25
  %939 = call float @__ocml_exp2_f32(float %938), !dbg !25
  %940 = insertelement <4 x float> poison, float %939, i64 0, !dbg !27
  %941 = fadd float %937, %939, !dbg !26
  %942 = extractelement <4 x float> %746, i64 1, !dbg !25
  %943 = call float @__ocml_exp2_f32(float %942), !dbg !25
  %944 = insertelement <4 x float> %940, float %943, i64 1, !dbg !27
  %945 = fadd float %941, %943, !dbg !26
  %946 = extractelement <4 x float> %746, i64 2, !dbg !25
  %947 = call float @__ocml_exp2_f32(float %946), !dbg !25
  %948 = insertelement <4 x float> %944, float %947, i64 2, !dbg !27
  %949 = fadd float %945, %947, !dbg !26
  %950 = extractelement <4 x float> %746, i64 3, !dbg !25
  %951 = call float @__ocml_exp2_f32(float %950), !dbg !25
  %952 = insertelement <4 x float> %948, float %951, i64 3, !dbg !27
  %math.result692 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %952) #9, !dbg !27, !frisk.register_lifetimes !1
  %953 = fptrunc <4 x float> %math.result692 to <4 x half>, !dbg !27
  %math.packed693 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %953) #9, !dbg !26, !frisk.register_lifetimes !1
  %954 = fadd float %949, %951, !dbg !26
  %955 = extractelement <4 x float> %747, i64 0, !dbg !25
  %956 = call float @__ocml_exp2_f32(float %955), !dbg !25
  %957 = insertelement <4 x float> poison, float %956, i64 0, !dbg !27
  %958 = fadd float %954, %956, !dbg !26
  %959 = extractelement <4 x float> %747, i64 1, !dbg !25
  %960 = call float @__ocml_exp2_f32(float %959), !dbg !25
  %961 = insertelement <4 x float> %957, float %960, i64 1, !dbg !27
  %962 = fadd float %958, %960, !dbg !26
  %963 = extractelement <4 x float> %747, i64 2, !dbg !25
  %964 = call float @__ocml_exp2_f32(float %963), !dbg !25
  %965 = insertelement <4 x float> %961, float %964, i64 2, !dbg !27
  %966 = fadd float %962, %964, !dbg !26
  %967 = extractelement <4 x float> %747, i64 3, !dbg !25
  %968 = call float @__ocml_exp2_f32(float %967), !dbg !25
  %969 = insertelement <4 x float> %965, float %968, i64 3, !dbg !27
  %math.result694 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %969) #9, !dbg !27, !frisk.register_lifetimes !1
  %970 = fptrunc <4 x float> %math.result694 to <4 x half>, !dbg !27
  %math.packed695 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %970) #9, !dbg !26, !frisk.register_lifetimes !1
  %971 = fadd float %966, %968, !dbg !26
  %972 = extractelement <4 x float> %748, i64 0, !dbg !25
  %973 = call float @__ocml_exp2_f32(float %972), !dbg !25
  %974 = insertelement <4 x float> poison, float %973, i64 0, !dbg !27
  %975 = fadd float %971, %973, !dbg !26
  %976 = extractelement <4 x float> %748, i64 1, !dbg !25
  %977 = call float @__ocml_exp2_f32(float %976), !dbg !25
  %978 = insertelement <4 x float> %974, float %977, i64 1, !dbg !27
  %979 = fadd float %975, %977, !dbg !26
  %980 = extractelement <4 x float> %748, i64 2, !dbg !25
  %981 = call float @__ocml_exp2_f32(float %980), !dbg !25
  %982 = insertelement <4 x float> %978, float %981, i64 2, !dbg !27
  %983 = fadd float %979, %981, !dbg !26
  %984 = extractelement <4 x float> %748, i64 3, !dbg !25
  %985 = call float @__ocml_exp2_f32(float %984), !dbg !25
  %986 = insertelement <4 x float> %982, float %985, i64 3, !dbg !27
  %math.result696 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %986) #9, !dbg !27, !frisk.register_lifetimes !1
  %987 = fptrunc <4 x float> %math.result696 to <4 x half>, !dbg !27
  %math.packed697 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %987) #9, !dbg !26, !frisk.register_lifetimes !1
  %988 = fadd float %983, %985, !dbg !26
  %989 = extractelement <4 x float> %749, i64 0, !dbg !25
  %990 = call float @__ocml_exp2_f32(float %989), !dbg !25
  %991 = insertelement <4 x float> poison, float %990, i64 0, !dbg !27
  %992 = fadd float %988, %990, !dbg !26
  %993 = extractelement <4 x float> %749, i64 1, !dbg !25
  %994 = call float @__ocml_exp2_f32(float %993), !dbg !25
  %995 = insertelement <4 x float> %991, float %994, i64 1, !dbg !27
  %996 = fadd float %992, %994, !dbg !26
  %997 = extractelement <4 x float> %749, i64 2, !dbg !25
  %998 = call float @__ocml_exp2_f32(float %997), !dbg !25
  %999 = insertelement <4 x float> %995, float %998, i64 2, !dbg !27
  %1000 = fadd float %996, %998, !dbg !26
  %1001 = extractelement <4 x float> %749, i64 3, !dbg !25
  %1002 = call float @__ocml_exp2_f32(float %1001), !dbg !25
  %1003 = insertelement <4 x float> %999, float %1002, i64 3, !dbg !27
  %math.result698 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1003) #9, !dbg !27, !frisk.register_lifetimes !1
  %1004 = fptrunc <4 x float> %math.result698 to <4 x half>, !dbg !27
  %math.packed699 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1004) #9, !dbg !26, !frisk.register_lifetimes !1
  %1005 = fadd float %1000, %1002, !dbg !26
  %1006 = extractelement <4 x float> %750, i64 0, !dbg !25
  %1007 = call float @__ocml_exp2_f32(float %1006), !dbg !25
  %1008 = insertelement <4 x float> poison, float %1007, i64 0, !dbg !27
  %1009 = fadd float %1005, %1007, !dbg !26
  %1010 = extractelement <4 x float> %750, i64 1, !dbg !25
  %1011 = call float @__ocml_exp2_f32(float %1010), !dbg !25
  %1012 = insertelement <4 x float> %1008, float %1011, i64 1, !dbg !27
  %1013 = fadd float %1009, %1011, !dbg !26
  %1014 = extractelement <4 x float> %750, i64 2, !dbg !25
  %1015 = call float @__ocml_exp2_f32(float %1014), !dbg !25
  %1016 = insertelement <4 x float> %1012, float %1015, i64 2, !dbg !27
  %1017 = fadd float %1013, %1015, !dbg !26
  %1018 = extractelement <4 x float> %750, i64 3, !dbg !25
  %1019 = call float @__ocml_exp2_f32(float %1018), !dbg !25
  %1020 = insertelement <4 x float> %1016, float %1019, i64 3, !dbg !27
  %math.result700 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1020) #9, !dbg !27, !frisk.register_lifetimes !1
  %1021 = fptrunc <4 x float> %math.result700 to <4 x half>, !dbg !27
  %math.packed701 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1021) #9, !dbg !26, !frisk.register_lifetimes !1
  %1022 = fadd float %1017, %1019, !dbg !26
  %1023 = extractelement <4 x float> %751, i64 0, !dbg !25
  %1024 = call float @__ocml_exp2_f32(float %1023), !dbg !25
  %1025 = insertelement <4 x float> poison, float %1024, i64 0, !dbg !27
  %1026 = fadd float %1022, %1024, !dbg !26
  %1027 = extractelement <4 x float> %751, i64 1, !dbg !25
  %1028 = call float @__ocml_exp2_f32(float %1027), !dbg !25
  %1029 = insertelement <4 x float> %1025, float %1028, i64 1, !dbg !27
  %1030 = fadd float %1026, %1028, !dbg !26
  %1031 = extractelement <4 x float> %751, i64 2, !dbg !25
  %1032 = call float @__ocml_exp2_f32(float %1031), !dbg !25
  %1033 = insertelement <4 x float> %1029, float %1032, i64 2, !dbg !27
  %1034 = fadd float %1030, %1032, !dbg !26
  %1035 = extractelement <4 x float> %751, i64 3, !dbg !25
  %1036 = call float @__ocml_exp2_f32(float %1035), !dbg !25
  %1037 = insertelement <4 x float> %1033, float %1036, i64 3, !dbg !27
  %math.result702 = call <4 x float> asm sideeffect "", "=v,0"(<4 x float> %1037) #9, !dbg !27, !frisk.register_lifetimes !1
  %1038 = fptrunc <4 x float> %math.result702 to <4 x half>, !dbg !27
  %math.packed703 = call <4 x half> asm sideeffect "", "=v,0"(<4 x half> %1038) #9, !dbg !26, !frisk.register_lifetimes !1
  %1039 = fadd float %1034, %1036, !dbg !26
  %1040 = bitcast float %903 to i32, !dbg !26
  %1041 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1040), !dbg !26
  %1042 = bitcast i32 %1041 to float, !dbg !26
  %1043 = fadd float %903, %1042, !dbg !26
  %1044 = bitcast float %1043 to i32, !dbg !26
  %1045 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1044), !dbg !26
  %1046 = bitcast i32 %1045 to float, !dbg !26
  %1047 = fadd float %1043, %1046, !dbg !26
  %1048 = fadd float %1047, %759, !dbg !26
  %1049 = insertelement <1 x float> poison, float %1048, i64 0, !dbg !6
  %1050 = insertvalue [2 x <1 x float>] poison, <1 x float> %1049, 0, !dbg !6
  %1051 = bitcast float %1039 to i32, !dbg !26
  %1052 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1051), !dbg !26
  %1053 = bitcast i32 %1052 to float, !dbg !26
  %1054 = fadd float %1039, %1053, !dbg !26
  %1055 = bitcast float %1054 to i32, !dbg !26
  %1056 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1055), !dbg !26
  %1057 = bitcast i32 %1056 to float, !dbg !26
  %1058 = fadd float %1054, %1057, !dbg !26
  %1059 = fadd float %1058, %767, !dbg !26
  %1060 = insertelement <1 x float> poison, float %1059, i64 0, !dbg !6
  %1061 = insertvalue [2 x <1 x float>] %1050, <1 x float> %1060, 1, !dbg !6
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %address.local1508 = shl nuw nsw i32 %address.thread783, 3, !dbg !14
  %address.local1509 = and i32 %address.local1508, 120, !dbg !14
  %address.local1510 = shl nuw nsw i32 %address.thread783, 5, !dbg !14
  %address.local1511 = and i32 %address.local1510, 1536, !dbg !14
  %address.local1512 = or disjoint i32 %address.local1509, %address.local1511, !dbg !14
  %address.local1513 = zext nneg i32 %address.local1512 to i64, !dbg !14
  %address.local1514 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1513, !dbg !14
  %1062 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1514, i32 0) #0, !dbg !14
  %1063 = bitcast <4 x i32> %1062 to <8 x half>, !dbg !14
  %address.local1515 = shl nuw nsw i32 %address.thread783, 3, !dbg !14
  %address.local1516 = and i32 %address.local1515, 120, !dbg !14
  %address.local1517 = shl nuw nsw i32 %address.thread783, 5, !dbg !14
  %address.local1518 = and i32 %address.local1517, 1536, !dbg !14
  %address.local1519 = or disjoint i32 %address.local1516, %address.local1518, !dbg !14
  %address.local1520 = zext nneg i32 %address.local1519 to i64, !dbg !14
  %address.local1521 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1520, !dbg !14
  %address.local1522 = getelementptr i8, ptr addrspace(3) %address.local1521, i64 256, !dbg !14
  %1064 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1522, i32 0) #0, !dbg !14
  %1065 = bitcast <4 x i32> %1064 to <8 x half>, !dbg !14
  %address.local1523 = shl nuw nsw i32 %address.thread783, 3, !dbg !14
  %address.local1524 = and i32 %address.local1523, 120, !dbg !14
  %address.local1525 = shl nuw nsw i32 %address.thread783, 5, !dbg !14
  %address.local1526 = and i32 %address.local1525, 1536, !dbg !14
  %address.local1527 = or disjoint i32 %address.local1524, %address.local1526, !dbg !14
  %address.local1528 = zext nneg i32 %address.local1527 to i64, !dbg !14
  %address.local1529 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1528, !dbg !14
  %address.local1530 = getelementptr i8, ptr addrspace(3) %address.local1529, i64 512, !dbg !14
  %1066 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1530, i32 0) #0, !dbg !14
  %1067 = bitcast <4 x i32> %1066 to <8 x half>, !dbg !14
  %address.local1531 = shl nuw nsw i32 %address.thread783, 3, !dbg !14
  %address.local1532 = and i32 %address.local1531, 120, !dbg !14
  %address.local1533 = shl nuw nsw i32 %address.thread783, 5, !dbg !14
  %address.local1534 = and i32 %address.local1533, 1536, !dbg !14
  %address.local1535 = or disjoint i32 %address.local1532, %address.local1534, !dbg !14
  %address.local1536 = zext nneg i32 %address.local1535 to i64, !dbg !14
  %address.local1537 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1536, !dbg !14
  %address.local1538 = getelementptr i8, ptr addrspace(3) %address.local1537, i64 768, !dbg !14
  %1068 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1538, i32 0) #0, !dbg !14
  %1069 = bitcast <4 x i32> %1068 to <8 x half>, !dbg !14
  %address.local1539 = shl nuw nsw i32 %address.thread783, 3, !dbg !14
  %address.local1540 = and i32 %address.local1539, 120, !dbg !14
  %address.local1541 = shl nuw nsw i32 %address.thread783, 5, !dbg !14
  %address.local1542 = and i32 %address.local1541, 1536, !dbg !14
  %address.local1543 = or disjoint i32 %address.local1540, %address.local1542, !dbg !14
  %address.local1544 = zext nneg i32 %address.local1543 to i64, !dbg !14
  %address.local1545 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1544, !dbg !14
  %address.local1546 = getelementptr i8, ptr addrspace(3) %address.local1545, i64 4096, !dbg !14
  %1070 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1546, i32 0) #0, !dbg !14
  %1071 = bitcast <4 x i32> %1070 to <8 x half>, !dbg !14
  %address.local1547 = shl nuw nsw i32 %address.thread783, 3, !dbg !14
  %address.local1548 = and i32 %address.local1547, 120, !dbg !14
  %address.local1549 = shl nuw nsw i32 %address.thread783, 5, !dbg !14
  %address.local1550 = and i32 %address.local1549, 1536, !dbg !14
  %address.local1551 = or disjoint i32 %address.local1548, %address.local1550, !dbg !14
  %address.local1552 = zext nneg i32 %address.local1551 to i64, !dbg !14
  %address.local1553 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1552, !dbg !14
  %address.local1554 = getelementptr i8, ptr addrspace(3) %address.local1553, i64 4352, !dbg !14
  %1072 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1554, i32 0) #0, !dbg !14
  %1073 = bitcast <4 x i32> %1072 to <8 x half>, !dbg !14
  %address.local1555 = shl nuw nsw i32 %address.thread783, 3, !dbg !14
  %address.local1556 = and i32 %address.local1555, 120, !dbg !14
  %address.local1557 = shl nuw nsw i32 %address.thread783, 5, !dbg !14
  %address.local1558 = and i32 %address.local1557, 1536, !dbg !14
  %address.local1559 = or disjoint i32 %address.local1556, %address.local1558, !dbg !14
  %address.local1560 = zext nneg i32 %address.local1559 to i64, !dbg !14
  %address.local1561 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1560, !dbg !14
  %address.local1562 = getelementptr i8, ptr addrspace(3) %address.local1561, i64 4608, !dbg !14
  %1074 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1562, i32 0) #0, !dbg !14
  %1075 = bitcast <4 x i32> %1074 to <8 x half>, !dbg !14
  %address.local1563 = shl nuw nsw i32 %address.thread783, 3, !dbg !14
  %address.local1564 = and i32 %address.local1563, 120, !dbg !14
  %address.local1565 = shl nuw nsw i32 %address.thread783, 5, !dbg !14
  %address.local1566 = and i32 %address.local1565, 1536, !dbg !14
  %address.local1567 = or disjoint i32 %address.local1564, %address.local1566, !dbg !14
  %address.local1568 = zext nneg i32 %address.local1567 to i64, !dbg !14
  %address.local1569 = getelementptr [2 x i8], ptr addrspace(3) @shm_0, i64 %address.local1568, !dbg !14
  %address.local1570 = getelementptr i8, ptr addrspace(3) %address.local1569, i64 4864, !dbg !14
  %1076 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1570, i32 0) #0, !dbg !14
  %1077 = bitcast <4 x i32> %1076 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1078 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %245, !dbg !15
  %address.local1571 = getelementptr [2 x i8], ptr addrspace(1) %1078, i64 %address.local1498, !dbg !15
  %1079 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1080 = ptrtoint ptr addrspace(1) %address.local1571 to i64, !dbg !15
  %1081 = sub i64 %1080, %1079, !dbg !15
  %dma.buffer.offset1906 = trunc i64 %1081 to i32, !dbg !15
  %1082 = trunc i64 %1079 to i32, !dbg !15
  %1083 = insertelement <4 x i32> poison, i32 %1082, i64 0, !dbg !15
  %1084 = lshr i64 %1079, 32, !dbg !15
  %1085 = trunc i64 %1084 to i32, !dbg !15
  %1086 = insertelement <4 x i32> %1083, i32 %1085, i64 1, !dbg !15
  %1087 = insertelement <4 x i32> %1086, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1907 = insertelement <4 x i32> %1087, i32 131072, i64 3, !dbg !15
  %1088 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %85, i32 %dma.buffer.offset1906, <4 x i32> %dma.buffer.rsrc1907) #11, !dbg !15
  %address.local1572 = getelementptr [2 x i8], ptr addrspace(1) %1078, i64 %address.local1506, !dbg !15
  %1089 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1090 = ptrtoint ptr addrspace(1) %address.local1572 to i64, !dbg !15
  %1091 = sub i64 %1090, %1089, !dbg !15
  %dma.buffer.offset1904 = trunc i64 %1091 to i32, !dbg !15
  %1092 = trunc i64 %1089 to i32, !dbg !15
  %1093 = insertelement <4 x i32> poison, i32 %1092, i64 0, !dbg !15
  %1094 = lshr i64 %1089, 32, !dbg !15
  %1095 = trunc i64 %1094 to i32, !dbg !15
  %1096 = insertelement <4 x i32> %1093, i32 %1095, i64 1, !dbg !15
  %1097 = insertelement <4 x i32> %1096, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1905 = insertelement <4 x i32> %1097, i32 131072, i64 3, !dbg !15
  %1098 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %88, i32 %dma.buffer.offset1904, <4 x i32> %dma.buffer.rsrc1905) #11, !dbg !15
  %1099 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1100 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1099, <4 x float> zeroinitializer) #9, !dbg !14
  %1101 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1102 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed675, <4 x half> %1101, <4 x float> %1100) #9, !dbg !14
  %1103 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1104 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed677, <4 x half> %1103, <4 x float> %1102) #9, !dbg !14
  %1105 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1106 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed679, <4 x half> %1105, <4 x float> %1104) #9, !dbg !14
  %1107 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1108 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed681, <4 x half> %1107, <4 x float> %1106) #9, !dbg !14
  %1109 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1110 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed683, <4 x half> %1109, <4 x float> %1108) #9, !dbg !14
  %1111 = shufflevector <8 x half> %1075, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1112 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed685, <4 x half> %1111, <4 x float> %1110) #9, !dbg !14
  %1113 = shufflevector <8 x half> %1077, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1114 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed687, <4 x half> %1113, <4 x float> %1112) #9, !dbg !14
  %1115 = extractvalue [2 x <8 x float>] %128, 0, !dbg !28
  %1116 = shufflevector <8 x float> %1115, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1117 = fmul <4 x float> %1116, %756, !dbg !28
  %1118 = fadd <4 x float> %1117, %1114, !dbg !29
  %1119 = shufflevector <8 x half> %1063, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1120 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1119, <4 x float> zeroinitializer) #9, !dbg !14
  %1121 = shufflevector <8 x half> %1065, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1122 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed675, <4 x half> %1121, <4 x float> %1120) #9, !dbg !14
  %1123 = shufflevector <8 x half> %1067, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1124 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed677, <4 x half> %1123, <4 x float> %1122) #9, !dbg !14
  %1125 = shufflevector <8 x half> %1069, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1126 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed679, <4 x half> %1125, <4 x float> %1124) #9, !dbg !14
  %1127 = shufflevector <8 x half> %1071, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1128 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed681, <4 x half> %1127, <4 x float> %1126) #9, !dbg !14
  %1129 = shufflevector <8 x half> %1073, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1130 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed683, <4 x half> %1129, <4 x float> %1128) #9, !dbg !14
  %1131 = shufflevector <8 x half> %1075, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1132 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed685, <4 x half> %1131, <4 x float> %1130) #9, !dbg !14
  %1133 = shufflevector <8 x half> %1077, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1134 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed687, <4 x half> %1133, <4 x float> %1132) #9, !dbg !14
  %1135 = shufflevector <8 x float> %1115, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1136 = fmul <4 x float> %1135, %756, !dbg !28
  %1137 = fadd <4 x float> %1136, %1134, !dbg !29
  %1138 = shufflevector <4 x float> %1118, <4 x float> %1137, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1139 = insertvalue [2 x <8 x float>] poison, <8 x float> %1138, 0, !dbg !6
  %1140 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed689, <4 x half> %1099, <4 x float> zeroinitializer) #9, !dbg !14
  %1141 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed691, <4 x half> %1101, <4 x float> %1140) #9, !dbg !14
  %1142 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed693, <4 x half> %1103, <4 x float> %1141) #9, !dbg !14
  %1143 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed695, <4 x half> %1105, <4 x float> %1142) #9, !dbg !14
  %1144 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed697, <4 x half> %1107, <4 x float> %1143) #9, !dbg !14
  %1145 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed699, <4 x half> %1109, <4 x float> %1144) #9, !dbg !14
  %1146 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed701, <4 x half> %1111, <4 x float> %1145) #9, !dbg !14
  %1147 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed703, <4 x half> %1113, <4 x float> %1146) #9, !dbg !14
  %1148 = extractvalue [2 x <8 x float>] %128, 1, !dbg !28
  %1149 = shufflevector <8 x float> %1148, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1150 = fmul <4 x float> %1149, %764, !dbg !28
  %1151 = fadd <4 x float> %1150, %1147, !dbg !29
  %1152 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed689, <4 x half> %1119, <4 x float> zeroinitializer) #9, !dbg !14
  %1153 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed691, <4 x half> %1121, <4 x float> %1152) #9, !dbg !14
  %1154 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed693, <4 x half> %1123, <4 x float> %1153) #9, !dbg !14
  %1155 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed695, <4 x half> %1125, <4 x float> %1154) #9, !dbg !14
  %1156 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed697, <4 x half> %1127, <4 x float> %1155) #9, !dbg !14
  %1157 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed699, <4 x half> %1129, <4 x float> %1156) #9, !dbg !14
  %1158 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed701, <4 x half> %1131, <4 x float> %1157) #9, !dbg !14
  %1159 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed703, <4 x half> %1133, <4 x float> %1158) #9, !dbg !14
  %1160 = shufflevector <8 x float> %1148, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1161 = fmul <4 x float> %1160, %764, !dbg !28
  %1162 = fadd <4 x float> %1161, %1159, !dbg !29
  %1163 = shufflevector <4 x float> %1151, <4 x float> %1162, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1164 = insertvalue [2 x <8 x float>] %1139, <8 x float> %1163, 1, !dbg !6
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1165 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1514, i32 0) #0, !dbg !14
  %1166 = bitcast <4 x i32> %1165 to <8 x half>, !dbg !14
  %1167 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1522, i32 0) #0, !dbg !14
  %1168 = bitcast <4 x i32> %1167 to <8 x half>, !dbg !14
  %1169 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1530, i32 0) #0, !dbg !14
  %1170 = bitcast <4 x i32> %1169 to <8 x half>, !dbg !14
  %1171 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1538, i32 0) #0, !dbg !14
  %1172 = bitcast <4 x i32> %1171 to <8 x half>, !dbg !14
  %1173 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1546, i32 0) #0, !dbg !14
  %1174 = bitcast <4 x i32> %1173 to <8 x half>, !dbg !14
  %1175 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1554, i32 0) #0, !dbg !14
  %1176 = bitcast <4 x i32> %1175 to <8 x half>, !dbg !14
  %1177 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1562, i32 0) #0, !dbg !14
  %1178 = bitcast <4 x i32> %1177 to <8 x half>, !dbg !14
  %1179 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1570, i32 0) #0, !dbg !14
  %1180 = bitcast <4 x i32> %1179 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1181 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %351, !dbg !15
  %address.local1573 = getelementptr [2 x i8], ptr addrspace(1) %1181, i64 %address.local1498, !dbg !15
  %1182 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1183 = ptrtoint ptr addrspace(1) %address.local1573 to i64, !dbg !15
  %1184 = sub i64 %1183, %1182, !dbg !15
  %dma.buffer.offset1902 = trunc i64 %1184 to i32, !dbg !15
  %1185 = trunc i64 %1182 to i32, !dbg !15
  %1186 = insertelement <4 x i32> poison, i32 %1185, i64 0, !dbg !15
  %1187 = lshr i64 %1182, 32, !dbg !15
  %1188 = trunc i64 %1187 to i32, !dbg !15
  %1189 = insertelement <4 x i32> %1186, i32 %1188, i64 1, !dbg !15
  %1190 = insertelement <4 x i32> %1189, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1903 = insertelement <4 x i32> %1190, i32 131072, i64 3, !dbg !15
  %1191 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %85, i32 %dma.buffer.offset1902, <4 x i32> %dma.buffer.rsrc1903) #11, !dbg !15
  %address.local1574 = getelementptr [2 x i8], ptr addrspace(1) %1181, i64 %address.local1506, !dbg !15
  %1192 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1193 = ptrtoint ptr addrspace(1) %address.local1574 to i64, !dbg !15
  %1194 = sub i64 %1193, %1192, !dbg !15
  %dma.buffer.offset1900 = trunc i64 %1194 to i32, !dbg !15
  %1195 = trunc i64 %1192 to i32, !dbg !15
  %1196 = insertelement <4 x i32> poison, i32 %1195, i64 0, !dbg !15
  %1197 = lshr i64 %1192, 32, !dbg !15
  %1198 = trunc i64 %1197 to i32, !dbg !15
  %1199 = insertelement <4 x i32> %1196, i32 %1198, i64 1, !dbg !15
  %1200 = insertelement <4 x i32> %1199, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1901 = insertelement <4 x i32> %1200, i32 131072, i64 3, !dbg !15
  %1201 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %88, i32 %dma.buffer.offset1900, <4 x i32> %dma.buffer.rsrc1901) #11, !dbg !15
  %1202 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1203 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1202, <4 x float> zeroinitializer) #9, !dbg !14
  %1204 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1205 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed675, <4 x half> %1204, <4 x float> %1203) #9, !dbg !14
  %1206 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1207 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed677, <4 x half> %1206, <4 x float> %1205) #9, !dbg !14
  %1208 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1209 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed679, <4 x half> %1208, <4 x float> %1207) #9, !dbg !14
  %1210 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1211 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed681, <4 x half> %1210, <4 x float> %1209) #9, !dbg !14
  %1212 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1213 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed683, <4 x half> %1212, <4 x float> %1211) #9, !dbg !14
  %1214 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1215 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed685, <4 x half> %1214, <4 x float> %1213) #9, !dbg !14
  %1216 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1217 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed687, <4 x half> %1216, <4 x float> %1215) #9, !dbg !14
  %1218 = extractvalue [2 x <8 x float>] %131, 0, !dbg !28
  %1219 = shufflevector <8 x float> %1218, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1220 = fmul <4 x float> %1219, %756, !dbg !28
  %1221 = fadd <4 x float> %1220, %1217, !dbg !29
  %1222 = shufflevector <8 x half> %1166, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1223 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1222, <4 x float> zeroinitializer) #9, !dbg !14
  %1224 = shufflevector <8 x half> %1168, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1225 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed675, <4 x half> %1224, <4 x float> %1223) #9, !dbg !14
  %1226 = shufflevector <8 x half> %1170, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1227 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed677, <4 x half> %1226, <4 x float> %1225) #9, !dbg !14
  %1228 = shufflevector <8 x half> %1172, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1229 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed679, <4 x half> %1228, <4 x float> %1227) #9, !dbg !14
  %1230 = shufflevector <8 x half> %1174, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1231 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed681, <4 x half> %1230, <4 x float> %1229) #9, !dbg !14
  %1232 = shufflevector <8 x half> %1176, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1233 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed683, <4 x half> %1232, <4 x float> %1231) #9, !dbg !14
  %1234 = shufflevector <8 x half> %1178, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1235 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed685, <4 x half> %1234, <4 x float> %1233) #9, !dbg !14
  %1236 = shufflevector <8 x half> %1180, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1237 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed687, <4 x half> %1236, <4 x float> %1235) #9, !dbg !14
  %1238 = shufflevector <8 x float> %1218, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1239 = fmul <4 x float> %1238, %756, !dbg !28
  %1240 = fadd <4 x float> %1239, %1237, !dbg !29
  %1241 = shufflevector <4 x float> %1221, <4 x float> %1240, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1242 = insertvalue [2 x <8 x float>] poison, <8 x float> %1241, 0, !dbg !6
  %1243 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed689, <4 x half> %1202, <4 x float> zeroinitializer) #9, !dbg !14
  %1244 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed691, <4 x half> %1204, <4 x float> %1243) #9, !dbg !14
  %1245 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed693, <4 x half> %1206, <4 x float> %1244) #9, !dbg !14
  %1246 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed695, <4 x half> %1208, <4 x float> %1245) #9, !dbg !14
  %1247 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed697, <4 x half> %1210, <4 x float> %1246) #9, !dbg !14
  %1248 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed699, <4 x half> %1212, <4 x float> %1247) #9, !dbg !14
  %1249 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed701, <4 x half> %1214, <4 x float> %1248) #9, !dbg !14
  %1250 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed703, <4 x half> %1216, <4 x float> %1249) #9, !dbg !14
  %1251 = extractvalue [2 x <8 x float>] %131, 1, !dbg !28
  %1252 = shufflevector <8 x float> %1251, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1253 = fmul <4 x float> %1252, %764, !dbg !28
  %1254 = fadd <4 x float> %1253, %1250, !dbg !29
  %1255 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed689, <4 x half> %1222, <4 x float> zeroinitializer) #9, !dbg !14
  %1256 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed691, <4 x half> %1224, <4 x float> %1255) #9, !dbg !14
  %1257 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed693, <4 x half> %1226, <4 x float> %1256) #9, !dbg !14
  %1258 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed695, <4 x half> %1228, <4 x float> %1257) #9, !dbg !14
  %1259 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed697, <4 x half> %1230, <4 x float> %1258) #9, !dbg !14
  %1260 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed699, <4 x half> %1232, <4 x float> %1259) #9, !dbg !14
  %1261 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed701, <4 x half> %1234, <4 x float> %1260) #9, !dbg !14
  %1262 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed703, <4 x half> %1236, <4 x float> %1261) #9, !dbg !14
  %1263 = shufflevector <8 x float> %1251, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1264 = fmul <4 x float> %1263, %764, !dbg !28
  %1265 = fadd <4 x float> %1264, %1262, !dbg !29
  %1266 = shufflevector <4 x float> %1254, <4 x float> %1265, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1267 = insertvalue [2 x <8 x float>] %1242, <8 x float> %1266, 1, !dbg !6
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1268 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1514, i32 0) #0, !dbg !14
  %1269 = bitcast <4 x i32> %1268 to <8 x half>, !dbg !14
  %1270 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1522, i32 0) #0, !dbg !14
  %1271 = bitcast <4 x i32> %1270 to <8 x half>, !dbg !14
  %1272 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1530, i32 0) #0, !dbg !14
  %1273 = bitcast <4 x i32> %1272 to <8 x half>, !dbg !14
  %1274 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1538, i32 0) #0, !dbg !14
  %1275 = bitcast <4 x i32> %1274 to <8 x half>, !dbg !14
  %1276 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1546, i32 0) #0, !dbg !14
  %1277 = bitcast <4 x i32> %1276 to <8 x half>, !dbg !14
  %1278 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1554, i32 0) #0, !dbg !14
  %1279 = bitcast <4 x i32> %1278 to <8 x half>, !dbg !14
  %1280 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1562, i32 0) #0, !dbg !14
  %1281 = bitcast <4 x i32> %1280 to <8 x half>, !dbg !14
  %1282 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1570, i32 0) #0, !dbg !14
  %1283 = bitcast <4 x i32> %1282 to <8 x half>, !dbg !14
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1284 = getelementptr [2 x i8], ptr addrspace(1) %2, i64 %457, !dbg !15
  %address.local1575 = getelementptr [2 x i8], ptr addrspace(1) %1284, i64 %address.local1498, !dbg !15
  %1285 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1286 = ptrtoint ptr addrspace(1) %address.local1575 to i64, !dbg !15
  %1287 = sub i64 %1286, %1285, !dbg !15
  %dma.buffer.offset1898 = trunc i64 %1287 to i32, !dbg !15
  %1288 = trunc i64 %1285 to i32, !dbg !15
  %1289 = insertelement <4 x i32> poison, i32 %1288, i64 0, !dbg !15
  %1290 = lshr i64 %1285, 32, !dbg !15
  %1291 = trunc i64 %1290 to i32, !dbg !15
  %1292 = insertelement <4 x i32> %1289, i32 %1291, i64 1, !dbg !15
  %1293 = insertelement <4 x i32> %1292, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc1899 = insertelement <4 x i32> %1293, i32 131072, i64 3, !dbg !15
  %1294 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %85, i32 %dma.buffer.offset1898, <4 x i32> %dma.buffer.rsrc1899) #11, !dbg !15
  %address.local1576 = getelementptr [2 x i8], ptr addrspace(1) %1284, i64 %address.local1506, !dbg !15
  %1295 = ptrtoint ptr addrspace(1) %2 to i64, !dbg !15
  %1296 = ptrtoint ptr addrspace(1) %address.local1576 to i64, !dbg !15
  %1297 = sub i64 %1296, %1295, !dbg !15
  %dma.buffer.offset = trunc i64 %1297 to i32, !dbg !15
  %1298 = trunc i64 %1295 to i32, !dbg !15
  %1299 = insertelement <4 x i32> poison, i32 %1298, i64 0, !dbg !15
  %1300 = lshr i64 %1295, 32, !dbg !15
  %1301 = trunc i64 %1300 to i32, !dbg !15
  %1302 = insertelement <4 x i32> %1299, i32 %1301, i64 1, !dbg !15
  %1303 = insertelement <4 x i32> %1302, i32 33554432, i64 2, !dbg !15
  %dma.buffer.rsrc = insertelement <4 x i32> %1303, i32 131072, i64 3, !dbg !15
  %1304 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %88, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #11, !dbg !15
  %1305 = shufflevector <8 x half> %1269, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1306 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1305, <4 x float> zeroinitializer) #9, !dbg !14
  %1307 = shufflevector <8 x half> %1271, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1308 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed675, <4 x half> %1307, <4 x float> %1306) #9, !dbg !14
  %1309 = shufflevector <8 x half> %1273, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1310 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed677, <4 x half> %1309, <4 x float> %1308) #9, !dbg !14
  %1311 = shufflevector <8 x half> %1275, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1312 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed679, <4 x half> %1311, <4 x float> %1310) #9, !dbg !14
  %1313 = shufflevector <8 x half> %1277, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1314 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed681, <4 x half> %1313, <4 x float> %1312) #9, !dbg !14
  %1315 = shufflevector <8 x half> %1279, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1316 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed683, <4 x half> %1315, <4 x float> %1314) #9, !dbg !14
  %1317 = shufflevector <8 x half> %1281, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1318 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed685, <4 x half> %1317, <4 x float> %1316) #9, !dbg !14
  %1319 = shufflevector <8 x half> %1283, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1320 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed687, <4 x half> %1319, <4 x float> %1318) #9, !dbg !14
  %1321 = extractvalue [2 x <8 x float>] %132, 0, !dbg !28
  %1322 = shufflevector <8 x float> %1321, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1323 = fmul <4 x float> %1322, %756, !dbg !28
  %1324 = fadd <4 x float> %1323, %1320, !dbg !29
  %1325 = shufflevector <8 x half> %1269, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1326 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1325, <4 x float> zeroinitializer) #9, !dbg !14
  %1327 = shufflevector <8 x half> %1271, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1328 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed675, <4 x half> %1327, <4 x float> %1326) #9, !dbg !14
  %1329 = shufflevector <8 x half> %1273, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1330 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed677, <4 x half> %1329, <4 x float> %1328) #9, !dbg !14
  %1331 = shufflevector <8 x half> %1275, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1332 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed679, <4 x half> %1331, <4 x float> %1330) #9, !dbg !14
  %1333 = shufflevector <8 x half> %1277, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1334 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed681, <4 x half> %1333, <4 x float> %1332) #9, !dbg !14
  %1335 = shufflevector <8 x half> %1279, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1336 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed683, <4 x half> %1335, <4 x float> %1334) #9, !dbg !14
  %1337 = shufflevector <8 x half> %1281, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1338 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed685, <4 x half> %1337, <4 x float> %1336) #9, !dbg !14
  %1339 = shufflevector <8 x half> %1283, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1340 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed687, <4 x half> %1339, <4 x float> %1338) #9, !dbg !14
  %1341 = shufflevector <8 x float> %1321, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1342 = fmul <4 x float> %1341, %756, !dbg !28
  %1343 = fadd <4 x float> %1342, %1340, !dbg !29
  %1344 = shufflevector <4 x float> %1324, <4 x float> %1343, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1345 = insertvalue [2 x <8 x float>] poison, <8 x float> %1344, 0, !dbg !6
  %1346 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed689, <4 x half> %1305, <4 x float> zeroinitializer) #9, !dbg !14
  %1347 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed691, <4 x half> %1307, <4 x float> %1346) #9, !dbg !14
  %1348 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed693, <4 x half> %1309, <4 x float> %1347) #9, !dbg !14
  %1349 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed695, <4 x half> %1311, <4 x float> %1348) #9, !dbg !14
  %1350 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed697, <4 x half> %1313, <4 x float> %1349) #9, !dbg !14
  %1351 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed699, <4 x half> %1315, <4 x float> %1350) #9, !dbg !14
  %1352 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed701, <4 x half> %1317, <4 x float> %1351) #9, !dbg !14
  %1353 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed703, <4 x half> %1319, <4 x float> %1352) #9, !dbg !14
  %1354 = extractvalue [2 x <8 x float>] %132, 1, !dbg !28
  %1355 = shufflevector <8 x float> %1354, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1356 = fmul <4 x float> %1355, %764, !dbg !28
  %1357 = fadd <4 x float> %1356, %1353, !dbg !29
  %1358 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed689, <4 x half> %1325, <4 x float> zeroinitializer) #9, !dbg !14
  %1359 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed691, <4 x half> %1327, <4 x float> %1358) #9, !dbg !14
  %1360 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed693, <4 x half> %1329, <4 x float> %1359) #9, !dbg !14
  %1361 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed695, <4 x half> %1331, <4 x float> %1360) #9, !dbg !14
  %1362 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed697, <4 x half> %1333, <4 x float> %1361) #9, !dbg !14
  %1363 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed699, <4 x half> %1335, <4 x float> %1362) #9, !dbg !14
  %1364 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed701, <4 x half> %1337, <4 x float> %1363) #9, !dbg !14
  %1365 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed703, <4 x half> %1339, <4 x float> %1364) #9, !dbg !14
  %1366 = shufflevector <8 x float> %1354, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1367 = fmul <4 x float> %1366, %764, !dbg !28
  %1368 = fadd <4 x float> %1367, %1365, !dbg !29
  %1369 = shufflevector <4 x float> %1357, <4 x float> %1368, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1370 = insertvalue [2 x <8 x float>] %1345, <8 x float> %1369, 1, !dbg !6
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #11, !dbg !15
  fence syncscope("workgroup") release, !dbg !15
  call void @llvm.amdgcn.s.barrier(), !dbg !15
  fence syncscope("workgroup") acquire, !dbg !15
  %1371 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1514, i32 0) #0, !dbg !14
  %1372 = bitcast <4 x i32> %1371 to <8 x half>, !dbg !14
  %1373 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1522, i32 0) #0, !dbg !14
  %1374 = bitcast <4 x i32> %1373 to <8 x half>, !dbg !14
  %1375 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1530, i32 0) #0, !dbg !14
  %1376 = bitcast <4 x i32> %1375 to <8 x half>, !dbg !14
  %1377 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1538, i32 0) #0, !dbg !14
  %1378 = bitcast <4 x i32> %1377 to <8 x half>, !dbg !14
  %1379 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1546, i32 0) #0, !dbg !14
  %1380 = bitcast <4 x i32> %1379 to <8 x half>, !dbg !14
  %1381 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1554, i32 0) #0, !dbg !14
  %1382 = bitcast <4 x i32> %1381 to <8 x half>, !dbg !14
  %1383 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1562, i32 0) #0, !dbg !14
  %1384 = bitcast <4 x i32> %1383 to <8 x half>, !dbg !14
  %1385 = call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3) %address.local1570, i32 0) #0, !dbg !14
  %1386 = bitcast <4 x i32> %1385 to <8 x half>, !dbg !14
  %1387 = shufflevector <8 x half> %1372, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1388 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1387, <4 x float> zeroinitializer) #9, !dbg !14
  %1389 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1390 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed675, <4 x half> %1389, <4 x float> %1388) #9, !dbg !14
  %1391 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1392 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed677, <4 x half> %1391, <4 x float> %1390) #9, !dbg !14
  %1393 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1394 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed679, <4 x half> %1393, <4 x float> %1392) #9, !dbg !14
  %1395 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1396 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed681, <4 x half> %1395, <4 x float> %1394) #9, !dbg !14
  %1397 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1398 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed683, <4 x half> %1397, <4 x float> %1396) #9, !dbg !14
  %1399 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1400 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed685, <4 x half> %1399, <4 x float> %1398) #9, !dbg !14
  %1401 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !14
  %1402 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed687, <4 x half> %1401, <4 x float> %1400) #9, !dbg !14
  %1403 = extractvalue [2 x <8 x float>] %133, 0, !dbg !28
  %1404 = shufflevector <8 x float> %1403, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1405 = fmul <4 x float> %1404, %756, !dbg !28
  %1406 = fadd <4 x float> %1405, %1402, !dbg !29
  %1407 = shufflevector <8 x half> %1372, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1408 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed, <4 x half> %1407, <4 x float> zeroinitializer) #9, !dbg !14
  %1409 = shufflevector <8 x half> %1374, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1410 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed675, <4 x half> %1409, <4 x float> %1408) #9, !dbg !14
  %1411 = shufflevector <8 x half> %1376, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1412 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed677, <4 x half> %1411, <4 x float> %1410) #9, !dbg !14
  %1413 = shufflevector <8 x half> %1378, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1414 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed679, <4 x half> %1413, <4 x float> %1412) #9, !dbg !14
  %1415 = shufflevector <8 x half> %1380, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1416 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed681, <4 x half> %1415, <4 x float> %1414) #9, !dbg !14
  %1417 = shufflevector <8 x half> %1382, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1418 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed683, <4 x half> %1417, <4 x float> %1416) #9, !dbg !14
  %1419 = shufflevector <8 x half> %1384, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1420 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed685, <4 x half> %1419, <4 x float> %1418) #9, !dbg !14
  %1421 = shufflevector <8 x half> %1386, <8 x half> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !14
  %1422 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed687, <4 x half> %1421, <4 x float> %1420) #9, !dbg !14
  %1423 = shufflevector <8 x float> %1403, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1424 = fmul <4 x float> %1423, %756, !dbg !28
  %1425 = fadd <4 x float> %1424, %1422, !dbg !29
  %1426 = shufflevector <4 x float> %1406, <4 x float> %1425, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1427 = insertvalue [2 x <8 x float>] poison, <8 x float> %1426, 0, !dbg !6
  %1428 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed689, <4 x half> %1387, <4 x float> zeroinitializer) #9, !dbg !14
  %1429 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed691, <4 x half> %1389, <4 x float> %1428) #9, !dbg !14
  %1430 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed693, <4 x half> %1391, <4 x float> %1429) #9, !dbg !14
  %1431 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed695, <4 x half> %1393, <4 x float> %1430) #9, !dbg !14
  %1432 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed697, <4 x half> %1395, <4 x float> %1431) #9, !dbg !14
  %1433 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed699, <4 x half> %1397, <4 x float> %1432) #9, !dbg !14
  %1434 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed701, <4 x half> %1399, <4 x float> %1433) #9, !dbg !14
  %1435 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed703, <4 x half> %1401, <4 x float> %1434) #9, !dbg !14
  %1436 = extractvalue [2 x <8 x float>] %133, 1, !dbg !28
  %1437 = shufflevector <8 x float> %1436, <8 x float> poison, <4 x i32> <i32 0, i32 1, i32 2, i32 3>, !dbg !28
  %1438 = fmul <4 x float> %1437, %764, !dbg !28
  %1439 = fadd <4 x float> %1438, %1435, !dbg !29
  %1440 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed689, <4 x half> %1407, <4 x float> zeroinitializer) #9, !dbg !14
  %1441 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed691, <4 x half> %1409, <4 x float> %1440) #9, !dbg !14
  %1442 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed693, <4 x half> %1411, <4 x float> %1441) #9, !dbg !14
  %1443 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed695, <4 x half> %1413, <4 x float> %1442) #9, !dbg !14
  %1444 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed697, <4 x half> %1415, <4 x float> %1443) #9, !dbg !14
  %1445 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed699, <4 x half> %1417, <4 x float> %1444) #9, !dbg !14
  %1446 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed701, <4 x half> %1419, <4 x float> %1445) #9, !dbg !14
  %1447 = call <4 x float> asm "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\0A\09", "=v,v,v,0"(<4 x half> %math.packed703, <4 x half> %1421, <4 x float> %1446) #9, !dbg !14
  %1448 = shufflevector <8 x float> %1436, <8 x float> poison, <4 x i32> <i32 4, i32 5, i32 6, i32 7>, !dbg !28
  %1449 = fmul <4 x float> %1448, %764, !dbg !28
  %1450 = fadd <4 x float> %1449, %1447, !dbg !29
  %1451 = shufflevector <4 x float> %1439, <4 x float> %1450, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>, !dbg !6
  %1452 = insertvalue [2 x <8 x float>] %1427, <8 x float> %1451, 1, !dbg !6
  %1453 = insertvalue [2 x <1 x float>] poison, <1 x float> %730, 0, !dbg !6
  %1454 = insertvalue [2 x <1 x float>] %1453, <1 x float> %733, 1, !dbg !6
  fence syncscope("workgroup") release, !dbg !6
  call void @llvm.amdgcn.s.barrier(), !dbg !6
  fence syncscope("workgroup") acquire, !dbg !6
  %1455 = add nuw nsw i64 %127, 1, !dbg !6
  br label %126, !dbg !6

1456:                                             ; preds = %126
  %1457 = and i32 %31, 16, !dbg !6
  %.not = icmp eq i32 %1457, 0, !dbg !6
  %1458 = and i32 %31, 32, !dbg !6
  %.not669 = icmp eq i32 %1458, 0, !dbg !6
  %1459 = extractvalue [2 x <8 x float>] %133, 0, !dbg !6
  %1460 = extractelement <8 x float> %1459, i64 0, !dbg !6
  %1461 = extractelement <8 x float> %1459, i64 1, !dbg !6
  %1462 = extractelement <8 x float> %1459, i64 2, !dbg !6
  %1463 = extractelement <8 x float> %1459, i64 3, !dbg !6
  %1464 = select i1 %.not, float %1461, float %1460, !dbg !6
  %1465 = bitcast float %1464 to i32, !dbg !6
  %1466 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1465), !dbg !6
  %1467 = bitcast i32 %1466 to float, !dbg !6
  %1468 = select i1 %.not, float %1460, float %1467, !dbg !6
  %1469 = select i1 %.not, float %1467, float %1461, !dbg !6
  %1470 = select i1 %.not, float %1463, float %1462, !dbg !6
  %1471 = bitcast float %1470 to i32, !dbg !6
  %1472 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1471), !dbg !6
  %1473 = bitcast i32 %1472 to float, !dbg !6
  %1474 = select i1 %.not, float %1462, float %1473, !dbg !6
  %1475 = select i1 %.not, float %1473, float %1463, !dbg !6
  %1476 = select i1 %.not669, float %1474, float %1468, !dbg !6
  %1477 = bitcast float %1476 to i32, !dbg !6
  %1478 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1477), !dbg !6
  %1479 = bitcast i32 %1478 to float, !dbg !6
  %1480 = select i1 %.not669, float %1468, float %1479, !dbg !6
  %1481 = select i1 %.not669, float %1479, float %1474, !dbg !6
  %1482 = select i1 %.not669, float %1475, float %1469, !dbg !6
  %1483 = bitcast float %1482 to i32, !dbg !6
  %1484 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1483), !dbg !6
  %1485 = bitcast i32 %1484 to float, !dbg !6
  %1486 = select i1 %.not669, float %1469, float %1485, !dbg !6
  %1487 = select i1 %.not669, float %1485, float %1475, !dbg !6
  %1488 = extractelement <8 x float> %1459, i64 4, !dbg !6
  %1489 = extractelement <8 x float> %1459, i64 5, !dbg !6
  %1490 = extractelement <8 x float> %1459, i64 6, !dbg !6
  %1491 = extractelement <8 x float> %1459, i64 7, !dbg !6
  %1492 = select i1 %.not, float %1489, float %1488, !dbg !6
  %1493 = bitcast float %1492 to i32, !dbg !6
  %1494 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1493), !dbg !6
  %1495 = bitcast i32 %1494 to float, !dbg !6
  %1496 = select i1 %.not, float %1488, float %1495, !dbg !6
  %1497 = select i1 %.not, float %1495, float %1489, !dbg !6
  %1498 = select i1 %.not, float %1491, float %1490, !dbg !6
  %1499 = bitcast float %1498 to i32, !dbg !6
  %1500 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1499), !dbg !6
  %1501 = bitcast i32 %1500 to float, !dbg !6
  %1502 = select i1 %.not, float %1490, float %1501, !dbg !6
  %1503 = select i1 %.not, float %1501, float %1491, !dbg !6
  %1504 = select i1 %.not669, float %1502, float %1496, !dbg !6
  %1505 = bitcast float %1504 to i32, !dbg !6
  %1506 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1505), !dbg !6
  %1507 = bitcast i32 %1506 to float, !dbg !6
  %1508 = select i1 %.not669, float %1496, float %1507, !dbg !6
  %1509 = select i1 %.not669, float %1507, float %1502, !dbg !6
  %1510 = select i1 %.not669, float %1503, float %1497, !dbg !6
  %1511 = bitcast float %1510 to i32, !dbg !6
  %1512 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1511), !dbg !6
  %1513 = bitcast i32 %1512 to float, !dbg !6
  %1514 = select i1 %.not669, float %1497, float %1513, !dbg !6
  %1515 = select i1 %.not669, float %1513, float %1503, !dbg !6
  %1516 = extractvalue [2 x <8 x float>] %133, 1, !dbg !6
  %1517 = extractelement <8 x float> %1516, i64 0, !dbg !6
  %1518 = extractelement <8 x float> %1516, i64 1, !dbg !6
  %1519 = extractelement <8 x float> %1516, i64 2, !dbg !6
  %1520 = extractelement <8 x float> %1516, i64 3, !dbg !6
  %1521 = select i1 %.not, float %1518, float %1517, !dbg !6
  %1522 = bitcast float %1521 to i32, !dbg !6
  %1523 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1522), !dbg !6
  %1524 = bitcast i32 %1523 to float, !dbg !6
  %1525 = select i1 %.not, float %1517, float %1524, !dbg !6
  %1526 = select i1 %.not, float %1524, float %1518, !dbg !6
  %1527 = select i1 %.not, float %1520, float %1519, !dbg !6
  %1528 = bitcast float %1527 to i32, !dbg !6
  %1529 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1528), !dbg !6
  %1530 = bitcast i32 %1529 to float, !dbg !6
  %1531 = select i1 %.not, float %1519, float %1530, !dbg !6
  %1532 = select i1 %.not, float %1530, float %1520, !dbg !6
  %1533 = select i1 %.not669, float %1531, float %1525, !dbg !6
  %1534 = bitcast float %1533 to i32, !dbg !6
  %1535 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1534), !dbg !6
  %1536 = bitcast i32 %1535 to float, !dbg !6
  %1537 = select i1 %.not669, float %1525, float %1536, !dbg !6
  %1538 = select i1 %.not669, float %1536, float %1531, !dbg !6
  %1539 = select i1 %.not669, float %1532, float %1526, !dbg !6
  %1540 = bitcast float %1539 to i32, !dbg !6
  %1541 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1540), !dbg !6
  %1542 = bitcast i32 %1541 to float, !dbg !6
  %1543 = select i1 %.not669, float %1526, float %1542, !dbg !6
  %1544 = select i1 %.not669, float %1542, float %1532, !dbg !6
  %1545 = extractelement <8 x float> %1516, i64 4, !dbg !6
  %1546 = extractelement <8 x float> %1516, i64 5, !dbg !6
  %1547 = extractelement <8 x float> %1516, i64 6, !dbg !6
  %1548 = extractelement <8 x float> %1516, i64 7, !dbg !6
  %1549 = select i1 %.not, float %1546, float %1545, !dbg !6
  %1550 = bitcast float %1549 to i32, !dbg !6
  %1551 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1550), !dbg !6
  %1552 = bitcast i32 %1551 to float, !dbg !6
  %1553 = select i1 %.not, float %1545, float %1552, !dbg !6
  %1554 = select i1 %.not, float %1552, float %1546, !dbg !6
  %1555 = select i1 %.not, float %1548, float %1547, !dbg !6
  %1556 = bitcast float %1555 to i32, !dbg !6
  %1557 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1556), !dbg !6
  %1558 = bitcast i32 %1557 to float, !dbg !6
  %1559 = select i1 %.not, float %1547, float %1558, !dbg !6
  %1560 = select i1 %.not, float %1558, float %1548, !dbg !6
  %1561 = select i1 %.not669, float %1559, float %1553, !dbg !6
  %1562 = bitcast float %1561 to i32, !dbg !6
  %1563 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1562), !dbg !6
  %1564 = bitcast i32 %1563 to float, !dbg !6
  %1565 = select i1 %.not669, float %1553, float %1564, !dbg !6
  %1566 = select i1 %.not669, float %1564, float %1559, !dbg !6
  %1567 = select i1 %.not669, float %1560, float %1554, !dbg !6
  %1568 = bitcast float %1567 to i32, !dbg !6
  %1569 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1568), !dbg !6
  %1570 = bitcast i32 %1569 to float, !dbg !6
  %1571 = select i1 %.not669, float %1554, float %1570, !dbg !6
  %1572 = select i1 %.not669, float %1570, float %1560, !dbg !6
  %1573 = extractvalue [2 x <8 x float>] %132, 0, !dbg !6
  %1574 = extractelement <8 x float> %1573, i64 0, !dbg !6
  %1575 = extractelement <8 x float> %1573, i64 1, !dbg !6
  %1576 = extractelement <8 x float> %1573, i64 2, !dbg !6
  %1577 = extractelement <8 x float> %1573, i64 3, !dbg !6
  %1578 = select i1 %.not, float %1575, float %1574, !dbg !6
  %1579 = bitcast float %1578 to i32, !dbg !6
  %1580 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1579), !dbg !6
  %1581 = bitcast i32 %1580 to float, !dbg !6
  %1582 = select i1 %.not, float %1574, float %1581, !dbg !6
  %1583 = select i1 %.not, float %1581, float %1575, !dbg !6
  %1584 = select i1 %.not, float %1577, float %1576, !dbg !6
  %1585 = bitcast float %1584 to i32, !dbg !6
  %1586 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1585), !dbg !6
  %1587 = bitcast i32 %1586 to float, !dbg !6
  %1588 = select i1 %.not, float %1576, float %1587, !dbg !6
  %1589 = select i1 %.not, float %1587, float %1577, !dbg !6
  %1590 = select i1 %.not669, float %1588, float %1582, !dbg !6
  %1591 = bitcast float %1590 to i32, !dbg !6
  %1592 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1591), !dbg !6
  %1593 = bitcast i32 %1592 to float, !dbg !6
  %1594 = select i1 %.not669, float %1582, float %1593, !dbg !6
  %1595 = select i1 %.not669, float %1593, float %1588, !dbg !6
  %1596 = select i1 %.not669, float %1589, float %1583, !dbg !6
  %1597 = bitcast float %1596 to i32, !dbg !6
  %1598 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1597), !dbg !6
  %1599 = bitcast i32 %1598 to float, !dbg !6
  %1600 = select i1 %.not669, float %1583, float %1599, !dbg !6
  %1601 = select i1 %.not669, float %1599, float %1589, !dbg !6
  %1602 = extractelement <8 x float> %1573, i64 4, !dbg !6
  %1603 = extractelement <8 x float> %1573, i64 5, !dbg !6
  %1604 = extractelement <8 x float> %1573, i64 6, !dbg !6
  %1605 = extractelement <8 x float> %1573, i64 7, !dbg !6
  %1606 = select i1 %.not, float %1603, float %1602, !dbg !6
  %1607 = bitcast float %1606 to i32, !dbg !6
  %1608 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1607), !dbg !6
  %1609 = bitcast i32 %1608 to float, !dbg !6
  %1610 = select i1 %.not, float %1602, float %1609, !dbg !6
  %1611 = select i1 %.not, float %1609, float %1603, !dbg !6
  %1612 = select i1 %.not, float %1605, float %1604, !dbg !6
  %1613 = bitcast float %1612 to i32, !dbg !6
  %1614 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1613), !dbg !6
  %1615 = bitcast i32 %1614 to float, !dbg !6
  %1616 = select i1 %.not, float %1604, float %1615, !dbg !6
  %1617 = select i1 %.not, float %1615, float %1605, !dbg !6
  %1618 = select i1 %.not669, float %1616, float %1610, !dbg !6
  %1619 = bitcast float %1618 to i32, !dbg !6
  %1620 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1619), !dbg !6
  %1621 = bitcast i32 %1620 to float, !dbg !6
  %1622 = select i1 %.not669, float %1610, float %1621, !dbg !6
  %1623 = select i1 %.not669, float %1621, float %1616, !dbg !6
  %1624 = select i1 %.not669, float %1617, float %1611, !dbg !6
  %1625 = bitcast float %1624 to i32, !dbg !6
  %1626 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1625), !dbg !6
  %1627 = bitcast i32 %1626 to float, !dbg !6
  %1628 = select i1 %.not669, float %1611, float %1627, !dbg !6
  %1629 = select i1 %.not669, float %1627, float %1617, !dbg !6
  %1630 = extractvalue [2 x <8 x float>] %132, 1, !dbg !6
  %1631 = extractelement <8 x float> %1630, i64 0, !dbg !6
  %1632 = extractelement <8 x float> %1630, i64 1, !dbg !6
  %1633 = extractelement <8 x float> %1630, i64 2, !dbg !6
  %1634 = extractelement <8 x float> %1630, i64 3, !dbg !6
  %1635 = select i1 %.not, float %1632, float %1631, !dbg !6
  %1636 = bitcast float %1635 to i32, !dbg !6
  %1637 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1636), !dbg !6
  %1638 = bitcast i32 %1637 to float, !dbg !6
  %1639 = select i1 %.not, float %1631, float %1638, !dbg !6
  %1640 = select i1 %.not, float %1638, float %1632, !dbg !6
  %1641 = select i1 %.not, float %1634, float %1633, !dbg !6
  %1642 = bitcast float %1641 to i32, !dbg !6
  %1643 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1642), !dbg !6
  %1644 = bitcast i32 %1643 to float, !dbg !6
  %1645 = select i1 %.not, float %1633, float %1644, !dbg !6
  %1646 = select i1 %.not, float %1644, float %1634, !dbg !6
  %1647 = select i1 %.not669, float %1645, float %1639, !dbg !6
  %1648 = bitcast float %1647 to i32, !dbg !6
  %1649 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1648), !dbg !6
  %1650 = bitcast i32 %1649 to float, !dbg !6
  %1651 = select i1 %.not669, float %1639, float %1650, !dbg !6
  %1652 = select i1 %.not669, float %1650, float %1645, !dbg !6
  %1653 = select i1 %.not669, float %1646, float %1640, !dbg !6
  %1654 = bitcast float %1653 to i32, !dbg !6
  %1655 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1654), !dbg !6
  %1656 = bitcast i32 %1655 to float, !dbg !6
  %1657 = select i1 %.not669, float %1640, float %1656, !dbg !6
  %1658 = select i1 %.not669, float %1656, float %1646, !dbg !6
  %1659 = extractelement <8 x float> %1630, i64 4, !dbg !6
  %1660 = extractelement <8 x float> %1630, i64 5, !dbg !6
  %1661 = extractelement <8 x float> %1630, i64 6, !dbg !6
  %1662 = extractelement <8 x float> %1630, i64 7, !dbg !6
  %1663 = select i1 %.not, float %1660, float %1659, !dbg !6
  %1664 = bitcast float %1663 to i32, !dbg !6
  %1665 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1664), !dbg !6
  %1666 = bitcast i32 %1665 to float, !dbg !6
  %1667 = select i1 %.not, float %1659, float %1666, !dbg !6
  %1668 = select i1 %.not, float %1666, float %1660, !dbg !6
  %1669 = select i1 %.not, float %1662, float %1661, !dbg !6
  %1670 = bitcast float %1669 to i32, !dbg !6
  %1671 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1670), !dbg !6
  %1672 = bitcast i32 %1671 to float, !dbg !6
  %1673 = select i1 %.not, float %1661, float %1672, !dbg !6
  %1674 = select i1 %.not, float %1672, float %1662, !dbg !6
  %1675 = select i1 %.not669, float %1673, float %1667, !dbg !6
  %1676 = bitcast float %1675 to i32, !dbg !6
  %1677 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1676), !dbg !6
  %1678 = bitcast i32 %1677 to float, !dbg !6
  %1679 = select i1 %.not669, float %1667, float %1678, !dbg !6
  %1680 = select i1 %.not669, float %1678, float %1673, !dbg !6
  %1681 = select i1 %.not669, float %1674, float %1668, !dbg !6
  %1682 = bitcast float %1681 to i32, !dbg !6
  %1683 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1682), !dbg !6
  %1684 = bitcast i32 %1683 to float, !dbg !6
  %1685 = select i1 %.not669, float %1668, float %1684, !dbg !6
  %1686 = select i1 %.not669, float %1684, float %1674, !dbg !6
  %1687 = extractvalue [2 x <8 x float>] %131, 0, !dbg !6
  %1688 = extractelement <8 x float> %1687, i64 0, !dbg !6
  %1689 = extractelement <8 x float> %1687, i64 1, !dbg !6
  %1690 = extractelement <8 x float> %1687, i64 2, !dbg !6
  %1691 = extractelement <8 x float> %1687, i64 3, !dbg !6
  %1692 = select i1 %.not, float %1689, float %1688, !dbg !6
  %1693 = bitcast float %1692 to i32, !dbg !6
  %1694 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1693), !dbg !6
  %1695 = bitcast i32 %1694 to float, !dbg !6
  %1696 = select i1 %.not, float %1688, float %1695, !dbg !6
  %1697 = select i1 %.not, float %1695, float %1689, !dbg !6
  %1698 = select i1 %.not, float %1691, float %1690, !dbg !6
  %1699 = bitcast float %1698 to i32, !dbg !6
  %1700 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1699), !dbg !6
  %1701 = bitcast i32 %1700 to float, !dbg !6
  %1702 = select i1 %.not, float %1690, float %1701, !dbg !6
  %1703 = select i1 %.not, float %1701, float %1691, !dbg !6
  %1704 = select i1 %.not669, float %1702, float %1696, !dbg !6
  %1705 = bitcast float %1704 to i32, !dbg !6
  %1706 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1705), !dbg !6
  %1707 = bitcast i32 %1706 to float, !dbg !6
  %1708 = select i1 %.not669, float %1696, float %1707, !dbg !6
  %1709 = select i1 %.not669, float %1707, float %1702, !dbg !6
  %1710 = select i1 %.not669, float %1703, float %1697, !dbg !6
  %1711 = bitcast float %1710 to i32, !dbg !6
  %1712 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1711), !dbg !6
  %1713 = bitcast i32 %1712 to float, !dbg !6
  %1714 = select i1 %.not669, float %1697, float %1713, !dbg !6
  %1715 = select i1 %.not669, float %1713, float %1703, !dbg !6
  %1716 = extractelement <8 x float> %1687, i64 4, !dbg !6
  %1717 = extractelement <8 x float> %1687, i64 5, !dbg !6
  %1718 = extractelement <8 x float> %1687, i64 6, !dbg !6
  %1719 = extractelement <8 x float> %1687, i64 7, !dbg !6
  %1720 = select i1 %.not, float %1717, float %1716, !dbg !6
  %1721 = bitcast float %1720 to i32, !dbg !6
  %1722 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1721), !dbg !6
  %1723 = bitcast i32 %1722 to float, !dbg !6
  %1724 = select i1 %.not, float %1716, float %1723, !dbg !6
  %1725 = select i1 %.not, float %1723, float %1717, !dbg !6
  %1726 = select i1 %.not, float %1719, float %1718, !dbg !6
  %1727 = bitcast float %1726 to i32, !dbg !6
  %1728 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1727), !dbg !6
  %1729 = bitcast i32 %1728 to float, !dbg !6
  %1730 = select i1 %.not, float %1718, float %1729, !dbg !6
  %1731 = select i1 %.not, float %1729, float %1719, !dbg !6
  %1732 = select i1 %.not669, float %1730, float %1724, !dbg !6
  %1733 = bitcast float %1732 to i32, !dbg !6
  %1734 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1733), !dbg !6
  %1735 = bitcast i32 %1734 to float, !dbg !6
  %1736 = select i1 %.not669, float %1724, float %1735, !dbg !6
  %1737 = select i1 %.not669, float %1735, float %1730, !dbg !6
  %1738 = select i1 %.not669, float %1731, float %1725, !dbg !6
  %1739 = bitcast float %1738 to i32, !dbg !6
  %1740 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1739), !dbg !6
  %1741 = bitcast i32 %1740 to float, !dbg !6
  %1742 = select i1 %.not669, float %1725, float %1741, !dbg !6
  %1743 = select i1 %.not669, float %1741, float %1731, !dbg !6
  %1744 = extractvalue [2 x <8 x float>] %131, 1, !dbg !6
  %1745 = extractelement <8 x float> %1744, i64 0, !dbg !6
  %1746 = extractelement <8 x float> %1744, i64 1, !dbg !6
  %1747 = extractelement <8 x float> %1744, i64 2, !dbg !6
  %1748 = extractelement <8 x float> %1744, i64 3, !dbg !6
  %1749 = select i1 %.not, float %1746, float %1745, !dbg !6
  %1750 = bitcast float %1749 to i32, !dbg !6
  %1751 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1750), !dbg !6
  %1752 = bitcast i32 %1751 to float, !dbg !6
  %1753 = select i1 %.not, float %1745, float %1752, !dbg !6
  %1754 = select i1 %.not, float %1752, float %1746, !dbg !6
  %1755 = select i1 %.not, float %1748, float %1747, !dbg !6
  %1756 = bitcast float %1755 to i32, !dbg !6
  %1757 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1756), !dbg !6
  %1758 = bitcast i32 %1757 to float, !dbg !6
  %1759 = select i1 %.not, float %1747, float %1758, !dbg !6
  %1760 = select i1 %.not, float %1758, float %1748, !dbg !6
  %1761 = select i1 %.not669, float %1759, float %1753, !dbg !6
  %1762 = bitcast float %1761 to i32, !dbg !6
  %1763 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1762), !dbg !6
  %1764 = bitcast i32 %1763 to float, !dbg !6
  %1765 = select i1 %.not669, float %1753, float %1764, !dbg !6
  %1766 = select i1 %.not669, float %1764, float %1759, !dbg !6
  %1767 = select i1 %.not669, float %1760, float %1754, !dbg !6
  %1768 = bitcast float %1767 to i32, !dbg !6
  %1769 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1768), !dbg !6
  %1770 = bitcast i32 %1769 to float, !dbg !6
  %1771 = select i1 %.not669, float %1754, float %1770, !dbg !6
  %1772 = select i1 %.not669, float %1770, float %1760, !dbg !6
  %1773 = extractelement <8 x float> %1744, i64 4, !dbg !6
  %1774 = extractelement <8 x float> %1744, i64 5, !dbg !6
  %1775 = extractelement <8 x float> %1744, i64 6, !dbg !6
  %1776 = extractelement <8 x float> %1744, i64 7, !dbg !6
  %1777 = select i1 %.not, float %1774, float %1773, !dbg !6
  %1778 = bitcast float %1777 to i32, !dbg !6
  %1779 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1778), !dbg !6
  %1780 = bitcast i32 %1779 to float, !dbg !6
  %1781 = select i1 %.not, float %1773, float %1780, !dbg !6
  %1782 = select i1 %.not, float %1780, float %1774, !dbg !6
  %1783 = select i1 %.not, float %1776, float %1775, !dbg !6
  %1784 = bitcast float %1783 to i32, !dbg !6
  %1785 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1784), !dbg !6
  %1786 = bitcast i32 %1785 to float, !dbg !6
  %1787 = select i1 %.not, float %1775, float %1786, !dbg !6
  %1788 = select i1 %.not, float %1786, float %1776, !dbg !6
  %1789 = select i1 %.not669, float %1787, float %1781, !dbg !6
  %1790 = bitcast float %1789 to i32, !dbg !6
  %1791 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1790), !dbg !6
  %1792 = bitcast i32 %1791 to float, !dbg !6
  %1793 = select i1 %.not669, float %1781, float %1792, !dbg !6
  %1794 = select i1 %.not669, float %1792, float %1787, !dbg !6
  %1795 = select i1 %.not669, float %1788, float %1782, !dbg !6
  %1796 = bitcast float %1795 to i32, !dbg !6
  %1797 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1796), !dbg !6
  %1798 = bitcast i32 %1797 to float, !dbg !6
  %1799 = select i1 %.not669, float %1782, float %1798, !dbg !6
  %1800 = select i1 %.not669, float %1798, float %1788, !dbg !6
  %1801 = extractvalue [2 x <8 x float>] %128, 0, !dbg !6
  %1802 = extractelement <8 x float> %1801, i64 0, !dbg !6
  %1803 = extractelement <8 x float> %1801, i64 1, !dbg !6
  %1804 = extractelement <8 x float> %1801, i64 2, !dbg !6
  %1805 = extractelement <8 x float> %1801, i64 3, !dbg !6
  %1806 = select i1 %.not, float %1803, float %1802, !dbg !6
  %1807 = bitcast float %1806 to i32, !dbg !6
  %1808 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1807), !dbg !6
  %1809 = bitcast i32 %1808 to float, !dbg !6
  %1810 = select i1 %.not, float %1802, float %1809, !dbg !6
  %1811 = select i1 %.not, float %1809, float %1803, !dbg !6
  %1812 = select i1 %.not, float %1805, float %1804, !dbg !6
  %1813 = bitcast float %1812 to i32, !dbg !6
  %1814 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1813), !dbg !6
  %1815 = bitcast i32 %1814 to float, !dbg !6
  %1816 = select i1 %.not, float %1804, float %1815, !dbg !6
  %1817 = select i1 %.not, float %1815, float %1805, !dbg !6
  %1818 = select i1 %.not669, float %1816, float %1810, !dbg !6
  %1819 = bitcast float %1818 to i32, !dbg !6
  %1820 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1819), !dbg !6
  %1821 = select i1 %.not669, float %1817, float %1811, !dbg !6
  %1822 = bitcast float %1821 to i32, !dbg !6
  %1823 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1822), !dbg !6
  %1824 = extractelement <8 x float> %1801, i64 4, !dbg !6
  %1825 = extractelement <8 x float> %1801, i64 5, !dbg !6
  %1826 = extractelement <8 x float> %1801, i64 6, !dbg !6
  %1827 = extractelement <8 x float> %1801, i64 7, !dbg !6
  %1828 = select i1 %.not, float %1825, float %1824, !dbg !6
  %1829 = bitcast float %1828 to i32, !dbg !6
  %1830 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1829), !dbg !6
  %1831 = bitcast i32 %1830 to float, !dbg !6
  %1832 = select i1 %.not, float %1824, float %1831, !dbg !6
  %1833 = select i1 %.not, float %1831, float %1825, !dbg !6
  %1834 = select i1 %.not, float %1827, float %1826, !dbg !6
  %1835 = bitcast float %1834 to i32, !dbg !6
  %1836 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1835), !dbg !6
  %1837 = bitcast i32 %1836 to float, !dbg !6
  %1838 = select i1 %.not, float %1826, float %1837, !dbg !6
  %1839 = select i1 %.not, float %1837, float %1827, !dbg !6
  %1840 = select i1 %.not669, float %1838, float %1832, !dbg !6
  %1841 = bitcast float %1840 to i32, !dbg !6
  %1842 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1841), !dbg !6
  %1843 = select i1 %.not669, float %1839, float %1833, !dbg !6
  %1844 = bitcast float %1843 to i32, !dbg !6
  %1845 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1844), !dbg !6
  %1846 = extractvalue [2 x <8 x float>] %128, 1, !dbg !6
  %1847 = extractelement <8 x float> %1846, i64 0, !dbg !6
  %1848 = extractelement <8 x float> %1846, i64 1, !dbg !6
  %1849 = extractelement <8 x float> %1846, i64 2, !dbg !6
  %1850 = extractelement <8 x float> %1846, i64 3, !dbg !6
  %1851 = select i1 %.not, float %1848, float %1847, !dbg !6
  %1852 = bitcast float %1851 to i32, !dbg !6
  %1853 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1852), !dbg !6
  %1854 = bitcast i32 %1853 to float, !dbg !6
  %1855 = select i1 %.not, float %1847, float %1854, !dbg !6
  %1856 = select i1 %.not, float %1854, float %1848, !dbg !6
  %1857 = select i1 %.not, float %1850, float %1849, !dbg !6
  %1858 = bitcast float %1857 to i32, !dbg !6
  %1859 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1858), !dbg !6
  %1860 = bitcast i32 %1859 to float, !dbg !6
  %1861 = select i1 %.not, float %1849, float %1860, !dbg !6
  %1862 = select i1 %.not, float %1860, float %1850, !dbg !6
  %1863 = select i1 %.not669, float %1861, float %1855, !dbg !6
  %1864 = bitcast float %1863 to i32, !dbg !6
  %1865 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1864), !dbg !6
  %1866 = select i1 %.not669, float %1862, float %1856, !dbg !6
  %1867 = bitcast float %1866 to i32, !dbg !6
  %1868 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1867), !dbg !6
  %1869 = extractelement <8 x float> %1846, i64 4, !dbg !6
  %1870 = extractelement <8 x float> %1846, i64 5, !dbg !6
  %1871 = extractelement <8 x float> %1846, i64 6, !dbg !6
  %1872 = extractelement <8 x float> %1846, i64 7, !dbg !6
  %1873 = select i1 %.not, float %1870, float %1869, !dbg !6
  %1874 = bitcast float %1873 to i32, !dbg !6
  %1875 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1874), !dbg !6
  %1876 = bitcast i32 %1875 to float, !dbg !6
  %1877 = select i1 %.not, float %1869, float %1876, !dbg !6
  %1878 = select i1 %.not, float %1876, float %1870, !dbg !6
  %1879 = select i1 %.not, float %1872, float %1871, !dbg !6
  %1880 = bitcast float %1879 to i32, !dbg !6
  %1881 = call i32 @llvm.amdgcn.ds.bpermute(i32 %12, i32 %1880), !dbg !6
  %1882 = bitcast i32 %1881 to float, !dbg !6
  %1883 = select i1 %.not, float %1871, float %1882, !dbg !6
  %1884 = select i1 %.not, float %1882, float %1872, !dbg !6
  %1885 = select i1 %.not669, float %1883, float %1877, !dbg !6
  %1886 = bitcast float %1885 to i32, !dbg !6
  %1887 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1886), !dbg !6
  %1888 = select i1 %.not669, float %1884, float %1878, !dbg !6
  %1889 = bitcast float %1888 to i32, !dbg !6
  %1890 = call i32 @llvm.amdgcn.ds.bpermute(i32 %16, i32 %1889), !dbg !6
  %1891 = extractvalue [2 x <1 x float>] %129, 0, !dbg !17
  %1892 = shufflevector <1 x float> %1891, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  %1893 = extractvalue [2 x <1 x float>] %129, 1, !dbg !17
  %1894 = shufflevector <1 x float> %1893, <1 x float> poison, <4 x i32> zeroinitializer, !dbg !17
  br i1 true, label %1895, label %1961, !dbg !19

1895:                                             ; preds = %1456
  %1896 = bitcast i32 %1887 to float, !dbg !6
  %1897 = select i1 %.not669, float %1877, float %1896, !dbg !6
  %1898 = insertelement <4 x float> poison, float %1897, i64 0, !dbg !17
  %1899 = bitcast i32 %1890 to float, !dbg !6
  %1900 = select i1 %.not669, float %1878, float %1899, !dbg !6
  %1901 = insertelement <4 x float> %1898, float %1900, i64 1, !dbg !17
  %1902 = select i1 %.not669, float %1896, float %1883, !dbg !6
  %1903 = insertelement <4 x float> %1901, float %1902, i64 2, !dbg !17
  %1904 = select i1 %.not669, float %1899, float %1884, !dbg !6
  %1905 = insertelement <4 x float> %1903, float %1904, i64 3, !dbg !17
  %1906 = fdiv <4 x float> %1905, %1894, !dbg !17
  %1907 = fptrunc <4 x float> %1906 to <4 x half>, !dbg !18
  %1908 = extractelement <4 x half> %1907, i64 3, !dbg !18
  %1909 = extractelement <4 x half> %1907, i64 2, !dbg !18
  %1910 = extractelement <4 x half> %1907, i64 1, !dbg !18
  %1911 = extractelement <4 x half> %1907, i64 0, !dbg !18
  %1912 = bitcast i32 %1865 to float, !dbg !6
  %1913 = select i1 %.not669, float %1855, float %1912, !dbg !6
  %1914 = insertelement <4 x float> poison, float %1913, i64 0, !dbg !17
  %1915 = bitcast i32 %1868 to float, !dbg !6
  %1916 = select i1 %.not669, float %1856, float %1915, !dbg !6
  %1917 = insertelement <4 x float> %1914, float %1916, i64 1, !dbg !17
  %1918 = select i1 %.not669, float %1912, float %1861, !dbg !6
  %1919 = insertelement <4 x float> %1917, float %1918, i64 2, !dbg !17
  %1920 = select i1 %.not669, float %1915, float %1862, !dbg !6
  %1921 = insertelement <4 x float> %1919, float %1920, i64 3, !dbg !17
  %1922 = fdiv <4 x float> %1921, %1894, !dbg !17
  %1923 = fptrunc <4 x float> %1922 to <4 x half>, !dbg !18
  %1924 = extractelement <4 x half> %1923, i64 3, !dbg !18
  %1925 = extractelement <4 x half> %1923, i64 2, !dbg !18
  %1926 = extractelement <4 x half> %1923, i64 1, !dbg !18
  %1927 = extractelement <4 x half> %1923, i64 0, !dbg !18
  %1928 = bitcast i32 %1842 to float, !dbg !6
  %1929 = select i1 %.not669, float %1832, float %1928, !dbg !6
  %1930 = insertelement <4 x float> poison, float %1929, i64 0, !dbg !17
  %1931 = bitcast i32 %1845 to float, !dbg !6
  %1932 = select i1 %.not669, float %1833, float %1931, !dbg !6
  %1933 = insertelement <4 x float> %1930, float %1932, i64 1, !dbg !17
  %1934 = select i1 %.not669, float %1928, float %1838, !dbg !6
  %1935 = insertelement <4 x float> %1933, float %1934, i64 2, !dbg !17
  %1936 = select i1 %.not669, float %1931, float %1839, !dbg !6
  %1937 = insertelement <4 x float> %1935, float %1936, i64 3, !dbg !17
  %1938 = fdiv <4 x float> %1937, %1892, !dbg !17
  %1939 = fptrunc <4 x float> %1938 to <4 x half>, !dbg !18
  %1940 = extractelement <4 x half> %1939, i64 3, !dbg !18
  %1941 = extractelement <4 x half> %1939, i64 2, !dbg !18
  %1942 = extractelement <4 x half> %1939, i64 1, !dbg !18
  %1943 = extractelement <4 x half> %1939, i64 0, !dbg !18
  %1944 = bitcast i32 %1820 to float, !dbg !6
  %1945 = select i1 %.not669, float %1810, float %1944, !dbg !6
  %1946 = insertelement <4 x float> poison, float %1945, i64 0, !dbg !17
  %1947 = bitcast i32 %1823 to float, !dbg !6
  %1948 = select i1 %.not669, float %1811, float %1947, !dbg !6
  %1949 = insertelement <4 x float> %1946, float %1948, i64 1, !dbg !17
  %1950 = select i1 %.not669, float %1944, float %1816, !dbg !6
  %1951 = insertelement <4 x float> %1949, float %1950, i64 2, !dbg !17
  %1952 = select i1 %.not669, float %1947, float %1817, !dbg !6
  %1953 = insertelement <4 x float> %1951, float %1952, i64 3, !dbg !17
  %1954 = fdiv <4 x float> %1953, %1892, !dbg !17
  %1955 = fptrunc <4 x float> %1954 to <4 x half>, !dbg !18
  %1956 = extractelement <4 x half> %1955, i64 3, !dbg !18
  %1957 = extractelement <4 x half> %1955, i64 2, !dbg !18
  %1958 = extractelement <4 x half> %1955, i64 1, !dbg !18
  %1959 = extractelement <4 x half> %1955, i64 0, !dbg !18
  %1960 = shl nuw nsw i64 %26, 19, !dbg !19
  %address.thread1577 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local1578 = zext nneg i32 %address.thread1577 to i64, !dbg !13
  %address.local1579 = add nsw i64 %27, %address.local1578, !dbg !19
  %address.local1580 = lshr i32 %address.thread1577, 2, !dbg !19
  %address.local1581 = and i32 %address.local1580, 48, !dbg !19
  %address.local1582 = zext nneg i32 %address.local1581 to i64, !dbg !19
  %address.local1583 = add nsw i64 %address.local1579, %address.local1582, !dbg !19
  %address.local1584 = and i32 %address.thread1577, 240, !dbg !19
  %address.local1585 = sub nsw i32 0, %address.local1584, !dbg !19
  %address.local1586 = sext i32 %address.local1585 to i64, !dbg !19
  %address.local1587 = add nsw i64 %address.local1583, %address.local1586, !dbg !19
  %address.local1588 = add nsw i64 %address.local1587, %28, !dbg !19
  %address.local1589 = shl nuw nsw i64 %address.local1588, 7, !dbg !19
  %address.local1590 = add nuw nsw i64 %1960, %address.local1589, !dbg !19
  %address.local1591 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1590, !dbg !19
  %address.local1592 = lshr i32 %address.thread1577, 2, !dbg !19
  %address.local1593 = and i32 %address.local1592, 12, !dbg !19
  %address.local1594 = zext nneg i32 %address.local1593 to i64, !dbg !19
  %address.local1595 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1591, i64 %address.local1594, !dbg !19
  store half %1959, ptr addrspace(1) %address.local1595, align 2, !dbg !19
  %address.local1596 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1590, !dbg !19
  %address.local1597 = lshr i32 %address.thread1577, 2, !dbg !19
  %address.local1598 = and i32 %address.local1597, 12, !dbg !19
  %address.local1599 = or disjoint i32 %address.local1598, 1, !dbg !19
  %address.local1600 = zext nneg i32 %address.local1599 to i64, !dbg !19
  %address.local1601 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1596, i64 %address.local1600, !dbg !19
  store half %1958, ptr addrspace(1) %address.local1601, align 2, !dbg !19
  %address.local1602 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1590, !dbg !19
  %address.local1603 = lshr i32 %address.thread1577, 2, !dbg !19
  %address.local1604 = and i32 %address.local1603, 12, !dbg !19
  %address.local1605 = or disjoint i32 %address.local1604, 2, !dbg !19
  %address.local1606 = zext nneg i32 %address.local1605 to i64, !dbg !19
  %address.local1607 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1602, i64 %address.local1606, !dbg !19
  store half %1957, ptr addrspace(1) %address.local1607, align 2, !dbg !19
  %address.local1608 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1590, !dbg !19
  %address.local1609 = lshr i32 %address.thread1577, 2, !dbg !19
  %address.local1610 = and i32 %address.local1609, 12, !dbg !19
  %address.local1611 = or disjoint i32 %address.local1610, 3, !dbg !19
  %address.local1612 = zext nneg i32 %address.local1611 to i64, !dbg !19
  %address.local1613 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1608, i64 %address.local1612, !dbg !19
  store half %1956, ptr addrspace(1) %address.local1613, align 2, !dbg !19
  %address.local1614 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1590, !dbg !19
  %address.local1615 = lshr i32 %address.thread1577, 2, !dbg !19
  %address.local1616 = and i32 %address.local1615, 12, !dbg !19
  %address.local1617 = or disjoint i32 %address.local1616, 16, !dbg !19
  %address.local1618 = zext nneg i32 %address.local1617 to i64, !dbg !19
  %address.local1619 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1614, i64 %address.local1618, !dbg !19
  store half %1943, ptr addrspace(1) %address.local1619, align 2, !dbg !19
  %address.local1620 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1590, !dbg !19
  %address.local1621 = lshr i32 %address.thread1577, 2, !dbg !19
  %address.local1622 = and i32 %address.local1621, 12, !dbg !19
  %address.local1623 = or disjoint i32 %address.local1622, 17, !dbg !19
  %address.local1624 = zext nneg i32 %address.local1623 to i64, !dbg !19
  %address.local1625 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1620, i64 %address.local1624, !dbg !19
  store half %1942, ptr addrspace(1) %address.local1625, align 2, !dbg !19
  %address.local1626 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1590, !dbg !19
  %address.local1627 = lshr i32 %address.thread1577, 2, !dbg !19
  %address.local1628 = and i32 %address.local1627, 12, !dbg !19
  %address.local1629 = or disjoint i32 %address.local1628, 18, !dbg !19
  %address.local1630 = zext nneg i32 %address.local1629 to i64, !dbg !19
  %address.local1631 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1626, i64 %address.local1630, !dbg !19
  store half %1941, ptr addrspace(1) %address.local1631, align 2, !dbg !19
  %address.local1632 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1590, !dbg !19
  %address.local1633 = lshr i32 %address.thread1577, 2, !dbg !19
  %address.local1634 = and i32 %address.local1633, 12, !dbg !19
  %address.local1635 = or disjoint i32 %address.local1634, 19, !dbg !19
  %address.local1636 = zext nneg i32 %address.local1635 to i64, !dbg !19
  %address.local1637 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1632, i64 %address.local1636, !dbg !19
  store half %1940, ptr addrspace(1) %address.local1637, align 2, !dbg !19
  %address.local1638 = shl nsw i64 %address.local1588, 7, !dbg !19
  %address.local1639 = add i64 %address.local1638, 8192, !dbg !19
  %address.local1640 = add nuw nsw i64 %1960, %address.local1639, !dbg !19
  %address.local1641 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1640, !dbg !19
  %address.local1642 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1641, i64 %address.local1594, !dbg !19
  store half %1927, ptr addrspace(1) %address.local1642, align 2, !dbg !19
  %address.local1643 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1640, !dbg !19
  %address.local1644 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1643, i64 %address.local1600, !dbg !19
  store half %1926, ptr addrspace(1) %address.local1644, align 2, !dbg !19
  %address.local1645 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1640, !dbg !19
  %address.local1646 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1645, i64 %address.local1606, !dbg !19
  store half %1925, ptr addrspace(1) %address.local1646, align 2, !dbg !19
  %address.local1647 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1640, !dbg !19
  %address.local1648 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1647, i64 %address.local1612, !dbg !19
  store half %1924, ptr addrspace(1) %address.local1648, align 2, !dbg !19
  %address.local1649 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1640, !dbg !19
  %address.local1650 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1649, i64 %address.local1618, !dbg !19
  store half %1911, ptr addrspace(1) %address.local1650, align 2, !dbg !19
  %address.local1651 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1640, !dbg !19
  %address.local1652 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1651, i64 %address.local1624, !dbg !19
  store half %1910, ptr addrspace(1) %address.local1652, align 2, !dbg !19
  %address.local1653 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1640, !dbg !19
  %address.local1654 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1653, i64 %address.local1630, !dbg !19
  store half %1909, ptr addrspace(1) %address.local1654, align 2, !dbg !19
  %address.local1655 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1640, !dbg !19
  %address.local1656 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1655, i64 %address.local1636, !dbg !19
  store half %1908, ptr addrspace(1) %address.local1656, align 2, !dbg !19
  br label %1961, !dbg !19

1961:                                             ; preds = %1895, %1456
  br i1 true, label %1962, label %2004, !dbg !19

1962:                                             ; preds = %1961
  %1963 = insertelement <4 x float> poison, float %1793, i64 0, !dbg !17
  %1964 = insertelement <4 x float> %1963, float %1799, i64 1, !dbg !17
  %1965 = insertelement <4 x float> %1964, float %1794, i64 2, !dbg !17
  %1966 = insertelement <4 x float> %1965, float %1800, i64 3, !dbg !17
  %1967 = fdiv <4 x float> %1966, %1894, !dbg !17
  %1968 = fptrunc <4 x float> %1967 to <4 x half>, !dbg !18
  %1969 = extractelement <4 x half> %1968, i64 3, !dbg !18
  %1970 = extractelement <4 x half> %1968, i64 2, !dbg !18
  %1971 = extractelement <4 x half> %1968, i64 1, !dbg !18
  %1972 = extractelement <4 x half> %1968, i64 0, !dbg !18
  %1973 = insertelement <4 x float> poison, float %1765, i64 0, !dbg !17
  %1974 = insertelement <4 x float> %1973, float %1771, i64 1, !dbg !17
  %1975 = insertelement <4 x float> %1974, float %1766, i64 2, !dbg !17
  %1976 = insertelement <4 x float> %1975, float %1772, i64 3, !dbg !17
  %1977 = fdiv <4 x float> %1976, %1894, !dbg !17
  %1978 = fptrunc <4 x float> %1977 to <4 x half>, !dbg !18
  %1979 = extractelement <4 x half> %1978, i64 3, !dbg !18
  %1980 = extractelement <4 x half> %1978, i64 2, !dbg !18
  %1981 = extractelement <4 x half> %1978, i64 1, !dbg !18
  %1982 = extractelement <4 x half> %1978, i64 0, !dbg !18
  %1983 = insertelement <4 x float> poison, float %1736, i64 0, !dbg !17
  %1984 = insertelement <4 x float> %1983, float %1742, i64 1, !dbg !17
  %1985 = insertelement <4 x float> %1984, float %1737, i64 2, !dbg !17
  %1986 = insertelement <4 x float> %1985, float %1743, i64 3, !dbg !17
  %1987 = fdiv <4 x float> %1986, %1892, !dbg !17
  %1988 = fptrunc <4 x float> %1987 to <4 x half>, !dbg !18
  %1989 = extractelement <4 x half> %1988, i64 3, !dbg !18
  %1990 = extractelement <4 x half> %1988, i64 2, !dbg !18
  %1991 = extractelement <4 x half> %1988, i64 1, !dbg !18
  %1992 = extractelement <4 x half> %1988, i64 0, !dbg !18
  %1993 = insertelement <4 x float> poison, float %1708, i64 0, !dbg !17
  %1994 = insertelement <4 x float> %1993, float %1714, i64 1, !dbg !17
  %1995 = insertelement <4 x float> %1994, float %1709, i64 2, !dbg !17
  %1996 = insertelement <4 x float> %1995, float %1715, i64 3, !dbg !17
  %1997 = fdiv <4 x float> %1996, %1892, !dbg !17
  %1998 = fptrunc <4 x float> %1997 to <4 x half>, !dbg !18
  %1999 = extractelement <4 x half> %1998, i64 3, !dbg !18
  %2000 = extractelement <4 x half> %1998, i64 2, !dbg !18
  %2001 = extractelement <4 x half> %1998, i64 1, !dbg !18
  %2002 = extractelement <4 x half> %1998, i64 0, !dbg !18
  %2003 = shl nuw nsw i64 %34, 19, !dbg !19
  %address.thread1657 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local1658 = zext nneg i32 %address.thread1657 to i64, !dbg !13
  %address.local1659 = add nsw i64 %27, %address.local1658, !dbg !19
  %address.local1660 = lshr i32 %address.thread1657, 2, !dbg !19
  %address.local1661 = and i32 %address.local1660, 48, !dbg !19
  %address.local1662 = zext nneg i32 %address.local1661 to i64, !dbg !19
  %address.local1663 = add nsw i64 %address.local1659, %address.local1662, !dbg !19
  %address.local1664 = and i32 %address.thread1657, 240, !dbg !19
  %address.local1665 = sub nsw i32 0, %address.local1664, !dbg !19
  %address.local1666 = sext i32 %address.local1665 to i64, !dbg !19
  %address.local1667 = add nsw i64 %address.local1663, %address.local1666, !dbg !19
  %address.local1668 = add nsw i64 %address.local1667, %28, !dbg !19
  %address.local1669 = shl nuw nsw i64 %address.local1668, 7, !dbg !19
  %address.local1670 = add nuw nsw i64 %2003, %address.local1669, !dbg !19
  %address.local1671 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1670, !dbg !19
  %address.local1672 = lshr i32 %address.thread1657, 2, !dbg !19
  %address.local1673 = and i32 %address.local1672, 12, !dbg !19
  %address.local1674 = or disjoint i32 %address.local1673, 32, !dbg !19
  %address.local1675 = zext nneg i32 %address.local1674 to i64, !dbg !19
  %address.local1676 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1671, i64 %address.local1675, !dbg !19
  store half %2002, ptr addrspace(1) %address.local1676, align 2, !dbg !19
  %address.local1677 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1670, !dbg !19
  %address.local1678 = lshr i32 %address.thread1657, 2, !dbg !19
  %address.local1679 = and i32 %address.local1678, 12, !dbg !19
  %address.local1680 = or disjoint i32 %address.local1679, 33, !dbg !19
  %address.local1681 = zext nneg i32 %address.local1680 to i64, !dbg !19
  %address.local1682 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1677, i64 %address.local1681, !dbg !19
  store half %2001, ptr addrspace(1) %address.local1682, align 2, !dbg !19
  %address.local1683 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1670, !dbg !19
  %address.local1684 = lshr i32 %address.thread1657, 2, !dbg !19
  %address.local1685 = and i32 %address.local1684, 12, !dbg !19
  %address.local1686 = or disjoint i32 %address.local1685, 34, !dbg !19
  %address.local1687 = zext nneg i32 %address.local1686 to i64, !dbg !19
  %address.local1688 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1683, i64 %address.local1687, !dbg !19
  store half %2000, ptr addrspace(1) %address.local1688, align 2, !dbg !19
  %address.local1689 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1670, !dbg !19
  %address.local1690 = lshr i32 %address.thread1657, 2, !dbg !19
  %address.local1691 = and i32 %address.local1690, 12, !dbg !19
  %address.local1692 = or disjoint i32 %address.local1691, 35, !dbg !19
  %address.local1693 = zext nneg i32 %address.local1692 to i64, !dbg !19
  %address.local1694 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1689, i64 %address.local1693, !dbg !19
  store half %1999, ptr addrspace(1) %address.local1694, align 2, !dbg !19
  %address.local1695 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1670, !dbg !19
  %address.local1696 = lshr i32 %address.thread1657, 2, !dbg !19
  %address.local1697 = and i32 %address.local1696, 12, !dbg !19
  %address.local1698 = or disjoint i32 %address.local1697, 48, !dbg !19
  %address.local1699 = zext nneg i32 %address.local1698 to i64, !dbg !19
  %address.local1700 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1695, i64 %address.local1699, !dbg !19
  store half %1992, ptr addrspace(1) %address.local1700, align 2, !dbg !19
  %address.local1701 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1670, !dbg !19
  %address.local1702 = lshr i32 %address.thread1657, 2, !dbg !19
  %address.local1703 = and i32 %address.local1702, 12, !dbg !19
  %address.local1704 = or disjoint i32 %address.local1703, 49, !dbg !19
  %address.local1705 = zext nneg i32 %address.local1704 to i64, !dbg !19
  %address.local1706 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1701, i64 %address.local1705, !dbg !19
  store half %1991, ptr addrspace(1) %address.local1706, align 2, !dbg !19
  %address.local1707 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1670, !dbg !19
  %address.local1708 = lshr i32 %address.thread1657, 2, !dbg !19
  %address.local1709 = and i32 %address.local1708, 12, !dbg !19
  %address.local1710 = or disjoint i32 %address.local1709, 50, !dbg !19
  %address.local1711 = zext nneg i32 %address.local1710 to i64, !dbg !19
  %address.local1712 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1707, i64 %address.local1711, !dbg !19
  store half %1990, ptr addrspace(1) %address.local1712, align 2, !dbg !19
  %address.local1713 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1670, !dbg !19
  %address.local1714 = lshr i32 %address.thread1657, 2, !dbg !19
  %address.local1715 = or i32 %address.local1714, 51, !dbg !19
  %address.local1716 = zext nneg i32 %address.local1715 to i64, !dbg !19
  %address.local1717 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1713, i64 %address.local1716, !dbg !19
  store half %1989, ptr addrspace(1) %address.local1717, align 2, !dbg !19
  %address.local1718 = shl nsw i64 %address.local1668, 7, !dbg !19
  %address.local1719 = add i64 %address.local1718, 8192, !dbg !19
  %address.local1720 = add nuw nsw i64 %2003, %address.local1719, !dbg !19
  %address.local1721 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1720, !dbg !19
  %address.local1722 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1721, i64 %address.local1675, !dbg !19
  store half %1982, ptr addrspace(1) %address.local1722, align 2, !dbg !19
  %address.local1723 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1720, !dbg !19
  %address.local1724 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1723, i64 %address.local1681, !dbg !19
  store half %1981, ptr addrspace(1) %address.local1724, align 2, !dbg !19
  %address.local1725 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1720, !dbg !19
  %address.local1726 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1725, i64 %address.local1687, !dbg !19
  store half %1980, ptr addrspace(1) %address.local1726, align 2, !dbg !19
  %address.local1727 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1720, !dbg !19
  %address.local1728 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1727, i64 %address.local1693, !dbg !19
  store half %1979, ptr addrspace(1) %address.local1728, align 2, !dbg !19
  %address.local1729 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1720, !dbg !19
  %address.local1730 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1729, i64 %address.local1699, !dbg !19
  store half %1972, ptr addrspace(1) %address.local1730, align 2, !dbg !19
  %address.local1731 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1720, !dbg !19
  %address.local1732 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1731, i64 %address.local1705, !dbg !19
  store half %1971, ptr addrspace(1) %address.local1732, align 2, !dbg !19
  %address.local1733 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1720, !dbg !19
  %address.local1734 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1733, i64 %address.local1711, !dbg !19
  store half %1970, ptr addrspace(1) %address.local1734, align 2, !dbg !19
  %address.local1735 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1720, !dbg !19
  %address.local1736 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1735, i64 %address.local1716, !dbg !19
  store half %1969, ptr addrspace(1) %address.local1736, align 2, !dbg !19
  br label %2004, !dbg !19

2004:                                             ; preds = %1962, %1961
  br i1 true, label %2005, label %2047, !dbg !19

2005:                                             ; preds = %2004
  %2006 = insertelement <4 x float> poison, float %1679, i64 0, !dbg !17
  %2007 = insertelement <4 x float> %2006, float %1685, i64 1, !dbg !17
  %2008 = insertelement <4 x float> %2007, float %1680, i64 2, !dbg !17
  %2009 = insertelement <4 x float> %2008, float %1686, i64 3, !dbg !17
  %2010 = fdiv <4 x float> %2009, %1894, !dbg !17
  %2011 = fptrunc <4 x float> %2010 to <4 x half>, !dbg !18
  %2012 = extractelement <4 x half> %2011, i64 3, !dbg !18
  %2013 = extractelement <4 x half> %2011, i64 2, !dbg !18
  %2014 = extractelement <4 x half> %2011, i64 1, !dbg !18
  %2015 = extractelement <4 x half> %2011, i64 0, !dbg !18
  %2016 = insertelement <4 x float> poison, float %1651, i64 0, !dbg !17
  %2017 = insertelement <4 x float> %2016, float %1657, i64 1, !dbg !17
  %2018 = insertelement <4 x float> %2017, float %1652, i64 2, !dbg !17
  %2019 = insertelement <4 x float> %2018, float %1658, i64 3, !dbg !17
  %2020 = fdiv <4 x float> %2019, %1894, !dbg !17
  %2021 = fptrunc <4 x float> %2020 to <4 x half>, !dbg !18
  %2022 = extractelement <4 x half> %2021, i64 3, !dbg !18
  %2023 = extractelement <4 x half> %2021, i64 2, !dbg !18
  %2024 = extractelement <4 x half> %2021, i64 1, !dbg !18
  %2025 = extractelement <4 x half> %2021, i64 0, !dbg !18
  %2026 = insertelement <4 x float> poison, float %1622, i64 0, !dbg !17
  %2027 = insertelement <4 x float> %2026, float %1628, i64 1, !dbg !17
  %2028 = insertelement <4 x float> %2027, float %1623, i64 2, !dbg !17
  %2029 = insertelement <4 x float> %2028, float %1629, i64 3, !dbg !17
  %2030 = fdiv <4 x float> %2029, %1892, !dbg !17
  %2031 = fptrunc <4 x float> %2030 to <4 x half>, !dbg !18
  %2032 = extractelement <4 x half> %2031, i64 3, !dbg !18
  %2033 = extractelement <4 x half> %2031, i64 2, !dbg !18
  %2034 = extractelement <4 x half> %2031, i64 1, !dbg !18
  %2035 = extractelement <4 x half> %2031, i64 0, !dbg !18
  %2036 = insertelement <4 x float> poison, float %1594, i64 0, !dbg !17
  %2037 = insertelement <4 x float> %2036, float %1600, i64 1, !dbg !17
  %2038 = insertelement <4 x float> %2037, float %1595, i64 2, !dbg !17
  %2039 = insertelement <4 x float> %2038, float %1601, i64 3, !dbg !17
  %2040 = fdiv <4 x float> %2039, %1892, !dbg !17
  %2041 = fptrunc <4 x float> %2040 to <4 x half>, !dbg !18
  %2042 = extractelement <4 x half> %2041, i64 3, !dbg !18
  %2043 = extractelement <4 x half> %2041, i64 2, !dbg !18
  %2044 = extractelement <4 x half> %2041, i64 1, !dbg !18
  %2045 = extractelement <4 x half> %2041, i64 0, !dbg !18
  %2046 = shl nuw nsw i64 %34, 19, !dbg !19
  %address.thread1737 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local1738 = zext nneg i32 %address.thread1737 to i64, !dbg !13
  %address.local1739 = add nsw i64 %27, %address.local1738, !dbg !19
  %address.local1740 = lshr i32 %address.thread1737, 2, !dbg !19
  %address.local1741 = and i32 %address.local1740, 48, !dbg !19
  %address.local1742 = zext nneg i32 %address.local1741 to i64, !dbg !19
  %address.local1743 = add nsw i64 %address.local1739, %address.local1742, !dbg !19
  %address.local1744 = and i32 %address.thread1737, 240, !dbg !19
  %address.local1745 = sub nsw i32 0, %address.local1744, !dbg !19
  %address.local1746 = sext i32 %address.local1745 to i64, !dbg !19
  %address.local1747 = add nsw i64 %address.local1743, %address.local1746, !dbg !19
  %address.local1748 = add nsw i64 %address.local1747, %28, !dbg !19
  %address.local1749 = shl nuw nsw i64 %address.local1748, 7, !dbg !19
  %address.local1750 = add nuw nsw i64 %2046, %address.local1749, !dbg !19
  %address.local1751 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1750, !dbg !19
  %address.local1752 = lshr i32 %address.thread1737, 2, !dbg !19
  %address.local1753 = and i32 %address.local1752, 12, !dbg !19
  %address.local1754 = or disjoint i32 %address.local1753, 64, !dbg !19
  %address.local1755 = zext nneg i32 %address.local1754 to i64, !dbg !19
  %address.local1756 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1751, i64 %address.local1755, !dbg !19
  store half %2045, ptr addrspace(1) %address.local1756, align 2, !dbg !19
  %address.local1757 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1750, !dbg !19
  %address.local1758 = lshr i32 %address.thread1737, 2, !dbg !19
  %address.local1759 = and i32 %address.local1758, 12, !dbg !19
  %address.local1760 = or disjoint i32 %address.local1759, 65, !dbg !19
  %address.local1761 = zext nneg i32 %address.local1760 to i64, !dbg !19
  %address.local1762 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1757, i64 %address.local1761, !dbg !19
  store half %2044, ptr addrspace(1) %address.local1762, align 2, !dbg !19
  %address.local1763 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1750, !dbg !19
  %address.local1764 = lshr i32 %address.thread1737, 2, !dbg !19
  %address.local1765 = and i32 %address.local1764, 12, !dbg !19
  %address.local1766 = or disjoint i32 %address.local1765, 66, !dbg !19
  %address.local1767 = zext nneg i32 %address.local1766 to i64, !dbg !19
  %address.local1768 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1763, i64 %address.local1767, !dbg !19
  store half %2043, ptr addrspace(1) %address.local1768, align 2, !dbg !19
  %address.local1769 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1750, !dbg !19
  %address.local1770 = lshr i32 %address.thread1737, 2, !dbg !19
  %address.local1771 = and i32 %address.local1770, 12, !dbg !19
  %address.local1772 = or disjoint i32 %address.local1771, 67, !dbg !19
  %address.local1773 = zext nneg i32 %address.local1772 to i64, !dbg !19
  %address.local1774 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1769, i64 %address.local1773, !dbg !19
  store half %2042, ptr addrspace(1) %address.local1774, align 2, !dbg !19
  %address.local1775 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1750, !dbg !19
  %address.local1776 = lshr i32 %address.thread1737, 2, !dbg !19
  %address.local1777 = and i32 %address.local1776, 12, !dbg !19
  %address.local1778 = or disjoint i32 %address.local1777, 80, !dbg !19
  %address.local1779 = zext nneg i32 %address.local1778 to i64, !dbg !19
  %address.local1780 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1775, i64 %address.local1779, !dbg !19
  store half %2035, ptr addrspace(1) %address.local1780, align 2, !dbg !19
  %address.local1781 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1750, !dbg !19
  %address.local1782 = lshr i32 %address.thread1737, 2, !dbg !19
  %address.local1783 = and i32 %address.local1782, 12, !dbg !19
  %address.local1784 = or disjoint i32 %address.local1783, 81, !dbg !19
  %address.local1785 = zext nneg i32 %address.local1784 to i64, !dbg !19
  %address.local1786 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1781, i64 %address.local1785, !dbg !19
  store half %2034, ptr addrspace(1) %address.local1786, align 2, !dbg !19
  %address.local1787 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1750, !dbg !19
  %address.local1788 = lshr i32 %address.thread1737, 2, !dbg !19
  %address.local1789 = and i32 %address.local1788, 12, !dbg !19
  %address.local1790 = or disjoint i32 %address.local1789, 82, !dbg !19
  %address.local1791 = zext nneg i32 %address.local1790 to i64, !dbg !19
  %address.local1792 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1787, i64 %address.local1791, !dbg !19
  store half %2033, ptr addrspace(1) %address.local1792, align 2, !dbg !19
  %address.local1793 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1750, !dbg !19
  %address.local1794 = lshr i32 %address.thread1737, 2, !dbg !19
  %address.local1795 = and i32 %address.local1794, 12, !dbg !19
  %address.local1796 = or disjoint i32 %address.local1795, 83, !dbg !19
  %address.local1797 = zext nneg i32 %address.local1796 to i64, !dbg !19
  %address.local1798 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1793, i64 %address.local1797, !dbg !19
  store half %2032, ptr addrspace(1) %address.local1798, align 2, !dbg !19
  %address.local1799 = shl nsw i64 %address.local1748, 7, !dbg !19
  %address.local1800 = add i64 %address.local1799, 8192, !dbg !19
  %address.local1801 = add nuw nsw i64 %2046, %address.local1800, !dbg !19
  %address.local1802 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1801, !dbg !19
  %address.local1803 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1802, i64 %address.local1755, !dbg !19
  store half %2025, ptr addrspace(1) %address.local1803, align 2, !dbg !19
  %address.local1804 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1801, !dbg !19
  %address.local1805 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1804, i64 %address.local1761, !dbg !19
  store half %2024, ptr addrspace(1) %address.local1805, align 2, !dbg !19
  %address.local1806 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1801, !dbg !19
  %address.local1807 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1806, i64 %address.local1767, !dbg !19
  store half %2023, ptr addrspace(1) %address.local1807, align 2, !dbg !19
  %address.local1808 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1801, !dbg !19
  %address.local1809 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1808, i64 %address.local1773, !dbg !19
  store half %2022, ptr addrspace(1) %address.local1809, align 2, !dbg !19
  %address.local1810 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1801, !dbg !19
  %address.local1811 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1810, i64 %address.local1779, !dbg !19
  store half %2015, ptr addrspace(1) %address.local1811, align 2, !dbg !19
  %address.local1812 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1801, !dbg !19
  %address.local1813 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1812, i64 %address.local1785, !dbg !19
  store half %2014, ptr addrspace(1) %address.local1813, align 2, !dbg !19
  %address.local1814 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1801, !dbg !19
  %address.local1815 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1814, i64 %address.local1791, !dbg !19
  store half %2013, ptr addrspace(1) %address.local1815, align 2, !dbg !19
  %address.local1816 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1801, !dbg !19
  %address.local1817 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1816, i64 %address.local1797, !dbg !19
  store half %2012, ptr addrspace(1) %address.local1817, align 2, !dbg !19
  br label %2047, !dbg !19

2047:                                             ; preds = %2005, %2004
  br i1 true, label %2048, label %2090, !dbg !19

2048:                                             ; preds = %2047
  %2049 = insertelement <4 x float> poison, float %1565, i64 0, !dbg !17
  %2050 = insertelement <4 x float> %2049, float %1571, i64 1, !dbg !17
  %2051 = insertelement <4 x float> %2050, float %1566, i64 2, !dbg !17
  %2052 = insertelement <4 x float> %2051, float %1572, i64 3, !dbg !17
  %2053 = fdiv <4 x float> %2052, %1894, !dbg !17
  %2054 = fptrunc <4 x float> %2053 to <4 x half>, !dbg !18
  %2055 = extractelement <4 x half> %2054, i64 3, !dbg !18
  %2056 = extractelement <4 x half> %2054, i64 2, !dbg !18
  %2057 = extractelement <4 x half> %2054, i64 1, !dbg !18
  %2058 = extractelement <4 x half> %2054, i64 0, !dbg !18
  %2059 = insertelement <4 x float> poison, float %1537, i64 0, !dbg !17
  %2060 = insertelement <4 x float> %2059, float %1543, i64 1, !dbg !17
  %2061 = insertelement <4 x float> %2060, float %1538, i64 2, !dbg !17
  %2062 = insertelement <4 x float> %2061, float %1544, i64 3, !dbg !17
  %2063 = fdiv <4 x float> %2062, %1894, !dbg !17
  %2064 = fptrunc <4 x float> %2063 to <4 x half>, !dbg !18
  %2065 = extractelement <4 x half> %2064, i64 3, !dbg !18
  %2066 = extractelement <4 x half> %2064, i64 2, !dbg !18
  %2067 = extractelement <4 x half> %2064, i64 1, !dbg !18
  %2068 = extractelement <4 x half> %2064, i64 0, !dbg !18
  %2069 = insertelement <4 x float> poison, float %1508, i64 0, !dbg !17
  %2070 = insertelement <4 x float> %2069, float %1514, i64 1, !dbg !17
  %2071 = insertelement <4 x float> %2070, float %1509, i64 2, !dbg !17
  %2072 = insertelement <4 x float> %2071, float %1515, i64 3, !dbg !17
  %2073 = fdiv <4 x float> %2072, %1892, !dbg !17
  %2074 = fptrunc <4 x float> %2073 to <4 x half>, !dbg !18
  %2075 = extractelement <4 x half> %2074, i64 3, !dbg !18
  %2076 = extractelement <4 x half> %2074, i64 2, !dbg !18
  %2077 = extractelement <4 x half> %2074, i64 1, !dbg !18
  %2078 = extractelement <4 x half> %2074, i64 0, !dbg !18
  %2079 = insertelement <4 x float> poison, float %1480, i64 0, !dbg !17
  %2080 = insertelement <4 x float> %2079, float %1486, i64 1, !dbg !17
  %2081 = insertelement <4 x float> %2080, float %1481, i64 2, !dbg !17
  %2082 = insertelement <4 x float> %2081, float %1487, i64 3, !dbg !17
  %2083 = fdiv <4 x float> %2082, %1892, !dbg !17
  %2084 = fptrunc <4 x float> %2083 to <4 x half>, !dbg !18
  %2085 = extractelement <4 x half> %2084, i64 3, !dbg !18
  %2086 = extractelement <4 x half> %2084, i64 2, !dbg !18
  %2087 = extractelement <4 x half> %2084, i64 1, !dbg !18
  %2088 = extractelement <4 x half> %2084, i64 0, !dbg !18
  %2089 = shl nuw nsw i64 %34, 19, !dbg !19
  %address.thread1818 = call i32 asm sideeffect "", "=v,0"(i32 %31) #9, !dbg !19, !range !32, !frisk.register_lifetimes !1
  %address.local1819 = zext nneg i32 %address.thread1818 to i64, !dbg !13
  %address.local1820 = add nsw i64 %27, %address.local1819, !dbg !19
  %address.local1821 = lshr i32 %address.thread1818, 2, !dbg !19
  %address.local1822 = and i32 %address.local1821, 48, !dbg !19
  %address.local1823 = zext nneg i32 %address.local1822 to i64, !dbg !19
  %address.local1824 = add nsw i64 %address.local1820, %address.local1823, !dbg !19
  %address.local1825 = and i32 %address.thread1818, 240, !dbg !19
  %address.local1826 = sub nsw i32 0, %address.local1825, !dbg !19
  %address.local1827 = sext i32 %address.local1826 to i64, !dbg !19
  %address.local1828 = add nsw i64 %address.local1824, %address.local1827, !dbg !19
  %address.local1829 = add nsw i64 %address.local1828, %28, !dbg !19
  %address.local1830 = shl nuw nsw i64 %address.local1829, 7, !dbg !19
  %address.local1831 = add nuw nsw i64 %2089, %address.local1830, !dbg !19
  %address.local1832 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1831, !dbg !19
  %address.local1833 = lshr i32 %address.thread1818, 2, !dbg !19
  %address.local1834 = and i32 %address.local1833, 12, !dbg !19
  %address.local1835 = or disjoint i32 %address.local1834, 96, !dbg !19
  %address.local1836 = zext nneg i32 %address.local1835 to i64, !dbg !19
  %address.local1837 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1832, i64 %address.local1836, !dbg !19
  store half %2088, ptr addrspace(1) %address.local1837, align 2, !dbg !19
  %address.local1838 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1831, !dbg !19
  %address.local1839 = lshr i32 %address.thread1818, 2, !dbg !19
  %address.local1840 = and i32 %address.local1839, 12, !dbg !19
  %address.local1841 = or disjoint i32 %address.local1840, 97, !dbg !19
  %address.local1842 = zext nneg i32 %address.local1841 to i64, !dbg !19
  %address.local1843 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1838, i64 %address.local1842, !dbg !19
  store half %2087, ptr addrspace(1) %address.local1843, align 2, !dbg !19
  %address.local1844 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1831, !dbg !19
  %address.local1845 = lshr i32 %address.thread1818, 2, !dbg !19
  %address.local1846 = and i32 %address.local1845, 12, !dbg !19
  %address.local1847 = or disjoint i32 %address.local1846, 98, !dbg !19
  %address.local1848 = zext nneg i32 %address.local1847 to i64, !dbg !19
  %address.local1849 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1844, i64 %address.local1848, !dbg !19
  store half %2086, ptr addrspace(1) %address.local1849, align 2, !dbg !19
  %address.local1850 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1831, !dbg !19
  %address.local1851 = lshr i32 %address.thread1818, 2, !dbg !19
  %address.local1852 = and i32 %address.local1851, 12, !dbg !19
  %address.local1853 = or disjoint i32 %address.local1852, 99, !dbg !19
  %address.local1854 = zext nneg i32 %address.local1853 to i64, !dbg !19
  %address.local1855 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1850, i64 %address.local1854, !dbg !19
  store half %2085, ptr addrspace(1) %address.local1855, align 2, !dbg !19
  %address.local1856 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1831, !dbg !19
  %address.local1857 = lshr i32 %address.thread1818, 2, !dbg !19
  %address.local1858 = and i32 %address.local1857, 12, !dbg !19
  %address.local1859 = or disjoint i32 %address.local1858, 112, !dbg !19
  %address.local1860 = zext nneg i32 %address.local1859 to i64, !dbg !19
  %address.local1861 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1856, i64 %address.local1860, !dbg !19
  store half %2078, ptr addrspace(1) %address.local1861, align 2, !dbg !19
  %address.local1862 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1831, !dbg !19
  %address.local1863 = lshr i32 %address.thread1818, 2, !dbg !19
  %address.local1864 = and i32 %address.local1863, 12, !dbg !19
  %address.local1865 = or disjoint i32 %address.local1864, 113, !dbg !19
  %address.local1866 = zext nneg i32 %address.local1865 to i64, !dbg !19
  %address.local1867 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1862, i64 %address.local1866, !dbg !19
  store half %2077, ptr addrspace(1) %address.local1867, align 2, !dbg !19
  %address.local1868 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1831, !dbg !19
  %address.local1869 = lshr i32 %address.thread1818, 2, !dbg !19
  %address.local1870 = and i32 %address.local1869, 12, !dbg !19
  %address.local1871 = or disjoint i32 %address.local1870, 114, !dbg !19
  %address.local1872 = zext nneg i32 %address.local1871 to i64, !dbg !19
  %address.local1873 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1868, i64 %address.local1872, !dbg !19
  store half %2076, ptr addrspace(1) %address.local1873, align 2, !dbg !19
  %address.local1874 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1831, !dbg !19
  %address.local1875 = lshr i32 %address.thread1818, 2, !dbg !19
  %address.local1876 = or i32 %address.local1875, 115, !dbg !19
  %address.local1877 = zext nneg i32 %address.local1876 to i64, !dbg !19
  %address.local1878 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1874, i64 %address.local1877, !dbg !19
  store half %2075, ptr addrspace(1) %address.local1878, align 2, !dbg !19
  %address.local1879 = shl nsw i64 %address.local1829, 7, !dbg !19
  %address.local1880 = add i64 %address.local1879, 8192, !dbg !19
  %address.local1881 = add nuw nsw i64 %2089, %address.local1880, !dbg !19
  %address.local1882 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1881, !dbg !19
  %address.local1883 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1882, i64 %address.local1836, !dbg !19
  store half %2068, ptr addrspace(1) %address.local1883, align 2, !dbg !19
  %address.local1884 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1881, !dbg !19
  %address.local1885 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1884, i64 %address.local1842, !dbg !19
  store half %2067, ptr addrspace(1) %address.local1885, align 2, !dbg !19
  %address.local1886 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1881, !dbg !19
  %address.local1887 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1886, i64 %address.local1848, !dbg !19
  store half %2066, ptr addrspace(1) %address.local1887, align 2, !dbg !19
  %address.local1888 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1881, !dbg !19
  %address.local1889 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1888, i64 %address.local1854, !dbg !19
  store half %2065, ptr addrspace(1) %address.local1889, align 2, !dbg !19
  %address.local1890 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1881, !dbg !19
  %address.local1891 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1890, i64 %address.local1860, !dbg !19
  store half %2058, ptr addrspace(1) %address.local1891, align 2, !dbg !19
  %address.local1892 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1881, !dbg !19
  %address.local1893 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1892, i64 %address.local1866, !dbg !19
  store half %2057, ptr addrspace(1) %address.local1893, align 2, !dbg !19
  %address.local1894 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1881, !dbg !19
  %address.local1895 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1894, i64 %address.local1872, !dbg !19
  store half %2056, ptr addrspace(1) %address.local1895, align 2, !dbg !19
  %address.local1896 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %3, i64 %address.local1881, !dbg !19
  %address.local1897 = getelementptr inbounds nuw [2 x i8], ptr addrspace(1) %address.local1896, i64 %address.local1877, !dbg !19
  store half %2055, ptr addrspace(1) %address.local1897, align 2, !dbg !19
  br label %2090, !dbg !19

2090:                                             ; preds = %2048, %2047
  ret void, !dbg !5
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.z() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.y() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.amdgcn.workgroup.id.x() #3

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #3

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #5

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) captures(none), ptr addrspace(3) captures(none), i32 immarg, i32 immarg, i32 immarg) #6

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

attributes #0 = { convergent willreturn memory(read) }
attributes #1 = { "amdgpu-flat-work-group-size"="256,256" "amdgpu-waves-per-eu"="1" "frisk.register_comparisons"="1" "frisk.register_lifetimes"="1" "uniform-work-group-size" }
attributes #2 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #6 = { nocallback nofree nounwind willreturn }
attributes #7 = { convergent nocallback nofree nounwind willreturn }
attributes #8 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #9 = { convergent nounwind willreturn memory(none) }
attributes #10 = { convergent nounwind memory(none) }
attributes #11 = { convergent nounwind }

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!31}

!0 = !DIFile(filename: "input.mlir", directory: "/data2/xsl/DeepGenGraph/generated_ir/thread_approximate_math")
!1 = !{}
!2 = !DISubroutineType(types: !1)
!3 = distinct !DICompileUnit(language: DW_LANG_C, file: !0, producer: "DeepGenGraph MLIR", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!4 = distinct !DISubprogram(name: "Attn_p2", linkageName: "Attn_p2", scope: !0, file: !0, line: 23, type: !2, scopeLine: 23, spFlags: DISPFlagDefinition, unit: !3)
!5 = !DILocation(line: 23, column: 3, scope: !4)
!6 = !DILocation(line: 57, column: 24, scope: !4)
!8 = !DILocation(line: 74, column: 20, scope: !4)
!11 = !DILocation(line: 67, column: 22, scope: !4)
!12 = !DILocation(line: 68, column: 22, scope: !4)
!13 = !DILocation(line: 69, column: 24, scope: !4)
!14 = !DILocation(line: 87, column: 23, scope: !4)
!15 = !DILocation(line: 86, column: 20, scope: !4)
!16 = !DILocation(line: 70, column: 26, scope: !4)
!17 = !DILocation(line: 92, column: 23, scope: !4)
!18 = !DILocation(line: 93, column: 19, scope: !4)
!19 = !DILocation(line: 94, column: 9, scope: !4)
!20 = !DILocation(line: 71, column: 23, scope: !4)
!21 = !DILocation(line: 75, column: 22, scope: !4)
!22 = !DILocation(line: 76, column: 23, scope: !4)
!23 = !DILocation(line: 77, column: 20, scope: !4)
!24 = !DILocation(line: 78, column: 22, scope: !4)
!25 = !DILocation(line: 79, column: 22, scope: !4)
!26 = !DILocation(line: 80, column: 20, scope: !4)
!27 = !DILocation(line: 81, column: 16, scope: !4)
!28 = !DILocation(line: 88, column: 22, scope: !4)
!29 = !DILocation(line: 89, column: 20, scope: !4)
!30 = !{i32 256, i32 1, i32 1}
!31 = !{i32 2, !"Debug Info Version", i32 3}
!32 = !{i32 0, i32 256}
