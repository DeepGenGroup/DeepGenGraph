target triple = "amdgcn-amd-amdhsa"
@lds = addrspace(3) global [8192 x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1), ptr addrspace(3), i32, i32, i32)
declare void @llvm.amdgcn.asyncmark()
declare void @llvm.amdgcn.wait.asyncmark(i16)
define amdgpu_kernel void @dma_offset_probe(ptr addrspace(1) %q) "amdgpu-flat-work-group-size"="256,256" {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %row = lshr i32 %tid, 1
  %rb = shl i32 %row, 8
  %pair = and i32 %tid, 1
  %pb = shl i32 %pair, 2
  %src = or i32 %rb, %pb
  %base = getelementptr i8, ptr addrspace(1) %q, i32 %src
  %wave = and i32 %tid, 192
  %wb = shl i32 %wave, 2
  %src0 = getelementptr i8, ptr addrspace(1) %base, i64 0
  %dst0 = add i32 %wb, 0
  %lds0 = getelementptr i8, ptr addrspace(3) @lds, i32 %dst0
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %src0, ptr addrspace(3) %lds0, i32 4, i32 0, i32 0)
  %src1 = getelementptr i8, ptr addrspace(1) %base, i64 8
  %dst1 = add i32 %wb, 1024
  %lds1 = getelementptr i8, ptr addrspace(3) @lds, i32 %dst1
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %src1, ptr addrspace(3) %lds1, i32 4, i32 0, i32 0)
  %src2 = getelementptr i8, ptr addrspace(1) %base, i64 16
  %dst2 = add i32 %wb, 2048
  %lds2 = getelementptr i8, ptr addrspace(3) @lds, i32 %dst2
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %src2, ptr addrspace(3) %lds2, i32 4, i32 0, i32 0)
  %src3 = getelementptr i8, ptr addrspace(1) %base, i64 24
  %dst3 = add i32 %wb, 3072
  %lds3 = getelementptr i8, ptr addrspace(3) @lds, i32 %dst3
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %src3, ptr addrspace(3) %lds3, i32 4, i32 0, i32 0)
  %src4 = getelementptr i8, ptr addrspace(1) %base, i64 32
  %dst4 = add i32 %wb, 4096
  %lds4 = getelementptr i8, ptr addrspace(3) @lds, i32 %dst4
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %src4, ptr addrspace(3) %lds4, i32 4, i32 0, i32 0)
  %src5 = getelementptr i8, ptr addrspace(1) %base, i64 40
  %dst5 = add i32 %wb, 5120
  %lds5 = getelementptr i8, ptr addrspace(3) @lds, i32 %dst5
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %src5, ptr addrspace(3) %lds5, i32 4, i32 0, i32 0)
  %src6 = getelementptr i8, ptr addrspace(1) %base, i64 48
  %dst6 = add i32 %wb, 6144
  %lds6 = getelementptr i8, ptr addrspace(3) @lds, i32 %dst6
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %src6, ptr addrspace(3) %lds6, i32 4, i32 0, i32 0)
  %src7 = getelementptr i8, ptr addrspace(1) %base, i64 56
  %dst7 = add i32 %wb, 7168
  %lds7 = getelementptr i8, ptr addrspace(3) @lds, i32 %dst7
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %src7, ptr addrspace(3) %lds7, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  ret void
}
