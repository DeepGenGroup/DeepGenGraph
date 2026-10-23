; ModuleID = '/data2/xsl/DeepGenGraph/generated_ir/group8_reverse_k_gep/control.read_probe.ll'
source_filename = "/data2/xsl/DeepGenGraph/generated_ir/group8_reverse_k_gep/control.read_probe.ll"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@shm_0 = addrspace(3) global [16384 x i8] undef, align 16

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nounwind memory(read, argmem: readwrite, inaccessiblemem: none, target_mem: none)
define amdgpu_kernel void @probe(ptr addrspace(1) nofree writeonly captures(none) %out) local_unnamed_addr #1 {
  %probe.ssa24 = tail call i32 @llvm.amdgcn.workitem.id.x()
  %address.thread = tail call i32 asm sideeffect "", "=v,0"(i32 %probe.ssa24) #2
  %address.local179 = lshr i32 %address.thread, 2
  %address.local180 = and i32 %address.local179, 4
  %address.local181 = shl i32 %address.thread, 1
  %address.local182 = and i32 %address.local181, 24
  %address.local184 = shl nuw nsw i32 %address.thread, 5
  %address.local185 = and i32 %address.local184, 1024
  %address.local187 = shl nuw nsw i32 %address.thread, 7
  %address.local188 = and i32 %address.local187, 384
  %1 = getelementptr [2 x i8], ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i32 8192), i32 %address.local180
  %2 = getelementptr [2 x i8], ptr addrspace(3) %1, i32 %address.local182
  %3 = getelementptr [2 x i8], ptr addrspace(3) %2, i32 %address.local185
  %address.local191 = getelementptr [2 x i8], ptr addrspace(3) %3, i32 %address.local188
  %ksw.read0.address = ptrtoint ptr addrspace(3) %address.local191 to i32
  %ksw.read.bit = shl nuw nsw i32 %probe.ssa24, 6
  %ksw.read.mask = and i32 %ksw.read.bit, 64
  %ksw.read0.xor = xor i32 %ksw.read.mask, %ksw.read0.address
  %ksw.read0.pointer = inttoptr i32 %ksw.read0.xor to ptr addrspace(3)
  %address.local205 = getelementptr i8, ptr addrspace(3) %address.local191, i32 4096
  %ksw.read1.address = ptrtoint ptr addrspace(3) %address.local205 to i32
  %ksw.read1.xor = xor i32 %ksw.read.mask, %ksw.read1.address
  %ksw.read1.pointer = inttoptr i32 %ksw.read1.xor to ptr addrspace(3)
  %address.local207.idx = shl i32 %address.thread, 8
  %address.local207 = getelementptr i8, ptr addrspace(3) getelementptr inbounds nuw (i8, ptr addrspace(3) @shm_0, i32 8192), i32 %address.local207.idx
  %address.local208 = lshr i32 %address.thread, 2
  %4 = mul i32 %address.local208, -512
  %address.local210 = getelementptr [2 x i8], ptr addrspace(3) %address.local207, i32 %4
  %address.local213 = getelementptr [2 x i8], ptr addrspace(3) %address.local210, i32 %address.local182
  %5 = and i32 %address.local208, 12
  %address.local216 = getelementptr [2 x i8], ptr addrspace(3) %address.local213, i32 %5
  %6 = and i32 %address.thread, 32
  %.not = icmp eq i32 %6, 0
  %7 = select i1 %.not, i32 0, i32 1016
  %address.local220 = getelementptr [2 x i8], ptr addrspace(3) %address.local216, i32 %7
  %address.local221 = getelementptr i8, ptr addrspace(3) %address.local220, i32 64
  %ksw.read2.address = ptrtoint ptr addrspace(3) %address.local221 to i32
  %ksw.read2.xor = xor i32 %ksw.read.mask, %ksw.read2.address
  %ksw.read2.pointer = inttoptr i32 %ksw.read2.xor to ptr addrspace(3)
  %address.local223 = getelementptr [2 x i8], ptr addrspace(3) %address.local207, i32 %5
  %address.local224 = getelementptr [2 x i8], ptr addrspace(3) %address.local223, i32 %7
  %address.local225 = getelementptr [2 x i8], ptr addrspace(3) %address.local224, i32 %4
  %address.local226 = getelementptr [2 x i8], ptr addrspace(3) %address.local225, i32 %address.local182
  %address.local227 = getelementptr i8, ptr addrspace(3) %address.local226, i32 4160
  %ksw.read3.address = ptrtoint ptr addrspace(3) %address.local227 to i32
  %ksw.read3.xor = xor i32 %ksw.read.mask, %ksw.read3.address
  %ksw.read3.pointer = inttoptr i32 %ksw.read3.xor to ptr addrspace(3)
  %address.local233 = getelementptr i8, ptr addrspace(3) %address.local220, i32 128
  %ksw.read4.address = ptrtoint ptr addrspace(3) %address.local233 to i32
  %ksw.read4.xor = xor i32 %ksw.read.mask, %ksw.read4.address
  %ksw.read4.pointer = inttoptr i32 %ksw.read4.xor to ptr addrspace(3)
  %address.local239 = getelementptr i8, ptr addrspace(3) %address.local226, i32 4224
  %ksw.read5.address = ptrtoint ptr addrspace(3) %address.local239 to i32
  %ksw.read5.xor = xor i32 %ksw.read.mask, %ksw.read5.address
  %ksw.read5.pointer = inttoptr i32 %ksw.read5.xor to ptr addrspace(3)
  %address.local245 = getelementptr i8, ptr addrspace(3) %address.local220, i32 192
  %ksw.read6.address = ptrtoint ptr addrspace(3) %address.local245 to i32
  %ksw.read6.xor = xor i32 %ksw.read.mask, %ksw.read6.address
  %ksw.read6.pointer = inttoptr i32 %ksw.read6.xor to ptr addrspace(3)
  %address.local251 = getelementptr i8, ptr addrspace(3) %address.local226, i32 4288
  %ksw.read7.address = ptrtoint ptr addrspace(3) %address.local251 to i32
  %ksw.read7.xor = xor i32 %ksw.read.mask, %ksw.read7.address
  %ksw.read7.pointer = inttoptr i32 %ksw.read7.xor to ptr addrspace(3)
  %address.local253 = getelementptr i8, ptr addrspace(3) %address.local191, i32 1024
  %ksw.read8.address = ptrtoint ptr addrspace(3) %address.local253 to i32
  %ksw.read8.xor = xor i32 %ksw.read.mask, %ksw.read8.address
  %ksw.read8.pointer = inttoptr i32 %ksw.read8.xor to ptr addrspace(3)
  %address.local255 = getelementptr i8, ptr addrspace(3) %address.local191, i32 5120
  %ksw.read9.address = ptrtoint ptr addrspace(3) %address.local255 to i32
  %ksw.read9.xor = xor i32 %ksw.read.mask, %ksw.read9.address
  %ksw.read9.pointer = inttoptr i32 %ksw.read9.xor to ptr addrspace(3)
  %address.local257 = getelementptr i8, ptr addrspace(3) %address.local207, i32 1024
  %address.local258 = getelementptr [2 x i8], ptr addrspace(3) %address.local257, i32 %4
  %address.local259 = getelementptr [2 x i8], ptr addrspace(3) %address.local258, i32 %address.local182
  %address.local260 = getelementptr [2 x i8], ptr addrspace(3) %address.local259, i32 %5
  %address.local261 = getelementptr [2 x i8], ptr addrspace(3) %address.local260, i32 %7
  %address.local262 = getelementptr i8, ptr addrspace(3) %address.local261, i32 64
  %ksw.read10.address = ptrtoint ptr addrspace(3) %address.local262 to i32
  %ksw.read10.xor = xor i32 %ksw.read.mask, %ksw.read10.address
  %ksw.read10.pointer = inttoptr i32 %ksw.read10.xor to ptr addrspace(3)
  %address.local264 = getelementptr [2 x i8], ptr addrspace(3) %address.local257, i32 %5
  %address.local265 = getelementptr [2 x i8], ptr addrspace(3) %address.local264, i32 %7
  %address.local266 = getelementptr [2 x i8], ptr addrspace(3) %address.local265, i32 %4
  %address.local267 = getelementptr [2 x i8], ptr addrspace(3) %address.local266, i32 %address.local182
  %address.local268 = getelementptr i8, ptr addrspace(3) %address.local267, i32 4160
  %ksw.read11.address = ptrtoint ptr addrspace(3) %address.local268 to i32
  %ksw.read11.xor = xor i32 %ksw.read.mask, %ksw.read11.address
  %ksw.read11.pointer = inttoptr i32 %ksw.read11.xor to ptr addrspace(3)
  %address.local275 = getelementptr i8, ptr addrspace(3) %address.local261, i32 128
  %ksw.read12.address = ptrtoint ptr addrspace(3) %address.local275 to i32
  %ksw.read12.xor = xor i32 %ksw.read.mask, %ksw.read12.address
  %ksw.read12.pointer = inttoptr i32 %ksw.read12.xor to ptr addrspace(3)
  %address.local281 = getelementptr i8, ptr addrspace(3) %address.local267, i32 4224
  %ksw.read13.address = ptrtoint ptr addrspace(3) %address.local281 to i32
  %ksw.read13.xor = xor i32 %ksw.read.mask, %ksw.read13.address
  %ksw.read13.pointer = inttoptr i32 %ksw.read13.xor to ptr addrspace(3)
  %address.local288 = getelementptr i8, ptr addrspace(3) %address.local261, i32 192
  %ksw.read14.address = ptrtoint ptr addrspace(3) %address.local288 to i32
  %ksw.read14.xor = xor i32 %ksw.read.mask, %ksw.read14.address
  %ksw.read14.pointer = inttoptr i32 %ksw.read14.xor to ptr addrspace(3)
  %address.local294 = getelementptr i8, ptr addrspace(3) %address.local267, i32 4288
  %ksw.read15.address = ptrtoint ptr addrspace(3) %address.local294 to i32
  %ksw.read15.xor = xor i32 %ksw.read.mask, %ksw.read15.address
  %ksw.read15.pointer = inttoptr i32 %ksw.read15.xor to ptr addrspace(3)
  %probe.value0 = load <4 x half>, ptr addrspace(3) %ksw.read0.pointer, align 2
  %probe.value1 = load <4 x half>, ptr addrspace(3) %ksw.read1.pointer, align 2
  %probe.value2 = load <4 x half>, ptr addrspace(3) %ksw.read2.pointer, align 2
  %probe.value3 = load <4 x half>, ptr addrspace(3) %ksw.read3.pointer, align 2
  %probe.value4 = load <4 x half>, ptr addrspace(3) %ksw.read4.pointer, align 2
  %probe.value5 = load <4 x half>, ptr addrspace(3) %ksw.read5.pointer, align 2
  %probe.value6 = load <4 x half>, ptr addrspace(3) %ksw.read6.pointer, align 2
  %probe.value7 = load <4 x half>, ptr addrspace(3) %ksw.read7.pointer, align 2
  %probe.value8 = load <4 x half>, ptr addrspace(3) %ksw.read8.pointer, align 2
  %probe.value9 = load <4 x half>, ptr addrspace(3) %ksw.read9.pointer, align 2
  %probe.value10 = load <4 x half>, ptr addrspace(3) %ksw.read10.pointer, align 2
  %probe.value11 = load <4 x half>, ptr addrspace(3) %ksw.read11.pointer, align 2
  %probe.value12 = load <4 x half>, ptr addrspace(3) %ksw.read12.pointer, align 2
  %probe.value13 = load <4 x half>, ptr addrspace(3) %ksw.read13.pointer, align 2
  %probe.value14 = load <4 x half>, ptr addrspace(3) %ksw.read14.pointer, align 2
  %probe.value15 = load <4 x half>, ptr addrspace(3) %ksw.read15.pointer, align 2
  %probe.output.offset = shl nuw nsw i32 %probe.ssa24, 7
  %8 = zext nneg i32 %probe.output.offset to i64
  %probe.output.base = getelementptr i8, ptr addrspace(1) %out, i64 %8
  store <4 x half> %probe.value0, ptr addrspace(1) %probe.output.base, align 8
  %probe.out1 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 8
  store <4 x half> %probe.value1, ptr addrspace(1) %probe.out1, align 8
  %probe.out2 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 16
  store <4 x half> %probe.value2, ptr addrspace(1) %probe.out2, align 8
  %probe.out3 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 24
  store <4 x half> %probe.value3, ptr addrspace(1) %probe.out3, align 8
  %probe.out4 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 32
  store <4 x half> %probe.value4, ptr addrspace(1) %probe.out4, align 8
  %probe.out5 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 40
  store <4 x half> %probe.value5, ptr addrspace(1) %probe.out5, align 8
  %probe.out6 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 48
  store <4 x half> %probe.value6, ptr addrspace(1) %probe.out6, align 8
  %probe.out7 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 56
  store <4 x half> %probe.value7, ptr addrspace(1) %probe.out7, align 8
  %probe.out8 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 64
  store <4 x half> %probe.value8, ptr addrspace(1) %probe.out8, align 8
  %probe.out9 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 72
  store <4 x half> %probe.value9, ptr addrspace(1) %probe.out9, align 8
  %probe.out10 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 80
  store <4 x half> %probe.value10, ptr addrspace(1) %probe.out10, align 8
  %probe.out11 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 88
  store <4 x half> %probe.value11, ptr addrspace(1) %probe.out11, align 8
  %probe.out12 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 96
  store <4 x half> %probe.value12, ptr addrspace(1) %probe.out12, align 8
  %probe.out13 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 104
  store <4 x half> %probe.value13, ptr addrspace(1) %probe.out13, align 8
  %probe.out14 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 112
  store <4 x half> %probe.value14, ptr addrspace(1) %probe.out14, align 8
  %probe.out15 = getelementptr i8, ptr addrspace(1) %probe.output.base, i64 120
  store <4 x half> %probe.value15, ptr addrspace(1) %probe.out15, align 8
  ret void
}

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nounwind memory(read, argmem: readwrite, inaccessiblemem: none, target_mem: none) "amdgpu-flat-work-group-size"="256,256" "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" }
attributes #2 = { convergent nounwind memory(none) }
