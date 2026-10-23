; ModuleID = 'generated_ir/frag_prefetch/portable.off.ll'
source_filename = "<stdin>"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: mustprogress nounwind willreturn memory(argmem: readwrite)
define amdgpu_kernel void @test(ptr addrspace(3) nofree readonly captures(none) %p, ptr nofree writeonly captures(none) %out, i64 %index) local_unnamed_addr #0 {
  %1 = addrspacecast ptr %out to ptr addrspace(1)
  %pb0 = getelementptr i8, ptr addrspace(3) %p, i32 8
  %a0 = load <4 x half>, ptr addrspace(3) %p, align 8
  %b0 = load <4 x half>, ptr addrspace(3) %pb0, align 8
  %r0 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3", "=v,v,v,0"(<4 x half> %a0, <4 x half> %b0, <4 x float> zeroinitializer) #1
  %pa1 = getelementptr i8, ptr addrspace(3) %p, i32 16
  %pb1 = getelementptr i8, ptr addrspace(3) %p, i32 24
  %a1 = load <4 x half>, ptr addrspace(3) %pa1, align 8
  %b1 = load <4 x half>, ptr addrspace(3) %pb1, align 8
  %r1 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3", "=v,v,v,0"(<4 x half> %a1, <4 x half> %b1, <4 x float> %r0) #1
  %pa2 = getelementptr i8, ptr addrspace(3) %p, i32 32
  %pb2 = getelementptr i8, ptr addrspace(3) %p, i32 40
  %a2 = load <4 x half>, ptr addrspace(3) %pa2, align 8
  %b2 = load <4 x half>, ptr addrspace(3) %pb2, align 8
  %r2 = tail call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3", "=v,v,v,0"(<4 x half> %a2, <4 x half> %b2, <4 x float> %r1) #1
  store <4 x float> %r2, ptr addrspace(1) %1, align 16
  ret void
}

attributes #0 = { mustprogress nounwind willreturn memory(argmem: readwrite) "amdgpu-no-cluster-id-x" "amdgpu-no-cluster-id-y" "amdgpu-no-cluster-id-z" "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "amdgpu-no-wwm" }
attributes #1 = { convergent nounwind willreturn memory(none) }
