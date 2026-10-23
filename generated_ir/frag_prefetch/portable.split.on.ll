; ModuleID = '<stdin>'
source_filename = "<stdin>"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: memory(read)
declare void @opaque(ptr addrspace(3)) #0

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #1

; Function Attrs: convergent memory(read)
; Unknown intrinsic
declare <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3(ptr addrspace(3), i32) #2

define amdgpu_kernel void @test(ptr addrspace(3) %p, ptr %out, i64 %index, ptr addrspace(3) %q) #3 {
  %pa0 = getelementptr half, ptr addrspace(3) %p, i64 0
  %pb0 = getelementptr half, ptr addrspace(3) %q, i64 4
  %a0 = load <4 x half>, ptr addrspace(3) %pa0, align 8
  %b0 = load <4 x half>, ptr addrspace(3) %pb0, align 8
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %pa1 = getelementptr half, ptr addrspace(3) %p, i64 8
  %pb1 = getelementptr half, ptr addrspace(3) %q, i64 12
  %a1 = load <4 x half>, ptr addrspace(3) %pa1, align 8
  %b1 = load <4 x half>, ptr addrspace(3) %pb1, align 8
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %r0 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3", "=v,v,v,0"(<4 x half> %a0, <4 x half> %b0, <4 x float> zeroinitializer) #4
  call void @llvm.amdgcn.sched.barrier(i32 0)
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %pa2 = getelementptr half, ptr addrspace(3) %p, i64 16
  %pb2 = getelementptr half, ptr addrspace(3) %q, i64 20
  %a2 = load <4 x half>, ptr addrspace(3) %pa2, align 8
  %b2 = load <4 x half>, ptr addrspace(3) %pb2, align 8
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %r1 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3", "=v,v,v,0"(<4 x half> %a1, <4 x half> %b1, <4 x float> %r0) #4
  call void @llvm.amdgcn.sched.barrier(i32 0)
  %r2 = call <4 x float> asm "v_mfma_f32_16x16x16f16 $0, $1, $2, $3", "=v,v,v,0"(<4 x half> %a2, <4 x half> %b2, <4 x float> %r1) #4
  store <4 x float> %r2, ptr %out, align 16
  ret void
}

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.sched.barrier(i32 immarg) #1

attributes #0 = { memory(read) }
attributes #1 = { convergent nocallback nofree nounwind willreturn }
attributes #2 = { convergent memory(read) }
attributes #3 = { "frisk.frag_prefetch"="8" }
attributes #4 = { convergent nounwind willreturn memory(none) }
