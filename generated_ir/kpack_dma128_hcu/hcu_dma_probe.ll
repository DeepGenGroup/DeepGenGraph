; ModuleID = '<stdin>'
source_filename = "<stdin>"
target triple = "amdgcn-amd-amdhsa"

@shared = addrspace(3) global [1024 x i8] undef, align 16

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #0

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #0

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #1

define amdgpu_kernel void @hcu_dma_probe(ptr addrspace(1) %src) {
  %dst = getelementptr i8, ptr addrspace(3) @shared, i32 0
  %dma.source.bits = ptrtoint ptr addrspace(1) %src to i64
  %dma.source.opaque = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #3
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1)
  call void @llvm.hcu.global.load.dwordx4.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %dst, i32 0, i32 0, i32 0) #4
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
  call void @llvm.amdgcn.s.barrier()
  ret void
}

; Function Attrs: convergent nounwind willreturn
; Unknown intrinsic
declare void @llvm.hcu.global.load.dwordx4.lds(ptr addrspace(1), ptr addrspace(3), i32, i32, i32) #2

attributes #0 = { nocallback nofree nounwind willreturn }
attributes #1 = { convergent nocallback nofree nounwind willreturn }
attributes #2 = { convergent nounwind willreturn }
attributes #3 = { convergent nounwind memory(none) }
attributes #4 = { convergent }
