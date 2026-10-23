target triple = "amdgcn-amd-amdhsa"
@lds = addrspace(3) global [1024 x i8] undef, align 16
declare i32 @llvm.amdgcn.workitem.id.x()
declare void @llvm.amdgcn.s.barrier()
define amdgpu_kernel void @dma_asm_probe(<4 x i32> %rsrc) #0 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %offset = mul i32 %tid, 16
  %lds = ptrtoint ptr addrspace(3) @lds to i32
  %m0 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %lds, i32 %offset, <4 x i32> %rsrc) #1
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #1
  call void @llvm.amdgcn.s.barrier()
  ret void
}
attributes #0 = { "amdgpu-flat-work-group-size"="64,64" }
attributes #1 = { convergent nounwind }
