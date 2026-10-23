; ModuleID = '<stdin>'
source_filename = "<stdin>"
target triple = "amdgcn-amd-amdhsa"

@shm_0 = addrspace(3) global [4352 x i8] undef, align 16

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.asyncmark() #1

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.wait.asyncmark(i16 immarg) #1

; Function Attrs: convergent nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #2

define amdgpu_kernel void @copy_probe(ptr addrspace(1) %0, ptr addrspace(1) %1, ptr addrspace(1) %2, ptr addrspace(1) %3) #3 {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %active = icmp ult i32 %tid, 256
  br i1 %active, label %copy, label %done

copy:                                             ; preds = %4
  %permuted = xor i32 %tid, 37
  %source.offset = mul i32 %permuted, 32
  %source.plus = add i32 %source.offset, 16
  %src = getelementptr i8, ptr addrspace(1) %0, i32 %source.plus
  %wave = lshr i32 %tid, 6
  %base = mul i32 %wave, 1024
  %padded = add i32 %base, 128
  %dst = getelementptr i8, ptr addrspace(3) @shm_0, i32 %padded
  %5 = ptrtoint ptr addrspace(1) %0 to i64
  %6 = ptrtoint ptr addrspace(1) %src to i64
  %7 = sub i64 %6, %5
  %dma.buffer.offset = trunc i64 %7 to i32
  %8 = trunc i64 %5 to i32
  %9 = insertelement <4 x i32> poison, i32 %8, i64 0
  %10 = lshr i64 %5, 32
  %11 = trunc i64 %10 to i32
  %12 = insertelement <4 x i32> %9, i32 %11, i64 1
  %13 = insertelement <4 x i32> %12, i32 33554432, i64 2
  %dma.buffer.rsrc = insertelement <4 x i32> %13, i32 131072, i64 3
  %14 = ptrtoint ptr addrspace(3) %dst to i32
  %dma.buffer.lds = call i32 @llvm.amdgcn.readfirstlane.i32(i32 %14)
  %15 = call i32 asm sideeffect "s_mov_b32 $0, $1\0A\09buffer_load_dwordx4 $2, $3, 0 offen lds", "={m0},s,v,s,~{memory}"(i32 %dma.buffer.lds, i32 %dma.buffer.offset, <4 x i32> %dma.buffer.rsrc) #5
  br label %done

done:                                             ; preds = %copy, %4
  call void asm sideeffect "s_waitcnt vmcnt(0)", "~{memory}"() #5
  call void @llvm.amdgcn.s.barrier()
  br i1 %active, label %readback, label %exit

readback:                                         ; preds = %done
  %out.offset = mul i32 %tid, 16
  %read.offset = add i32 %out.offset, 128
  %read = getelementptr i8, ptr addrspace(3) @shm_0, i32 %read.offset
  %v = load <8 x half>, ptr addrspace(3) %read, align 16
  %out = getelementptr i8, ptr addrspace(1) %3, i32 %out.offset
  store <8 x half> %v, ptr addrspace(1) %out, align 16
  br label %exit

exit:                                             ; preds = %readback, %done
  ret void
}

; Function Attrs: convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.readfirstlane.i32(i32) #4

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nocallback nofree nounwind willreturn }
attributes #2 = { convergent nocallback nofree nounwind willreturn }
attributes #3 = { "amdgpu-flat-work-group-size"="256,256" }
attributes #4 = { convergent nocallback nocreateundeforpoison nofree nounwind willreturn memory(none) }
attributes #5 = { convergent nounwind }
