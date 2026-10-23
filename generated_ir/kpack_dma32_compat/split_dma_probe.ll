; ModuleID = '<stdin>'
source_filename = "<stdin>"
target triple = "amdgcn-amd-amdhsa"

@shm_0 = addrspace(3) global [4352 x i8] undef, align 16

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.amdgcn.workitem.id.x() #0

; Function Attrs: nocallback nofree nounwind willreturn
declare void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) captures(none), ptr addrspace(3) captures(none), i32 immarg, i32 immarg, i32 immarg) #1

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
  %5 = call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)
  %dma.lane = call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 %5)
  %6 = ptrtoint ptr addrspace(1) %src to i64
  %7 = trunc i64 %6 to i32
  %8 = lshr i64 %6, 32
  %9 = trunc i64 %8 to i32
  %10 = and i32 %dma.lane, 3
  %11 = shl i32 %10, 2
  %12 = and i32 %dma.lane, 60
  %13 = add i32 %12, 0
  %14 = call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %7)
  %15 = call i32 @llvm.amdgcn.ds.bpermute(i32 %13, i32 %9)
  %16 = zext i32 %15 to i64
  %17 = shl i64 %16, 32
  %18 = zext i32 %14 to i64
  %19 = or i64 %18, %17
  %20 = zext i32 %11 to i64
  %21 = add i64 %19, %20
  %22 = inttoptr i64 %21 to ptr addrspace(1)
  %23 = getelementptr i8, ptr addrspace(3) %dst, i32 0
  %dma.source.bits = ptrtoint ptr addrspace(1) %22 to i64
  %dma.source.opaque = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits) #6
  %dma.source.ptr = inttoptr i64 %dma.source.opaque to ptr addrspace(1)
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr, ptr addrspace(3) %23, i32 4, i32 0, i32 0)
  %24 = and i32 %dma.lane, 60
  %25 = add i32 %24, 64
  %26 = call i32 @llvm.amdgcn.ds.bpermute(i32 %25, i32 %7)
  %27 = call i32 @llvm.amdgcn.ds.bpermute(i32 %25, i32 %9)
  %28 = zext i32 %27 to i64
  %29 = shl i64 %28, 32
  %30 = zext i32 %26 to i64
  %31 = or i64 %30, %29
  %32 = zext i32 %11 to i64
  %33 = add i64 %31, %32
  %34 = inttoptr i64 %33 to ptr addrspace(1)
  %35 = getelementptr i8, ptr addrspace(3) %dst, i32 256
  %dma.source.bits1 = ptrtoint ptr addrspace(1) %34 to i64
  %dma.source.opaque2 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits1) #6
  %dma.source.ptr3 = inttoptr i64 %dma.source.opaque2 to ptr addrspace(1)
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr3, ptr addrspace(3) %35, i32 4, i32 0, i32 0)
  %36 = and i32 %dma.lane, 60
  %37 = add i32 %36, 128
  %38 = call i32 @llvm.amdgcn.ds.bpermute(i32 %37, i32 %7)
  %39 = call i32 @llvm.amdgcn.ds.bpermute(i32 %37, i32 %9)
  %40 = zext i32 %39 to i64
  %41 = shl i64 %40, 32
  %42 = zext i32 %38 to i64
  %43 = or i64 %42, %41
  %44 = zext i32 %11 to i64
  %45 = add i64 %43, %44
  %46 = inttoptr i64 %45 to ptr addrspace(1)
  %47 = getelementptr i8, ptr addrspace(3) %dst, i32 512
  %dma.source.bits4 = ptrtoint ptr addrspace(1) %46 to i64
  %dma.source.opaque5 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits4) #6
  %dma.source.ptr6 = inttoptr i64 %dma.source.opaque5 to ptr addrspace(1)
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr6, ptr addrspace(3) %47, i32 4, i32 0, i32 0)
  %48 = and i32 %dma.lane, 60
  %49 = add i32 %48, 192
  %50 = call i32 @llvm.amdgcn.ds.bpermute(i32 %49, i32 %7)
  %51 = call i32 @llvm.amdgcn.ds.bpermute(i32 %49, i32 %9)
  %52 = zext i32 %51 to i64
  %53 = shl i64 %52, 32
  %54 = zext i32 %50 to i64
  %55 = or i64 %54, %53
  %56 = zext i32 %11 to i64
  %57 = add i64 %55, %56
  %58 = inttoptr i64 %57 to ptr addrspace(1)
  %59 = getelementptr i8, ptr addrspace(3) %dst, i32 768
  %dma.source.bits7 = ptrtoint ptr addrspace(1) %58 to i64
  %dma.source.opaque8 = call i64 asm sideeffect "", "=v,0"(i64 %dma.source.bits7) #6
  %dma.source.ptr9 = inttoptr i64 %dma.source.opaque8 to ptr addrspace(1)
  call void @llvm.amdgcn.global.load.async.lds(ptr addrspace(1) %dma.source.ptr9, ptr addrspace(3) %59, i32 4, i32 0, i32 0)
  br label %done

done:                                             ; preds = %copy, %4
  call void @llvm.amdgcn.asyncmark()
  call void @llvm.amdgcn.wait.asyncmark(i16 0)
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

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32) #4

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32) #4

; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
declare i32 @llvm.amdgcn.ds.bpermute(i32, i32) #5

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { nocallback nofree nounwind willreturn }
attributes #2 = { convergent nocallback nofree nounwind willreturn }
attributes #3 = { "amdgpu-flat-work-group-size"="256,256" }
attributes #4 = { nocallback nocreateundeforpoison nofree nosync nounwind willreturn memory(none) }
attributes #5 = { convergent nocallback nofree nounwind willreturn memory(none) }
attributes #6 = { convergent nounwind memory(none) }
