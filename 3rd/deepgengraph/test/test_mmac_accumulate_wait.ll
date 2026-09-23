; Timing-control variant: eight s_nop 7 instructions before and after each MMAC.
; Only instruction spacing differs from test_mmac_accumulate.ll.
; The padding is deliberately conservative for diagnosis, not a measured ISA latency.
; DCU diagnostic independent of Frisk layouts, shared memory and K loops.
; Use the existing HsacoLauncher and GenHsaco.sh with this explicit input path.
; Expected f16 output, all dimensions:
;   seqLen=0:   one MMA,        16 * half(0.3)      = 4.80078125
;   seqLen=64:  dependent MMA,  32 * half(0.3)      = 9.6015625
;   seqLen=128: seeded MMA,    10 + 16 * half(0.3) = 14.796875
; Other blocks repeat the dependent-MMA result. This is a diagnostic kernel,
; not attention. Its four pointer arguments match the existing launcher ABI.

declare i32 @llvm.amdgcn.workitem.id.x()
declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workgroup.id.y()

define amdgpu_kernel void @Attn_p2(ptr addrspace(1) %q,
                                 ptr addrspace(1) %v,
                                 ptr addrspace(1) %k,
                                 ptr addrspace(1) %out) #0 !reqd_work_group_size !0 {
entry:
  %once = call <4 x float> asm sideeffect "s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09v_mmac_f32_16x16x16_f16 $0, $2, $1, $3\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7", "=v,v,v,0"(
      <4 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>,
      <4 x half> <half 0xH34CD, half 0xH34CD, half 0xH34CD, half 0xH34CD>,
      <4 x float> zeroinitializer)
  %twice = call <4 x float> asm sideeffect "s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09v_mmac_f32_16x16x16_f16 $0, $2, $1, $3\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7", "=v,v,v,0"(
      <4 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>,
      <4 x half> <half 0xH34CD, half 0xH34CD, half 0xH34CD, half 0xH34CD>,
      <4 x float> %once)
  %seeded = call <4 x float> asm sideeffect "s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09v_mmac_f32_16x16x16_f16 $0, $2, $1, $3\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7\0A\09s_nop 7", "=v,v,v,0"(
      <4 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>,
      <4 x half> <half 0xH34CD, half 0xH34CD, half 0xH34CD, half 0xH34CD>,
      <4 x float> <float 10.0, float 10.0, float 10.0, float 10.0>)
  %by = call i32 @llvm.amdgcn.workgroup.id.y()
  %bx = call i32 @llvm.amdgcn.workgroup.id.x()
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %first = icmp eq i32 %by, 0
  %third = icmp eq i32 %by, 2
  %base_result = select i1 %first, <4 x float> %once, <4 x float> %twice
  %result = select i1 %third, <4 x float> %seeded, <4 x float> %base_result
  %head_offset = mul i32 %bx, 524288
  %block_offset = mul i32 %by, 8192
  %thread_offset = mul i32 %tid, 64
  %base0 = add i32 %head_offset, %block_offset
  %base = add i32 %base0, %thread_offset
  br label %loop
loop:
  %i = phi i32 [ 0, %entry ], [ %next, %loop ]
  %element = and i32 %i, 3
  %value = extractelement <4 x float> %result, i32 %element
  %half_value = fptrunc float %value to half
  %offset = add i32 %base, %i
  %ptr = getelementptr half, ptr addrspace(1) %out, i32 %offset
  store half %half_value, ptr addrspace(1) %ptr, align 2
  %next = add i32 %i, 1
  %more = icmp ult i32 %next, 64
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="128,128" }
!0 = !{i32 128, i32 1, i32 1}
