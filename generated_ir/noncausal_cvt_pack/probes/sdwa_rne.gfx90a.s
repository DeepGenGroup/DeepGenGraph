	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx90a"
	.amdhsa_code_object_version 6
	.text
	.globl	probe                           ; -- Begin function probe
	.p2align	6
	.type	probe,@function
probe:                                  ; @probe
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	;;#ASMSTART
	v_cvt_f16_f32 v2, v0
	v_cvt_f16_f32_sdwa v2, v1 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
	;;#ASMEND
	v_mov_b32_e32 v0, v2
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	probe, .Lfunc_end0-probe
                                        ; -- End function
	.set .Lprobe.num_vgpr, 3
	.set .Lprobe.num_agpr, 0
	.set .Lprobe.numbered_sgpr, 32
	.set .Lprobe.num_named_barrier, 0
	.set .Lprobe.private_seg_size, 0
	.set .Lprobe.uses_vcc, 0
	.set .Lprobe.uses_flat_scratch, 0
	.set .Lprobe.has_dyn_sized_stack, 0
	.set .Lprobe.has_recursion, 0
	.set .Lprobe.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 28
; TotalNumSgprs: 36
; NumVgprs: 3
; NumAgprs: 0
; TotalNumVgprs: 3
; ScratchSize: 0
; MemoryBound: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 3
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx90a
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
