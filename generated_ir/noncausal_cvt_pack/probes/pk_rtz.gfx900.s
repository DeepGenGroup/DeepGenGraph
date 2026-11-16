	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	probe                           ; -- Begin function probe
	.p2align	6
	.type	probe,@function
probe:                                  ; @probe
; %bb.0:
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	;;#ASMSTART
	v_cvt_pkrtz_f16_f32 v0, v0, v1
	;;#ASMEND
	s_setpc_b64 s[30:31]
.Lfunc_end0:
	.size	probe, .Lfunc_end0-probe
                                        ; -- End function
	.set .Lprobe.num_vgpr, 2
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
; codeLenInByte = 16
; TotalNumSgprs: 36
; NumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 2
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:  []
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
