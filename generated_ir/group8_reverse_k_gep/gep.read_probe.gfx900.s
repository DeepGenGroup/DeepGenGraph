	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	probe                           ; -- Begin function probe
	.p2align	8
	.type	probe,@function
probe:                                  ; @probe
; %bb.0:
	v_mov_b32_e32 v1, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v2, 1, v1
	v_lshlrev_b32_e32 v3, 2, v1
	v_lshlrev_b32_e32 v4, 6, v1
	v_and_b32_e32 v2, 8, v2
	v_and_b32_e32 v3, 48, v3
	v_and_b32_e32 v4, 0x800, v4
	v_or3_b32 v2, v2, v3, v4
	v_lshlrev_b32_e32 v1, 8, v1
	v_lshlrev_b32_e32 v3, 6, v0
	v_and_b32_e32 v1, 0x300, v1
	v_and_b32_e32 v3, 64, v3
	v_or3_b32 v5, v2, v1, v3
	v_or_b32_e32 v1, 0x2000, v5
	v_xor_b32_e32 v6, 64, v1
	v_add_u32_e32 v15, 0x2000, v5
	v_add_u32_e32 v5, 0x3000, v5
	ds_read2_b64 v[1:4], v15 offset1:16
	ds_read2_b64 v[7:10], v5 offset1:16
	ds_read2_b64 v[11:14], v6 offset1:16
	v_add_u32_e32 v16, 0x1000, v6
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[17:20], v16 offset1:16
	ds_read2_b64 v[21:24], v15 offset0:128 offset1:144
	ds_read2_b64 v[27:30], v5 offset0:128 offset1:144
	ds_read2_b64 v[31:34], v6 offset0:128 offset1:144
	ds_read2_b64 v[37:40], v16 offset0:128 offset1:144
	v_lshlrev_b32_e32 v0, 7, v0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v6, v2
	v_mov_b32_e32 v15, v11
	v_mov_b32_e32 v16, v12
	v_mov_b32_e32 v25, v21
	v_mov_b32_e32 v26, v22
	v_mov_b32_e32 v35, v31
	v_mov_b32_e32 v36, v32
	global_store_dwordx4 v0, v[5:8], s[0:1]
	global_store_dwordx4 v0, v[15:18], s[0:1] offset:16
	v_mov_b32_e32 v5, v9
	v_mov_b32_e32 v6, v10
	v_mov_b32_e32 v15, v19
	v_mov_b32_e32 v16, v20
	global_store_dwordx4 v0, v[25:28], s[0:1] offset:64
	global_store_dwordx4 v0, v[35:38], s[0:1] offset:80
	v_mov_b32_e32 v25, v29
	v_mov_b32_e32 v26, v30
	v_mov_b32_e32 v35, v39
	v_mov_b32_e32 v36, v40
	global_store_dwordx4 v0, v[3:6], s[0:1] offset:32
	global_store_dwordx4 v0, v[13:16], s[0:1] offset:48
	global_store_dwordx4 v0, v[23:26], s[0:1] offset:96
	global_store_dwordx4 v0, v[33:36], s[0:1] offset:112
	s_endpgm
.Lfunc_end0:
	.size	probe, .Lfunc_end0-probe
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel probe
		.amdhsa_group_segment_fixed_size 16384
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 49
		.amdhsa_next_free_sgpr 98
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lprobe.num_vgpr, 41
	.set .Lprobe.num_agpr, 0
	.set .Lprobe.numbered_sgpr, 6
	.set .Lprobe.num_named_barrier, 0
	.set .Lprobe.private_seg_size, 0
	.set .Lprobe.uses_vcc, 0
	.set .Lprobe.uses_flat_scratch, 0
	.set .Lprobe.has_dyn_sized_stack, 0
	.set .Lprobe.has_recursion, 0
	.set .Lprobe.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 316
; TotalNumSgprs: 10
; NumVgprs: 41
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 16384 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 49
; Occupancy: 4
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .name:           out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           probe
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         probe.kd
    .uses_dynamic_stack: false
    .vgpr_count:     41
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgpu-amd-amdhsa-unknown-gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
