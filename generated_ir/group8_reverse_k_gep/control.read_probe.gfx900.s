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
	v_lshlrev_b32_e32 v4, 2, v1
	v_lshlrev_b32_e32 v5, 6, v1
	v_and_b32_e32 v3, 8, v2
	v_and_b32_e32 v4, 48, v4
	v_and_b32_e32 v5, 0x800, v5
	v_or3_b32 v3, v3, v4, v5
	v_lshlrev_b32_e32 v5, 8, v1
	v_lshlrev_b32_e32 v7, 6, v0
	v_and_b32_e32 v6, 0x300, v5
	v_and_b32_e32 v7, 64, v7
	v_bfe_i32 v1, v1, 5, 1
	v_or3_b32 v8, v3, v6, v7
	v_or_b32_e32 v6, v6, v4
	v_and_b32_e32 v2, 24, v2
	v_and_b32_e32 v1, 0x7f0, v1
	v_add3_u32 v6, v6, v2, v1
	v_or_b32_e32 v2, v5, v2
	v_and_b32_e32 v3, 0xfffffc00, v5
	v_add_u32_e32 v1, v2, v1
	v_sub_u32_e32 v1, v1, v3
	v_add_u32_e32 v1, v1, v4
	v_add_u32_e32 v2, 0x3040, v1
	v_xor_b32_e32 v5, v7, v2
	v_add_u32_e32 v2, 0x2080, v6
	v_xor_b32_e32 v15, v7, v2
	v_add_u32_e32 v2, 0x3080, v1
	v_xor_b32_e32 v17, v7, v2
	v_add_u32_e32 v2, 0x20c0, v6
	v_xor_b32_e32 v19, v7, v2
	v_add_u32_e32 v2, 0x30c0, v1
	v_xor_b32_e32 v21, v7, v2
	v_add_u32_e32 v2, 0x2440, v6
	v_xor_b32_e32 v23, v7, v2
	v_add_u32_e32 v2, 0x3440, v1
	v_xor_b32_e32 v25, v7, v2
	v_add_u32_e32 v2, 0x2480, v6
	v_add_u32_e32 v9, 0x2040, v6
	v_xor_b32_e32 v27, v7, v2
	v_add_u32_e32 v2, 0x3480, v1
	v_add_u32_e32 v1, 0x34c0, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	v_xor_b32_e32 v11, v7, v9
	v_xor_b32_e32 v29, v7, v2
	v_add_u32_e32 v2, 0x24c0, v6
	v_xor_b32_e32 v33, v7, v1
	v_xor_b32_e32 v6, v7, v2
	ds_read2st64_b64 v[1:4], v8 offset0:16 offset1:18
	ds_read2st64_b64 v[7:10], v8 offset0:24 offset1:26
	ds_read_b64 v[11:12], v11
	ds_read_b64 v[13:14], v5
	ds_read_b64 v[15:16], v15
	ds_read_b64 v[17:18], v17
	ds_read_b64 v[19:20], v19
	ds_read_b64 v[21:22], v21
	ds_read_b64 v[23:24], v23
	ds_read_b64 v[25:26], v25
	ds_read_b64 v[27:28], v27
	ds_read_b64 v[29:30], v29
	ds_read_b64 v[31:32], v6
	ds_read_b64 v[33:34], v33
	v_lshlrev_b32_e32 v0, 7, v0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v5, v1
	v_mov_b32_e32 v6, v2
	global_store_dwordx4 v0, v[5:8], s[0:1]
	global_store_dwordx4 v0, v[11:14], s[0:1] offset:16
	global_store_dwordx4 v0, v[15:18], s[0:1] offset:32
	global_store_dwordx4 v0, v[19:22], s[0:1] offset:48
	v_mov_b32_e32 v5, v9
	v_mov_b32_e32 v6, v10
	global_store_dwordx4 v0, v[3:6], s[0:1] offset:64
	global_store_dwordx4 v0, v[23:26], s[0:1] offset:80
	global_store_dwordx4 v0, v[27:30], s[0:1] offset:96
	global_store_dwordx4 v0, v[31:34], s[0:1] offset:112
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
	.set .Lprobe.num_vgpr, 35
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
; codeLenInByte = 480
; TotalNumSgprs: 10
; NumVgprs: 35
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
    .vgpr_count:     35
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgpu-amd-amdhsa-unknown-gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
