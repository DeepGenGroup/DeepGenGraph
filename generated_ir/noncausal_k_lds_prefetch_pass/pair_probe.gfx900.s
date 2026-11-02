	.amdgcn_target "amdgpu-amd-amdhsa-unknown-gfx900"
	.amdhsa_code_object_version 6
	.text
	.globl	pair0                           ; -- Begin function pair0
	.p2align	8
	.type	pair0,@function
pair0:                                  ; @pair0
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v2, 8, v3
	v_lshlrev_b32_e32 v4, 5, v3
	v_and_b32_e32 v1, 24, v1
	v_and_b32_e32 v2, 0x300, v2
	v_and_b32_e32 v4, 0x60, v4
	v_or3_b32 v4, v1, v2, v4
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_or_b32 v1, v0, 1, v1
	v_add_u32_e32 v1, 1, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_mov_b32_e32 v2, 0
	;;#ASMSTART
	;;#ASMEND
	v_bfe_u32 v2, v3, 1, 1
	v_add_lshl_u32 v1, v1, v2, 6
	v_lshrrev_b32_e32 v2, 2, v3
	v_and_b32_e32 v2, 12, v2
	v_sub_u32_e32 v1, v4, v1
	v_lshlrev_b32_e32 v4, 1, v2
	v_bfe_u32 v3, v3, 5, 1
	v_lshl_add_u32 v4, v1, 1, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v1, v1, 1, v3
	v_add_u32_e32 v2, 64, v1
	v_lshrrev_b32_e32 v5, 6, v2
	v_and_b32_e32 v5, 8, v5
	v_xor_b32_e32 v2, v5, v2
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v4, v3, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset1:16
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end0:
	.size	pair0, .Lfunc_end0-pair0
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair0
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
	.set .Lpair0.num_vgpr, 6
	.set .Lpair0.num_agpr, 0
	.set .Lpair0.numbered_sgpr, 6
	.set .Lpair0.num_named_barrier, 0
	.set .Lpair0.private_seg_size, 0
	.set .Lpair0.uses_vcc, 0
	.set .Lpair0.uses_flat_scratch, 0
	.set .Lpair0.has_dyn_sized_stack, 0
	.set .Lpair0.has_recursion, 0
	.set .Lpair0.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 204
; TotalNumSgprs: 10
; NumVgprs: 6
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
	.text
	.globl	pair1                           ; -- Begin function pair1
	.p2align	8
	.type	pair1,@function
pair1:                                  ; @pair1
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v2, 8, v3
	v_lshlrev_b32_e32 v4, 5, v3
	v_and_b32_e32 v1, 24, v1
	v_and_b32_e32 v2, 0x300, v2
	v_and_b32_e32 v4, 0x60, v4
	v_or3_b32 v4, v1, v2, v4
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_or_b32 v1, v0, 1, v1
	v_add_u32_e32 v1, 1, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_mov_b32_e32 v2, 0
	;;#ASMSTART
	;;#ASMEND
	v_bfe_u32 v2, v3, 1, 1
	v_add_lshl_u32 v1, v1, v2, 6
	v_lshrrev_b32_e32 v2, 2, v3
	v_and_b32_e32 v2, 12, v2
	v_sub_u32_e32 v1, v4, v1
	v_lshlrev_b32_e32 v4, 1, v2
	v_bfe_u32 v3, v3, 5, 1
	v_lshl_add_u32 v4, v1, 1, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v1, v1, 1, v3
	v_add_u32_e32 v2, 64, v1
	v_lshrrev_b32_e32 v5, 6, v2
	v_and_b32_e32 v5, 8, v5
	v_xor_b32_e32 v2, v5, v2
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v4, v3, v1
	v_add_u32_e32 v1, 0x1000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset1:16
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end1:
	.size	pair1, .Lfunc_end1-pair1
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair1
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
	.set .Lpair1.num_vgpr, 6
	.set .Lpair1.num_agpr, 0
	.set .Lpair1.numbered_sgpr, 6
	.set .Lpair1.num_named_barrier, 0
	.set .Lpair1.private_seg_size, 0
	.set .Lpair1.uses_vcc, 0
	.set .Lpair1.uses_flat_scratch, 0
	.set .Lpair1.has_dyn_sized_stack, 0
	.set .Lpair1.has_recursion, 0
	.set .Lpair1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 212
; TotalNumSgprs: 10
; NumVgprs: 6
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
	.text
	.globl	pair2                           ; -- Begin function pair2
	.p2align	8
	.type	pair2,@function
pair2:                                  ; @pair2
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 2, v3
	s_movk_i32 s0, 0xfb00
	v_mul_lo_u32 v1, v4, s0
	s_movk_i32 s0, 0x140
	v_lshlrev_b32_e32 v5, 30, v3
	v_and_b32_e32 v4, 12, v4
	v_mad_u64_u32 v[1:2], s[0:1], v3, s0, v[1:2]
	v_lshlrev_b32_e32 v2, 2, v3
	v_bfe_u32 v3, v3, 5, 1
	v_lshlrev_b32_e32 v6, 1, v4
	v_and_or_b32 v1, v2, 48, v1
	v_ashrrev_i32_e32 v2, 25, v5
	v_and_b32_e32 v2, 0xffffffc0, v2
	v_lshlrev_b32_e32 v5, 1, v2
	v_or_b32_e32 v2, v2, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_lshlrev_b32_e32 v2, 1, v2
	v_add3_u32 v5, v1, v5, v6
	v_add3_u32 v1, v1, v3, v2
	v_lshrrev_b32_e32 v2, 6, v1
	v_and_b32_e32 v2, 8, v2
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v5, v3, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset0:16 offset1:128
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end2:
	.size	pair2, .Lfunc_end2-pair2
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair2
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
	.set .Lpair2.num_vgpr, 7
	.set .Lpair2.num_agpr, 0
	.set .Lpair2.numbered_sgpr, 6
	.set .Lpair2.num_named_barrier, 0
	.set .Lpair2.private_seg_size, 0
	.set .Lpair2.uses_vcc, 0
	.set .Lpair2.uses_flat_scratch, 0
	.set .Lpair2.has_dyn_sized_stack, 0
	.set .Lpair2.has_recursion, 0
	.set .Lpair2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 172
; TotalNumSgprs: 10
; NumVgprs: 7
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
	.text
	.globl	pair3                           ; -- Begin function pair3
	.p2align	8
	.type	pair3,@function
pair3:                                  ; @pair3
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 2, v3
	s_movk_i32 s0, 0xfb00
	v_mul_lo_u32 v1, v4, s0
	s_movk_i32 s0, 0x140
	v_lshlrev_b32_e32 v5, 30, v3
	v_and_b32_e32 v4, 12, v4
	v_mad_u64_u32 v[1:2], s[0:1], v3, s0, v[1:2]
	v_lshlrev_b32_e32 v2, 2, v3
	v_bfe_u32 v3, v3, 5, 1
	v_lshlrev_b32_e32 v6, 1, v4
	v_and_or_b32 v1, v2, 48, v1
	v_ashrrev_i32_e32 v2, 25, v5
	v_and_b32_e32 v2, 0xffffffc0, v2
	v_lshlrev_b32_e32 v5, 1, v2
	v_or_b32_e32 v2, v2, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_lshlrev_b32_e32 v2, 1, v2
	v_add3_u32 v5, v1, v5, v6
	v_add3_u32 v1, v1, v3, v2
	v_lshrrev_b32_e32 v2, 6, v1
	v_and_b32_e32 v2, 8, v2
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v5, v3, v1
	v_add_u32_e32 v1, 0x1000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset0:16 offset1:128
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end3:
	.size	pair3, .Lfunc_end3-pair3
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair3
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
	.set .Lpair3.num_vgpr, 7
	.set .Lpair3.num_agpr, 0
	.set .Lpair3.numbered_sgpr, 6
	.set .Lpair3.num_named_barrier, 0
	.set .Lpair3.private_seg_size, 0
	.set .Lpair3.uses_vcc, 0
	.set .Lpair3.uses_flat_scratch, 0
	.set .Lpair3.has_dyn_sized_stack, 0
	.set .Lpair3.has_recursion, 0
	.set .Lpair3.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 180
; TotalNumSgprs: 10
; NumVgprs: 7
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
	.text
	.globl	pair4                           ; -- Begin function pair4
	.p2align	8
	.type	pair4,@function
pair4:                                  ; @pair4
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v2, 8, v3
	v_lshlrev_b32_e32 v4, 5, v3
	v_and_b32_e32 v1, 24, v1
	v_and_b32_e32 v2, 0x300, v2
	v_and_b32_e32 v4, 0x60, v4
	v_or3_b32 v4, v1, v2, v4
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_or_b32 v1, v0, 1, v1
	v_add_u32_e32 v1, 1, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_mov_b32_e32 v2, 0
	;;#ASMSTART
	;;#ASMEND
	v_bfe_u32 v2, v3, 1, 1
	v_add_lshl_u32 v1, v1, v2, 6
	v_lshrrev_b32_e32 v2, 2, v3
	v_and_b32_e32 v2, 12, v2
	v_sub_u32_e32 v1, v4, v1
	v_lshlrev_b32_e32 v4, 1, v2
	v_bfe_u32 v3, v3, 5, 1
	v_lshl_add_u32 v4, v1, 1, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v1, v1, 1, v3
	v_add_u32_e32 v2, 64, v1
	v_lshrrev_b32_e32 v5, 6, v2
	v_and_b32_e32 v5, 8, v5
	v_xor_b32_e32 v2, v5, v2
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v4, v3, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset0:128 offset1:144
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end4:
	.size	pair4, .Lfunc_end4-pair4
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair4
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
	.set .Lpair4.num_vgpr, 6
	.set .Lpair4.num_agpr, 0
	.set .Lpair4.numbered_sgpr, 6
	.set .Lpair4.num_named_barrier, 0
	.set .Lpair4.private_seg_size, 0
	.set .Lpair4.uses_vcc, 0
	.set .Lpair4.uses_flat_scratch, 0
	.set .Lpair4.has_dyn_sized_stack, 0
	.set .Lpair4.has_recursion, 0
	.set .Lpair4.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 204
; TotalNumSgprs: 10
; NumVgprs: 6
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
	.text
	.globl	pair5                           ; -- Begin function pair5
	.p2align	8
	.type	pair5,@function
pair5:                                  ; @pair5
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v2, 8, v3
	v_lshlrev_b32_e32 v4, 5, v3
	v_and_b32_e32 v1, 24, v1
	v_and_b32_e32 v2, 0x300, v2
	v_and_b32_e32 v4, 0x60, v4
	v_or3_b32 v4, v1, v2, v4
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_or_b32 v1, v0, 1, v1
	v_add_u32_e32 v1, 1, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_mov_b32_e32 v2, 0
	;;#ASMSTART
	;;#ASMEND
	v_bfe_u32 v2, v3, 1, 1
	v_add_lshl_u32 v1, v1, v2, 6
	v_lshrrev_b32_e32 v2, 2, v3
	v_and_b32_e32 v2, 12, v2
	v_sub_u32_e32 v1, v4, v1
	v_lshlrev_b32_e32 v4, 1, v2
	v_bfe_u32 v3, v3, 5, 1
	v_lshl_add_u32 v4, v1, 1, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v1, v1, 1, v3
	v_add_u32_e32 v2, 64, v1
	v_lshrrev_b32_e32 v5, 6, v2
	v_and_b32_e32 v5, 8, v5
	v_xor_b32_e32 v2, v5, v2
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v4, v3, v1
	v_add_u32_e32 v1, 0x1000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset0:128 offset1:144
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end5:
	.size	pair5, .Lfunc_end5-pair5
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair5
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
	.set .Lpair5.num_vgpr, 6
	.set .Lpair5.num_agpr, 0
	.set .Lpair5.numbered_sgpr, 6
	.set .Lpair5.num_named_barrier, 0
	.set .Lpair5.private_seg_size, 0
	.set .Lpair5.uses_vcc, 0
	.set .Lpair5.uses_flat_scratch, 0
	.set .Lpair5.has_dyn_sized_stack, 0
	.set .Lpair5.has_recursion, 0
	.set .Lpair5.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 212
; TotalNumSgprs: 10
; NumVgprs: 6
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
	.text
	.globl	pair6                           ; -- Begin function pair6
	.p2align	8
	.type	pair6,@function
pair6:                                  ; @pair6
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 2, v3
	s_movk_i32 s0, 0xfb00
	v_mul_lo_u32 v1, v4, s0
	s_movk_i32 s0, 0x140
	v_lshlrev_b32_e32 v5, 30, v3
	v_and_b32_e32 v4, 12, v4
	v_mad_u64_u32 v[1:2], s[0:1], v3, s0, v[1:2]
	v_lshlrev_b32_e32 v2, 2, v3
	v_bfe_u32 v3, v3, 5, 1
	v_lshlrev_b32_e32 v6, 1, v4
	v_and_or_b32 v1, v2, 48, v1
	v_ashrrev_i32_e32 v2, 25, v5
	v_and_b32_e32 v2, 0xffffffc0, v2
	v_lshlrev_b32_e32 v5, 1, v2
	v_or_b32_e32 v2, v2, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_lshlrev_b32_e32 v2, 1, v2
	v_add3_u32 v5, v1, v5, v6
	v_add3_u32 v1, v1, v3, v2
	v_lshrrev_b32_e32 v2, 6, v1
	v_and_b32_e32 v2, 8, v2
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v5, v3, v1
	v_add_u32_e32 v1, 0x80, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2st64_b64 v[1:4], v1 offset0:2 offset1:10
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end6:
	.size	pair6, .Lfunc_end6-pair6
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair6
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
	.set .Lpair6.num_vgpr, 7
	.set .Lpair6.num_agpr, 0
	.set .Lpair6.numbered_sgpr, 6
	.set .Lpair6.num_named_barrier, 0
	.set .Lpair6.private_seg_size, 0
	.set .Lpair6.uses_vcc, 0
	.set .Lpair6.uses_flat_scratch, 0
	.set .Lpair6.has_dyn_sized_stack, 0
	.set .Lpair6.has_recursion, 0
	.set .Lpair6.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 180
; TotalNumSgprs: 10
; NumVgprs: 7
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
	.text
	.globl	pair7                           ; -- Begin function pair7
	.p2align	8
	.type	pair7,@function
pair7:                                  ; @pair7
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 2, v3
	s_movk_i32 s0, 0xfb00
	v_mul_lo_u32 v1, v4, s0
	s_movk_i32 s0, 0x140
	v_lshlrev_b32_e32 v5, 30, v3
	v_and_b32_e32 v4, 12, v4
	v_mad_u64_u32 v[1:2], s[0:1], v3, s0, v[1:2]
	v_lshlrev_b32_e32 v2, 2, v3
	v_bfe_u32 v3, v3, 5, 1
	v_lshlrev_b32_e32 v6, 1, v4
	v_and_or_b32 v1, v2, 48, v1
	v_ashrrev_i32_e32 v2, 25, v5
	v_and_b32_e32 v2, 0xffffffc0, v2
	v_lshlrev_b32_e32 v5, 1, v2
	v_or_b32_e32 v2, v2, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_lshlrev_b32_e32 v2, 1, v2
	v_add3_u32 v5, v1, v5, v6
	v_add3_u32 v1, v1, v3, v2
	v_lshrrev_b32_e32 v2, 6, v1
	v_and_b32_e32 v2, 8, v2
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v5, v3, v1
	v_add_u32_e32 v1, 0x2000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset1:16
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end7:
	.size	pair7, .Lfunc_end7-pair7
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair7
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
	.set .Lpair7.num_vgpr, 7
	.set .Lpair7.num_agpr, 0
	.set .Lpair7.numbered_sgpr, 6
	.set .Lpair7.num_named_barrier, 0
	.set .Lpair7.private_seg_size, 0
	.set .Lpair7.uses_vcc, 0
	.set .Lpair7.uses_flat_scratch, 0
	.set .Lpair7.has_dyn_sized_stack, 0
	.set .Lpair7.has_recursion, 0
	.set .Lpair7.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 180
; TotalNumSgprs: 10
; NumVgprs: 7
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
	.text
	.globl	pair8                           ; -- Begin function pair8
	.p2align	8
	.type	pair8,@function
pair8:                                  ; @pair8
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 2, v3
	s_movk_i32 s0, 0xfb00
	v_mul_lo_u32 v1, v4, s0
	s_movk_i32 s0, 0x140
	v_lshlrev_b32_e32 v5, 30, v3
	v_and_b32_e32 v4, 12, v4
	v_mad_u64_u32 v[1:2], s[0:1], v3, s0, v[1:2]
	v_lshlrev_b32_e32 v2, 2, v3
	v_bfe_u32 v3, v3, 5, 1
	v_lshlrev_b32_e32 v6, 1, v4
	v_and_or_b32 v1, v2, 48, v1
	v_ashrrev_i32_e32 v2, 25, v5
	v_and_b32_e32 v2, 0xffffffc0, v2
	v_lshlrev_b32_e32 v5, 1, v2
	v_or_b32_e32 v2, v2, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_lshlrev_b32_e32 v2, 1, v2
	v_add3_u32 v5, v1, v5, v6
	v_add3_u32 v1, v1, v3, v2
	v_lshrrev_b32_e32 v2, 6, v1
	v_and_b32_e32 v2, 8, v2
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v5, v3, v1
	v_add_u32_e32 v1, 0x3000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset1:16
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end8:
	.size	pair8, .Lfunc_end8-pair8
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair8
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
	.set .Lpair8.num_vgpr, 7
	.set .Lpair8.num_agpr, 0
	.set .Lpair8.numbered_sgpr, 6
	.set .Lpair8.num_named_barrier, 0
	.set .Lpair8.private_seg_size, 0
	.set .Lpair8.uses_vcc, 0
	.set .Lpair8.uses_flat_scratch, 0
	.set .Lpair8.has_dyn_sized_stack, 0
	.set .Lpair8.has_recursion, 0
	.set .Lpair8.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 180
; TotalNumSgprs: 10
; NumVgprs: 7
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
	.text
	.globl	pair9                           ; -- Begin function pair9
	.p2align	8
	.type	pair9,@function
pair9:                                  ; @pair9
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v2, 8, v3
	v_lshlrev_b32_e32 v4, 5, v3
	v_and_b32_e32 v1, 24, v1
	v_and_b32_e32 v2, 0x300, v2
	v_and_b32_e32 v4, 0x60, v4
	v_or3_b32 v4, v1, v2, v4
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_or_b32 v1, v0, 1, v1
	v_add_u32_e32 v1, 1, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_mov_b32_e32 v2, 0
	;;#ASMSTART
	;;#ASMEND
	v_bfe_u32 v2, v3, 1, 1
	v_add_lshl_u32 v1, v1, v2, 6
	v_lshrrev_b32_e32 v2, 2, v3
	v_and_b32_e32 v2, 12, v2
	v_sub_u32_e32 v1, v4, v1
	v_lshlrev_b32_e32 v4, 1, v2
	v_bfe_u32 v3, v3, 5, 1
	v_lshl_add_u32 v4, v1, 1, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v1, v1, 1, v3
	v_add_u32_e32 v2, 64, v1
	v_lshrrev_b32_e32 v5, 6, v2
	v_and_b32_e32 v5, 8, v5
	v_xor_b32_e32 v2, v5, v2
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v4, v3, v1
	v_add_u32_e32 v1, 0x2000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset1:16
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end9:
	.size	pair9, .Lfunc_end9-pair9
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair9
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
	.set .Lpair9.num_vgpr, 6
	.set .Lpair9.num_agpr, 0
	.set .Lpair9.numbered_sgpr, 6
	.set .Lpair9.num_named_barrier, 0
	.set .Lpair9.private_seg_size, 0
	.set .Lpair9.uses_vcc, 0
	.set .Lpair9.uses_flat_scratch, 0
	.set .Lpair9.has_dyn_sized_stack, 0
	.set .Lpair9.has_recursion, 0
	.set .Lpair9.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 212
; TotalNumSgprs: 10
; NumVgprs: 6
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
	.text
	.globl	pair10                          ; -- Begin function pair10
	.p2align	8
	.type	pair10,@function
pair10:                                 ; @pair10
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v2, 8, v3
	v_lshlrev_b32_e32 v4, 5, v3
	v_and_b32_e32 v1, 24, v1
	v_and_b32_e32 v2, 0x300, v2
	v_and_b32_e32 v4, 0x60, v4
	v_or3_b32 v4, v1, v2, v4
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_or_b32 v1, v0, 1, v1
	v_add_u32_e32 v1, 1, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_mov_b32_e32 v2, 0
	;;#ASMSTART
	;;#ASMEND
	v_bfe_u32 v2, v3, 1, 1
	v_add_lshl_u32 v1, v1, v2, 6
	v_lshrrev_b32_e32 v2, 2, v3
	v_and_b32_e32 v2, 12, v2
	v_sub_u32_e32 v1, v4, v1
	v_lshlrev_b32_e32 v4, 1, v2
	v_bfe_u32 v3, v3, 5, 1
	v_lshl_add_u32 v4, v1, 1, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v1, v1, 1, v3
	v_add_u32_e32 v2, 64, v1
	v_lshrrev_b32_e32 v5, 6, v2
	v_and_b32_e32 v5, 8, v5
	v_xor_b32_e32 v2, v5, v2
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v4, v3, v1
	v_add_u32_e32 v1, 0x3000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset1:16
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end10:
	.size	pair10, .Lfunc_end10-pair10
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair10
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
	.set .Lpair10.num_vgpr, 6
	.set .Lpair10.num_agpr, 0
	.set .Lpair10.numbered_sgpr, 6
	.set .Lpair10.num_named_barrier, 0
	.set .Lpair10.private_seg_size, 0
	.set .Lpair10.uses_vcc, 0
	.set .Lpair10.uses_flat_scratch, 0
	.set .Lpair10.has_dyn_sized_stack, 0
	.set .Lpair10.has_recursion, 0
	.set .Lpair10.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 212
; TotalNumSgprs: 10
; NumVgprs: 6
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
	.text
	.globl	pair11                          ; -- Begin function pair11
	.p2align	8
	.type	pair11,@function
pair11:                                 ; @pair11
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 2, v3
	s_movk_i32 s0, 0xfb00
	v_mul_lo_u32 v1, v4, s0
	s_movk_i32 s0, 0x140
	v_lshlrev_b32_e32 v5, 30, v3
	v_and_b32_e32 v4, 12, v4
	v_mad_u64_u32 v[1:2], s[0:1], v3, s0, v[1:2]
	v_lshlrev_b32_e32 v2, 2, v3
	v_bfe_u32 v3, v3, 5, 1
	v_lshlrev_b32_e32 v6, 1, v4
	v_and_or_b32 v1, v2, 48, v1
	v_ashrrev_i32_e32 v2, 25, v5
	v_and_b32_e32 v2, 0xffffffc0, v2
	v_lshlrev_b32_e32 v5, 1, v2
	v_or_b32_e32 v2, v2, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_lshlrev_b32_e32 v2, 1, v2
	v_add3_u32 v5, v1, v5, v6
	v_add3_u32 v1, v1, v3, v2
	v_lshrrev_b32_e32 v2, 6, v1
	v_and_b32_e32 v2, 8, v2
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v5, v3, v1
	v_add_u32_e32 v1, 0x2000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset0:128 offset1:144
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end11:
	.size	pair11, .Lfunc_end11-pair11
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair11
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
	.set .Lpair11.num_vgpr, 7
	.set .Lpair11.num_agpr, 0
	.set .Lpair11.numbered_sgpr, 6
	.set .Lpair11.num_named_barrier, 0
	.set .Lpair11.private_seg_size, 0
	.set .Lpair11.uses_vcc, 0
	.set .Lpair11.uses_flat_scratch, 0
	.set .Lpair11.has_dyn_sized_stack, 0
	.set .Lpair11.has_recursion, 0
	.set .Lpair11.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 180
; TotalNumSgprs: 10
; NumVgprs: 7
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
	.text
	.globl	pair12                          ; -- Begin function pair12
	.p2align	8
	.type	pair12,@function
pair12:                                 ; @pair12
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 2, v3
	s_movk_i32 s0, 0xfb00
	v_mul_lo_u32 v1, v4, s0
	s_movk_i32 s0, 0x140
	v_lshlrev_b32_e32 v5, 30, v3
	v_and_b32_e32 v4, 12, v4
	v_mad_u64_u32 v[1:2], s[0:1], v3, s0, v[1:2]
	v_lshlrev_b32_e32 v2, 2, v3
	v_bfe_u32 v3, v3, 5, 1
	v_lshlrev_b32_e32 v6, 1, v4
	v_and_or_b32 v1, v2, 48, v1
	v_ashrrev_i32_e32 v2, 25, v5
	v_and_b32_e32 v2, 0xffffffc0, v2
	v_lshlrev_b32_e32 v5, 1, v2
	v_or_b32_e32 v2, v2, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_lshlrev_b32_e32 v2, 1, v2
	v_add3_u32 v5, v1, v5, v6
	v_add3_u32 v1, v1, v3, v2
	v_lshrrev_b32_e32 v2, 6, v1
	v_and_b32_e32 v2, 8, v2
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v5, v3, v1
	v_add_u32_e32 v1, 0x3000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset0:128 offset1:144
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end12:
	.size	pair12, .Lfunc_end12-pair12
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair12
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
	.set .Lpair12.num_vgpr, 7
	.set .Lpair12.num_agpr, 0
	.set .Lpair12.numbered_sgpr, 6
	.set .Lpair12.num_named_barrier, 0
	.set .Lpair12.private_seg_size, 0
	.set .Lpair12.uses_vcc, 0
	.set .Lpair12.uses_flat_scratch, 0
	.set .Lpair12.has_dyn_sized_stack, 0
	.set .Lpair12.has_recursion, 0
	.set .Lpair12.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 180
; TotalNumSgprs: 10
; NumVgprs: 7
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
	.text
	.globl	pair13                          ; -- Begin function pair13
	.p2align	8
	.type	pair13,@function
pair13:                                 ; @pair13
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v2, 8, v3
	v_lshlrev_b32_e32 v4, 5, v3
	v_and_b32_e32 v1, 24, v1
	v_and_b32_e32 v2, 0x300, v2
	v_and_b32_e32 v4, 0x60, v4
	v_or3_b32 v4, v1, v2, v4
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_or_b32 v1, v0, 1, v1
	v_add_u32_e32 v1, 1, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_mov_b32_e32 v2, 0
	;;#ASMSTART
	;;#ASMEND
	v_bfe_u32 v2, v3, 1, 1
	v_add_lshl_u32 v1, v1, v2, 6
	v_lshrrev_b32_e32 v2, 2, v3
	v_and_b32_e32 v2, 12, v2
	v_sub_u32_e32 v1, v4, v1
	v_lshlrev_b32_e32 v4, 1, v2
	v_bfe_u32 v3, v3, 5, 1
	v_lshl_add_u32 v4, v1, 1, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v1, v1, 1, v3
	v_add_u32_e32 v2, 64, v1
	v_lshrrev_b32_e32 v5, 6, v2
	v_and_b32_e32 v5, 8, v5
	v_xor_b32_e32 v2, v5, v2
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v4, v3, v1
	v_add_u32_e32 v1, 0x2000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset0:128 offset1:144
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end13:
	.size	pair13, .Lfunc_end13-pair13
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair13
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
	.set .Lpair13.num_vgpr, 6
	.set .Lpair13.num_agpr, 0
	.set .Lpair13.numbered_sgpr, 6
	.set .Lpair13.num_named_barrier, 0
	.set .Lpair13.private_seg_size, 0
	.set .Lpair13.uses_vcc, 0
	.set .Lpair13.uses_flat_scratch, 0
	.set .Lpair13.has_dyn_sized_stack, 0
	.set .Lpair13.has_recursion, 0
	.set .Lpair13.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 212
; TotalNumSgprs: 10
; NumVgprs: 6
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
	.text
	.globl	pair14                          ; -- Begin function pair14
	.p2align	8
	.type	pair14,@function
pair14:                                 ; @pair14
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v2, 8, v3
	v_lshlrev_b32_e32 v4, 5, v3
	v_and_b32_e32 v1, 24, v1
	v_and_b32_e32 v2, 0x300, v2
	v_and_b32_e32 v4, 0x60, v4
	v_or3_b32 v4, v1, v2, v4
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_or_b32 v1, v0, 1, v1
	v_add_u32_e32 v1, 1, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_mov_b32_e32 v2, 0
	;;#ASMSTART
	;;#ASMEND
	v_bfe_u32 v2, v3, 1, 1
	v_add_lshl_u32 v1, v1, v2, 6
	v_lshrrev_b32_e32 v2, 2, v3
	v_and_b32_e32 v2, 12, v2
	v_sub_u32_e32 v1, v4, v1
	v_lshlrev_b32_e32 v4, 1, v2
	v_bfe_u32 v3, v3, 5, 1
	v_lshl_add_u32 v4, v1, 1, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v1, v1, 1, v3
	v_add_u32_e32 v2, 64, v1
	v_lshrrev_b32_e32 v5, 6, v2
	v_and_b32_e32 v5, 8, v5
	v_xor_b32_e32 v2, v5, v2
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v4, v3, v1
	v_add_u32_e32 v1, 0x3000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset0:128 offset1:144
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end14:
	.size	pair14, .Lfunc_end14-pair14
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair14
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
	.set .Lpair14.num_vgpr, 6
	.set .Lpair14.num_agpr, 0
	.set .Lpair14.numbered_sgpr, 6
	.set .Lpair14.num_named_barrier, 0
	.set .Lpair14.private_seg_size, 0
	.set .Lpair14.uses_vcc, 0
	.set .Lpair14.uses_flat_scratch, 0
	.set .Lpair14.has_dyn_sized_stack, 0
	.set .Lpair14.has_recursion, 0
	.set .Lpair14.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 212
; TotalNumSgprs: 10
; NumVgprs: 6
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
	.text
	.globl	pair15                          ; -- Begin function pair15
	.p2align	8
	.type	pair15,@function
pair15:                                 ; @pair15
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 2, v3
	s_movk_i32 s0, 0xfb00
	v_mul_lo_u32 v1, v4, s0
	s_movk_i32 s0, 0x140
	v_lshlrev_b32_e32 v5, 30, v3
	v_and_b32_e32 v4, 12, v4
	v_mad_u64_u32 v[1:2], s[0:1], v3, s0, v[1:2]
	v_lshlrev_b32_e32 v2, 2, v3
	v_bfe_u32 v3, v3, 5, 1
	v_lshlrev_b32_e32 v6, 1, v4
	v_and_or_b32 v1, v2, 48, v1
	v_ashrrev_i32_e32 v2, 25, v5
	v_and_b32_e32 v2, 0xffffffc0, v2
	v_lshlrev_b32_e32 v5, 1, v2
	v_or_b32_e32 v2, v2, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_lshlrev_b32_e32 v2, 1, v2
	v_add3_u32 v5, v1, v5, v6
	v_add3_u32 v1, v1, v3, v2
	v_lshrrev_b32_e32 v2, 6, v1
	v_and_b32_e32 v2, 8, v2
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v5, v3, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset1:16
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end15:
	.size	pair15, .Lfunc_end15-pair15
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair15
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
	.set .Lpair15.num_vgpr, 7
	.set .Lpair15.num_agpr, 0
	.set .Lpair15.numbered_sgpr, 6
	.set .Lpair15.num_named_barrier, 0
	.set .Lpair15.private_seg_size, 0
	.set .Lpair15.uses_vcc, 0
	.set .Lpair15.uses_flat_scratch, 0
	.set .Lpair15.has_dyn_sized_stack, 0
	.set .Lpair15.has_recursion, 0
	.set .Lpair15.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 172
; TotalNumSgprs: 10
; NumVgprs: 7
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
	.text
	.globl	pair16                          ; -- Begin function pair16
	.p2align	8
	.type	pair16,@function
pair16:                                 ; @pair16
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 2, v3
	s_movk_i32 s0, 0xfb00
	v_mul_lo_u32 v1, v4, s0
	s_movk_i32 s0, 0x140
	v_lshlrev_b32_e32 v5, 30, v3
	v_and_b32_e32 v4, 12, v4
	v_mad_u64_u32 v[1:2], s[0:1], v3, s0, v[1:2]
	v_lshlrev_b32_e32 v2, 2, v3
	v_bfe_u32 v3, v3, 5, 1
	v_lshlrev_b32_e32 v6, 1, v4
	v_and_or_b32 v1, v2, 48, v1
	v_ashrrev_i32_e32 v2, 25, v5
	v_and_b32_e32 v2, 0xffffffc0, v2
	v_lshlrev_b32_e32 v5, 1, v2
	v_or_b32_e32 v2, v2, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_lshlrev_b32_e32 v2, 1, v2
	v_add3_u32 v5, v1, v5, v6
	v_add3_u32 v1, v1, v3, v2
	v_lshrrev_b32_e32 v2, 6, v1
	v_and_b32_e32 v2, 8, v2
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v5, v3, v1
	v_add_u32_e32 v1, 0x1000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset1:16
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end16:
	.size	pair16, .Lfunc_end16-pair16
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair16
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
	.set .Lpair16.num_vgpr, 7
	.set .Lpair16.num_agpr, 0
	.set .Lpair16.numbered_sgpr, 6
	.set .Lpair16.num_named_barrier, 0
	.set .Lpair16.private_seg_size, 0
	.set .Lpair16.uses_vcc, 0
	.set .Lpair16.uses_flat_scratch, 0
	.set .Lpair16.has_dyn_sized_stack, 0
	.set .Lpair16.has_recursion, 0
	.set .Lpair16.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 180
; TotalNumSgprs: 10
; NumVgprs: 7
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
	.text
	.globl	pair17                          ; -- Begin function pair17
	.p2align	8
	.type	pair17,@function
pair17:                                 ; @pair17
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v2, 8, v3
	v_lshlrev_b32_e32 v4, 5, v3
	v_and_b32_e32 v1, 24, v1
	v_and_b32_e32 v2, 0x300, v2
	v_and_b32_e32 v4, 0x60, v4
	v_or3_b32 v4, v1, v2, v4
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_or_b32 v1, v0, 1, v1
	v_add_u32_e32 v1, 1, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_mov_b32_e32 v2, 0
	;;#ASMSTART
	;;#ASMEND
	v_bfe_u32 v2, v3, 1, 1
	v_add_lshl_u32 v1, v1, v2, 6
	v_lshrrev_b32_e32 v2, 2, v3
	v_and_b32_e32 v2, 12, v2
	v_sub_u32_e32 v1, v4, v1
	v_lshlrev_b32_e32 v4, 1, v2
	v_bfe_u32 v3, v3, 5, 1
	v_lshl_add_u32 v4, v1, 1, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v1, v1, 1, v3
	v_add_u32_e32 v2, 64, v1
	v_lshrrev_b32_e32 v5, 6, v2
	v_and_b32_e32 v5, 8, v5
	v_xor_b32_e32 v2, v5, v2
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v4, v3, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset1:16
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end17:
	.size	pair17, .Lfunc_end17-pair17
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair17
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
	.set .Lpair17.num_vgpr, 6
	.set .Lpair17.num_agpr, 0
	.set .Lpair17.numbered_sgpr, 6
	.set .Lpair17.num_named_barrier, 0
	.set .Lpair17.private_seg_size, 0
	.set .Lpair17.uses_vcc, 0
	.set .Lpair17.uses_flat_scratch, 0
	.set .Lpair17.has_dyn_sized_stack, 0
	.set .Lpair17.has_recursion, 0
	.set .Lpair17.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 204
; TotalNumSgprs: 10
; NumVgprs: 6
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
	.text
	.globl	pair18                          ; -- Begin function pair18
	.p2align	8
	.type	pair18,@function
pair18:                                 ; @pair18
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v2, 8, v3
	v_lshlrev_b32_e32 v4, 5, v3
	v_and_b32_e32 v1, 24, v1
	v_and_b32_e32 v2, 0x300, v2
	v_and_b32_e32 v4, 0x60, v4
	v_or3_b32 v4, v1, v2, v4
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_or_b32 v1, v0, 1, v1
	v_add_u32_e32 v1, 1, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_mov_b32_e32 v2, 0
	;;#ASMSTART
	;;#ASMEND
	v_bfe_u32 v2, v3, 1, 1
	v_add_lshl_u32 v1, v1, v2, 6
	v_lshrrev_b32_e32 v2, 2, v3
	v_and_b32_e32 v2, 12, v2
	v_sub_u32_e32 v1, v4, v1
	v_lshlrev_b32_e32 v4, 1, v2
	v_bfe_u32 v3, v3, 5, 1
	v_lshl_add_u32 v4, v1, 1, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v1, v1, 1, v3
	v_add_u32_e32 v2, 64, v1
	v_lshrrev_b32_e32 v5, 6, v2
	v_and_b32_e32 v5, 8, v5
	v_xor_b32_e32 v2, v5, v2
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v4, v3, v1
	v_add_u32_e32 v1, 0x1000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset1:16
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end18:
	.size	pair18, .Lfunc_end18-pair18
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair18
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
	.set .Lpair18.num_vgpr, 6
	.set .Lpair18.num_agpr, 0
	.set .Lpair18.numbered_sgpr, 6
	.set .Lpair18.num_named_barrier, 0
	.set .Lpair18.private_seg_size, 0
	.set .Lpair18.uses_vcc, 0
	.set .Lpair18.uses_flat_scratch, 0
	.set .Lpair18.has_dyn_sized_stack, 0
	.set .Lpair18.has_recursion, 0
	.set .Lpair18.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 212
; TotalNumSgprs: 10
; NumVgprs: 6
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
	.text
	.globl	pair19                          ; -- Begin function pair19
	.p2align	8
	.type	pair19,@function
pair19:                                 ; @pair19
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 2, v3
	s_movk_i32 s0, 0xfb00
	v_mul_lo_u32 v1, v4, s0
	s_movk_i32 s0, 0x140
	v_lshlrev_b32_e32 v5, 30, v3
	v_and_b32_e32 v4, 12, v4
	v_mad_u64_u32 v[1:2], s[0:1], v3, s0, v[1:2]
	v_lshlrev_b32_e32 v2, 2, v3
	v_bfe_u32 v3, v3, 5, 1
	v_lshlrev_b32_e32 v6, 1, v4
	v_and_or_b32 v1, v2, 48, v1
	v_ashrrev_i32_e32 v2, 25, v5
	v_and_b32_e32 v2, 0xffffffc0, v2
	v_lshlrev_b32_e32 v5, 1, v2
	v_or_b32_e32 v2, v2, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_lshlrev_b32_e32 v2, 1, v2
	v_add3_u32 v5, v1, v5, v6
	v_add3_u32 v1, v1, v3, v2
	v_lshrrev_b32_e32 v2, 6, v1
	v_and_b32_e32 v2, 8, v2
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v5, v3, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset0:128 offset1:144
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end19:
	.size	pair19, .Lfunc_end19-pair19
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair19
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
	.set .Lpair19.num_vgpr, 7
	.set .Lpair19.num_agpr, 0
	.set .Lpair19.numbered_sgpr, 6
	.set .Lpair19.num_named_barrier, 0
	.set .Lpair19.private_seg_size, 0
	.set .Lpair19.uses_vcc, 0
	.set .Lpair19.uses_flat_scratch, 0
	.set .Lpair19.has_dyn_sized_stack, 0
	.set .Lpair19.has_recursion, 0
	.set .Lpair19.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 172
; TotalNumSgprs: 10
; NumVgprs: 7
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
	.text
	.globl	pair20                          ; -- Begin function pair20
	.p2align	8
	.type	pair20,@function
pair20:                                 ; @pair20
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshrrev_b32_e32 v4, 2, v3
	s_movk_i32 s0, 0xfb00
	v_mul_lo_u32 v1, v4, s0
	s_movk_i32 s0, 0x140
	v_lshlrev_b32_e32 v5, 30, v3
	v_and_b32_e32 v4, 12, v4
	v_mad_u64_u32 v[1:2], s[0:1], v3, s0, v[1:2]
	v_lshlrev_b32_e32 v2, 2, v3
	v_bfe_u32 v3, v3, 5, 1
	v_lshlrev_b32_e32 v6, 1, v4
	v_and_or_b32 v1, v2, 48, v1
	v_ashrrev_i32_e32 v2, 25, v5
	v_and_b32_e32 v2, 0xffffffc0, v2
	v_lshlrev_b32_e32 v5, 1, v2
	v_or_b32_e32 v2, v2, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_lshlrev_b32_e32 v2, 1, v2
	v_add3_u32 v5, v1, v5, v6
	v_add3_u32 v1, v1, v3, v2
	v_lshrrev_b32_e32 v2, 6, v1
	v_and_b32_e32 v2, 8, v2
	v_xor_b32_e32 v2, v2, v1
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v5, v3, v1
	v_add_u32_e32 v1, 0x1000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset0:128 offset1:144
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end20:
	.size	pair20, .Lfunc_end20-pair20
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair20
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
	.set .Lpair20.num_vgpr, 7
	.set .Lpair20.num_agpr, 0
	.set .Lpair20.numbered_sgpr, 6
	.set .Lpair20.num_named_barrier, 0
	.set .Lpair20.private_seg_size, 0
	.set .Lpair20.uses_vcc, 0
	.set .Lpair20.uses_flat_scratch, 0
	.set .Lpair20.has_dyn_sized_stack, 0
	.set .Lpair20.has_recursion, 0
	.set .Lpair20.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 180
; TotalNumSgprs: 10
; NumVgprs: 7
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
	.text
	.globl	pair21                          ; -- Begin function pair21
	.p2align	8
	.type	pair21,@function
pair21:                                 ; @pair21
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v2, 8, v3
	v_lshlrev_b32_e32 v4, 5, v3
	v_and_b32_e32 v1, 24, v1
	v_and_b32_e32 v2, 0x300, v2
	v_and_b32_e32 v4, 0x60, v4
	v_or3_b32 v4, v1, v2, v4
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_or_b32 v1, v0, 1, v1
	v_add_u32_e32 v1, 1, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_mov_b32_e32 v2, 0
	;;#ASMSTART
	;;#ASMEND
	v_bfe_u32 v2, v3, 1, 1
	v_add_lshl_u32 v1, v1, v2, 6
	v_lshrrev_b32_e32 v2, 2, v3
	v_and_b32_e32 v2, 12, v2
	v_sub_u32_e32 v1, v4, v1
	v_lshlrev_b32_e32 v4, 1, v2
	v_bfe_u32 v3, v3, 5, 1
	v_lshl_add_u32 v4, v1, 1, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v1, v1, 1, v3
	v_add_u32_e32 v2, 64, v1
	v_lshrrev_b32_e32 v5, 6, v2
	v_and_b32_e32 v5, 8, v5
	v_xor_b32_e32 v2, v5, v2
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v4, v3, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset0:128 offset1:144
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end21:
	.size	pair21, .Lfunc_end21-pair21
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair21
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
	.set .Lpair21.num_vgpr, 6
	.set .Lpair21.num_agpr, 0
	.set .Lpair21.numbered_sgpr, 6
	.set .Lpair21.num_named_barrier, 0
	.set .Lpair21.private_seg_size, 0
	.set .Lpair21.uses_vcc, 0
	.set .Lpair21.uses_flat_scratch, 0
	.set .Lpair21.has_dyn_sized_stack, 0
	.set .Lpair21.has_recursion, 0
	.set .Lpair21.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 204
; TotalNumSgprs: 10
; NumVgprs: 6
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
	.text
	.globl	pair22                          ; -- Begin function pair22
	.p2align	8
	.type	pair22,@function
pair22:                                 ; @pair22
; %bb.0:
	v_mov_b32_e32 v3, v0
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v1, 1, v3
	v_lshlrev_b32_e32 v2, 8, v3
	v_lshlrev_b32_e32 v4, 5, v3
	v_and_b32_e32 v1, 24, v1
	v_and_b32_e32 v2, 0x300, v2
	v_and_b32_e32 v4, 0x60, v4
	v_or3_b32 v4, v1, v2, v4
	v_lshlrev_b32_e32 v1, 2, v0
	v_and_b32_e32 v1, 12, v1
	v_and_or_b32 v1, v0, 1, v1
	v_add_u32_e32 v1, 1, v1
	v_lshrrev_b32_e32 v1, 1, v1
	v_mov_b32_e32 v2, 0
	;;#ASMSTART
	;;#ASMEND
	v_bfe_u32 v2, v3, 1, 1
	v_add_lshl_u32 v1, v1, v2, 6
	v_lshrrev_b32_e32 v2, 2, v3
	v_and_b32_e32 v2, 12, v2
	v_sub_u32_e32 v1, v4, v1
	v_lshlrev_b32_e32 v4, 1, v2
	v_bfe_u32 v3, v3, 5, 1
	v_lshl_add_u32 v4, v1, 1, v4
	v_mul_u32_u24_e32 v3, 0x7f0, v3
	v_add_u32_e32 v1, v1, v2
	v_lshl_add_u32 v1, v1, 1, v3
	v_add_u32_e32 v2, 64, v1
	v_lshrrev_b32_e32 v5, 6, v2
	v_and_b32_e32 v5, 8, v5
	v_xor_b32_e32 v2, v5, v2
	v_sub_u32_e32 v1, v2, v1
	v_add3_u32 v1, v4, v3, v1
	v_add_u32_e32 v1, 0x1000, v1
	s_load_dwordx2 s[0:1], s[4:5], 0x0
	ds_read2_b64 v[1:4], v1 offset0:128 offset1:144
	v_lshlrev_b32_e32 v0, 4, v0
	s_waitcnt lgkmcnt(0)
	global_store_dwordx4 v0, v[1:4], s[0:1]
	s_endpgm
.Lfunc_end22:
	.size	pair22, .Lfunc_end22-pair22
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel pair22
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
	.set .Lpair22.num_vgpr, 6
	.set .Lpair22.num_agpr, 0
	.set .Lpair22.numbered_sgpr, 6
	.set .Lpair22.num_named_barrier, 0
	.set .Lpair22.private_seg_size, 0
	.set .Lpair22.uses_vcc, 0
	.set .Lpair22.uses_flat_scratch, 0
	.set .Lpair22.has_dyn_sized_stack, 0
	.set .Lpair22.has_recursion, 0
	.set .Lpair22.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 212
; TotalNumSgprs: 10
; NumVgprs: 6
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
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair0
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair0.kd
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair1
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair1.kd
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair2
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair2.kd
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair3
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair3.kd
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair4
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair4.kd
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair5
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair5.kd
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair6
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair6.kd
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair7
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair7.kd
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair8
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair8.kd
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair9
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair9.kd
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair10
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair10.kd
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair11
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair11.kd
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair12
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair12.kd
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair13
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair13.kd
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair14
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair14.kd
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair15
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair15.kd
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair16
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair16.kd
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair17
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair17.kd
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair18
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair18.kd
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair19
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair19.kd
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair20
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair20.kd
    .uses_dynamic_stack: false
    .vgpr_count:     7
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair21
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair21.kd
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
  - .args:
      - .address_space:  global
        .name:           probe.out
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .max_flat_workgroup_size: 256
    .name:           pair22
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         pair22.kd
    .uses_dynamic_stack: false
    .vgpr_count:     6
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgpu-amd-amdhsa-unknown-gfx900
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
