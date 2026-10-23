	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx90a"
	.amdhsa_code_object_version 6
	.text
	.globl	test                            ; -- Begin function test
	.p2align	8
	.type	test,@function
test:                                   ; @test
; %bb.0:
	s_load_dword s2, s[4:5], 0x0
	s_load_dwordx2 s[0:1], s[4:5], 0x8
	v_mov_b32_e32 v0, 0
	v_mov_b32_e32 v1, v0
	v_mov_b32_e32 v2, v0
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, s2
	ds_read2_b64 v[4:7], v3 offset1:1
	ds_read2_b64 v[8:11], v3 offset0:2 offset1:3
	ds_read2_b64 v[12:15], v3 offset0:4 offset1:5
	v_mov_b32_e32 v3, v0
	v_pk_mov_b32 v[18:19], v[2:3], v[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[16:17], v[0:1], v[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[16:19], v[4:5], v[6:7], v[16:19]
	;;#ASMEND
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[16:19], v[8:9], v[10:11], v[16:19]
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[16:19], v[12:13], v[14:15], v[16:19]
	;;#ASMEND
	global_store_dwordx4 v0, v[16:19], s[0:1]
	s_endpgm
.Lfunc_end0:
	.size	test, .Lfunc_end0-test
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel test
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 24
		.amdhsa_user_sgpr_count 6
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_flat_scratch_init 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 20
		.amdhsa_next_free_sgpr 6
		.amdhsa_accum_offset 20
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
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
	.set .Ltest.num_vgpr, 20
	.set .Ltest.num_agpr, 0
	.set .Ltest.numbered_sgpr, 6
	.set .Ltest.num_named_barrier, 0
	.set .Ltest.private_seg_size, 0
	.set .Ltest.uses_vcc, 0
	.set .Ltest.uses_flat_scratch, 0
	.set .Ltest.has_dyn_sized_stack, 0
	.set .Ltest.has_recursion, 0
	.set .Ltest.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 128
; TotalNumSgprs: 10
; NumVgprs: 20
; NumAgprs: 0
; TotalNumVgprs: 20
; ScratchSize: 0
; MemoryBound: 1
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 1
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 10
; NumVGPRsForWavesPerEU: 20
; AccumOffset: 20
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 4
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.text
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  local
        .name:           p
        .offset:         0
        .pointee_align:  1
        .size:           4
        .value_kind:     dynamic_shared_pointer
      - .address_space:  generic
        .name:           out
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .name:           index
        .offset:         16
        .size:           8
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .max_flat_workgroup_size: 1024
    .name:           test
    .private_segment_fixed_size: 0
    .sgpr_count:     10
    .sgpr_spill_count: 0
    .symbol:         test.kd
    .uses_dynamic_stack: false
    .vgpr_count:     20
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx90a
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
