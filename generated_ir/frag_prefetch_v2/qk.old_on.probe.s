	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx90a"
	.amdhsa_code_object_version 6
	.text
	.globl	Attn_p2                         ; -- Begin function Attn_p2
	.p2align	8
	.type	Attn_p2,@function
Attn_p2:                                ; @Attn_p2
.Lfunc_begin0:
	.cfi_sections .debug_frame
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	.file	1 "3rd/deepgengraph/test" "test_input_sd16.mlir"
	.loc	1 22 3 prologue_end             ; test_input_sd16.mlir:22:3
	s_ashr_i32 s13, s15, 31
	.loc	1 66 22                         ; test_input_sd16.mlir:66:22
	s_xor_b32 s4, s13, s15
	s_ashr_i32 s5, s4, 31
	s_lshr_b32 s5, s5, 27
	s_add_i32 s4, s4, s5
	s_ashr_i32 s4, s4, 5
	.loc	1 22 3                          ; test_input_sd16.mlir:22:3
	s_load_dwordx4 s[0:3], s[8:9], 0x0
	s_load_dwordx2 s[10:11], s[8:9], 0x18
	.loc	1 66 22                         ; test_input_sd16.mlir:66:22
	s_xor_b32 s4, s4, s13
	s_add_i32 s5, s4, s14
	s_mov_b32 s12, s15
	s_lshl_b32 s5, s5, 20
	s_lshl_b32 s4, s4, 20
	s_lshl_b32 s15, s15, 15
	s_and_b32 s5, s5, 0x1f00000
	s_sub_i32 s4, s15, s4
	s_add_i32 s16, s5, s4
	s_waitcnt lgkmcnt(0)
	s_mov_b32 s4, s0
	s_mov_b32 s5, s1
	.loc	1 67 22                         ; test_input_sd16.mlir:67:22
	s_mov_b32 s0, s13
	s_mov_b32 s1, s13
	s_xor_b64 s[0:1], s[0:1], s[12:13]
	s_lshr_b32 s1, s1, 20
	s_add_i32 s0, s0, s1
	s_lshr_b32 s0, s0, 12
	s_xor_b32 s0, s0, s13
	.loc	1 66 22                         ; test_input_sd16.mlir:66:22
	;;#ASMSTART
	;;#ASMEND
	v_lshlrev_b32_e32 v10, 7, v0
	.loc	1 67 22                         ; test_input_sd16.mlir:67:22
	s_add_i32 s0, s0, s14
	s_lshr_b64 s[8:9], s[12:13], 5
	.loc	1 66 22                         ; test_input_sd16.mlir:66:22
	v_lshlrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v2, 0x3f80, v10
	v_lshrrev_b32_e32 v3, 4, v0
	v_or_b32_e32 v5, 0x100, v0
	.loc	1 67 22                         ; test_input_sd16.mlir:67:22
	s_lshl_b32 s0, s0, 19
	s_lshl_b32 s9, s8, 20
	.loc	1 66 22                         ; test_input_sd16.mlir:66:22
	v_and_b32_e32 v1, 0xc00, v1
	v_and_or_b32 v3, v3, 8, v2
	v_lshlrev_b32_e32 v6, 4, v5
	v_lshrrev_b32_e32 v5, 4, v5
	.loc	1 67 22                         ; test_input_sd16.mlir:67:22
	s_and_b32 s1, s8, 0x7f
	s_and_b32 s0, s0, 0xf80000
	.loc	1 66 22                         ; test_input_sd16.mlir:66:22
	v_lshl_or_b32 v4, v3, 1, s16
	s_mov_b32 s7, 0x20000
	s_brev_b32 s6, 64
	v_and_b32_e32 v6, 0x1c00, v6
	v_and_or_b32 v2, v5, 24, v2
	.loc	1 67 22                         ; test_input_sd16.mlir:67:22
	s_sub_i32 s8, s15, s9
	s_or_b32 s9, s0, s1
	.loc	1 66 22                         ; test_input_sd16.mlir:66:22
	v_readfirstlane_b32 s12, v1
	;;#ASMSTART
	s_mov_b32 m0, s12
	buffer_load_dwordx4 v4, s[4:7], 0 offen lds
	;;#ASMEND
	v_lshl_or_b32 v5, v2, 1, s16
	.loc	1 67 22                         ; test_input_sd16.mlir:67:22
	v_or_b32_e32 v7, 0x2000, v1
	v_add_u32_e32 v3, s9, v3
	.loc	1 66 22                         ; test_input_sd16.mlir:66:22
	v_readfirstlane_b32 s12, v6
	;;#ASMSTART
	s_mov_b32 m0, s12
	buffer_load_dwordx4 v5, s[4:7], 0 offen lds
	;;#ASMEND
	.loc	1 67 22                         ; test_input_sd16.mlir:67:22
	v_add_u32_e32 v1, s9, v2
	v_lshl_add_u32 v3, v3, 1, s8
	s_mov_b32 s0, s2
	s_mov_b32 s1, s3
	s_mov_b32 s2, s6
	s_mov_b32 s3, s7
	v_or_b32_e32 v8, 0x2000, v6
	.loc	1 66 22                         ; test_input_sd16.mlir:66:22
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	s_barrier
	.loc	1 67 22                         ; test_input_sd16.mlir:67:22
	v_readfirstlane_b32 s4, v7
	;;#ASMSTART
	s_mov_b32 m0, s4
	buffer_load_dwordx4 v3, s[0:3], 0 offen lds
	;;#ASMEND
	v_lshl_add_u32 v1, v1, 1, s8
	v_readfirstlane_b32 s4, v8
	;;#ASMSTART
	s_mov_b32 m0, s4
	buffer_load_dwordx4 v1, s[0:3], 0 offen lds
	;;#ASMEND
	.loc	1 68 24                         ; test_input_sd16.mlir:68:24
	v_lshlrev_b32_e32 v1, 5, v0
	v_lshlrev_b32_e32 v2, 3, v0
	v_lshlrev_b32_e32 v11, 1, v0
	v_lshrrev_b32_e32 v16, 2, v0
	v_and_b32_e32 v1, 0x400, v1
	s_movk_i32 s0, 0x180
	v_and_b32_e32 v12, 4, v16
	v_and_b32_e32 v2, 0x78, v2
	v_and_or_b32 v3, v11, s0, v1
	v_or3_b32 v2, v2, v12, v3
	v_lshlrev_b32_e32 v6, 1, v2
	ds_read2st64_b64 v[2:5], v6 offset1:2
	ds_read2st64_b64 v[6:9], v6 offset0:8 offset1:10
	s_movk_i32 s1, 0x3015
	.loc	1 69 26                         ; test_input_sd16.mlir:69:26
	v_and_b32_e32 v17, 24, v11
	.loc	1 67 22                         ; test_input_sd16.mlir:67:22
	;;#ASMSTART
	s_waitcnt vmcnt(0)
	;;#ASMEND
	.loc	1 68 24                         ; test_input_sd16.mlir:68:24
	s_waitcnt lgkmcnt(1)
	v_pk_mul_f16 v40, v2, s1 op_sel_hi:[1,0]
	.loc	1 69 26                         ; test_input_sd16.mlir:69:26
	v_and_or_b32 v2, v10, s0, v17
	v_or3_b32 v1, v2, v12, v1
	v_lshlrev_b32_e32 v58, 1, v1
	.loc	1 67 22                         ; test_input_sd16.mlir:67:22
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 69 26                         ; test_input_sd16.mlir:69:26
	ds_read_b64 v[10:11], v58 offset:8192
	.loc	1 22 3                          ; test_input_sd16.mlir:22:3
	v_mov_b32_e32 v72, 0
	.loc	1 68 24                         ; test_input_sd16.mlir:68:24
	v_pk_mul_f16 v41, v3, s1 op_sel_hi:[1,0]
	v_pk_mul_f16 v43, v7, s1 op_sel_hi:[1,0]
	v_pk_mul_f16 v42, v6, s1 op_sel_hi:[1,0]
	v_pk_mul_f16 v69, v5, s1 op_sel_hi:[1,0]
	v_pk_mul_f16 v68, v4, s1 op_sel_hi:[1,0]
	v_pk_mul_f16 v15, v9, s1 op_sel_hi:[1,0]
	v_pk_mul_f16 v14, v8, s1 op_sel_hi:[1,0]
	; sched_barrier mask(0x00000000)
	.loc	1 69 26                         ; test_input_sd16.mlir:69:26
	ds_read_b64 v[12:13], v58 offset:12288
	; sched_barrier mask(0x00000000)
	s_mov_b32 s0, 0
	s_mov_b32 s1, s0
	s_mov_b32 s2, s0
	s_mov_b32 s3, s0
	v_pk_mov_b32 v[8:9], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[6:7], s[0:1], s[0:1] op_sel:[0,1]
	v_pk_mov_b32 v[4:5], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[2:3], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[6:9], v[40:41], v[10:11], v[6:9]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_lshlrev_b32_e32 v1, 8, v0
	v_lshlrev_b32_e32 v10, 10, v16
	v_lshlrev_b32_e32 v16, 1, v17
	.loc	1 68 24                         ; test_input_sd16.mlir:68:24
	v_lshrrev_b32_e32 v17, 1, v0
	.loc	1 69 26                         ; test_input_sd16.mlir:69:26
	v_sub_u32_e32 v11, v1, v10
	v_and_b32_e32 v17, 24, v17
	.loc	1 68 24                         ; test_input_sd16.mlir:68:24
	v_bfe_i32 v0, v0, 5, 1
	.loc	1 69 26                         ; test_input_sd16.mlir:69:26
	v_or_b32_e32 v11, v11, v16
	v_and_b32_e32 v0, 0x7f0, v0
	v_or_b32_e32 v1, v1, v17
	v_add3_u32 v73, v11, v17, v0
	v_add_u32_e32 v0, v1, v0
	v_sub_u32_e32 v0, v0, v10
	v_add_u32_e32 v74, v0, v16
	v_add_u32_e32 v0, 0x2000, v73
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[6:9], v[42:43], v[12:13], v[6:9]
	
	;;#ASMEND
	v_pk_mov_b32 v[12:13], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[10:11], s[0:1], s[0:1] op_sel:[0,1]
	ds_read2_b64 v[16:19], v0 offset0:8 offset1:16
	v_add_u32_e32 v0, 0x3000, v74
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[10:13], v[40:41], v[16:17], v[10:13]
	
	;;#ASMEND
	v_pk_mov_b32 v[26:27], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[24:25], s[0:1], s[0:1] op_sel:[0,1]
	ds_read2_b64 v[20:23], v0 offset0:8 offset1:16
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[10:13], v[42:43], v[20:21], v[10:13]
	
	;;#ASMEND
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[24:27], v[40:41], v[18:19], v[24:27]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v73 offset:8384
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[24:27], v[42:43], v[22:23], v[24:27]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[20:21], v74 offset:12480
	v_pk_mov_b32 v[18:19], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[16:17], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[16:19], v[40:41], v[0:1], v[16:19]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v58 offset:9216
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[16:19], v[42:43], v[20:21], v[16:19]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[28:29], v58 offset:13312
	v_pk_mov_b32 v[22:23], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[20:21], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[20:23], v[40:41], v[0:1], v[20:23]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v73 offset:9280
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[20:23], v[42:43], v[28:29], v[20:23]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[32:33], v74 offset:13376
	v_pk_mov_b32 v[30:31], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[28:29], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[28:31], v[40:41], v[0:1], v[28:31]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v73 offset:9344
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[28:31], v[42:43], v[32:33], v[28:31]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[36:37], v74 offset:13440
	v_pk_mov_b32 v[34:35], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[32:33], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[32:35], v[40:41], v[0:1], v[32:35]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v73 offset:9408
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[32:35], v[42:43], v[36:37], v[32:35]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[44:45], v74 offset:13504
	v_pk_mov_b32 v[38:39], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[36:37], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[36:39], v[40:41], v[0:1], v[36:39]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v58 offset:8192
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[36:39], v[42:43], v[44:45], v[36:39]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[44:45], v58 offset:12288
	v_pk_mov_b32 v[42:43], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[40:41], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[40:43], v[68:69], v[0:1], v[40:43]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v73 offset:8256
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[40:43], v[14:15], v[44:45], v[40:43]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[48:49], v74 offset:12352
	v_pk_mov_b32 v[46:47], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[44:45], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[44:47], v[68:69], v[0:1], v[44:47]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v73 offset:8320
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[44:47], v[14:15], v[48:49], v[44:47]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[52:53], v74 offset:12416
	v_pk_mov_b32 v[50:51], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[48:49], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[48:51], v[68:69], v[0:1], v[48:51]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v73 offset:8384
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[48:51], v[14:15], v[52:53], v[48:51]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[56:57], v74 offset:12480
	v_pk_mov_b32 v[54:55], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[52:53], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[52:55], v[68:69], v[0:1], v[52:55]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v58 offset:9216
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[52:55], v[14:15], v[56:57], v[52:55]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[60:61], v58 offset:13312
	v_pk_mov_b32 v[58:59], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[56:57], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[56:59], v[68:69], v[0:1], v[56:59]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v73 offset:9280
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[56:59], v[14:15], v[60:61], v[56:59]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[64:65], v74 offset:13376
	v_pk_mov_b32 v[62:63], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[60:61], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[60:63], v[68:69], v[0:1], v[60:63]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v73 offset:9344
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[60:63], v[14:15], v[64:65], v[60:63]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[70:71], v74 offset:13440
	v_pk_mov_b32 v[66:67], s[2:3], s[2:3] op_sel:[0,1]
	v_pk_mov_b32 v[64:65], s[0:1], s[0:1] op_sel:[0,1]
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[64:67], v[68:69], v[0:1], v[64:67]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b64 v[0:1], v73 offset:9408
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[64:67], v[14:15], v[70:71], v[64:67]
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[2:5], v[68:69], v[0:1], v[2:5]
	
	;;#ASMEND
	ds_read_b64 v[70:71], v74 offset:13504
	global_store_dwordx4 v72, v[36:39], s[10:11] offset:112
	global_store_dwordx4 v72, v[32:35], s[10:11] offset:96
	global_store_dwordx4 v72, v[28:31], s[10:11] offset:80
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	v_mfma_f32_16x16x16f16 v[2:5], v[14:15], v[70:71], v[2:5]
	
	;;#ASMEND
	global_store_dwordx4 v72, v[20:23], s[10:11] offset:64
	global_store_dwordx4 v72, v[16:19], s[10:11] offset:48
	global_store_dwordx4 v72, v[24:27], s[10:11] offset:32
	global_store_dwordx4 v72, v[10:13], s[10:11] offset:16
	global_store_dwordx4 v72, v[6:9], s[10:11]
	global_store_dwordx4 v72, v[60:63], s[10:11] offset:208
	global_store_dwordx4 v72, v[56:59], s[10:11] offset:192
	global_store_dwordx4 v72, v[52:55], s[10:11] offset:176
	global_store_dwordx4 v72, v[48:51], s[10:11] offset:160
	global_store_dwordx4 v72, v[44:47], s[10:11] offset:144
	global_store_dwordx4 v72, v[40:43], s[10:11] offset:128
	global_store_dwordx4 v72, v[64:67], s[10:11] offset:224
	global_store_dwordx4 v72, v[2:5], s[10:11] offset:240
	s_endpgm
.Ltmp0:
.Lfunc_end0:
	.size	Attn_p2, .Lfunc_end0-Attn_p2
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel Attn_p2
		.amdhsa_group_segment_fixed_size 16384
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 288
		.amdhsa_user_sgpr_count 14
		.amdhsa_user_sgpr_private_segment_buffer 1
		.amdhsa_user_sgpr_dispatch_ptr 1
		.amdhsa_user_sgpr_queue_ptr 1
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 1
		.amdhsa_user_sgpr_flat_scratch_init 1
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_system_sgpr_private_segment_wavefront_offset 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 97
		.amdhsa_next_free_sgpr 98
		.amdhsa_accum_offset 76
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
	.set .LAttn_p2.num_vgpr, 75
	.set .LAttn_p2.num_agpr, 0
	.set .LAttn_p2.numbered_sgpr, 17
	.set .LAttn_p2.num_named_barrier, 0
	.set .LAttn_p2.private_seg_size, 0
	.set .LAttn_p2.uses_vcc, 0
	.set .LAttn_p2.uses_flat_scratch, 0
	.set .LAttn_p2.has_dyn_sized_stack, 0
	.set .LAttn_p2.has_recursion, 0
	.set .LAttn_p2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 1700
; TotalNumSgprs: 21
; NumVgprs: 75
; NumAgprs: 0
; TotalNumVgprs: 75
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 16384 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 12
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 97
; AccumOffset: 76
; Occupancy: 4
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 14
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 18
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
	.section	.debug_abbrev,"",@progbits
	.byte	1                               ; Abbreviation Code
	.byte	17                              ; DW_TAG_compile_unit
	.byte	0                               ; DW_CHILDREN_no
	.byte	37                              ; DW_AT_producer
	.byte	14                              ; DW_FORM_strp
	.byte	19                              ; DW_AT_language
	.byte	5                               ; DW_FORM_data2
	.byte	3                               ; DW_AT_name
	.byte	14                              ; DW_FORM_strp
	.byte	16                              ; DW_AT_stmt_list
	.byte	23                              ; DW_FORM_sec_offset
	.byte	27                              ; DW_AT_comp_dir
	.byte	14                              ; DW_FORM_strp
	.byte	17                              ; DW_AT_low_pc
	.byte	1                               ; DW_FORM_addr
	.byte	18                              ; DW_AT_high_pc
	.byte	6                               ; DW_FORM_data4
	.byte	0                               ; EOM(1)
	.byte	0                               ; EOM(2)
	.byte	0                               ; EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 ; Length of Unit
.Ldebug_info_start0:
	.short	4                               ; DWARF version number
	.long	.debug_abbrev                   ; Offset Into Abbrev. Section
	.byte	8                               ; Address Size (in bytes)
	.byte	1                               ; Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
	.long	.Linfo_string0                  ; DW_AT_producer
	.short	2                               ; DW_AT_language
	.long	.Linfo_string1                  ; DW_AT_name
	.long	.Lline_table_start0             ; DW_AT_stmt_list
	.long	.Linfo_string2                  ; DW_AT_comp_dir
	.quad	.Lfunc_begin0                   ; DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       ; DW_AT_high_pc
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"DeepGenGraph MLIR"             ; string offset=0 ; DeepGenGraph MLIR
.Linfo_string1:
	.asciz	"test_input_sd16.mlir"          ; string offset=18 ; test_input_sd16.mlir
.Linfo_string2:
	.asciz	"3rd/deepgengraph/test"         ; string offset=39 ; 3rd/deepgengraph/test
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .name:           v0
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           v1
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           v2
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .name:           v3
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
      - .offset:         32
        .size:           4
        .value_kind:     hidden_block_count_x
      - .offset:         36
        .size:           4
        .value_kind:     hidden_block_count_y
      - .offset:         40
        .size:           4
        .value_kind:     hidden_block_count_z
      - .offset:         44
        .size:           2
        .value_kind:     hidden_group_size_x
      - .offset:         46
        .size:           2
        .value_kind:     hidden_group_size_y
      - .offset:         48
        .size:           2
        .value_kind:     hidden_group_size_z
      - .offset:         50
        .size:           2
        .value_kind:     hidden_remainder_x
      - .offset:         52
        .size:           2
        .value_kind:     hidden_remainder_y
      - .offset:         54
        .size:           2
        .value_kind:     hidden_remainder_z
      - .offset:         72
        .size:           8
        .value_kind:     hidden_global_offset_x
      - .offset:         80
        .size:           8
        .value_kind:     hidden_global_offset_y
      - .offset:         88
        .size:           8
        .value_kind:     hidden_global_offset_z
      - .offset:         96
        .size:           2
        .value_kind:     hidden_grid_dims
      - .offset:         112
        .size:           8
        .value_kind:     hidden_hostcall_buffer
      - .offset:         120
        .size:           8
        .value_kind:     hidden_multigrid_sync_arg
      - .offset:         128
        .size:           8
        .value_kind:     hidden_heap_v1
      - .offset:         136
        .size:           8
        .value_kind:     hidden_default_queue
      - .offset:         144
        .size:           8
        .value_kind:     hidden_completion_action
      - .offset:         232
        .size:           8
        .value_kind:     hidden_queue_ptr
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 288
    .max_flat_workgroup_size: 256
    .name:           Attn_p2
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     21
    .sgpr_spill_count: 0
    .symbol:         Attn_p2.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     75
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx90a
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
