[BEGIN] 2026/10/14 18:33:22
root@dcu1:/workspace/deepgengraph# cat k_wait.hsaco.amdgcn 
	.amdgcn_target "amdgcn-amd-amdhsa--gfx936"
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
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; 
	.cfi_undefined 16
	.file	1 "/data2/xsl/DeepGenGraph/3rd/deepgengraph/test" "test_input_spidD.mlir"
	.loc	1 22 3 prologue_end             ; test_input_spidD.mlir:22:3
	s_ashr_i32 s17, s7, 31
	.loc	1 66 22                         ; test_input_spidD.mlir:66:22
	s_xor_b32 s0, s17, s7
	s_ashr_i32 s1, s0, 31
	s_lshr_b32 s1, s1, 27
	s_add_i32 s0, s0, s1
	s_ashr_i32 s0, s0, 5
	s_xor_b32 s20, s0, s17
	s_mov_b32 s16, s7
	.loc	1 22 3                          ; test_input_spidD.mlir:22:3
	s_load_dwordx8 s[8:15], s[4:5], 0x0
	.loc	1 66 22                         ; test_input_spidD.mlir:66:22
	s_ashr_i32 s21, s20, 31
	.loc	1 46 17                         ; test_input_spidD.mlir:46:17
	s_lshl_b64 s[2:3], s[16:17], 7
	.loc	1 66 22                         ; test_input_spidD.mlir:66:22
	s_add_u32 s7, s20, s6
	s_lshl_b64 s[0:1], s[20:21], 12
	s_sub_u32 s30, 0, s0
	s_subb_u32 s31, 0, s1
	s_mov_b64 s[18:19], 0
	.loc	1 68 24                         ; test_input_spidD.mlir:68:24
	v_mov_b32_e32 v196, 0
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	s_cmp_gt_i32 s16, -1
	v_mbcnt_lo_u32_b32 v209, -1, 0
	s_cbranch_scc0 .LBB0_9
; %bb.1:                                ; %.lr.ph
	.loc	1 66 22                         ; test_input_spidD.mlir:66:22
	s_and_b32 s22, s7, 31
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	s_or_b32 s0, s2, 0x7f
	v_mov_b32_e32 v1, s0
	s_add_u32 s0, s2, 0x80
	v_mov_b32_e32 v2, s3
	s_addc_u32 s1, s3, 0
	s_nop 0
	v_cmp_ge_u64_e64 s[0:1], s[0:1], v[1:2]
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	v_lshrrev_b32_e32 v1, 3, v0
	v_lshrrev_b32_e32 v2, 1, v0
	v_mul_u32_u24_e32 v1, 0xffdc, v1
	v_lshl_add_u32 v2, v0, 2, v2
	v_add_u32_e32 v1, v2, v1
	v_add_u16_e32 v5, 1, v1
	v_ashrrev_i16_e32 v6, 15, v5
	v_lshrrev_b16_e32 v6, 14, v6
	v_add_u16_e32 v5, v5, v6
	v_ashrrev_i16_e32 v5, 2, v5
	v_sub_u16_e32 v5, 0, v5
	v_add_u16_e32 v1, 2, v1
	v_bfe_i32 v197, v5, 0, 16
	v_ashrrev_i16_e32 v5, 15, v1
	v_lshrrev_b16_e32 v5, 14, v5
	v_add_u16_e32 v1, v1, v5
	v_ashrrev_i16_e32 v1, 2, v1
	v_sub_u16_e32 v1, 0, v1
	.loc	1 68 24                         ; test_input_spidD.mlir:68:24
	v_lshrrev_b32_e32 v3, 4, v0
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	v_bfe_i32 v199, v1, 0, 16
	.loc	1 69 26                         ; test_input_spidD.mlir:69:26
	v_lshrrev_b32_e32 v1, 2, v0
	v_bfe_u32 v3, v3, 1, 1
	v_mul_u32_u24_e32 v6, 0xffee, v1
	v_add3_u32 v2, v2, v6, v3
	v_lshrrev_b16_e32 v3, 15, v2
	v_add_u16_e32 v2, v2, v3
	v_ashrrev_i16_e32 v2, 1, v2
	v_sub_u16_e32 v2, 0, v2
	v_bfe_u32 v5, v0, 2, 2
	v_bfe_i32 v201, v2, 0, 16
	v_mul_u32_u24_e32 v2, 18, v0
	v_mul_u32_u24_e32 v3, 0xffb8, v1
	v_add3_u32 v2, v3, v2, v5
	v_ashrrev_i16_e32 v3, 15, v2
	v_lshrrev_b16_e32 v3, 14, v3
	v_add_u16_e32 v2, v2, v3
	v_ashrrev_i16_e32 v2, 2, v2
	v_sub_u16_e32 v2, 0, v2
	v_bfe_i32 v203, v2, 0, 16
	v_mov_b32_e32 v2, s3
	v_add_co_u32_e32 v3, vcc, s2, v0
	v_addc_co_u32_e32 v2, vcc, 0, v2, vcc
	v_and_b32_e32 v5, 0xf0, v0
	v_sub_co_u32_e32 v3, vcc, v3, v5
	v_and_b32_e32 v1, 48, v1
	v_subbrev_co_u32_e32 v2, vcc, 0, v2, vcc
	v_add_co_u32_e32 v1, vcc, v3, v1
	s_mov_b32 s36, 0
	.loc	1 66 22                         ; test_input_spidD.mlir:66:22
	s_add_u32 s4, s30, s2
	v_addc_co_u32_e32 v2, vcc, 0, v2, vcc
	s_addc_u32 s5, s31, s3
	s_lshl_b32 s22, s22, 20
	v_add_co_u32_e32 v205, vcc, 1, v1
	s_mov_b32 s42, s36
	s_mov_b32 s43, s36
	v_addc_co_u32_e32 v206, vcc, 0, v2, vcc
	s_movk_i32 s23, 0x41
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	s_waitcnt lgkmcnt(0)
	s_add_u32 s22, s8, s22
	s_mov_b32 s37, s36
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s40, s36
	s_mov_b32 s41, s36
	v_mov_b64_e32 v[191:192], s[42:43]
	v_add_co_u32_e32 v207, vcc, s23, v1
	s_addc_u32 s23, s9, 0
	s_lshl_b64 s[8:9], s[16:17], 15
	v_mov_b64_e32 v[185:186], s[36:37]
	s_add_u32 s22, s22, s8
	v_mov_b64_e32 v[189:190], s[40:41]
	v_mov_b64_e32 v[187:188], s[38:39]
	s_addc_u32 s23, s23, s9
	s_lshl_b64 s[8:9], s[20:21], 20
	v_mov_b64_e32 v[153:154], v[185:186]
	v_mov_b64_e32 v[177:178], v[185:186]
	v_mov_b64_e32 v[145:146], v[185:186]
	v_mov_b64_e32 v[169:170], v[185:186]
	v_mov_b64_e32 v[137:138], v[185:186]
	v_mov_b64_e32 v[161:162], v[185:186]
	v_mov_b64_e32 v[129:130], v[185:186]
	.loc	1 68 24                         ; test_input_spidD.mlir:68:24
	v_bfe_u32 v4, v0, 4, 2
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	s_sub_u32 s33, s22, s8
	v_mov_b64_e32 v[155:156], v[187:188]
	v_mov_b64_e32 v[157:158], v[189:190]
	v_mov_b64_e32 v[159:160], v[191:192]
	v_mov_b64_e32 v[179:180], v[187:188]
	v_mov_b64_e32 v[181:182], v[189:190]
	v_mov_b64_e32 v[183:184], v[191:192]
	v_mov_b64_e32 v[147:148], v[187:188]
	v_mov_b64_e32 v[149:150], v[189:190]
	v_mov_b64_e32 v[151:152], v[191:192]
	v_mov_b64_e32 v[171:172], v[187:188]
	v_mov_b64_e32 v[173:174], v[189:190]
	v_mov_b64_e32 v[175:176], v[191:192]
	v_mov_b64_e32 v[139:140], v[187:188]
	v_mov_b64_e32 v[141:142], v[189:190]
	v_mov_b64_e32 v[143:144], v[191:192]
	v_mov_b64_e32 v[163:164], v[187:188]
	v_mov_b64_e32 v[165:166], v[189:190]
	v_mov_b64_e32 v[167:168], v[191:192]
	v_mov_b64_e32 v[131:132], v[187:188]
	v_mov_b64_e32 v[133:134], v[189:190]
	v_mov_b64_e32 v[135:136], v[191:192]
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	v_ashrrev_i32_e32 v198, 31, v197
	v_ashrrev_i32_e32 v200, 31, v199
	.loc	1 69 26                         ; test_input_spidD.mlir:69:26
	v_ashrrev_i32_e32 v202, 31, v201
	v_ashrrev_i32_e32 v204, 31, v203
	v_lshlrev_b32_e32 v211, 4, v4
	v_addc_co_u32_e32 v208, vcc, 0, v2, vcc
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	s_subb_u32 s34, s23, s9
	s_mov_b32 s35, 0x30153015
	s_movk_i32 s68, 0xf980
	s_movk_i32 s69, 0xffee
	s_movk_i32 s70, 0xffb8
	v_mbcnt_hi_u32_b32 v210, -1, v209
	s_mov_b32 s71, 0x5040100
	s_movk_i32 s72, 0xfde4
	s_mov_b32 s73, 0x7060302
	s_movk_i32 s74, 0xfcc4
	v_mov_b32_e32 v212, 6
	v_mov_b32_e32 v213, 0x200
	v_mov_b32_e32 v214, 0x248
	v_mov_b32_e32 v215, 0xff800000
	v_mov_b32_e32 v193, v196
	v_mov_b32_e32 v194, v196
.LBB0_2:                                ; =>This Loop Header: Depth=1
                                        ;     Child Loop BB0_3 Depth 2
	.loc	1 0 24 is_stmt 0                ; test_input_spidD.mlir:0:24
	s_mov_b32 s37, s36
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	s_lshr_b64 s[20:21], s[18:19], 5
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s40, s36
	s_mov_b32 s41, s36
	s_mov_b32 s42, s36
	s_mov_b32 s43, s36
	s_mov_b32 s44, s36
	s_mov_b32 s45, s36
	s_mov_b32 s46, s36
	s_mov_b32 s47, s36
	s_mov_b32 s48, s36
	s_mov_b32 s49, s36
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
	s_mov_b32 s52, s36
	s_mov_b32 s53, s36
	s_mov_b32 s54, s36
	s_mov_b32 s55, s36
	s_mov_b32 s56, s36
	s_mov_b32 s57, s36
	s_mov_b32 s58, s36
	s_mov_b32 s59, s36
	s_mov_b32 s60, s36
	s_mov_b32 s61, s36
	s_mov_b32 s62, s36
	s_mov_b32 s63, s36
	s_mov_b32 s64, s36
	s_mov_b32 s65, s36
	s_mov_b32 s66, s36
	s_mov_b32 s67, s36
	v_mov_b64_e32 v[33:34], s[36:37]
	s_lshl_b64 s[22:23], s[18:19], 7
	.loc	1 67 22 is_stmt 1               ; test_input_spidD.mlir:67:22
	s_lshl_b64 s[8:9], s[20:21], 12
	v_mov_b64_e32 v[35:36], s[38:39]
	v_mov_b64_e32 v[37:38], s[40:41]
	v_mov_b64_e32 v[39:40], s[42:43]
	v_mov_b64_e32 v[41:42], s[44:45]
	v_mov_b64_e32 v[43:44], s[46:47]
	v_mov_b64_e32 v[45:46], s[48:49]
	v_mov_b64_e32 v[47:48], s[50:51]
	v_mov_b64_e32 v[49:50], s[52:53]
	v_mov_b64_e32 v[51:52], s[54:55]
	v_mov_b64_e32 v[53:54], s[56:57]
	v_mov_b64_e32 v[55:56], s[58:59]
	v_mov_b64_e32 v[57:58], s[60:61]
	v_mov_b64_e32 v[59:60], s[62:63]
	v_mov_b64_e32 v[61:62], s[64:65]
	v_mov_b64_e32 v[63:64], s[66:67]
	s_sub_u32 s8, s22, s8
	s_nop 0
	v_mov_b64_e32 v[1:2], v[33:34]
	s_subb_u32 s9, s23, s9
	v_mov_b64_e32 v[3:4], v[35:36]
	v_mov_b64_e32 v[5:6], v[37:38]
	v_mov_b64_e32 v[7:8], v[39:40]
	v_mov_b64_e32 v[9:10], v[41:42]
	v_mov_b64_e32 v[11:12], v[43:44]
	v_mov_b64_e32 v[13:14], v[45:46]
	v_mov_b64_e32 v[15:16], v[47:48]
	v_mov_b64_e32 v[17:18], v[49:50]
	v_mov_b64_e32 v[19:20], v[51:52]
	v_mov_b64_e32 v[21:22], v[53:54]
	v_mov_b64_e32 v[23:24], v[55:56]
	v_mov_b64_e32 v[25:26], v[57:58]
	v_mov_b64_e32 v[27:28], v[59:60]
	v_mov_b64_e32 v[29:30], v[61:62]
	v_mov_b64_e32 v[31:32], v[63:64]
	s_lshl_b64 s[8:9], s[8:9], 8
	s_mov_b64 s[24:25], 0
	s_mov_b64 s[26:27], s[20:21]
	s_mov_b64 s[28:29], s[18:19]
.LBB0_3:                                ;   Parent Loop BB0_2 Depth=1
                                        ; =>  This Inner Loop Header: Depth=2
	; sched_barrier mask(0x00000000)
	.loc	1 66 22                         ; test_input_spidD.mlir:66:22
	v_mov_b32_e32 v195, v0
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_lshrrev_b32_e32 v73, 1, v195
	v_mov_b32_e32 v74, v196
	s_nop 1
	v_lshlrev_b64 v[65:66], 8, v[73:74]
	v_and_b32_e32 v67, 1, v195
	s_add_u32 s37, s33, s24
	v_lshlrev_b32_e32 v87, 2, v195
	v_lshl_or_b32 v65, v67, 2, v65
	s_addc_u32 s38, s34, s25
	v_and_b32_e32 v69, 0x300, v87
	v_mov_b32_e32 v67, s38
	v_add_co_u32_e32 v65, vcc, s37, v65
	v_readfirstlane_b32 s21, v69
	v_addc_co_u32_e32 v66, vcc, v67, v66, vcc
	; sched_barrier mask(0x00000000)
	s_nop 1
	v_mov_b64_e32 v[67:68], v[65:66]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	v_or_b32_e32 v70, 0x100, v195
	global_load_dword v[67:68], off lds
	v_lshlrev_b32_e32 v67, 2, v70
	v_and_b32_e32 v71, 0x700, v67
	v_add_co_u32_e32 v67, vcc, 8, v65
	v_readfirstlane_b32 s21, v71
	v_addc_co_u32_e32 v68, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	s_nop 0
	global_load_dword v[67:68], off lds
	v_or_b32_e32 v67, 0x800, v69
	v_readfirstlane_b32 s21, v67
	v_add_co_u32_e32 v67, vcc, 16, v65
	v_addc_co_u32_e32 v68, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	v_or_b32_e32 v72, 0x300, v195
	global_load_dword v[67:68], off lds
	v_lshlrev_b32_e32 v67, 2, v72
	v_and_b32_e32 v74, 0xf00, v67
	v_add_co_u32_e32 v67, vcc, 24, v65
	v_readfirstlane_b32 s21, v74
	v_addc_co_u32_e32 v68, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	s_nop 0
	global_load_dword v[67:68], off lds
	v_or_b32_e32 v67, 0x1000, v69
	v_readfirstlane_b32 s21, v67
	v_add_co_u32_e32 v67, vcc, 32, v65
	v_addc_co_u32_e32 v68, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	v_or_b32_e32 v75, 0x500, v195
	global_load_dword v[67:68], off lds
	v_lshlrev_b32_e32 v67, 2, v75
	v_and_b32_e32 v76, 0x1700, v67
	v_add_co_u32_e32 v67, vcc, 40, v65
	v_readfirstlane_b32 s21, v76
	v_addc_co_u32_e32 v68, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	s_nop 0
	global_load_dword v[67:68], off lds
	v_or_b32_e32 v67, 0x1800, v69
	v_readfirstlane_b32 s21, v67
	v_add_co_u32_e32 v67, vcc, 48, v65
	v_addc_co_u32_e32 v68, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	v_add_co_u32_e32 v65, vcc, 56, v65
	global_load_dword v[67:68], off lds
	v_or_b32_e32 v67, 0x700, v195
	v_lshlrev_b32_e32 v68, 2, v67
	v_and_b32_e32 v68, 0x1f00, v68
	v_readfirstlane_b32 s21, v68
	v_addc_co_u32_e32 v66, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	.loc	1 67 22                         ; test_input_spidD.mlir:67:22
	s_lshr_b32 s21, s28, 12
	.loc	1 66 22                         ; test_input_spidD.mlir:66:22
	global_load_dword v[65:66], off lds
	.loc	1 67 22                         ; test_input_spidD.mlir:67:22
	s_add_i32 s21, s21, s6
	s_lshl_b32 s21, s21, 20
	s_and_b32 s37, s26, 0x7f
	s_and_b32 s21, s21, 0x1f00000
	s_add_u32 s21, s10, s21
	s_addc_u32 s38, s11, 0
	s_lshl_b32 s37, s37, 1
	s_add_u32 s21, s21, s37
	s_addc_u32 s37, s38, 0
	s_add_u32 s21, s21, s8
	v_lshlrev_b32_e32 v88, 4, v195
	v_lshrrev_b32_e32 v79, 4, v195
	v_lshrrev_b32_e32 v80, 7, v195
	s_addc_u32 s37, s37, s9
	v_or_b32_e32 v65, 0x2000, v69
	v_and_b32_e32 v89, 32, v88
	v_sub_u32_e32 v66, v79, v80
	v_readfirstlane_b32 s38, v65
	v_mov_b32_e32 v65, s37
	v_add_co_u32_e32 v77, vcc, s21, v89
	v_lshrrev_b32_e32 v85, 2, v195
	v_lshlrev_b32_e32 v66, 2, v66
	v_lshrrev_b32_e32 v82, 5, v195
	v_addc_co_u32_e32 v78, vcc, 0, v65, vcc
	v_and_b32_e32 v65, 56, v85
	v_and_b32_e32 v81, 4, v66
	v_sub_u32_e32 v66, v85, v82
	v_and_or_b32 v83, v66, 3, v65
	v_or_b32_e32 v65, v83, v81
	v_lshlrev_b32_e32 v84, 8, v65
	v_add_co_u32_e32 v65, vcc, v77, v84
	v_addc_co_u32_e32 v66, vcc, 0, v78, vcc
	v_and_b32_e32 v90, 4, v87
	v_add_co_u32_e32 v65, vcc, v65, v90
	.loc	1 66 22                         ; test_input_spidD.mlir:66:22
	; asyncmark
	; wait_asyncmark(0)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	.loc	1 67 22                         ; test_input_spidD.mlir:67:22
	v_mov_b32_e32 v86, v196
	v_addc_co_u32_e32 v66, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s38
	s_nop 0
	global_load_dword v[65:66], off lds
	v_or_b32_e32 v65, 0x2000, v71
	v_readfirstlane_b32 s21, v65
	v_lshrrev_b32_e32 v65, 2, v70
	v_and_b32_e32 v66, 0x78, v65
	v_sub_u32_e32 v65, v65, v82
	v_and_b32_e32 v65, 3, v65
	v_add_co_u32_e32 v71, vcc, v77, v90
	v_or3_b32 v65, v66, v65, v81
	v_addc_co_u32_e32 v91, vcc, 0, v78, vcc
	v_lshlrev_b32_e32 v65, 8, v65
	v_add_co_u32_e32 v65, vcc, v71, v65
	v_addc_co_u32_e32 v66, vcc, 0, v91, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	s_nop 0
	global_load_dword v[65:66], off lds
	v_or_b32_e32 v65, 0x2800, v69
	v_readfirstlane_b32 s21, v65
	v_add_co_u32_e32 v65, vcc, v71, v84
	v_addc_co_u32_e32 v66, vcc, 0, v91, vcc
	v_add_co_u32_e32 v65, vcc, 8, v65
	v_addc_co_u32_e32 v66, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	s_nop 0
	global_load_dword v[65:66], off lds
	v_or_b32_e32 v65, 0x2000, v74
	v_readfirstlane_b32 s21, v65
	v_lshrrev_b32_e32 v65, 2, v72
	v_and_b32_e32 v66, 0x78, v65
	v_sub_u32_e32 v65, v65, v82
	v_and_b32_e32 v65, 3, v65
	v_or3_b32 v65, v66, v65, v81
	v_lshlrev_b32_e32 v65, 8, v65
	v_add_co_u32_e32 v65, vcc, v71, v65
	v_addc_co_u32_e32 v66, vcc, 0, v91, vcc
	v_add_co_u32_e32 v65, vcc, 8, v65
	v_addc_co_u32_e32 v66, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	s_nop 0
	global_load_dword v[65:66], off lds
	v_or_b32_e32 v65, 0x3000, v69
	v_readfirstlane_b32 s21, v65
	v_not_b32_e32 v65, v80
	v_add_lshl_u32 v65, v79, v65, 2
	v_and_b32_e32 v70, 4, v65
	v_or_b32_e32 v65, v83, v70
	v_lshlrev_b32_e32 v72, 8, v65
	v_add_co_u32_e32 v65, vcc, v71, v72
	v_addc_co_u32_e32 v66, vcc, 0, v91, vcc
	v_add_co_u32_e32 v65, vcc, 16, v65
	v_addc_co_u32_e32 v66, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	s_nop 0
	global_load_dword v[65:66], off lds
	v_or_b32_e32 v65, 0x2000, v76
	v_readfirstlane_b32 s21, v65
	v_lshrrev_b32_e32 v65, 2, v75
	v_and_b32_e32 v66, 0x78, v65
	v_sub_u32_e32 v65, v65, v82
	v_and_b32_e32 v65, 3, v65
	v_or3_b32 v65, v66, v65, v70
	v_lshlrev_b32_e32 v65, 8, v65
	v_add_co_u32_e32 v65, vcc, v77, v65
	v_addc_co_u32_e32 v66, vcc, 0, v78, vcc
	v_add_co_u32_e32 v65, vcc, v65, v90
	v_addc_co_u32_e32 v66, vcc, 0, v66, vcc
	v_add_co_u32_e32 v65, vcc, 16, v65
	v_addc_co_u32_e32 v66, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	s_nop 0
	global_load_dword v[65:66], off lds
	v_or_b32_e32 v65, 0x3800, v69
	v_readfirstlane_b32 s21, v65
	v_add_co_u32_e32 v65, vcc, v77, v72
	v_addc_co_u32_e32 v66, vcc, 0, v78, vcc
	v_add_co_u32_e32 v65, vcc, v65, v90
	v_addc_co_u32_e32 v66, vcc, 0, v66, vcc
	v_add_co_u32_e32 v65, vcc, 24, v65
	v_addc_co_u32_e32 v66, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	s_nop 0
	global_load_dword v[65:66], off lds
	v_or_b32_e32 v65, 0x2000, v68
	v_readfirstlane_b32 s21, v65
	v_lshrrev_b32_e32 v65, 2, v67
	v_and_b32_e32 v66, 0x78, v65
	v_sub_u32_e32 v65, v65, v82
	v_and_b32_e32 v65, 3, v65
	v_or3_b32 v65, v66, v65, v70
	v_lshlrev_b32_e32 v65, 8, v65
	v_add_co_u32_e32 v65, vcc, v77, v65
	v_addc_co_u32_e32 v66, vcc, 0, v78, vcc
	v_add_co_u32_e32 v65, vcc, v65, v90
	v_addc_co_u32_e32 v66, vcc, 0, v66, vcc
	v_add_co_u32_e32 v65, vcc, 24, v65
	v_addc_co_u32_e32 v66, vcc, 0, v66, vcc
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	s_mov_b32 m0, s21
	.loc	1 68 24                         ; test_input_spidD.mlir:68:24
	v_lshlrev_b32_e32 v67, 6, v195
	.loc	1 67 22                         ; test_input_spidD.mlir:67:22
	global_load_dword v[65:66], off lds
	.loc	1 68 24                         ; test_input_spidD.mlir:68:24
	v_and_b32_e32 v65, 0xc0, v195
	v_lshlrev_b32_e32 v66, 3, v195
	v_lshlrev_b32_e32 v65, 1, v65
	v_and_b32_e32 v66, 0x78, v66
	v_and_b32_e32 v67, 0xc00, v67
	v_or3_b32 v69, v65, v66, v67
	.loc	1 67 22                         ; test_input_spidD.mlir:67:22
	; asyncmark
	.loc	1 68 24                         ; test_input_spidD.mlir:68:24
	ds_read2st64_b64 v[65:68], v69 offset1:1
	ds_read2st64_b64 v[69:72], v69 offset0:8 offset1:9
	.loc	1 67 22                         ; test_input_spidD.mlir:67:22
	; wait_asyncmark(0)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	.loc	1 68 24                         ; test_input_spidD.mlir:68:24
	v_pk_mul_f16 v98, v66, s35
	v_pk_mul_f16 v97, v65, s35
	v_pk_mul_f16 v100, v70, s35
	v_pk_mul_f16 v99, v69, s35
	v_pk_mul_f16 v102, v68, s35
	v_pk_mul_f16 v101, v67, s35
	v_pk_mul_f16 v104, v72, s35
	v_pk_mul_f16 v103, v71, s35
	; sched_barrier mask(0x00000000)
	.loc	1 69 26                         ; test_input_spidD.mlir:69:26
	v_mov_b64_e32 v[65:66], v[201:202]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_lshlrev_b32_e32 v67, 6, v65
	; sched_barrier mask(0x00000000)
	v_mov_b64_e32 v[65:66], v[203:204]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_mul_lo_u32 v77, v85, s68
	.loc	1 68 24                         ; test_input_spidD.mlir:68:24
	v_bfe_u32 v78, v195, 4, 2
	.loc	1 69 26                         ; test_input_spidD.mlir:69:26
	v_and_b32_e32 v80, 32, v195
	v_mul_u32_u24_e32 v76, 0x190, v195
	v_lshl_or_b32 v91, v78, 10, v80
	v_bfe_u32 v90, v195, 2, 2
	v_lshlrev_b32_e32 v68, 5, v73
	v_add_u32_e32 v69, v91, v76
	v_lshlrev_b32_e32 v66, 3, v90
	v_add3_u32 v68, v69, v77, v68
	v_lshlrev_b32_e32 v65, 5, v65
	v_add3_u32 v66, v68, v66, v67
	v_add_lshl_u32 v81, v66, v65, 1
	ds_read_b128 v[65:68], v81 offset:8192
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[33:36], v[97:98], v[65:66], v[33:36]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 68 24                         ; test_input_spidD.mlir:68:24
	v_lshlrev_b64 v[74:75], 2, v[195:196]
	.loc	1 69 26                         ; test_input_spidD.mlir:69:26
	v_lshlrev_b64 v[69:70], 4, v[85:86]
	v_bfe_u32 v92, v195, 1, 1
	v_sub_co_u32_e32 v69, vcc, v74, v69
	v_subb_co_u32_e32 v70, vcc, v75, v70, vcc
	v_or_b32_e32 v69, v69, v92
	v_bfe_u32 v93, v79, 1, 1
	v_add_co_u32_e32 v75, vcc, v69, v93
	v_addc_co_u32_e32 v79, vcc, 0, v70, vcc
	v_add_co_u32_e32 v69, vcc, 1, v75
	v_addc_co_u32_e32 v70, vcc, 0, v79, vcc
	v_lshrrev_b32_e32 v70, 31, v70
	v_add_lshl_u32 v69, v69, v70, 6
	v_and_b32_e32 v82, 0xffffff80, v69
	v_lshlrev_b64 v[69:70], 4, v[195:196]
	v_lshlrev_b64 v[71:72], 6, v[85:86]
	v_lshlrev_b32_e32 v96, 1, v89
	v_sub_co_u32_e32 v69, vcc, v69, v71
	v_or_b32_e32 v84, v69, v90
	.loc	1 66 22                         ; test_input_spidD.mlir:66:22
	v_lshlrev_b32_e32 v69, 1, v195
	.loc	1 69 26                         ; test_input_spidD.mlir:69:26
	v_subb_co_u32_e32 v83, vcc, v70, v72, vcc
	v_and_b32_e32 v94, 6, v69
	v_add_co_u32_e32 v70, vcc, v84, v94
	v_addc_co_u32_e32 v71, vcc, 0, v83, vcc
	v_ashrrev_i32_e32 v71, 31, v71
	v_lshrrev_b32_e32 v71, 30, v71
	v_add_lshl_u32 v70, v70, v71, 4
	v_lshlrev_b32_e32 v71, 8, v195
	v_and_b32_e32 v70, 0xffffffc0, v70
	v_and_b32_e32 v71, 0x300, v71
	v_add_u32_e32 v70, v82, v70
	v_mul_u32_u24_e32 v82, 0x90, v94
	v_and_or_b32 v86, v69, 24, v71
	v_sub_u32_e32 v70, v82, v70
	v_lshlrev_b32_e32 v95, 1, v86
	v_add3_u32 v69, v70, v95, v96
	v_lshlrev_b32_e32 v105, 11, v78
	v_lshlrev_b32_e32 v106, 1, v80
	v_add3_u32 v69, v69, v105, v106
	ds_read_b128 v[69:72], v69 offset:8256
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[33:36], v[99:100], v[67:68], v[33:36]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[37:40], v[97:98], v[69:70], v[37:40]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[37:40], v[99:100], v[71:72], v[37:40]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_mul_lo_u32 v107, v85, s70
	v_mul_lo_u32 v78, v85, s69
	v_mul_u32_u24_e32 v80, 18, v195
	v_add_u32_e32 v74, v73, v74
	v_add3_u32 v107, v107, v80, v90
	v_add3_u32 v108, v74, v78, v93
	v_add_u16_e32 v78, 1, v107
	v_ashrrev_i16_e32 v80, 15, v78
	v_lshrrev_b16_e32 v74, 15, v108
	v_lshrrev_b16_e32 v80, 14, v80
	v_add_u16_e32 v74, v108, v74
	v_add_u16_e32 v78, v78, v80
	v_ashrrev_i16_e32 v74, 1, v74
	v_ashrrev_i16_e32 v78, 2, v78
	v_sub_u16_e32 v74, 0, v74
	v_sub_u16_e32 v78, 0, v78
	v_bfe_i32 v74, v74, 0, 16
	v_lshlrev_b32_sdwa v78, v212, sext(v78) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_lshl_add_u32 v74, v74, 7, v78
	v_lshlrev_b32_e32 v109, 1, v76
	v_lshlrev_b32_e32 v110, 1, v77
	v_add3_u32 v74, v74, v109, v110
	v_lshlrev_b32_e32 v111, 4, v90
	v_lshlrev_b32_e32 v112, 6, v73
	v_add3_u32 v73, v74, v111, v112
	v_add3_u32 v77, v73, v105, v106
	v_add_co_u32_e32 v73, vcc, 3, v75
	v_addc_co_u32_e32 v74, vcc, 0, v79, vcc
	v_lshrrev_b32_e32 v74, 31, v74
	v_add_lshl_u32 v73, v73, v74, 6
	v_or_b32_e32 v74, 1, v94
	v_add_co_u32_e32 v74, vcc, v84, v74
	v_addc_co_u32_e32 v75, vcc, 0, v83, vcc
	v_ashrrev_i32_e32 v75, 31, v75
	v_lshrrev_b32_e32 v75, 30, v75
	v_add_lshl_u32 v74, v74, v75, 4
	v_and_b32_e32 v73, 0xffffff80, v73
	v_and_b32_e32 v74, 0xffffffc0, v74
	v_add_u32_e32 v73, v73, v74
	v_sub_u32_e32 v73, v82, v73
	v_add3_u32 v73, v73, v95, v96
	v_add3_u32 v73, v73, v105, v106
	ds_read_b128 v[73:76], v73 offset:8528
	ds_read_b128 v[77:80], v77 offset:8336
	ds_read_b128 v[81:84], v81 offset:9216
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[41:44], v[97:98], v[77:78], v[41:44]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_and_or_b32 v87, v87, 12, v92
	v_and_b32_e32 v92, 48, v88
	v_or_b32_e32 v95, 8, v94
	v_and_or_b32 v85, v85, 2, v92
	v_or_b32_e32 v86, v86, v89
	v_add_lshl_u32 v89, v93, v87, 5
	v_add_u32_e32 v85, v95, v85
	v_add_u32_e32 v87, 0x260, v89
	v_lshl_add_u32 v85, v85, 3, v213
	v_mul_u32_u24_e32 v96, 0x48, v95
	v_and_b32_e32 v87, 0x7c0, v87
	v_and_b32_e32 v85, 0x7e0, v85
	v_add_u32_e32 v85, v87, v85
	v_add3_u32 v91, v86, v96, v91
	v_sub_u32_e32 v85, v91, v85
	v_lshlrev_b32_e32 v85, 1, v85
	ds_read_b128 v[85:88], v85 offset:10432
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[41:44], v[99:100], v[79:80], v[41:44]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_add_u16_e32 v93, 18, v108
	v_lshrrev_b16_e32 v95, 15, v93
	v_add_u16_e32 v93, v93, v95
	v_add_u16_e32 v95, 57, v107
	v_ashrrev_i16_e32 v96, 15, v95
	v_lshrrev_b16_e32 v96, 14, v96
	v_add_u16_e32 v95, v95, v96
	v_ashrrev_i16_e32 v93, 1, v93
	v_ashrrev_i16_e32 v95, 2, v95
	v_or_b32_e32 v90, v90, v92
	v_sub_u16_e32 v93, 0, v93
	v_sub_u16_e32 v95, 0, v95
	v_add_u32_e32 v90, v90, v94
	v_bfe_i32 v93, v93, 0, 16
	v_lshlrev_b32_sdwa v95, v212, sext(v95) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:WORD_0
	v_add_u32_e32 v89, 0x2a0, v89
	v_lshl_add_u32 v90, v90, 3, v214
	v_lshl_add_u32 v93, v93, 7, v95
	v_and_b32_e32 v89, 0x7c0, v89
	v_and_b32_e32 v90, 0x7e0, v90
	v_add3_u32 v93, v93, v109, v110
	v_add_u32_e32 v89, v89, v90
	v_add3_u32 v93, v93, v111, v112
	v_sub_u32_e32 v89, v91, v89
	v_add3_u32 v93, v93, v105, v106
	v_lshlrev_b32_e32 v94, 1, v89
	ds_read_b128 v[89:92], v93 offset:11408
	ds_read_b128 v[93:96], v94 offset:10704
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[45:48], v[97:98], v[73:74], v[45:48]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[45:48], v[99:100], v[75:76], v[45:48]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[49:52], v[97:98], v[81:82], v[49:52]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[49:52], v[99:100], v[83:84], v[49:52]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[53:56], v[97:98], v[85:86], v[53:56]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[53:56], v[99:100], v[87:88], v[53:56]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[57:60], v[97:98], v[89:90], v[57:60]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[57:60], v[99:100], v[91:92], v[57:60]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[61:64], v[97:98], v[93:94], v[61:64]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[61:64], v[99:100], v[95:96], v[61:64]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[1:4], v[101:102], v[65:66], v[1:4]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[1:4], v[103:104], v[67:68], v[1:4]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[5:8], v[101:102], v[69:70], v[5:8]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[5:8], v[103:104], v[71:72], v[5:8]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[9:12], v[101:102], v[77:78], v[9:12]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[9:12], v[103:104], v[79:80], v[9:12]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[13:16], v[101:102], v[73:74], v[13:16]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[13:16], v[103:104], v[75:76], v[13:16]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[17:20], v[101:102], v[81:82], v[17:20]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[17:20], v[103:104], v[83:84], v[17:20]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[21:24], v[101:102], v[85:86], v[21:24]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[21:24], v[103:104], v[87:88], v[21:24]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[25:28], v[101:102], v[89:90], v[25:28]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[25:28], v[103:104], v[91:92], v[25:28]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[29:32], v[101:102], v[93:94], v[29:32]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[29:32], v[103:104], v[95:96], v[29:32]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 59 17                         ; test_input_spidD.mlir:59:17
	s_add_u32 s28, s28, 0x400
	s_addc_u32 s29, s29, 0
	s_add_u32 s24, s24, 64
	s_addc_u32 s25, s25, 0
	s_add_u32 s26, s26, 32
	s_addc_u32 s27, s27, 0
	s_cmp_lg_u64 s[24:25], 0x100
	s_barrier
	s_cbranch_scc1 .LBB0_3
; %bb.4:                                ;   in Loop: Header=BB0_2 Depth=1
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	s_or_b32 s21, s22, 0x7f
	v_mov_b32_e32 v65, s21
	v_mov_b32_e32 v66, s23
	s_nop 1
	v_cmp_ge_u64_e32 vcc, s[2:3], v[65:66]
	s_and_b64 s[24:25], s[0:1], vcc
	s_and_b64 vcc, exec, s[24:25]
	s_cbranch_vccnz .LBB0_6
; %bb.5:                                ;   in Loop: Header=BB0_2 Depth=1
	v_mov_b32_e32 v66, s23
	v_or_b32_e32 v65, s22, v211
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 1
	v_cmp_gt_u64_e32 vcc, v[205:206], v[65:66]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v67, 1, v65
	v_mov_b32_e32 v68, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v97, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[67:68]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v69, 2, v65
	v_mov_b32_e32 v70, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v98, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[69:70]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v71, 3, v65
	v_mov_b32_e32 v72, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v99, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[71:72]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v73, 4, v65
	v_mov_b32_e32 v74, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v100, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[73:74]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v75, 5, v65
	v_mov_b32_e32 v76, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v101, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[75:76]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v77, 6, v65
	v_mov_b32_e32 v78, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v102, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[77:78]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v79, 7, v65
	v_mov_b32_e32 v80, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v103, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[79:80]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v81, 8, v65
	v_mov_b32_e32 v82, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v104, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[81:82]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v83, 9, v65
	v_mov_b32_e32 v84, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v105, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[83:84]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v85, 10, v65
	v_mov_b32_e32 v86, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v106, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[85:86]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v87, 11, v65
	v_mov_b32_e32 v88, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v107, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[87:88]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v89, 12, v65
	v_mov_b32_e32 v90, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v108, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[89:90]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v91, 13, v65
	v_mov_b32_e32 v92, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v109, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[91:92]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v93, 14, v65
	v_mov_b32_e32 v94, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v110, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[93:94]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v95, 15, v65
	v_mov_b32_e32 v96, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v111, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[95:96]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v216, 64, v65
	v_mov_b32_e32 v217, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v112, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[216:217]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v218, 0x41, v65
	v_mov_b32_e32 v219, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v113, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[218:219]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v220, 0x42, v65
	v_mov_b32_e32 v221, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v114, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[220:221]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v222, 0x43, v65
	v_mov_b32_e32 v223, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v115, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[222:223]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v224, 0x44, v65
	v_mov_b32_e32 v225, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v116, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[224:225]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v226, 0x45, v65
	v_mov_b32_e32 v227, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v117, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[226:227]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v228, 0x46, v65
	v_mov_b32_e32 v229, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v118, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[228:229]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v230, 0x47, v65
	v_mov_b32_e32 v231, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v119, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[230:231]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v232, 0x48, v65
	v_mov_b32_e32 v233, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v120, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[232:233]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v234, 0x49, v65
	v_mov_b32_e32 v235, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v121, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[234:235]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v236, 0x4a, v65
	v_mov_b32_e32 v237, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v122, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[236:237]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v238, 0x4b, v65
	v_mov_b32_e32 v239, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v123, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[238:239]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v240, 0x4c, v65
	v_mov_b32_e32 v241, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v124, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[240:241]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v242, 0x4d, v65
	v_mov_b32_e32 v243, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v125, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[242:243]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v244, 0x4e, v65
	v_mov_b32_e32 v245, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v126, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[244:245]
	.loc	1 73 19                         ; test_input_spidD.mlir:73:19
	v_or_b32_e32 v246, 0x4f, v65
	v_mov_b32_e32 v247, s23
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v127, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	s_nop 0
	v_cmp_gt_u64_e32 vcc, v[205:206], v[246:247]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v128, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[65:66]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v65, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[67:68]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v66, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[69:70]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v67, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[71:72]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v68, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[73:74]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v69, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[75:76]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v70, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[77:78]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v71, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[79:80]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v72, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[81:82]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v73, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[83:84]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v74, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[85:86]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v75, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[87:88]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v76, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[89:90]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v77, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[91:92]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v78, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[93:94]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v79, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[95:96]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v80, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[216:217]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v81, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[218:219]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v82, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[220:221]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v83, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[222:223]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v84, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[224:225]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v85, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[226:227]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v86, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[228:229]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v87, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[230:231]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v88, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[232:233]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v89, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[234:235]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v90, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[236:237]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v91, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[238:239]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v92, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[240:241]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v93, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[242:243]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v94, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[244:245]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v95, v215, 0, vcc
	.loc	1 76 23                         ; test_input_spidD.mlir:76:23
	v_cmp_gt_u64_e32 vcc, v[207:208], v[246:247]
	.loc	1 77 26                         ; test_input_spidD.mlir:77:26
	v_cndmask_b32_e64 v96, v215, 0, vcc
	s_branch .LBB0_7
.LBB0_6:                                ;   in Loop: Header=BB0_2 Depth=1
	.loc	1 0 26 is_stmt 0                ; test_input_spidD.mlir:0:26
	s_mov_b32 s66, s36
	s_mov_b32 s67, s36
	s_mov_b32 s37, s36
	s_mov_b32 s38, s36
	s_mov_b32 s39, s36
	s_mov_b32 s40, s36
	s_mov_b32 s41, s36
	s_mov_b32 s42, s36
	s_mov_b32 s43, s36
	s_mov_b32 s44, s36
	s_mov_b32 s45, s36
	s_mov_b32 s46, s36
	s_mov_b32 s47, s36
	s_mov_b32 s48, s36
	s_mov_b32 s49, s36
	s_mov_b32 s50, s36
	s_mov_b32 s51, s36
	s_mov_b32 s52, s36
	s_mov_b32 s53, s36
	s_mov_b32 s54, s36
	s_mov_b32 s55, s36
	s_mov_b32 s56, s36
	s_mov_b32 s57, s36
	s_mov_b32 s58, s36
	s_mov_b32 s59, s36
	s_mov_b32 s60, s36
	s_mov_b32 s61, s36
	s_mov_b32 s62, s36
	s_mov_b32 s63, s36
	s_mov_b32 s64, s36
	s_mov_b32 s65, s36
	v_mov_b64_e32 v[127:128], s[66:67]
	v_mov_b64_e32 v[97:98], s[36:37]
	v_mov_b64_e32 v[125:126], s[64:65]
	v_mov_b64_e32 v[123:124], s[62:63]
	v_mov_b64_e32 v[121:122], s[60:61]
	v_mov_b64_e32 v[119:120], s[58:59]
	v_mov_b64_e32 v[117:118], s[56:57]
	v_mov_b64_e32 v[115:116], s[54:55]
	v_mov_b64_e32 v[113:114], s[52:53]
	v_mov_b64_e32 v[111:112], s[50:51]
	v_mov_b64_e32 v[109:110], s[48:49]
	v_mov_b64_e32 v[107:108], s[46:47]
	v_mov_b64_e32 v[105:106], s[44:45]
	v_mov_b64_e32 v[103:104], s[42:43]
	v_mov_b64_e32 v[101:102], s[40:41]
	v_mov_b64_e32 v[99:100], s[38:39]
	s_nop 1
	v_mov_b64_e32 v[65:66], v[97:98]
	v_mov_b64_e32 v[67:68], v[99:100]
	v_mov_b64_e32 v[69:70], v[101:102]
	v_mov_b64_e32 v[71:72], v[103:104]
	v_mov_b64_e32 v[73:74], v[105:106]
	v_mov_b64_e32 v[75:76], v[107:108]
	v_mov_b64_e32 v[77:78], v[109:110]
	v_mov_b64_e32 v[79:80], v[111:112]
	v_mov_b64_e32 v[81:82], v[113:114]
	v_mov_b64_e32 v[83:84], v[115:116]
	v_mov_b64_e32 v[85:86], v[117:118]
	v_mov_b64_e32 v[87:88], v[119:120]
	v_mov_b64_e32 v[89:90], v[121:122]
	v_mov_b64_e32 v[91:92], v[123:124]
	v_mov_b64_e32 v[93:94], v[125:126]
	v_mov_b64_e32 v[95:96], v[127:128]
.LBB0_7:                                ;   in Loop: Header=BB0_2 Depth=1
	.loc	1 92 20 is_stmt 1               ; test_input_spidD.mlir:92:20
	s_add_i32 s20, s20, s6
	s_lshl_b32 s20, s20, 20
	s_and_b32 s20, s20, 0x1f00000
	s_add_u32 s20, s12, s20
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	v_pk_add_f32 v[63:64], v[63:64], v[127:128]
	v_pk_add_f32 v[216:217], v[61:62], v[125:126]
	v_pk_add_f32 v[123:124], v[59:60], v[123:124]
	v_pk_add_f32 v[121:122], v[57:58], v[121:122]
	v_pk_add_f32 v[119:120], v[55:56], v[119:120]
	v_pk_add_f32 v[117:118], v[53:54], v[117:118]
	v_pk_add_f32 v[115:116], v[51:52], v[115:116]
	v_pk_add_f32 v[113:114], v[49:50], v[113:114]
	v_pk_add_f32 v[47:48], v[47:48], v[111:112]
	v_pk_add_f32 v[45:46], v[45:46], v[109:110]
	v_pk_add_f32 v[43:44], v[43:44], v[107:108]
	v_pk_add_f32 v[41:42], v[41:42], v[105:106]
	v_pk_add_f32 v[39:40], v[39:40], v[103:104]
	v_pk_add_f32 v[37:38], v[37:38], v[101:102]
	v_pk_add_f32 v[35:36], v[35:36], v[99:100]
	v_pk_add_f32 v[33:34], v[33:34], v[97:98]
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	s_addc_u32 s21, s13, 0
	; sched_barrier mask(0x00000000)
	v_mov_b32_e32 v195, v0
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_lshrrev_b32_e32 v49, 3, v195
	v_mov_b32_e32 v50, v196
	s_nop 1
	v_lshlrev_b64 v[51:52], 10, v[49:50]
	v_mov_b32_e32 v53, s21
	v_add_co_u32_e32 v51, vcc, s20, v51
	v_addc_co_u32_e32 v52, vcc, v53, v52, vcc
	v_mov_b32_e32 v53, s9
	v_add_co_u32_e32 v60, vcc, s8, v51
	v_addc_co_u32_e32 v61, vcc, v52, v53, vcc
	.loc	1 68 24                         ; test_input_spidD.mlir:68:24
	v_lshlrev_b64 v[51:52], 2, v[195:196]
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	v_lshlrev_b64 v[53:54], 5, v[49:50]
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v97, v33
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	v_sub_co_u32_e32 v53, vcc, v51, v53
	v_subb_co_u32_e32 v54, vcc, v52, v54, vcc
	s_nop 1
	v_lshlrev_b64 v[52:53], 1, v[53:54]
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v98, v34
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	v_add_co_u32_e32 v58, vcc, v60, v52
	v_addc_co_u32_e32 v59, vcc, v61, v53, vcc
	global_load_dwordx2 v[52:53], v[58:59], off
	global_load_dwordx2 v[54:55], v[58:59], off offset:256
	global_load_dwordx2 v[56:57], v[58:59], off offset:512
	s_nop 0
	global_load_dwordx2 v[58:59], v[58:59], off offset:768
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v99, v35
	v_exp_f32_e32 v100, v36
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	s_nop 1
	v_mov_b64_e32 v[33:34], v[97:98]
	v_mov_b64_e32 v[35:36], v[99:100]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v36, v36
	v_cvt_f16_f32_e32 v35, v35
	v_cvt_f16_f32_e32 v50, v34
	v_cvt_f16_f32_e32 v33, v33
	v_pack_b32_f16 v34, v35, v36
	v_pack_b32_f16 v33, v33, v50
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v101, v37
	v_exp_f32_e32 v102, v38
	v_exp_f32_e32 v103, v39
	v_exp_f32_e32 v104, v40
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	s_nop 1
	v_mov_b64_e32 v[35:36], v[101:102]
	v_mov_b64_e32 v[37:38], v[103:104]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v38, v38
	v_cvt_f16_f32_e32 v37, v37
	v_cvt_f16_f32_e32 v39, v36
	v_cvt_f16_f32_e32 v35, v35
	v_pack_b32_f16 v36, v37, v38
	v_pack_b32_f16 v35, v35, v39
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v105, v41
	v_exp_f32_e32 v106, v42
	v_exp_f32_e32 v107, v43
	v_exp_f32_e32 v108, v44
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	s_nop 1
	v_mov_b64_e32 v[37:38], v[105:106]
	v_mov_b64_e32 v[39:40], v[107:108]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v40, v40
	v_cvt_f16_f32_e32 v39, v39
	v_cvt_f16_f32_e32 v41, v38
	v_cvt_f16_f32_e32 v37, v37
	v_pack_b32_f16 v38, v39, v40
	v_pack_b32_f16 v37, v37, v41
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v109, v45
	v_exp_f32_e32 v110, v46
	v_exp_f32_e32 v111, v47
	v_exp_f32_e32 v112, v48
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	s_nop 1
	v_mov_b64_e32 v[39:40], v[109:110]
	v_mov_b64_e32 v[41:42], v[111:112]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v42, v42
	v_cvt_f16_f32_e32 v41, v41
	v_cvt_f16_f32_e32 v43, v40
	v_cvt_f16_f32_e32 v39, v39
	v_pack_b32_f16 v40, v41, v42
	v_pack_b32_f16 v39, v39, v43
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v113, v113
	v_exp_f32_e32 v114, v114
	v_exp_f32_e32 v115, v115
	v_exp_f32_e32 v116, v116
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	s_nop 1
	v_mov_b64_e32 v[41:42], v[113:114]
	v_mov_b64_e32 v[43:44], v[115:116]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v44, v44
	v_cvt_f16_f32_e32 v43, v43
	v_cvt_f16_f32_e32 v42, v42
	v_cvt_f16_f32_e32 v41, v41
	v_pack_b32_f16 v44, v43, v44
	v_pack_b32_f16 v43, v41, v42
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v117, v117
	v_exp_f32_e32 v118, v118
	v_exp_f32_e32 v119, v119
	v_exp_f32_e32 v120, v120
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	s_nop 1
	v_mov_b64_e32 v[45:46], v[117:118]
	v_mov_b64_e32 v[47:48], v[119:120]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v41, v48
	v_cvt_f16_f32_e32 v42, v47
	v_cvt_f16_f32_e32 v47, v46
	v_cvt_f16_f32_e32 v45, v45
	v_pack_b32_f16 v46, v42, v41
	v_pack_b32_f16 v45, v45, v47
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v121, v121
	v_exp_f32_e32 v122, v122
	v_exp_f32_e32 v123, v123
	v_exp_f32_e32 v124, v124
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	s_nop 1
	v_mov_b64_e32 v[127:128], v[123:124]
	v_mov_b64_e32 v[125:126], v[121:122]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v41, v128
	v_cvt_f16_f32_e32 v42, v127
	v_cvt_f16_f32_e32 v47, v126
	v_cvt_f16_f32_e32 v48, v125
	v_pack_b32_f16 v42, v42, v41
	v_pack_b32_f16 v41, v48, v47
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v125, v216
	v_exp_f32_e32 v126, v217
	v_exp_f32_e32 v127, v63
	v_exp_f32_e32 v128, v64
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	v_mov_b32_e32 v219, v128
	v_mov_b32_e32 v218, v127
	v_mov_b32_e32 v217, v126
	v_mov_b32_e32 v216, v125
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v47, v219
	v_cvt_f16_f32_e32 v48, v218
	v_cvt_f16_f32_e32 v50, v217
	v_cvt_f16_f32_e32 v62, v216
	v_pack_b32_f16 v48, v48, v47
	v_pack_b32_f16 v47, v62, v50
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_pk_add_f32 v[19:20], v[19:20], v[83:84]
	v_pk_add_f32 v[3:4], v[3:4], v[67:68]
	v_pk_add_f32 v[1:2], v[1:2], v[65:66]
	v_pk_add_f32 v[31:32], v[31:32], v[95:96]
	v_pk_add_f32 v[29:30], v[29:30], v[93:94]
	v_pk_add_f32 v[27:28], v[27:28], v[91:92]
	v_pk_add_f32 v[25:26], v[25:26], v[89:90]
	v_pk_add_f32 v[23:24], v[23:24], v[87:88]
	v_pk_add_f32 v[62:63], v[21:22], v[85:86]
	v_pk_add_f32 v[21:22], v[17:18], v[81:82]
	v_pk_add_f32 v[79:80], v[15:16], v[79:80]
	v_pk_add_f32 v[77:78], v[13:14], v[77:78]
	v_pk_add_f32 v[75:76], v[11:12], v[75:76]
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v11, v1
	v_exp_f32_e32 v12, v2
	v_exp_f32_e32 v13, v3
	v_exp_f32_e32 v14, v4
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	v_pk_add_f32 v[9:10], v[9:10], v[73:74]
	v_pk_add_f32 v[7:8], v[7:8], v[71:72]
	v_pk_add_f32 v[5:6], v[5:6], v[69:70]
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	v_mov_b64_e32 v[1:2], v[11:12]
	v_mov_b64_e32 v[3:4], v[13:14]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v4, v4
	v_cvt_f16_f32_e32 v3, v3
	v_cvt_f16_f32_e32 v15, v2
	v_cvt_f16_f32_e32 v1, v1
	v_pack_b32_f16 v2, v3, v4
	v_pack_b32_f16 v1, v1, v15
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v5, v5
	v_exp_f32_e32 v6, v6
	v_exp_f32_e32 v7, v7
	v_exp_f32_e32 v8, v8
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	s_nop 1
	v_mov_b64_e32 v[17:18], v[7:8]
	v_mov_b64_e32 v[15:16], v[5:6]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v3, v18
	v_cvt_f16_f32_e32 v4, v17
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v15, v15
	v_pack_b32_f16 v4, v4, v3
	v_pack_b32_f16 v3, v15, v16
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 86 20                         ; test_input_spidD.mlir:86:20
	v_mov_b32_e32 v66, v97
	v_mov_b32_e32 v67, v11
	v_mov_b32_e32 v11, v98
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v15, v10
	.loc	1 86 20                         ; test_input_spidD.mlir:86:20
	s_nop 0
	v_pk_add_f32 v[10:11], v[66:67], v[11:12]
	v_mov_b32_e32 v12, v99
	s_nop 1
	v_pk_add_f32 v[10:11], v[12:13], v[10:11]
	v_mov_b32_e32 v13, v100
	s_nop 1
	v_pk_add_f32 v[10:11], v[13:14], v[10:11]
	v_mov_b32_e32 v12, v101
	v_mov_b32_e32 v13, v5
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v9, v9
	.loc	1 86 20                         ; test_input_spidD.mlir:86:20
	s_nop 0
	v_pk_add_f32 v[10:11], v[12:13], v[10:11]
	v_mov_b32_e32 v5, v102
	s_nop 1
	v_pk_add_f32 v[10:11], v[5:6], v[10:11]
	v_mov_b32_e32 v6, v103
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v65, v75
	.loc	1 86 20                         ; test_input_spidD.mlir:86:20
	s_nop 0
	v_pk_add_f32 v[5:6], v[6:7], v[10:11]
	v_mov_b32_e32 v7, v104
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v17, v76
	.loc	1 86 20                         ; test_input_spidD.mlir:86:20
	s_nop 0
	v_pk_add_f32 v[5:6], v[7:8], v[5:6]
	v_mov_b32_e32 v8, v105
	s_nop 1
	v_pk_add_f32 v[5:6], v[8:9], v[5:6]
	v_mov_b32_e32 v14, v106
	s_nop 1
	v_pk_add_f32 v[5:6], v[14:15], v[5:6]
	v_mov_b32_e32 v64, v107
	s_nop 1
	v_pk_add_f32 v[5:6], v[64:65], v[5:6]
	v_mov_b32_e32 v16, v108
	s_nop 1
	v_pk_add_f32 v[66:67], v[16:17], v[5:6]
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	v_mov_b32_e32 v14, v9
	v_mov_b32_e32 v16, v65
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v5, v17
	v_cvt_f16_f32_e32 v6, v16
	v_cvt_f16_f32_e32 v7, v15
	v_cvt_f16_f32_e32 v8, v14
	v_pack_b32_f16 v6, v6, v5
	v_pack_b32_f16 v5, v8, v7
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v11, v77
	v_exp_f32_e32 v12, v78
	v_exp_f32_e32 v13, v79
	v_exp_f32_e32 v14, v80
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	s_nop 1
	v_mov_b64_e32 v[7:8], v[11:12]
	v_mov_b64_e32 v[9:10], v[13:14]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v10, v10
	v_cvt_f16_f32_e32 v9, v9
	v_cvt_f16_f32_e32 v15, v8
	v_cvt_f16_f32_e32 v7, v7
	v_pack_b32_f16 v8, v9, v10
	v_pack_b32_f16 v7, v7, v15
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v17, v19
	v_exp_f32_e32 v18, v20
	v_exp_f32_e32 v15, v21
	v_exp_f32_e32 v16, v22
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	s_nop 1
	v_mov_b64_e32 v[21:22], v[17:18]
	v_mov_b64_e32 v[19:20], v[15:16]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v9, v22
	v_cvt_f16_f32_e32 v10, v21
	v_cvt_f16_f32_e32 v20, v20
	v_cvt_f16_f32_e32 v19, v19
	v_pack_b32_f16 v10, v10, v9
	v_pack_b32_f16 v9, v19, v20
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 86 20                         ; test_input_spidD.mlir:86:20
	v_mov_b32_e32 v20, v109
	v_mov_b32_e32 v21, v11
	s_nop 1
	v_pk_add_f32 v[20:21], v[20:21], v[66:67]
	v_mov_b32_e32 v11, v110
	s_nop 1
	v_pk_add_f32 v[20:21], v[11:12], v[20:21]
	v_mov_b32_e32 v12, v111
	s_nop 1
	v_pk_add_f32 v[11:12], v[12:13], v[20:21]
	v_mov_b32_e32 v13, v112
	s_nop 1
	v_pk_add_f32 v[11:12], v[13:14], v[11:12]
	v_mov_b32_e32 v13, v113
	v_mov_b32_e32 v14, v15
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v22, v62
	.loc	1 86 20                         ; test_input_spidD.mlir:86:20
	s_nop 0
	v_pk_add_f32 v[11:12], v[13:14], v[11:12]
	v_mov_b32_e32 v15, v114
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v19, v63
	.loc	1 86 20                         ; test_input_spidD.mlir:86:20
	s_nop 0
	v_pk_add_f32 v[11:12], v[15:16], v[11:12]
	v_mov_b32_e32 v16, v115
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v63, v23
	.loc	1 86 20                         ; test_input_spidD.mlir:86:20
	s_nop 0
	v_pk_add_f32 v[11:12], v[16:17], v[11:12]
	v_mov_b32_e32 v17, v116
	s_nop 1
	v_pk_add_f32 v[11:12], v[17:18], v[11:12]
	v_mov_b32_e32 v21, v117
	s_nop 1
	v_pk_add_f32 v[11:12], v[21:22], v[11:12]
	v_mov_b32_e32 v18, v118
	s_nop 1
	v_pk_add_f32 v[11:12], v[18:19], v[11:12]
	v_mov_b32_e32 v62, v119
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	v_mov_b32_e32 v18, v22
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v21, v24
	.loc	1 86 20                         ; test_input_spidD.mlir:86:20
	v_pk_add_f32 v[64:65], v[62:63], v[11:12]
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	v_mov_b32_e32 v20, v63
	; sched_barrier mask(0x00000000)
	v_mov_b32_e32 v11, v18
	v_mov_b32_e32 v12, v19
	v_mov_b32_e32 v13, v20
	v_mov_b32_e32 v14, v21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v15, v12
	v_cvt_f16_f32_e32 v11, v11
	v_pack_b32_f16 v12, v13, v14
	v_pack_b32_f16 v11, v11, v15
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v22, v25
	v_exp_f32_e32 v23, v26
	v_exp_f32_e32 v24, v27
	v_exp_f32_e32 v25, v28
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	v_mov_b32_e32 v13, v22
	v_mov_b32_e32 v14, v23
	v_mov_b32_e32 v15, v24
	v_mov_b32_e32 v16, v25
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v15, v15
	v_cvt_f16_f32_e32 v17, v14
	v_cvt_f16_f32_e32 v13, v13
	v_pack_b32_f16 v14, v15, v16
	v_pack_b32_f16 v13, v13, v17
	; sched_barrier mask(0x00000000)
	.loc	1 84 21                         ; test_input_spidD.mlir:84:21
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 85 22                         ; test_input_spidD.mlir:85:22
	v_exp_f32_e32 v26, v29
	v_exp_f32_e32 v27, v30
	v_exp_f32_e32 v28, v31
	v_exp_f32_e32 v29, v32
	; sched_barrier mask(0x00000000)
	.loc	1 87 16                         ; test_input_spidD.mlir:87:16
	v_mov_b32_e32 v15, v26
	v_mov_b32_e32 v16, v27
	v_mov_b32_e32 v17, v28
	v_mov_b32_e32 v18, v29
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v19, v16
	v_cvt_f16_f32_e32 v15, v15
	v_pack_b32_f16 v16, v17, v18
	v_pack_b32_f16 v15, v15, v19
	; sched_barrier mask(0x00000000)
	.loc	1 86 20                         ; test_input_spidD.mlir:86:20
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v17, 64, v210
	v_xor_b32_e32 v18, 16, v210
	v_and_b32_e32 v30, 0xc0, v17
	v_cmp_lt_u32_e32 vcc, v18, v30
	v_cndmask_b32_e32 v19, v210, v18, vcc
	v_mov_b32_e32 v20, v120
	v_lshlrev_b32_e32 v31, 2, v19
	s_nop 0
	v_pk_add_f32 v[19:20], v[20:21], v[64:65]
	v_mov_b32_e32 v21, v121
	s_nop 1
	v_pk_add_f32 v[19:20], v[21:22], v[19:20]
	v_mov_b32_e32 v22, v122
	s_nop 1
	v_pk_add_f32 v[19:20], v[22:23], v[19:20]
	v_mov_b32_e32 v23, v123
	s_nop 1
	v_pk_add_f32 v[19:20], v[23:24], v[19:20]
	v_mov_b32_e32 v24, v124
	s_nop 1
	v_pk_add_f32 v[19:20], v[24:25], v[19:20]
	v_mov_b32_e32 v21, v125
	v_mov_b32_e32 v22, v26
	s_nop 1
	v_pk_add_f32 v[19:20], v[21:22], v[19:20]
	v_mov_b32_e32 v26, v126
	s_nop 1
	v_pk_add_f32 v[19:20], v[26:27], v[19:20]
	v_mov_b32_e32 v27, v127
	s_nop 1
	v_pk_add_f32 v[19:20], v[27:28], v[19:20]
	v_mov_b32_e32 v28, v128
	s_nop 1
	v_pk_add_f32 v[20:21], v[28:29], v[19:20]
	ds_bpermute_b32 v22, v31, v20
	ds_bpermute_b32 v23, v31, v21
	v_xor_b32_e32 v19, 32, v210
	v_cmp_lt_u32_e32 vcc, v19, v30
	v_cndmask_b32_e32 v24, v210, v19, vcc
	v_lshlrev_b32_e32 v24, 2, v24
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[20:21], v[20:21], v[22:23]
	ds_bpermute_b32 v22, v24, v20
	ds_bpermute_b32 v23, v24, v21
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	v_lshrrev_b32_e32 v27, 1, v195
	v_lshrrev_b32_e32 v28, 4, v195
	v_lshlrev_b32_e32 v26, 5, v195
	.loc	1 86 20                         ; test_input_spidD.mlir:86:20
	s_waitcnt lgkmcnt(0)
	v_pk_add_f32 v[20:21], v[20:21], v[22:23]
	v_pk_add_f32 v[193:194], v[193:194], v[20:21]
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	v_mul_u32_u24_e32 v20, 0x48, v27
	v_mul_u32_u24_e32 v23, 0x118, v28
	v_add3_u32 v23, v23, v20, v26
	s_nop 1
	v_mad_u64_u32 v[24:25], s[8:9], v49, s72, v[23:24]
	v_and_b32_e32 v20, 7, v195
	v_add_lshl_u32 v20, v28, v20, 5
	v_and_b32_e32 v29, 0x3c0, v20
	v_sub_u32_e32 v20, v24, v29
	s_waitcnt vmcnt(2)
	v_perm_b32 v21, v54, v52, s71
	s_waitcnt vmcnt(0)
	v_perm_b32 v22, v58, v56, s71
	v_lshlrev_b32_e32 v20, 1, v20
	ds_write_b64 v20, v[21:22]
	v_perm_b32 v24, v54, v52, s73
	v_perm_b32 v25, v58, v56, s73
	; sched_barrier mask(0x00000000)
	v_mov_b64_e32 v[21:22], v[197:198]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_mul_lo_u32 v22, v49, s74
	v_and_b32_e32 v30, 0x3e0, v51
	v_add_u32_e32 v26, v23, v26
	v_perm_b32 v23, v55, v53, s71
	v_add3_u32 v22, v26, v22, v30
	v_sub_u32_e32 v22, v22, v29
	v_lshlrev_b32_e32 v29, 1, v22
	v_lshl_add_u32 v21, v21, 6, v29
	ds_write_b64 v21, v[24:25] offset:16
	v_perm_b32 v24, v59, v57, s71
	; sched_barrier mask(0x00000000)
	v_mov_b64_e32 v[25:26], v[199:200]
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v22, v25, 6, v29
	ds_write_b64 v22, v[23:24] offset:32
	v_mul_u32_u24_e32 v23, 0xffdc, v49
	v_add3_u32 v23, v27, v51, v23
	v_add_u16_e32 v23, 3, v23
	v_ashrrev_i16_e32 v26, 15, v23
	v_lshrrev_b16_e32 v26, 14, v26
	v_add_u16_e32 v23, v23, v26
	v_ashrrev_i16_e32 v23, 2, v23
	v_sub_u16_e32 v23, 0, v23
	v_bfe_i32 v23, v23, 0, 16
	v_perm_b32 v24, v55, v53, s73
	v_perm_b32 v25, v59, v57, s73
	v_lshl_add_u32 v23, v23, 6, v29
	ds_write_b64 v23, v[24:25] offset:48
	.loc	1 93 23                         ; test_input_spidD.mlir:93:23
	v_lshlrev_b32_e32 v24, 7, v28
	v_bfe_u32 v30, v195, 3, 1
	v_sub_u32_e32 v24, 0, v24
	v_mul_u32_u24_e32 v26, 0x48, v30
	v_or_b32_e32 v27, v26, v24
	v_lshlrev_b32_e32 v32, 3, v195
	.loc	1 69 26                         ; test_input_spidD.mlir:69:26
	v_and_b32_e32 v25, 15, v195
	.loc	1 93 23                         ; test_input_spidD.mlir:93:23
	v_add_u32_e32 v29, v27, v32
	v_and_b32_e32 v31, 0x60, v32
	v_mul_i32_i24_e32 v49, 0xffffffc0, v30
	v_lshlrev_b32_e32 v24, 1, v29
	v_lshlrev_b32_e32 v57, 1, v31
	v_lshlrev_b32_e32 v58, 1, v49
	v_add_lshl_u32 v59, v30, v25, 4
	v_add3_u32 v24, v24, v57, v58
	v_and_b32_e32 v62, 0x1c0, v59
	.loc	1 68 24                         ; test_input_spidD.mlir:68:24
	v_bfe_u32 v78, v195, 4, 2
	.loc	1 93 23                         ; test_input_spidD.mlir:93:23
	v_sub_u32_e32 v24, v24, v62
	v_lshlrev_b32_e32 v70, 10, v78
	v_add_u32_e32 v24, v24, v70
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 93 23                         ; test_input_spidD.mlir:93:23
	ds_read_b128 v[49:52], v24
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[161:164], v[33:34], v[49:50], v[161:164]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_lshlrev_b32_e32 v53, 2, v195
	v_and_b32_e32 v53, 0xc0, v53
	v_mul_u32_u24_e32 v25, 0x240, v78
	v_add_u32_e32 v79, v31, v53
	v_add_u32_e32 v25, v25, v32
	v_add_u32_e32 v53, 32, v79
	v_and_b32_e32 v53, 0x1c0, v53
	v_add3_u32 v31, v27, v31, v25
	v_sub_u32_e32 v25, v31, v53
	v_lshlrev_b32_e32 v25, 1, v25
	v_sub_u32_e32 v25, v25, v62
	ds_read_b128 v[53:56], v25 offset:576
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[161:164], v[35:36], v[51:52], v[161:164]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[161:164], v[37:38], v[53:54], v[161:164]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[161:164], v[39:40], v[55:56], v[161:164]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_lshl_add_u32 v27, v195, 4, v70
	v_lshlrev_b32_e32 v28, 8, v28
	v_sub_u32_e32 v27, v27, v28
	v_lshl_add_u32 v26, v26, 1, v27
	v_add_u32_e32 v27, 0x120, v79
	v_and_b32_e32 v28, 32, v32
	v_and_b32_e32 v27, 0x3c0, v27
	v_add_lshl_u32 v28, v28, v29, 1
	v_add_u32_e32 v29, 0x120, v59
	v_add3_u32 v26, v26, v57, v58
	v_sub_u32_e32 v27, v31, v27
	v_and_b32_e32 v57, 0x3c0, v29
	v_sub_u32_e32 v26, v26, v62
	v_lshlrev_b32_e32 v27, 1, v27
	v_sub_u32_e32 v28, v28, v57
	v_sub_u32_e32 v27, v27, v62
	ds_read_b128 v[62:65], v26 offset:4096
	ds_read_b128 v[66:69], v27 offset:5184
	v_add_u32_e32 v28, v28, v70
	ds_read_b128 v[70:73], v28 offset:544
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[161:164], v[43:44], v[62:63], v[161:164]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v29, 0xa0, v79
	v_and_b32_e32 v29, 0x3c0, v29
	v_sub_u32_e32 v29, v31, v29
	v_lshlrev_b32_e32 v29, 1, v29
	v_sub_u32_e32 v29, v29, v57
	ds_read_b128 v[74:77], v29 offset:1376
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[161:164], v[45:46], v[64:65], v[161:164]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_add_lshl_u32 v30, v30, v78, 6
	v_add_u32_e32 v58, 0x1a0, v79
	v_sub_u32_e32 v30, v31, v30
	v_and_b32_e32 v58, 0x3c0, v58
	v_lshlrev_b32_e32 v30, 1, v30
	v_sub_u32_e32 v31, v31, v58
	v_sub_u32_e32 v30, v30, v57
	v_lshlrev_b32_e32 v31, 1, v31
	v_sub_u32_e32 v31, v31, v57
	ds_read_b128 v[78:81], v30 offset:4640
	ds_read_b128 v[82:85], v31 offset:5984
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[161:164], v[41:42], v[66:67], v[161:164]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[161:164], v[47:48], v[68:69], v[161:164]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[165:168], v[33:34], v[70:71], v[165:168]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[165:168], v[35:36], v[72:73], v[165:168]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[165:168], v[37:38], v[74:75], v[165:168]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[165:168], v[39:40], v[76:77], v[165:168]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[165:168], v[43:44], v[78:79], v[165:168]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[165:168], v[45:46], v[80:81], v[165:168]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[165:168], v[41:42], v[82:83], v[165:168]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[165:168], v[47:48], v[84:85], v[165:168]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[129:132], v[1:2], v[49:50], v[129:132]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[129:132], v[3:4], v[51:52], v[129:132]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[129:132], v[5:6], v[53:54], v[129:132]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[129:132], v[7:8], v[55:56], v[129:132]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[129:132], v[9:10], v[62:63], v[129:132]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[129:132], v[11:12], v[64:65], v[129:132]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[129:132], v[13:14], v[66:67], v[129:132]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[129:132], v[15:16], v[68:69], v[129:132]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[133:136], v[1:2], v[70:71], v[133:136]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[133:136], v[3:4], v[72:73], v[133:136]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[133:136], v[5:6], v[74:75], v[133:136]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[133:136], v[7:8], v[76:77], v[133:136]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[133:136], v[9:10], v[78:79], v[133:136]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[133:136], v[11:12], v[80:81], v[133:136]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[133:136], v[13:14], v[82:83], v[133:136]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[133:136], v[15:16], v[84:85], v[133:136]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	v_and_b32_e32 v32, 56, v32
	v_add_co_u32_e32 v81, vcc, v60, v32
	v_addc_co_u32_e32 v82, vcc, 0, v61, vcc
	s_barrier
	global_load_dwordx2 v[49:50], v[81:82], off offset:64
	global_load_dwordx2 v[51:52], v[81:82], off offset:320
	global_load_dwordx2 v[53:54], v[81:82], off offset:576
	global_load_dwordx2 v[55:56], v[81:82], off offset:832
	s_waitcnt vmcnt(2)
	v_perm_b32 v57, v51, v49, s71
	v_perm_b32 v59, v51, v49, s73
	s_waitcnt vmcnt(0)
	v_perm_b32 v58, v55, v53, s71
	v_perm_b32 v61, v52, v50, s71
	v_perm_b32 v49, v52, v50, s73
	v_perm_b32 v50, v56, v54, s73
	v_perm_b32 v60, v55, v53, s73
	v_perm_b32 v62, v56, v54, s71
	ds_write_b64 v20, v[57:58]
	ds_write_b64 v21, v[59:60] offset:16
	ds_write_b64 v22, v[61:62] offset:32
	ds_write_b64 v23, v[49:50] offset:48
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 93 23                         ; test_input_spidD.mlir:93:23
	ds_read_b128 v[49:52], v24
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[169:172], v[33:34], v[49:50], v[169:172]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[53:56], v25 offset:576
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[169:172], v[35:36], v[51:52], v[169:172]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[169:172], v[37:38], v[53:54], v[169:172]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[169:172], v[39:40], v[55:56], v[169:172]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[57:60], v26 offset:4096
	ds_read_b128 v[61:64], v27 offset:5184
	ds_read_b128 v[65:68], v28 offset:544
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[169:172], v[43:44], v[57:58], v[169:172]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[69:72], v29 offset:1376
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[169:172], v[45:46], v[59:60], v[169:172]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[73:76], v30 offset:4640
	ds_read_b128 v[77:80], v31 offset:5984
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[169:172], v[41:42], v[61:62], v[169:172]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[169:172], v[47:48], v[63:64], v[169:172]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[173:176], v[33:34], v[65:66], v[173:176]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[173:176], v[35:36], v[67:68], v[173:176]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[173:176], v[37:38], v[69:70], v[173:176]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[173:176], v[39:40], v[71:72], v[173:176]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[173:176], v[43:44], v[73:74], v[173:176]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[173:176], v[45:46], v[75:76], v[173:176]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[173:176], v[41:42], v[77:78], v[173:176]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[173:176], v[47:48], v[79:80], v[173:176]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[137:140], v[1:2], v[49:50], v[137:140]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[137:140], v[3:4], v[51:52], v[137:140]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[137:140], v[5:6], v[53:54], v[137:140]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[137:140], v[7:8], v[55:56], v[137:140]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[137:140], v[9:10], v[57:58], v[137:140]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[137:140], v[11:12], v[59:60], v[137:140]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[137:140], v[13:14], v[61:62], v[137:140]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[137:140], v[15:16], v[63:64], v[137:140]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[141:144], v[1:2], v[65:66], v[141:144]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[141:144], v[3:4], v[67:68], v[141:144]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[141:144], v[5:6], v[69:70], v[141:144]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[141:144], v[7:8], v[71:72], v[141:144]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[141:144], v[9:10], v[73:74], v[141:144]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[141:144], v[11:12], v[75:76], v[141:144]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[141:144], v[13:14], v[77:78], v[141:144]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[141:144], v[15:16], v[79:80], v[141:144]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	s_barrier
	global_load_dwordx2 v[49:50], v[81:82], off offset:128
	global_load_dwordx2 v[51:52], v[81:82], off offset:384
	global_load_dwordx2 v[53:54], v[81:82], off offset:640
	global_load_dwordx2 v[55:56], v[81:82], off offset:896
	s_waitcnt vmcnt(2)
	v_perm_b32 v57, v51, v49, s71
	v_perm_b32 v59, v51, v49, s73
	s_waitcnt vmcnt(0)
	v_perm_b32 v58, v55, v53, s71
	v_perm_b32 v61, v52, v50, s71
	v_perm_b32 v49, v52, v50, s73
	v_perm_b32 v50, v56, v54, s73
	v_perm_b32 v60, v55, v53, s73
	v_perm_b32 v62, v56, v54, s71
	ds_write_b64 v20, v[57:58]
	ds_write_b64 v21, v[59:60] offset:16
	ds_write_b64 v22, v[61:62] offset:32
	ds_write_b64 v23, v[49:50] offset:48
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 93 23                         ; test_input_spidD.mlir:93:23
	ds_read_b128 v[49:52], v24
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[177:180], v[33:34], v[49:50], v[177:180]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[53:56], v25 offset:576
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[177:180], v[35:36], v[51:52], v[177:180]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[177:180], v[37:38], v[53:54], v[177:180]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[177:180], v[39:40], v[55:56], v[177:180]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[57:60], v26 offset:4096
	ds_read_b128 v[61:64], v27 offset:5184
	ds_read_b128 v[65:68], v28 offset:544
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[177:180], v[43:44], v[57:58], v[177:180]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[69:72], v29 offset:1376
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[177:180], v[45:46], v[59:60], v[177:180]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[73:76], v30 offset:4640
	ds_read_b128 v[77:80], v31 offset:5984
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[177:180], v[41:42], v[61:62], v[177:180]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[177:180], v[47:48], v[63:64], v[177:180]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[181:184], v[33:34], v[65:66], v[181:184]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[181:184], v[35:36], v[67:68], v[181:184]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[181:184], v[37:38], v[69:70], v[181:184]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[181:184], v[39:40], v[71:72], v[181:184]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[181:184], v[43:44], v[73:74], v[181:184]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[181:184], v[45:46], v[75:76], v[181:184]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[181:184], v[41:42], v[77:78], v[181:184]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[181:184], v[47:48], v[79:80], v[181:184]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[145:148], v[1:2], v[49:50], v[145:148]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[145:148], v[3:4], v[51:52], v[145:148]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[145:148], v[5:6], v[53:54], v[145:148]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[145:148], v[7:8], v[55:56], v[145:148]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[145:148], v[9:10], v[57:58], v[145:148]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[145:148], v[11:12], v[59:60], v[145:148]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[145:148], v[13:14], v[61:62], v[145:148]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[145:148], v[15:16], v[63:64], v[145:148]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[149:152], v[1:2], v[65:66], v[149:152]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[149:152], v[3:4], v[67:68], v[149:152]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[149:152], v[5:6], v[69:70], v[149:152]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[149:152], v[7:8], v[71:72], v[149:152]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[149:152], v[9:10], v[73:74], v[149:152]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[149:152], v[11:12], v[75:76], v[149:152]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[149:152], v[13:14], v[77:78], v[149:152]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[149:152], v[15:16], v[79:80], v[149:152]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 92 20                         ; test_input_spidD.mlir:92:20
	s_barrier
	global_load_dwordx2 v[49:50], v[81:82], off offset:192
	global_load_dwordx2 v[51:52], v[81:82], off offset:448
	global_load_dwordx2 v[53:54], v[81:82], off offset:704
	global_load_dwordx2 v[55:56], v[81:82], off offset:960
	s_waitcnt vmcnt(2)
	v_perm_b32 v57, v51, v49, s71
	v_perm_b32 v59, v51, v49, s73
	s_waitcnt vmcnt(0)
	v_perm_b32 v58, v55, v53, s71
	v_perm_b32 v60, v55, v53, s73
	v_perm_b32 v61, v52, v50, s71
	v_perm_b32 v62, v56, v54, s71
	v_perm_b32 v49, v52, v50, s73
	v_perm_b32 v50, v56, v54, s73
	ds_write_b64 v20, v[57:58]
	ds_write_b64 v21, v[59:60] offset:16
	ds_write_b64 v22, v[61:62] offset:32
	ds_write_b64 v23, v[49:50] offset:48
	s_waitcnt lgkmcnt(0)
	s_barrier
	.loc	1 93 23                         ; test_input_spidD.mlir:93:23
	ds_read_b128 v[20:23], v24
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[185:188], v[33:34], v[20:21], v[185:188]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[49:52], v25 offset:576
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[185:188], v[35:36], v[22:23], v[185:188]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[185:188], v[37:38], v[49:50], v[185:188]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[185:188], v[39:40], v[51:52], v[185:188]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[53:56], v26 offset:4096
	ds_read_b128 v[24:27], v27 offset:5184
	ds_read_b128 v[57:60], v28 offset:544
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[185:188], v[43:44], v[53:54], v[185:188]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[61:64], v29 offset:1376
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[185:188], v[45:46], v[55:56], v[185:188]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[65:68], v30 offset:4640
	ds_read_b128 v[28:31], v31 offset:5984
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(4)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[185:188], v[41:42], v[24:25], v[185:188]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[185:188], v[47:48], v[26:27], v[185:188]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(3)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[189:192], v[33:34], v[57:58], v[189:192]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[189:192], v[35:36], v[59:60], v[189:192]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(2)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[189:192], v[37:38], v[61:62], v[189:192]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[189:192], v[39:40], v[63:64], v[189:192]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[189:192], v[43:44], v[65:66], v[189:192]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[189:192], v[45:46], v[67:68], v[189:192]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[189:192], v[41:42], v[28:29], v[189:192]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[189:192], v[47:48], v[30:31], v[189:192]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[153:156], v[1:2], v[20:21], v[153:156]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[153:156], v[3:4], v[22:23], v[153:156]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[153:156], v[5:6], v[49:50], v[153:156]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[153:156], v[7:8], v[51:52], v[153:156]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[153:156], v[9:10], v[53:54], v[153:156]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[153:156], v[11:12], v[55:56], v[153:156]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[153:156], v[13:14], v[24:25], v[153:156]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[153:156], v[15:16], v[26:27], v[153:156]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[157:160], v[1:2], v[57:58], v[157:160]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[157:160], v[3:4], v[59:60], v[157:160]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[157:160], v[5:6], v[61:62], v[157:160]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[157:160], v[7:8], v[63:64], v[157:160]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[157:160], v[9:10], v[65:66], v[157:160]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[157:160], v[11:12], v[67:68], v[157:160]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[157:160], v[13:14], v[28:29], v[157:160]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	s_nop 1
	v_mmac_f32_16x16x16_f16 v[157:160], v[15:16], v[30:31], v[157:160]
	s_nop 1
	
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	s_add_u32 s8, s18, 1
	s_addc_u32 s9, s19, 0
	s_cmp_lg_u64 s[18:19], s[16:17]
	s_barrier
	s_cbranch_scc0 .LBB0_10
; %bb.8:                                ;   in Loop: Header=BB0_2 Depth=1
	.loc	1 0 24 is_stmt 0                ; test_input_spidD.mlir:0:24
	s_mov_b64 s[18:19], s[8:9]
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	s_branch .LBB0_2
.LBB0_9:
                                        ; implicit-def: $vgpr185_vgpr186_vgpr187_vgpr188_vgpr189_vgpr190_vgpr191_vgpr192
                                        ; implicit-def: $vgpr153_vgpr154_vgpr155_vgpr156_vgpr157_vgpr158_vgpr159_vgpr160
                                        ; implicit-def: $vgpr177_vgpr178_vgpr179_vgpr180_vgpr181_vgpr182_vgpr183_vgpr184
                                        ; implicit-def: $vgpr145_vgpr146_vgpr147_vgpr148_vgpr149_vgpr150_vgpr151_vgpr152
                                        ; implicit-def: $vgpr169_vgpr170_vgpr171_vgpr172_vgpr173_vgpr174_vgpr175_vgpr176
                                        ; implicit-def: $vgpr137_vgpr138_vgpr139_vgpr140_vgpr141_vgpr142_vgpr143_vgpr144
                                        ; implicit-def: $vgpr193
                                        ; implicit-def: $sgpr4_sgpr5
                                        ; implicit-def: $vgpr210
                                        ; implicit-def: $vgpr17
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr19
                                        ; implicit-def: $vgpr161_vgpr162_vgpr163_vgpr164_vgpr165_vgpr166_vgpr167_vgpr168
                                        ; implicit-def: $vgpr129_vgpr130_vgpr131_vgpr132_vgpr133_vgpr134_vgpr135_vgpr136
	.loc	1 0 24                          ; test_input_spidD.mlir:0:24
	s_cbranch_execnz .LBB0_11
	s_branch .LBB0_12
.LBB0_10:                               ; %Flow
	s_branch .LBB0_12
.LBB0_11:                               ; %.._crit_edge_crit_edge
	s_mov_b32 s16, 0
	s_mov_b32 s22, s16
	s_mov_b32 s23, s16
	s_mov_b32 s17, s16
	s_mov_b32 s18, s16
	s_mov_b32 s19, s16
	s_mov_b32 s20, s16
	s_mov_b32 s21, s16
	v_mov_b64_e32 v[167:168], s[22:23]
	v_mov_b64_e32 v[161:162], s[16:17]
	v_mov_b64_e32 v[165:166], s[20:21]
	v_mov_b64_e32 v[163:164], s[18:19]
	s_nop 1
	v_mov_b64_e32 v[129:130], v[161:162]
	v_mov_b64_e32 v[175:176], v[167:168]
	v_mov_b64_e32 v[137:138], v[161:162]
	v_mov_b64_e32 v[183:184], v[167:168]
	v_mov_b64_e32 v[145:146], v[161:162]
	v_mov_b64_e32 v[191:192], v[167:168]
	v_mov_b64_e32 v[153:154], v[161:162]
	.loc	1 99 9 is_stmt 1                ; test_input_spidD.mlir:99:9
	s_add_u32 s4, s30, s2
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	v_mbcnt_hi_u32_b32 v210, -1, v209
	v_mov_b64_e32 v[131:132], v[163:164]
	v_mov_b64_e32 v[133:134], v[165:166]
	v_mov_b64_e32 v[135:136], v[167:168]
	v_mov_b64_e32 v[173:174], v[165:166]
	v_mov_b64_e32 v[171:172], v[163:164]
	v_mov_b64_e32 v[169:170], v[161:162]
	v_mov_b64_e32 v[139:140], v[163:164]
	v_mov_b64_e32 v[141:142], v[165:166]
	v_mov_b64_e32 v[143:144], v[167:168]
	v_mov_b64_e32 v[181:182], v[165:166]
	v_mov_b64_e32 v[179:180], v[163:164]
	v_mov_b64_e32 v[177:178], v[161:162]
	v_mov_b64_e32 v[147:148], v[163:164]
	v_mov_b64_e32 v[149:150], v[165:166]
	v_mov_b64_e32 v[151:152], v[167:168]
	v_mov_b64_e32 v[189:190], v[165:166]
	v_mov_b64_e32 v[187:188], v[163:164]
	v_mov_b64_e32 v[185:186], v[161:162]
	v_mov_b64_e32 v[155:156], v[163:164]
	v_mov_b64_e32 v[157:158], v[165:166]
	v_mov_b64_e32 v[159:160], v[167:168]
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	s_addc_u32 s5, s31, s3
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	v_add_u32_e32 v17, 64, v210
	v_xor_b32_e32 v18, 16, v210
	v_xor_b32_e32 v19, 32, v210
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v193, 0
.LBB0_12:                               ; %._crit_edge
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	v_and_b32_e32 v2, 0xc0, v17
	v_and_b32_e32 v1, 16, v0
	v_cmp_lt_u32_e32 vcc, v18, v2
	v_cmp_eq_u32_e64 s[2:3], 0, v1
	v_cndmask_b32_e32 v3, v210, v18, vcc
	v_cndmask_b32_e64 v1, v185, v186, s[2:3]
	v_lshlrev_b32_e32 v48, 2, v3
	v_cndmask_b32_e64 v3, v187, v188, s[2:3]
	ds_bpermute_b32 v1, v48, v1
	ds_bpermute_b32 v3, v48, v3
	v_and_b32_e32 v4, 32, v0
	v_cmp_lt_u32_e32 vcc, v19, v2
	v_cmp_eq_u32_e64 s[0:1], 0, v4
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v5, v1, v185, s[2:3]
	v_cndmask_b32_e64 v6, v3, v187, s[2:3]
	v_cndmask_b32_e32 v2, v210, v19, vcc
	v_cndmask_b32_e64 v4, v5, v6, s[0:1]
	v_lshlrev_b32_e32 v49, 2, v2
	ds_bpermute_b32 v2, v49, v4
	v_cndmask_b32_e64 v7, v188, v3, s[2:3]
	v_cndmask_b32_e64 v3, v177, v178, s[2:3]
	v_cndmask_b32_e64 v4, v186, v1, s[2:3]
	v_cndmask_b32_e64 v39, v131, v132, s[2:3]
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v1, v2, v5, s[0:1]
	ds_bpermute_b32 v5, v48, v3
	v_cndmask_b32_e64 v3, v179, v180, s[2:3]
	ds_bpermute_b32 v8, v48, v3
	v_cndmask_b32_e64 v3, v6, v2, s[0:1]
	v_cndmask_b32_e64 v2, v4, v7, s[0:1]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v9, v5, v177, s[2:3]
	ds_bpermute_b32 v6, v49, v2
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v10, v8, v179, s[2:3]
	v_cndmask_b32_e64 v2, v9, v10, s[0:1]
	ds_bpermute_b32 v11, v49, v2
	v_cndmask_b32_e64 v8, v180, v8, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v2, v6, v4, s[0:1]
	v_cndmask_b32_e64 v4, v7, v6, s[0:1]
	v_cndmask_b32_e64 v7, v169, v170, s[2:3]
	v_cndmask_b32_e64 v6, v178, v5, s[2:3]
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v5, v11, v9, s[0:1]
	ds_bpermute_b32 v9, v48, v7
	v_cndmask_b32_e64 v7, v171, v172, s[2:3]
	ds_bpermute_b32 v12, v48, v7
	v_cndmask_b32_e64 v7, v10, v11, s[0:1]
	v_cndmask_b32_e64 v10, v6, v8, s[0:1]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v11, v9, v169, s[2:3]
	ds_bpermute_b32 v10, v49, v10
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v13, v12, v171, s[2:3]
	v_cndmask_b32_e64 v14, v11, v13, s[0:1]
	ds_bpermute_b32 v14, v49, v14
	v_cndmask_b32_e64 v9, v170, v9, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v6, v10, v6, s[0:1]
	v_cndmask_b32_e64 v8, v8, v10, s[0:1]
	v_cndmask_b32_e64 v10, v172, v12, s[2:3]
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v17, v14, v11, s[0:1]
	v_cndmask_b32_e64 v11, v9, v10, s[0:1]
	ds_bpermute_b32 v11, v49, v11
	v_cndmask_b32_e64 v13, v13, v14, s[0:1]
	v_cndmask_b32_e64 v12, v161, v162, s[2:3]
	v_cndmask_b32_e64 v14, v163, v164, s[2:3]
	ds_bpermute_b32 v12, v48, v12
	ds_bpermute_b32 v15, v48, v14
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v18, v11, v9, s[0:1]
	v_cndmask_b32_e64 v14, v10, v11, s[0:1]
	v_cndmask_b32_e64 v9, v189, v190, s[2:3]
	v_cndmask_b32_e64 v10, v191, v192, s[2:3]
	ds_bpermute_b32 v9, v48, v9
	ds_bpermute_b32 v10, v48, v10
	s_waitcnt lgkmcnt(3)
	v_cndmask_b32_e64 v61, v12, v161, s[2:3]
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v63, v15, v163, s[2:3]
	v_cndmask_b32_e64 v11, v61, v63, s[0:1]
	v_cndmask_b32_e64 v62, v162, v12, s[2:3]
	ds_bpermute_b32 v65, v49, v11
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v11, v9, v189, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v12, v10, v191, s[2:3]
	v_cndmask_b32_e64 v64, v164, v15, s[2:3]
	v_cndmask_b32_e64 v15, v11, v12, s[0:1]
	ds_bpermute_b32 v15, v49, v15
	v_cndmask_b32_e64 v19, v192, v10, s[2:3]
	v_cndmask_b32_e64 v10, v181, v182, s[2:3]
	v_cndmask_b32_e64 v16, v62, v64, s[0:1]
	ds_bpermute_b32 v20, v48, v10
	v_cndmask_b32_e64 v10, v183, v184, s[2:3]
	ds_bpermute_b32 v66, v49, v16
	v_cndmask_b32_e64 v16, v190, v9, s[2:3]
	ds_bpermute_b32 v21, v48, v10
	v_cndmask_b32_e64 v10, v16, v19, s[0:1]
	s_waitcnt lgkmcnt(3)
	v_cndmask_b32_e64 v9, v15, v11, s[0:1]
	v_cndmask_b32_e64 v11, v12, v15, s[0:1]
	ds_bpermute_b32 v12, v49, v10
	s_waitcnt lgkmcnt(3)
	v_cndmask_b32_e64 v15, v20, v181, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v22, v21, v183, s[2:3]
	v_cndmask_b32_e64 v10, v15, v22, s[0:1]
	ds_bpermute_b32 v23, v49, v10
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v10, v12, v16, s[0:1]
	v_cndmask_b32_e64 v12, v19, v12, s[0:1]
	v_cndmask_b32_e64 v19, v173, v174, s[2:3]
	v_cndmask_b32_e64 v16, v182, v20, s[2:3]
	v_cndmask_b32_e64 v20, v184, v21, s[2:3]
	ds_bpermute_b32 v21, v48, v19
	v_cndmask_b32_e64 v19, v175, v176, s[2:3]
	ds_bpermute_b32 v24, v48, v19
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v19, v22, v23, s[0:1]
	v_cndmask_b32_e64 v22, v16, v20, s[0:1]
	v_cndmask_b32_e64 v15, v23, v15, s[0:1]
	ds_bpermute_b32 v22, v49, v22
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v23, v21, v173, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v25, v24, v175, s[2:3]
	v_cndmask_b32_e64 v26, v23, v25, s[0:1]
	ds_bpermute_b32 v26, v49, v26
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v16, v22, v16, s[0:1]
	v_cndmask_b32_e64 v20, v20, v22, s[0:1]
	v_cndmask_b32_e64 v21, v174, v21, s[2:3]
	v_cndmask_b32_e64 v22, v176, v24, s[2:3]
	v_cndmask_b32_e64 v24, v21, v22, s[0:1]
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v23, v26, v23, s[0:1]
	v_cndmask_b32_e64 v25, v25, v26, s[0:1]
	ds_bpermute_b32 v26, v49, v24
	v_cndmask_b32_e64 v24, v165, v166, s[2:3]
	ds_bpermute_b32 v27, v48, v24
	v_cndmask_b32_e64 v24, v167, v168, s[2:3]
	ds_bpermute_b32 v28, v48, v24
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v24, v26, v21, s[0:1]
	v_cndmask_b32_e64 v26, v22, v26, s[0:1]
	v_cndmask_b32_e64 v21, v153, v154, s[2:3]
	v_cndmask_b32_e64 v22, v155, v156, s[2:3]
	ds_bpermute_b32 v21, v48, v21
	ds_bpermute_b32 v22, v48, v22
	s_waitcnt lgkmcnt(3)
	v_cndmask_b32_e64 v55, v27, v165, s[2:3]
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v57, v28, v167, s[2:3]
	v_cndmask_b32_e64 v56, v166, v27, s[2:3]
	v_cndmask_b32_e64 v27, v55, v57, s[0:1]
	v_cndmask_b32_e64 v58, v168, v28, s[2:3]
	ds_bpermute_b32 v59, v49, v27
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v27, v21, v153, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v28, v22, v155, s[2:3]
	v_cndmask_b32_e64 v29, v27, v28, s[0:1]
	ds_bpermute_b32 v29, v49, v29
	v_cndmask_b32_e64 v31, v156, v22, s[2:3]
	v_cndmask_b32_e64 v22, v145, v146, s[2:3]
	v_cndmask_b32_e64 v30, v56, v58, s[0:1]
	ds_bpermute_b32 v32, v48, v22
	v_cndmask_b32_e64 v22, v147, v148, s[2:3]
	ds_bpermute_b32 v60, v49, v30
	v_cndmask_b32_e64 v30, v154, v21, s[2:3]
	ds_bpermute_b32 v33, v48, v22
	v_cndmask_b32_e64 v22, v30, v31, s[0:1]
	s_waitcnt lgkmcnt(3)
	v_cndmask_b32_e64 v21, v29, v27, s[0:1]
	v_cndmask_b32_e64 v27, v28, v29, s[0:1]
	ds_bpermute_b32 v28, v49, v22
	s_waitcnt lgkmcnt(3)
	v_cndmask_b32_e64 v29, v32, v145, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v34, v33, v147, s[2:3]
	v_cndmask_b32_e64 v22, v29, v34, s[0:1]
	ds_bpermute_b32 v35, v49, v22
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v22, v28, v30, s[0:1]
	v_cndmask_b32_e64 v28, v31, v28, s[0:1]
	v_cndmask_b32_e64 v31, v137, v138, s[2:3]
	v_cndmask_b32_e64 v30, v146, v32, s[2:3]
	v_cndmask_b32_e64 v32, v148, v33, s[2:3]
	ds_bpermute_b32 v33, v48, v31
	v_cndmask_b32_e64 v31, v139, v140, s[2:3]
	ds_bpermute_b32 v36, v48, v31
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v29, v35, v29, s[0:1]
	v_cndmask_b32_e64 v31, v34, v35, s[0:1]
	v_cndmask_b32_e64 v34, v30, v32, s[0:1]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v35, v33, v137, s[2:3]
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v37, v36, v139, s[2:3]
	ds_bpermute_b32 v34, v49, v34
	v_cndmask_b32_e64 v38, v35, v37, s[0:1]
	ds_bpermute_b32 v38, v49, v38
	v_cndmask_b32_e64 v36, v140, v36, s[2:3]
	ds_bpermute_b32 v39, v48, v39
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v30, v34, v30, s[0:1]
	v_cndmask_b32_e64 v32, v32, v34, s[0:1]
	v_cndmask_b32_e64 v34, v138, v33, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v33, v38, v35, s[0:1]
	v_cndmask_b32_e64 v35, v37, v38, s[0:1]
	v_cndmask_b32_e64 v37, v34, v36, s[0:1]
	v_cndmask_b32_e64 v38, v129, v130, s[2:3]
	ds_bpermute_b32 v37, v49, v37
	ds_bpermute_b32 v38, v48, v38
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v70, v132, v39, s[2:3]
	v_cndmask_b32_e64 v69, v39, v131, s[2:3]
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_div_scale_f32 v52, s[8:9], v194, v194, 1.0
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v34, v37, v34, s[0:1]
	v_cndmask_b32_e64 v36, v36, v37, s[0:1]
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v67, v38, v129, s[2:3]
	v_cndmask_b32_e64 v68, v130, v38, s[2:3]
	v_cndmask_b32_e64 v37, v157, v158, s[2:3]
	v_cndmask_b32_e64 v38, v159, v160, s[2:3]
	ds_bpermute_b32 v37, v48, v37
	ds_bpermute_b32 v38, v48, v38
	v_cndmask_b32_e64 v42, v68, v70, s[0:1]
	ds_bpermute_b32 v72, v49, v42
	v_cndmask_b32_e64 v39, v67, v69, s[0:1]
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v42, v158, v37, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v43, v160, v38, s[2:3]
	v_cndmask_b32_e64 v40, v38, v159, s[2:3]
	v_cndmask_b32_e64 v38, v42, v43, s[0:1]
	ds_bpermute_b32 v71, v49, v39
	v_cndmask_b32_e64 v39, v37, v157, s[2:3]
	ds_bpermute_b32 v44, v49, v38
	v_cndmask_b32_e64 v38, v149, v150, s[2:3]
	v_cndmask_b32_e64 v41, v39, v40, s[0:1]
	ds_bpermute_b32 v45, v48, v38
	v_cndmask_b32_e64 v38, v151, v152, s[2:3]
	ds_bpermute_b32 v41, v49, v41
	ds_bpermute_b32 v46, v48, v38
	s_waitcnt lgkmcnt(3)
	v_cndmask_b32_e64 v38, v44, v42, s[0:1]
	v_cndmask_b32_e64 v47, v141, v142, s[2:3]
	ds_bpermute_b32 v47, v48, v47
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v37, v41, v39, s[0:1]
	v_cndmask_b32_e64 v39, v40, v41, s[0:1]
	v_cndmask_b32_e64 v40, v43, v44, s[0:1]
	v_cndmask_b32_e64 v41, v45, v149, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v42, v46, v151, s[2:3]
	v_cndmask_b32_e64 v44, v150, v45, s[2:3]
	v_cndmask_b32_e64 v45, v152, v46, s[2:3]
	v_cndmask_b32_e64 v43, v41, v42, s[0:1]
	v_cndmask_b32_e64 v46, v44, v45, s[0:1]
	ds_bpermute_b32 v43, v49, v43
	ds_bpermute_b32 v46, v49, v46
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_div_scale_f32 v73, vcc, 1.0, v194, 1.0
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v50, v47, v141, s[2:3]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v41, v43, v41, s[0:1]
	v_cndmask_b32_e64 v43, v42, v43, s[0:1]
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v42, v46, v44, s[0:1]
	v_cndmask_b32_e64 v44, v143, v144, s[2:3]
	ds_bpermute_b32 v44, v48, v44
	v_cndmask_b32_e64 v51, v142, v47, s[2:3]
	v_cndmask_b32_e64 v55, v59, v55, s[0:1]
	v_cndmask_b32_e64 v57, v57, v59, s[0:1]
	v_cndmask_b32_e64 v56, v60, v56, s[0:1]
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v47, v44, v143, s[2:3]
	v_cndmask_b32_e64 v53, v144, v44, s[2:3]
	v_cndmask_b32_e64 v44, v50, v47, s[0:1]
	ds_bpermute_b32 v54, v49, v44
	v_cndmask_b32_e64 v44, v51, v53, s[0:1]
	ds_bpermute_b32 v74, v49, v44
	v_cndmask_b32_e64 v44, v133, v134, s[2:3]
	ds_bpermute_b32 v75, v48, v44
	v_cndmask_b32_e64 v44, v45, v46, s[0:1]
	s_waitcnt lgkmcnt(2)
	v_cndmask_b32_e64 v45, v54, v50, s[0:1]
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v46, v74, v51, s[0:1]
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_rcp_f32_e32 v51, v52
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	v_cndmask_b32_e64 v47, v47, v54, s[0:1]
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v50, v75, v133, s[2:3]
	v_cndmask_b32_e64 v54, v134, v75, s[2:3]
	v_cndmask_b32_e64 v75, v135, v136, s[2:3]
	ds_bpermute_b32 v48, v48, v75
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_fma_f32 v75, -v52, v51, 1.0
	v_fmac_f32_e32 v51, v75, v51
	v_mul_f32_e32 v75, v73, v51
	v_fma_f32 v76, -v52, v75, v73
	v_fmac_f32_e32 v75, v76, v51
	v_fma_f32 v52, -v52, v75, v73
	v_div_fmas_f32 v73, v52, v51, v75
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v51, v48, v135, s[2:3]
	v_cndmask_b32_e64 v52, v136, v48, s[2:3]
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_div_scale_f32 v77, s[2:3], v193, v193, 1.0
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	v_cndmask_b32_e64 v48, v50, v51, s[0:1]
	ds_bpermute_b32 v75, v49, v48
	v_cndmask_b32_e64 v48, v54, v52, s[0:1]
	ds_bpermute_b32 v76, v49, v48
	v_cndmask_b32_e64 v48, v53, v74, s[0:1]
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_div_scale_f32 v74, vcc, 1.0, v193, 1.0
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	s_waitcnt lgkmcnt(1)
	v_cndmask_b32_e64 v49, v75, v50, s[0:1]
	s_waitcnt lgkmcnt(0)
	v_cndmask_b32_e64 v50, v76, v54, s[0:1]
	v_cndmask_b32_e64 v51, v51, v75, s[0:1]
	v_cndmask_b32_e64 v52, v52, v76, s[0:1]
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_div_fixup_f32 v53, v73, v194, 1.0
	v_mov_b32_e32 v54, v53
	s_nop 1
	v_pk_mul_f32 v[51:52], v[53:54], v[51:52]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v73, v52
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_rcp_f32_e32 v52, v77
	v_pk_mul_f32 v[49:50], v[53:54], v[49:50]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v78, v49
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	v_cndmask_b32_e64 v49, v71, v67, s[0:1]
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_fma_f32 v67, -v77, v52, 1.0
	v_fmac_f32_e32 v52, v67, v52
	v_mul_f32_e32 v67, v74, v52
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v76, v50
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	v_cndmask_b32_e64 v50, v72, v68, s[0:1]
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_fma_f32 v68, -v77, v67, v74
	v_fmac_f32_e32 v67, v68, v52
	v_fma_f32 v68, -v77, v67, v74
	v_div_fmas_f32 v67, v68, v52, v67
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v75, v51
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	v_cndmask_b32_e64 v51, v69, v71, s[0:1]
	v_cndmask_b32_e64 v52, v70, v72, s[0:1]
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[49:50], v[53:54], v[49:50]
	s_nop 0
	v_pk_mul_f32 v[51:52], v[53:54], v[51:52]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v52, v52
	v_cvt_f16_f32_e32 v51, v51
	v_cvt_f16_f32_e32 v68, v50
	v_cvt_f16_f32_e32 v69, v49
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	v_cndmask_b32_e64 v58, v58, v60, s[0:1]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_pack_b32_f16 v52, v51, v52
	v_pack_b32_f16 v50, v75, v73
	v_pack_b32_f16 v51, v69, v68
	v_pack_b32_f16 v49, v78, v76
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_div_fixup_f32 v59, v67, v193, 1.0
	v_mov_b32_e32 v60, v59
	s_nop 1
	v_pk_mul_f32 v[55:56], v[59:60], v[55:56]
	v_pk_mul_f32 v[57:58], v[59:60], v[57:58]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v67, v58
	v_cvt_f16_f32_e32 v68, v57
	v_cvt_f16_f32_e32 v69, v56
	v_cvt_f16_f32_e32 v70, v55
	.loc	1 56 24                         ; test_input_spidD.mlir:56:24
	v_cndmask_b32_e64 v55, v65, v61, s[0:1]
	v_cndmask_b32_e64 v56, v66, v62, s[0:1]
	v_cndmask_b32_e64 v57, v63, v65, s[0:1]
	v_cndmask_b32_e64 v58, v64, v66, s[0:1]
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[55:56], v[59:60], v[55:56]
	s_nop 0
	v_pk_mul_f32 v[57:58], v[59:60], v[57:58]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v58, v58
	v_cvt_f16_f32_e32 v57, v57
	v_cvt_f16_f32_e32 v61, v56
	v_cvt_f16_f32_e32 v62, v55
	v_pack_b32_f16 v56, v68, v67
	v_pack_b32_f16 v55, v70, v69
	v_pack_b32_f16 v58, v57, v58
	v_pack_b32_f16 v57, v62, v61
	; sched_barrier mask(0x00000000)
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_lshrrev_b32_e32 v63, 2, v0
	v_and_b32_e32 v62, 0xf0, v0
	v_mov_b32_e32 v64, s5
	v_add_co_u32_e32 v0, vcc, s4, v0
	v_addc_co_u32_e32 v64, vcc, 0, v64, vcc
	v_sub_co_u32_e32 v0, vcc, v0, v62
	v_and_b32_e32 v61, 48, v63
	v_subbrev_co_u32_e32 v62, vcc, 0, v64, vcc
	v_add_co_u32_e32 v61, vcc, v0, v61
	s_lshl_b32 s0, s7, 20
	v_addc_co_u32_e32 v62, vcc, 0, v62, vcc
	s_and_b32 s0, s0, 0x1f00000
	s_add_u32 s0, s14, s0
	v_lshlrev_b64 v[61:62], 8, v[61:62]
	s_addc_u32 s1, s15, 0
	v_mov_b32_e32 v0, s1
	v_add_co_u32_e32 v64, vcc, s0, v61
	v_lshlrev_b32_e32 v63, 1, v63
	v_addc_co_u32_e32 v65, vcc, v0, v62, vcc
	v_and_b32_e32 v0, 24, v63
	v_add_co_u32_e32 v61, vcc, v64, v0
	v_addc_co_u32_e32 v62, vcc, 0, v65, vcc
	s_movk_i32 s0, 0x4000
	global_store_dwordx2 v[61:62], v[57:58], off
	global_store_dwordx2 v[61:62], v[55:56], off offset:32
	v_add_co_u32_e32 v57, vcc, s0, v64
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[17:18], v[59:60], v[17:18]
	v_pk_mul_f32 v[13:14], v[59:60], v[13:14]
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	v_addc_co_u32_e32 v58, vcc, 0, v65, vcc
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[45:46], v[53:54], v[45:46]
	v_pk_mul_f32 v[33:34], v[53:54], v[33:34]
	v_pk_mul_f32 v[23:24], v[59:60], v[23:24]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v14, v14
	v_cvt_f16_f32_e32 v13, v13
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v17, v17
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	v_add_co_u32_e32 v55, vcc, v57, v0
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v0, v46
	v_cvt_f16_f32_e32 v66, v45
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[45:46], v[53:54], v[47:48]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v34, v34
	v_cvt_f16_f32_e32 v33, v33
	v_cvt_f16_f32_e32 v47, v24
	v_cvt_f16_f32_e32 v48, v23
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[25:26], v[59:60], v[25:26]
	v_pk_mul_f32 v[35:36], v[53:54], v[35:36]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v25, v25
	v_cvt_f16_f32_e32 v36, v36
	v_cvt_f16_f32_e32 v35, v35
	v_cvt_f16_f32_e32 v26, v26
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	v_addc_co_u32_e32 v56, vcc, 0, v58, vcc
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_pack_b32_f16 v14, v13, v14
	v_pack_b32_f16 v13, v17, v18
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	v_or_b32_e32 v17, 0x66, v63
	global_store_dwordx2 v[55:56], v[51:52], off
	global_store_dwordx2 v[55:56], v[49:50], off offset:32
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v45, v45
	v_pack_b32_f16 v23, v33, v34
	v_pack_b32_f16 v33, v48, v47
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	global_store_dwordx2 v[61:62], v[13:14], off offset:64
	global_store_short v[61:62], v25, off offset:100
	global_store_dword v[61:62], v33, off offset:96
	v_add_co_u32_e32 v13, vcc, v64, v17
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v46, v46
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	v_addc_co_u32_e32 v14, vcc, 0, v65, vcc
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_pack_b32_f16 v0, v66, v0
	v_pack_b32_f16 v24, v35, v36
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	global_store_short v[13:14], v26, off
	global_store_dwordx2 v[55:56], v[23:24], off offset:64
	global_store_short v[55:56], v45, off offset:100
	global_store_dword v[55:56], v0, off offset:96
	v_add_co_u32_e32 v13, vcc, v57, v17
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[17:18], v[53:54], v[43:44]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v0, v18
	v_cvt_f16_f32_e32 v23, v17
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	v_addc_co_u32_e32 v14, vcc, 0, v58, vcc
	global_store_short v[13:14], v46, off
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[13:14], v[53:54], v[41:42]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v24, v14
	v_cvt_f16_f32_e32 v25, v13
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[13:14], v[53:54], v[29:30]
	v_pk_mul_f32 v[15:16], v[59:60], v[15:16]
	v_pk_mul_f32 v[19:20], v[59:60], v[19:20]
	v_pk_mul_f32 v[5:6], v[59:60], v[5:6]
	v_pk_mul_f32 v[7:8], v[59:60], v[7:8]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v26, v14
	v_pack_b32_f16 v14, v23, v0
	v_cvt_f16_f32_e32 v0, v20
	v_cvt_f16_f32_e32 v19, v19
	v_cvt_f16_f32_e32 v16, v16
	v_cvt_f16_f32_e32 v15, v15
	v_cvt_f16_f32_e32 v8, v8
	v_cvt_f16_f32_e32 v7, v7
	v_cvt_f16_f32_e32 v20, v6
	v_cvt_f16_f32_e32 v23, v5
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[17:18], v[53:54], v[31:32]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_pack_b32_f16 v6, v19, v0
	v_pack_b32_f16 v5, v15, v16
	v_pack_b32_f16 v8, v7, v8
	v_pack_b32_f16 v7, v23, v20
	v_cvt_f16_f32_e32 v18, v18
	v_cvt_f16_f32_e32 v17, v17
	v_cvt_f16_f32_e32 v29, v13
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	global_store_dwordx2 v[61:62], v[7:8], off offset:128
	global_store_dwordx2 v[61:62], v[5:6], off offset:160
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[5:6], v[53:54], v[37:38]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v0, v6
	v_cvt_f16_f32_e32 v7, v5
	v_pack_b32_f16 v13, v25, v24
	v_pack_b32_f16 v18, v17, v18
	v_pack_b32_f16 v17, v29, v26
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	global_store_dwordx2 v[55:56], v[17:18], off offset:128
	global_store_dwordx2 v[55:56], v[13:14], off offset:160
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_pack_b32_f16 v13, v7, v0
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[7:8], v[53:54], v[27:28]
	v_pk_mul_f32 v[5:6], v[53:54], v[39:40]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v0, v8
	v_cvt_f16_f32_e32 v7, v7
	v_cvt_f16_f32_e32 v14, v6
	v_cvt_f16_f32_e32 v15, v5
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[5:6], v[53:54], v[21:22]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v8, v6
	v_cvt_f16_f32_e32 v16, v5
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[5:6], v[59:60], v[9:10]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v9, v6
	v_pack_b32_f16 v6, v7, v0
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[0:1], v[59:60], v[1:2]
	v_pk_mul_f32 v[2:3], v[59:60], v[3:4]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v3, v3
	v_cvt_f16_f32_e32 v2, v2
	v_cvt_f16_f32_e32 v4, v1
	v_cvt_f16_f32_e32 v0, v0
	v_cvt_f16_f32_e32 v10, v5
	v_pack_b32_f16 v5, v16, v8
	.loc	1 97 23                         ; test_input_spidD.mlir:97:23
	v_pk_mul_f32 v[7:8], v[59:60], v[11:12]
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_cvt_f16_f32_e32 v7, v7
	v_cvt_f16_f32_e32 v8, v8
	v_pack_b32_f16 v1, v2, v3
	v_pack_b32_f16 v0, v0, v4
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	v_or_b32_e32 v2, 0xe6, v63
	.loc	1 98 19                         ; test_input_spidD.mlir:98:19
	v_pack_b32_f16 v9, v10, v9
	.loc	1 99 9                          ; test_input_spidD.mlir:99:9
	global_store_dwordx2 v[61:62], v[0:1], off offset:192
	global_store_short v[61:62], v7, off offset:228
	global_store_dword v[61:62], v9, off offset:224
	v_add_co_u32_e32 v0, vcc, v64, v2
	v_addc_co_u32_e32 v1, vcc, 0, v65, vcc
	global_store_short v[0:1], v8, off
	global_store_dwordx2 v[55:56], v[5:6], off offset:192
	global_store_short v[55:56], v15, off offset:228
	global_store_dword v[55:56], v13, off offset:224
	v_add_co_u32_e32 v0, vcc, v57, v2
	v_addc_co_u32_e32 v1, vcc, 0, v58, vcc
	global_store_short v[0:1], v14, off
	.loc	1 22 3                          ; test_input_spidD.mlir:22:3
	s_endpgm
.Ltmp0:
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel Attn_p2
		.amdhsa_group_segment_fixed_size 16384
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 32
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
		.amdhsa_system_sgpr_workgroup_id_y 1
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 248
		.amdhsa_next_free_sgpr 98
		.amdhsa_reserve_vcc 1
		.amdhsa_reserve_flat_scratch 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_load_store_out_of_order 0
		.amdhsa_matrix_excp 0
		.amdhsa_cancellation_excp 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	Attn_p2, .Lfunc_end0-Attn_p2
	.cfi_endproc
                                        ; -- End function
	.set Attn_p2.num_vgpr, 248
	.set Attn_p2.num_agpr, 0
	.set Attn_p2.numbered_sgpr, 75
	.set Attn_p2.num_named_barrier, 0
	.set Attn_p2.private_seg_size, 0
	.set Attn_p2.uses_vcc, 1
	.set Attn_p2.uses_flat_scratch, 0
	.set Attn_p2.has_dyn_sized_stack, 0
	.set Attn_p2.has_recursion, 0
	.set Attn_p2.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 15804
; TotalNumSgprs: 79
; NumVgprs: 248
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 16384 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 61
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 248
; Occupancy: 1
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 6
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
	.text
	.protected	__ockl_devmem_request   ; -- Begin function __ockl_devmem_request
	.weak	__ockl_devmem_request
	.p2align	2
	.type	__ockl_devmem_request,@function
__ockl_devmem_request:                  ; @__ockl_devmem_request
.Lfunc_begin1:
	.cfi_startproc
; %bb.0:
	.cfi_llvm_def_aspace_cfa 64, 0, 6
	.cfi_llvm_register_pair 16, 62, 32, 63, 32
	.cfi_undefined 2560
	.cfi_undefined 2561
	.cfi_undefined 2562
	.cfi_undefined 2563
	.cfi_undefined 2564
	.cfi_undefined 2565
	.cfi_undefined 2566
	.cfi_undefined 2567
	.cfi_undefined 2568
	.cfi_undefined 2569
	.cfi_undefined 2570
	.cfi_undefined 2571
	.cfi_undefined 2572
	.cfi_undefined 2573
	.cfi_undefined 2574
	.cfi_undefined 2575
	.cfi_undefined 36
	.cfi_undefined 37
	.cfi_undefined 38
	.cfi_undefined 39
	.cfi_undefined 40
	.cfi_undefined 41
	.cfi_undefined 42
	.cfi_undefined 43
	.cfi_undefined 44
	.cfi_undefined 45
	.cfi_undefined 46
	.cfi_undefined 47
	.cfi_undefined 48
	.cfi_undefined 49
	.cfi_undefined 50
	.cfi_undefined 51
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_load_dwordx2 s[8:9], s[8:9], 0x18
	v_mbcnt_lo_u32_b32 v4, -1, 0
	v_mbcnt_hi_u32_b32 v8, -1, v4
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_readfirstlane_b32 s4, v8
	v_mov_b32_e32 v10, 0
	v_cmp_ne_u32_e32 vcc, s4, v8
	v_mov_b32_e32 v11, 0
	v_cmp_eq_u32_e64 s[4:5], s4, v8
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB1_6
; %bb.1:
	v_mov_b32_e32 v4, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[12:13], v4, s[8:9] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_wbinvl1_vol
	global_load_dwordx2 v[5:6], v4, s[8:9] offset:40
	global_load_dwordx2 v[9:10], v4, s[8:9]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v5, v5, v12
	v_and_b32_e32 v6, v6, v13
	v_mul_lo_u32 v6, v6, 24
	v_mul_hi_u32 v7, v5, 24
	v_mul_lo_u32 v5, v5, 24
	v_add_u32_e32 v6, v7, v6
	s_waitcnt vmcnt(0)
	v_add_co_u32_e64 v5, s[6:7], v9, v5
	v_addc_co_u32_e64 v6, s[6:7], v10, v6, s[6:7]
	global_load_dwordx2 v[10:11], v[5:6], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v4, v[10:13], s[8:9] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_wbinvl1_vol
	v_cmp_ne_u64_e64 s[6:7], v[10:11], v[12:13]
	s_and_saveexec_b64 s[12:13], s[6:7]
	s_cbranch_execz .LBB1_5
; %bb.2:                                ; %.preheader6.i.i.preheader
	s_mov_b64 s[14:15], 0
.LBB1_3:                                ; %.preheader6.i.i
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_load_dwordx2 v[5:6], v4, s[8:9] offset:40
	global_load_dwordx2 v[14:15], v4, s[8:9]
	v_mov_b64_e32 v[12:13], v[10:11]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v5, v5, v12
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[9:10], s[6:7], v5, 24, v[14:15]
	v_and_b32_e32 v6, v6, v13
	v_mov_b32_e32 v5, v10
	s_nop 1
	v_mad_u64_u32 v[5:6], s[6:7], v6, 24, v[5:6]
	v_mov_b32_e32 v10, v5
	global_load_dwordx2 v[10:11], v[9:10], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[10:11], v4, v[10:13], s[8:9] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_wbinvl1_vol
	v_cmp_eq_u64_e64 s[6:7], v[10:11], v[12:13]
	s_or_b64 s[14:15], s[6:7], s[14:15]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB1_3
; %bb.4:                                ; %Flow26
	s_or_b64 exec, exec, s[14:15]
.LBB1_5:                                ; %Flow28
	s_or_b64 exec, exec, s[12:13]
.LBB1_6:                                ; %.loopexit7.i.i
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v9, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[12:13], v9, s[8:9] offset:40
	global_load_dwordx4 v[4:7], v9, s[8:9]
	v_readfirstlane_b32 s11, v11
	v_readfirstlane_b32 s10, v10
	s_mov_b64 s[12:13], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s6, v12
	v_readfirstlane_b32 s7, v13
	s_and_b64 s[14:15], s[6:7], s[10:11]
	s_lshl_b64 s[6:7], s[14:15], 12
	v_mov_b32_e32 v11, s7
	s_waitcnt vmcnt(0)
	v_add_co_u32_e64 v10, s[6:7], s6, v6
	v_addc_co_u32_e64 v11, s[6:7], v7, v11, s[6:7]
                                        ; implicit-def: $vgpr6_vgpr7
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB1_8
; %bb.7:
	v_lshlrev_b64 v[6:7], 6, v[8:9]
	s_mov_b32 s16, 0
	v_add_co_u32_e32 v6, vcc, v10, v6
	v_addc_co_u32_e32 v7, vcc, v11, v7, vcc
	s_mov_b32 s17, s16
	global_store_dwordx4 v[6:7], v[0:3], off
	s_mov_b32 s18, s16
	s_mov_b32 s19, s16
	v_mov_b64_e32 v[0:1], s[16:17]
	v_mov_b64_e32 v[2:3], s[18:19]
	global_store_dwordx4 v[6:7], v[0:3], off offset:16
	global_store_dwordx4 v[6:7], v[0:3], off offset:32
	global_store_dwordx4 v[6:7], v[0:3], off offset:48
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr10
                                        ; implicit-def: $vgpr11
                                        ; implicit-def: $vgpr0
.LBB1_8:                                ; %Flow24
	s_or_saveexec_b64 s[6:7], s[6:7]
	s_mul_i32 s16, s15, 24
	s_mul_hi_u32 s17, s14, 24
	s_mul_i32 s18, s14, 24
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB1_16
; %bb.9:
	s_add_i32 s14, s17, s16
	v_mov_b32_e32 v7, s14
	v_add_co_u32_e32 v6, vcc, s18, v4
	v_mov_b64_e32 v[12:13], s[12:13]
	v_mov_b32_e32 v9, 0
	v_addc_co_u32_e32 v7, vcc, v5, v7, vcc
	v_mov_b32_e32 v14, 3
	v_mov_b32_e32 v15, 1
	global_store_dwordx4 v[6:7], v[12:15], off offset:8
	v_lshlrev_b64 v[6:7], 6, v[8:9]
	s_mov_b32 s12, 0
	v_add_co_u32_e32 v6, vcc, v10, v6
	v_addc_co_u32_e32 v7, vcc, v11, v7, vcc
	s_mov_b32 s13, s12
	global_store_dwordx4 v[6:7], v[0:3], off
	s_mov_b32 s14, s12
	s_mov_b32 s15, s12
	v_mov_b64_e32 v[0:1], s[12:13]
	v_mov_b64_e32 v[2:3], s[14:15]
	global_store_dwordx4 v[6:7], v[0:3], off offset:16
	global_store_dwordx4 v[6:7], v[0:3], off offset:32
	global_store_dwordx4 v[6:7], v[0:3], off offset:48
	global_load_dwordx2 v[14:15], v9, s[8:9] offset:32 glc
	s_nop 0
	global_load_dwordx2 v[0:1], v9, s[8:9] offset:40
	v_mov_b32_e32 v12, s10
	v_mov_b32_e32 v13, s11
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v0, s10, v0
	v_and_b32_e32 v1, s11, v1
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v2, v0, 24
	v_mul_lo_u32 v0, v0, 24
	v_add_u32_e32 v1, v2, v1
	v_add_co_u32_e32 v10, vcc, v4, v0
	v_addc_co_u32_e32 v11, vcc, v5, v1, vcc
	global_store_dwordx2 v[10:11], v[14:15], off
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v9, v[12:15], s[8:9] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[14:15]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB1_12
; %bb.10:                               ; %.preheader4.i.i.preheader
	s_mov_b64 s[14:15], 0
.LBB1_11:                               ; %.preheader4.i.i
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[10:11], v[2:3], off
	v_mov_b32_e32 v0, s10
	v_mov_b32_e32 v1, s11
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v9, v[0:3], s[8:9] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_or_b64 s[14:15], vcc, s[14:15]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB1_11
.LBB1_12:                               ; %Flow22
	s_or_b64 exec, exec, s[12:13]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[8:9] offset:16
	s_mov_b64 s[12:13], exec
	v_mbcnt_lo_u32_b32 v2, s12, 0
	v_mbcnt_hi_u32_b32 v2, s13, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB1_14
; %bb.13:
	s_bcnt1_i32_b64 s12, s[12:13]
	v_mov_b32_e32 v2, s12
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8
.LBB1_14:
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB1_16
; %bb.15:
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s12, v0
	s_and_b32 m0, s12, 0xff
	global_store_dwordx2 v[2:3], v[0:1], off
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB1_16:                               ; %Flow25
	s_or_b64 exec, exec, s[6:7]
	s_add_i32 s6, s17, s16
	v_mov_b32_e32 v1, s6
	v_add_co_u32_e32 v0, vcc, s18, v4
	v_addc_co_u32_e32 v1, vcc, v5, v1, vcc
	s_branch .LBB1_20
.LBB1_17:                               ;   in Loop: Header=BB1_20 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v2
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB1_19
; %bb.18:                               ;   in Loop: Header=BB1_20 Depth=1
	s_mov_b64 s[6:7], 0
	s_sleep 1
	s_and_b64 s[6:7], s[6:7], exec
	s_cbranch_scc0 .LBB1_20
	s_branch .LBB1_22
.LBB1_19:                               ;   in Loop: Header=BB1_20 Depth=1
	s_mov_b64 s[6:7], -1
	s_and_b64 s[6:7], s[6:7], exec
	s_cbranch_scc1 .LBB1_22
.LBB1_20:                               ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB1_17
; %bb.21:                               ;   in Loop: Header=BB1_20 Depth=1
	global_load_dword v2, v[0:1], off offset:20 glc
	s_waitcnt vmcnt(0)
	buffer_wbinvl1_vol
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB1_17
.LBB1_22:
	global_load_dwordx4 v[0:3], v[6:7], off
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB1_25
; %bb.23:
	v_mov_b32_e32 v8, 0
	global_load_dwordx2 v[2:3], v8, s[8:9] offset:40
	global_load_dwordx2 v[11:12], v8, s[8:9] offset:24 glc
	global_load_dwordx2 v[4:5], v8, s[8:9]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s12, v2
	v_readfirstlane_b32 s13, v3
	s_add_u32 s14, s12, 1
	s_addc_u32 s15, s13, 0
	s_add_u32 s4, s14, s10
	s_addc_u32 s5, s15, s11
	s_cmp_eq_u64 s[4:5], 0
	s_cselect_b32 s5, s15, s5
	s_cselect_b32 s4, s14, s4
	s_and_b64 s[10:11], s[4:5], s[12:13]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_mov_b32_e32 v2, s11
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v6, vcc, s10, v4
	v_addc_co_u32_e32 v7, vcc, v5, v2, vcc
	v_mov_b32_e32 v9, s4
	global_store_dwordx2 v[6:7], v[11:12], off
	v_mov_b32_e32 v10, s5
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[4:5], v8, v[9:12], s[8:9] offset:24 glc
	s_mov_b64 s[10:11], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[4:5], v[11:12]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB1_25
.LBB1_24:                               ; %.preheader.i.i
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[6:7], v[4:5], off
	v_mov_b32_e32 v2, s4
	v_mov_b32_e32 v3, s5
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v8, v[2:5], s[8:9] offset:24 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[2:3], v[4:5]
	v_mov_b64_e32 v[4:5], v[2:3]
	s_or_b64 s[10:11], vcc, s[10:11]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB1_24
.LBB1_25:                               ; %__ockl_hostcall_preview.exit
	s_or_b64 exec, exec, s[6:7]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	__ockl_devmem_request, .Lfunc_end1-__ockl_devmem_request
	.cfi_endproc
                                        ; -- End function
	.set __ockl_devmem_request.num_vgpr, 16
	.set __ockl_devmem_request.num_agpr, 0
	.set __ockl_devmem_request.numbered_sgpr, 32
	.set __ockl_devmem_request.num_named_barrier, 0
	.set __ockl_devmem_request.private_seg_size, 0
	.set __ockl_devmem_request.uses_vcc, 1
	.set __ockl_devmem_request.uses_flat_scratch, 0
	.set __ockl_devmem_request.has_dyn_sized_stack, 0
	.set __ockl_devmem_request.has_recursion, 0
	.set __ockl_devmem_request.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 1304
; TotalNumSgprs: 36
; NumVgprs: 16
; ScratchSize: 0
; MemoryBound: 0
	.text
	.protected	__ockl_dm_init_v1       ; -- Begin function __ockl_dm_init_v1
	.weak	__ockl_dm_init_v1
	.p2align	2
	.type	__ockl_dm_init_v1,@function
__ockl_dm_init_v1:                      ; @__ockl_dm_init_v1
.Lfunc_begin2:
	.cfi_startproc
; %bb.0:
	.cfi_llvm_def_aspace_cfa 64, 0, 6
	.cfi_llvm_register_pair 16, 62, 32, 63, 32
	.cfi_undefined 2560
	.cfi_undefined 2561
	.cfi_undefined 2564
	.cfi_undefined 2565
	.cfi_undefined 2566
	.cfi_undefined 2567
	.cfi_undefined 2568
	.cfi_undefined 2569
	.cfi_undefined 2570
	.cfi_undefined 2571
	.cfi_undefined 2572
	.cfi_undefined 2573
	.cfi_undefined 2574
	.cfi_undefined 36
	.cfi_undefined 37
	.cfi_undefined 38
	.cfi_undefined 39
	.cfi_undefined 40
	.cfi_undefined 41
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	v_mov_b32_e32 v8, v1
	v_mov_b32_e32 v7, v0
	v_and_b32_e32 v6, 0x3ff, v31
	s_mov_b32 s4, 0
	v_cmp_ne_u32_e32 vcc, 0, v4
	s_and_saveexec_b64 s[8:9], vcc
	s_cbranch_execnz .LBB2_3
; %bb.1:
	s_or_b64 exec, exec, s[8:9]
	v_cmp_eq_u32_e32 vcc, 0, v6
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execnz .LBB2_4
.LBB2_2:
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.LBB2_3:
	v_lshlrev_b32_e32 v0, 4, v6
	v_add_co_u32_e32 v0, vcc, v7, v0
	s_mov_b32 s6, s4
	s_mov_b32 s7, s4
	v_addc_co_u32_e32 v1, vcc, 0, v8, vcc
	s_mov_b32 s5, s4
	v_mov_b64_e32 v[11:12], s[6:7]
	v_mov_b64_e32 v[9:10], s[4:5]
	v_add_co_u32_e32 v13, vcc, 0x1000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x2000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x3000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x4000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x5000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x6000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x7000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x8000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x9000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0xa000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0xb000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0xc000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0xd000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0xe000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0xf000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x10000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x11000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x12000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x13000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x14000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x15000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x16000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x17000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x18000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x19000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x1a000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x1b000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x1c000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x1d000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	v_add_co_u32_e32 v13, vcc, 0x1e000, v0
	v_addc_co_u32_e32 v14, vcc, 0, v1, vcc
	global_store_dwordx4 v[0:1], v[9:12], off
	v_add_co_u32_e32 v0, vcc, 0x1f000, v0
	v_addc_co_u32_e32 v1, vcc, 0, v1, vcc
	global_store_dwordx4 v[13:14], v[9:12], off
	global_store_dwordx4 v[0:1], v[9:12], off
	s_or_b64 exec, exec, s[8:9]
	v_cmp_eq_u32_e32 vcc, 0, v6
	s_and_saveexec_b64 s[4:5], vcc
	s_cbranch_execz .LBB2_2
.LBB2_4:
	v_mov_b32_e32 v6, 0
	s_nop 1
	v_lshlrev_b64 v[0:1], 21, v[5:6]
	v_add_co_u32_e32 v0, vcc, v0, v2
	v_addc_co_u32_e32 v1, vcc, v1, v3, vcc
	v_add_co_u32_e32 v4, vcc, 0x1a000, v7
	v_addc_co_u32_e32 v5, vcc, 0, v8, vcc
	global_store_dwordx2 v[4:5], v[2:3], off offset:2048
	global_store_dwordx4 v[4:5], v[0:3], off offset:2056
	s_or_b64 exec, exec, s[4:5]
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end2:
	.size	__ockl_dm_init_v1, .Lfunc_end2-__ockl_dm_init_v1
	.cfi_endproc
                                        ; -- End function
	.set __ockl_dm_init_v1.num_vgpr, 32
	.set __ockl_dm_init_v1.num_agpr, 0
	.set __ockl_dm_init_v1.numbered_sgpr, 32
	.set __ockl_dm_init_v1.num_named_barrier, 0
	.set __ockl_dm_init_v1.private_seg_size, 0
	.set __ockl_dm_init_v1.uses_vcc, 1
	.set __ockl_dm_init_v1.uses_flat_scratch, 0
	.set __ockl_dm_init_v1.has_dyn_sized_stack, 0
	.set __ockl_dm_init_v1.has_recursion, 0
	.set __ockl_dm_init_v1.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 804
; TotalNumSgprs: 36
; NumVgprs: 32
; ScratchSize: 0
; MemoryBound: 1
	.text
	.protected	__ockl_sanitizer_report ; -- Begin function __ockl_sanitizer_report
	.weak	__ockl_sanitizer_report
	.p2align	2
	.type	__ockl_sanitizer_report,@function
__ockl_sanitizer_report:                ; @__ockl_sanitizer_report
.Lfunc_begin3:
	.cfi_startproc
; %bb.0:
	.cfi_llvm_def_aspace_cfa 64, 0, 6
	.cfi_llvm_register_pair 16, 62, 32, 63, 32
	.cfi_undefined 2560
	.cfi_undefined 2561
	.cfi_undefined 2562
	.cfi_undefined 2563
	.cfi_undefined 2564
	.cfi_undefined 2565
	.cfi_undefined 2566
	.cfi_undefined 2567
	.cfi_undefined 2568
	.cfi_undefined 2569
	.cfi_undefined 2570
	.cfi_undefined 2571
	.cfi_undefined 2572
	.cfi_undefined 2573
	.cfi_undefined 2574
	.cfi_undefined 2575
	.cfi_undefined 2576
	.cfi_undefined 2577
	.cfi_undefined 2578
	.cfi_undefined 2579
	.cfi_undefined 2580
	.cfi_undefined 2581
	.cfi_undefined 2582
	.cfi_undefined 2583
	.cfi_undefined 2584
	.cfi_undefined 2585
	.cfi_undefined 2586
	.cfi_undefined 2587
	.cfi_undefined 36
	.cfi_undefined 37
	.cfi_undefined 38
	.cfi_undefined 39
	.cfi_undefined 40
	.cfi_undefined 41
	.cfi_undefined 42
	.cfi_undefined 43
	.cfi_undefined 44
	.cfi_undefined 45
	.cfi_undefined 46
	.cfi_undefined 47
	.cfi_undefined 48
	.cfi_undefined 49
	.cfi_undefined 50
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_load_dwordx2 s[8:9], s[8:9], 0x18
	v_mbcnt_lo_u32_b32 v16, -1, 0
	v_mbcnt_hi_u32_b32 v20, -1, v16
	; sched_barrier mask(0x00000000)
	;;#ASMSTART
	;;#ASMEND
	; sched_barrier mask(0x00000000)
	v_readfirstlane_b32 s4, v20
	v_mov_b32_e32 v22, 0
	v_cmp_ne_u32_e32 vcc, s4, v20
	v_mov_b32_e32 v23, 0
	v_cmp_eq_u32_e64 s[4:5], s4, v20
	s_and_saveexec_b64 s[10:11], s[4:5]
	s_cbranch_execz .LBB3_6
; %bb.1:
	v_mov_b32_e32 v16, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[24:25], v16, s[8:9] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_wbinvl1_vol
	global_load_dwordx2 v[17:18], v16, s[8:9] offset:40
	global_load_dwordx2 v[21:22], v16, s[8:9]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v17, v17, v24
	v_and_b32_e32 v18, v18, v25
	v_mul_lo_u32 v18, v18, 24
	v_mul_hi_u32 v19, v17, 24
	v_mul_lo_u32 v17, v17, 24
	v_add_u32_e32 v18, v19, v18
	s_waitcnt vmcnt(0)
	v_add_co_u32_e64 v17, s[6:7], v21, v17
	v_addc_co_u32_e64 v18, s[6:7], v22, v18, s[6:7]
	global_load_dwordx2 v[22:23], v[17:18], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v16, v[22:25], s[8:9] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_wbinvl1_vol
	v_cmp_ne_u64_e64 s[6:7], v[22:23], v[24:25]
	s_and_saveexec_b64 s[12:13], s[6:7]
	s_cbranch_execz .LBB3_5
; %bb.2:                                ; %.preheader6.i.i.preheader
	s_mov_b64 s[14:15], 0
.LBB3_3:                                ; %.preheader6.i.i
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_load_dwordx2 v[17:18], v16, s[8:9] offset:40
	global_load_dwordx2 v[26:27], v16, s[8:9]
	v_mov_b64_e32 v[24:25], v[22:23]
	s_waitcnt vmcnt(1)
	v_and_b32_e32 v17, v17, v24
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[21:22], s[6:7], v17, 24, v[26:27]
	v_and_b32_e32 v18, v18, v25
	v_mov_b32_e32 v17, v22
	s_nop 1
	v_mad_u64_u32 v[17:18], s[6:7], v18, 24, v[17:18]
	v_mov_b32_e32 v22, v17
	global_load_dwordx2 v[22:23], v[21:22], off glc
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[22:23], v16, v[22:25], s[8:9] offset:24 glc
	s_waitcnt vmcnt(0)
	buffer_wbinvl1_vol
	v_cmp_eq_u64_e64 s[6:7], v[22:23], v[24:25]
	s_or_b64 s[14:15], s[6:7], s[14:15]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB3_3
; %bb.4:                                ; %Flow26
	s_or_b64 exec, exec, s[14:15]
.LBB3_5:                                ; %Flow28
	s_or_b64 exec, exec, s[12:13]
.LBB3_6:                                ; %.loopexit7.i.i
	s_or_b64 exec, exec, s[10:11]
	v_mov_b32_e32 v21, 0
	s_waitcnt lgkmcnt(0)
	global_load_dwordx2 v[24:25], v21, s[8:9] offset:40
	global_load_dwordx4 v[16:19], v21, s[8:9]
	v_readfirstlane_b32 s11, v23
	v_readfirstlane_b32 s10, v22
	s_mov_b64 s[12:13], exec
	s_waitcnt vmcnt(1)
	v_readfirstlane_b32 s6, v24
	v_readfirstlane_b32 s7, v25
	s_and_b64 s[14:15], s[6:7], s[10:11]
	s_lshl_b64 s[6:7], s[14:15], 12
	v_mov_b32_e32 v22, s7
	s_waitcnt vmcnt(0)
	v_add_co_u32_e64 v18, s[6:7], s6, v18
	v_addc_co_u32_e64 v19, s[6:7], v19, v22, s[6:7]
	s_and_saveexec_b64 s[6:7], vcc
	s_xor_b64 s[6:7], exec, s[6:7]
	s_cbranch_execz .LBB3_8
; %bb.7:
	v_lshlrev_b64 v[20:21], 6, v[20:21]
	v_add_co_u32_e32 v18, vcc, v18, v20
	v_addc_co_u32_e32 v19, vcc, v19, v21, vcc
	global_store_dwordx4 v[18:19], v[0:3], off
	global_store_dwordx4 v[18:19], v[4:7], off offset:16
	global_store_dwordx4 v[18:19], v[8:11], off offset:32
	global_store_dwordx4 v[18:19], v[12:15], off offset:48
                                        ; implicit-def: $vgpr0
                                        ; implicit-def: $vgpr4
                                        ; implicit-def: $vgpr8
                                        ; implicit-def: $vgpr12
                                        ; implicit-def: $vgpr20
                                        ; implicit-def: $vgpr18
                                        ; implicit-def: $vgpr19
.LBB3_8:                                ; %Flow24
	s_or_saveexec_b64 s[6:7], s[6:7]
	s_mul_i32 s16, s15, 24
	s_mul_hi_u32 s17, s14, 24
	s_mul_i32 s18, s14, 24
	s_xor_b64 exec, exec, s[6:7]
	s_cbranch_execz .LBB3_16
; %bb.9:
	s_add_i32 s14, s17, s16
	v_mov_b32_e32 v21, s14
	v_add_co_u32_e32 v25, vcc, s18, v16
	v_addc_co_u32_e32 v26, vcc, v17, v21, vcc
	v_mov_b64_e32 v[21:22], s[12:13]
	v_mov_b32_e32 v23, 4
	v_mov_b32_e32 v24, 1
	global_store_dwordx4 v[25:26], v[21:24], off offset:8
	s_nop 1
	v_mov_b32_e32 v21, 0
	s_nop 1
	v_lshlrev_b64 v[22:23], 6, v[20:21]
	v_add_co_u32_e32 v18, vcc, v18, v22
	v_addc_co_u32_e32 v19, vcc, v19, v23, vcc
	global_store_dwordx4 v[18:19], v[0:3], off
	global_store_dwordx4 v[18:19], v[4:7], off offset:16
	global_store_dwordx4 v[18:19], v[8:11], off offset:32
	global_store_dwordx4 v[18:19], v[12:15], off offset:48
	global_load_dwordx2 v[8:9], v21, s[8:9] offset:32 glc
	s_nop 0
	global_load_dwordx2 v[0:1], v21, s[8:9] offset:40
	v_mov_b32_e32 v6, s10
	v_mov_b32_e32 v7, s11
	s_waitcnt vmcnt(0)
	v_and_b32_e32 v0, s10, v0
	v_and_b32_e32 v1, s11, v1
	v_mul_lo_u32 v1, v1, 24
	v_mul_hi_u32 v2, v0, 24
	v_mul_lo_u32 v0, v0, 24
	v_add_u32_e32 v1, v2, v1
	v_add_co_u32_e32 v4, vcc, v16, v0
	v_addc_co_u32_e32 v5, vcc, v17, v1, vcc
	global_store_dwordx2 v[4:5], v[8:9], off
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v21, v[6:9], s[8:9] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[8:9]
	s_and_saveexec_b64 s[12:13], vcc
	s_cbranch_execz .LBB3_12
; %bb.10:                               ; %.preheader4.i.i.preheader
	s_mov_b64 s[14:15], 0
.LBB3_11:                               ; %.preheader4.i.i
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s10
	v_mov_b32_e32 v1, s11
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v21, v[0:3], s[8:9] offset:32 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_or_b64 s[14:15], vcc, s[14:15]
	s_andn2_b64 exec, exec, s[14:15]
	s_cbranch_execnz .LBB3_11
.LBB3_12:                               ; %Flow22
	s_or_b64 exec, exec, s[12:13]
	v_mov_b32_e32 v3, 0
	global_load_dwordx2 v[0:1], v3, s[8:9] offset:16
	s_mov_b64 s[12:13], exec
	v_mbcnt_lo_u32_b32 v2, s12, 0
	v_mbcnt_hi_u32_b32 v2, s13, v2
	v_cmp_eq_u32_e32 vcc, 0, v2
	s_and_saveexec_b64 s[14:15], vcc
	s_cbranch_execz .LBB3_14
; %bb.13:
	s_bcnt1_i32_b64 s12, s[12:13]
	v_mov_b32_e32 v2, s12
	s_waitcnt vmcnt(0)
	global_atomic_add_x2 v[0:1], v[2:3], off offset:8
.LBB3_14:
	s_or_b64 exec, exec, s[14:15]
	s_waitcnt vmcnt(0)
	global_load_dwordx2 v[2:3], v[0:1], off offset:16
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, 0, v[2:3]
	s_cbranch_vccnz .LBB3_16
; %bb.15:
	global_load_dword v0, v[0:1], off offset:24
	v_mov_b32_e32 v1, 0
	s_waitcnt vmcnt(0)
	v_readfirstlane_b32 s12, v0
	s_and_b32 m0, s12, 0xff
	global_store_dwordx2 v[2:3], v[0:1], off
	s_sendmsg sendmsg(MSG_INTERRUPT)
.LBB3_16:                               ; %Flow25
	s_or_b64 exec, exec, s[6:7]
	s_add_i32 s6, s17, s16
	v_mov_b32_e32 v1, s6
	v_add_co_u32_e32 v0, vcc, s18, v16
	v_addc_co_u32_e32 v1, vcc, v17, v1, vcc
	s_branch .LBB3_20
.LBB3_17:                               ;   in Loop: Header=BB3_20 Depth=1
	s_or_b64 exec, exec, s[6:7]
	v_readfirstlane_b32 s6, v2
	s_cmp_eq_u32 s6, 0
	s_cbranch_scc1 .LBB3_19
; %bb.18:                               ;   in Loop: Header=BB3_20 Depth=1
	s_mov_b64 s[6:7], 0
	s_sleep 1
	s_and_b64 s[6:7], s[6:7], exec
	s_cbranch_scc0 .LBB3_20
	s_branch .LBB3_22
.LBB3_19:                               ;   in Loop: Header=BB3_20 Depth=1
	s_mov_b64 s[6:7], -1
	s_and_b64 s[6:7], s[6:7], exec
	s_cbranch_scc1 .LBB3_22
.LBB3_20:                               ; =>This Inner Loop Header: Depth=1
	v_mov_b32_e32 v2, 1
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB3_17
; %bb.21:                               ;   in Loop: Header=BB3_20 Depth=1
	global_load_dword v2, v[0:1], off offset:20 glc
	s_waitcnt vmcnt(0)
	buffer_wbinvl1_vol
	v_and_b32_e32 v2, 1, v2
	s_branch .LBB3_17
.LBB3_22:
	s_and_saveexec_b64 s[6:7], s[4:5]
	s_cbranch_execz .LBB3_25
; %bb.23:
	v_mov_b32_e32 v6, 0
	global_load_dwordx2 v[0:1], v6, s[8:9] offset:40
	global_load_dwordx2 v[9:10], v6, s[8:9] offset:24 glc
	global_load_dwordx2 v[2:3], v6, s[8:9]
	s_waitcnt vmcnt(2)
	v_readfirstlane_b32 s12, v0
	v_readfirstlane_b32 s13, v1
	s_add_u32 s14, s12, 1
	s_addc_u32 s15, s13, 0
	s_add_u32 s4, s14, s10
	s_addc_u32 s5, s15, s11
	s_cmp_eq_u64 s[4:5], 0
	s_cselect_b32 s5, s15, s5
	s_cselect_b32 s4, s14, s4
	s_and_b64 s[10:11], s[4:5], s[12:13]
	s_mul_i32 s11, s11, 24
	s_mul_hi_u32 s12, s10, 24
	s_mul_i32 s10, s10, 24
	s_add_i32 s11, s12, s11
	v_mov_b32_e32 v0, s11
	s_waitcnt vmcnt(0)
	v_add_co_u32_e32 v4, vcc, s10, v2
	v_addc_co_u32_e32 v5, vcc, v3, v0, vcc
	v_mov_b32_e32 v7, s4
	global_store_dwordx2 v[4:5], v[9:10], off
	v_mov_b32_e32 v8, s5
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[2:3], v6, v[7:10], s[8:9] offset:24 glc
	s_mov_b64 s[10:11], 0
	s_waitcnt vmcnt(0)
	v_cmp_ne_u64_e32 vcc, v[2:3], v[9:10]
	s_and_b64 exec, exec, vcc
	s_cbranch_execz .LBB3_25
.LBB3_24:                               ; %.preheader.i.i
                                        ; =>This Inner Loop Header: Depth=1
	s_sleep 1
	global_store_dwordx2 v[4:5], v[2:3], off
	v_mov_b32_e32 v0, s4
	v_mov_b32_e32 v1, s5
	s_waitcnt vmcnt(0)
	global_atomic_cmpswap_x2 v[0:1], v6, v[0:3], s[8:9] offset:24 glc
	s_waitcnt vmcnt(0)
	v_cmp_eq_u64_e32 vcc, v[0:1], v[2:3]
	v_mov_b64_e32 v[2:3], v[0:1]
	s_or_b64 s[10:11], vcc, s[10:11]
	s_andn2_b64 exec, exec, s[10:11]
	s_cbranch_execnz .LBB3_24
.LBB3_25:                               ; %__ockl_hostcall_preview.exit
	s_or_b64 exec, exec, s[6:7]
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end3:
	.size	__ockl_sanitizer_report, .Lfunc_end3-__ockl_sanitizer_report
	.cfi_endproc
                                        ; -- End function
	.set __ockl_sanitizer_report.num_vgpr, 28
	.set __ockl_sanitizer_report.num_agpr, 0
	.set __ockl_sanitizer_report.numbered_sgpr, 32
	.set __ockl_sanitizer_report.num_named_barrier, 0
	.set __ockl_sanitizer_report.private_seg_size, 0
	.set __ockl_sanitizer_report.uses_vcc, 1
	.set __ockl_sanitizer_report.uses_flat_scratch, 0
	.set __ockl_sanitizer_report.has_dyn_sized_stack, 0
	.set __ockl_sanitizer_report.has_recursion, 0
	.set __ockl_sanitizer_report.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 1256
; TotalNumSgprs: 36
; NumVgprs: 28
; ScratchSize: 0
; MemoryBound: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 32
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 32
	.section	.AMDGPU.csdata,"",@progbits
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
	.asciz	"DeepGenGraph MLIR"             ; string offset=0
.Linfo_string1:
	.asciz	"test_input_spidD.mlir"         ; string offset=18
.Linfo_string2:
	.asciz	"/data2/xsl/DeepGenGraph/3rd/deepgengraph/test" ; string offset=40
	.ident	"clang version 17.0.0"
	.section	".note.GNU-stack","",@progbits
	.amdgpu_metadata
---
amdhsa.kernels:
  - .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
    .group_segment_fixed_size: 16384
    .kernarg_segment_align: 8
    .kernarg_segment_size: 32
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           Attn_p2
    .private_segment_fixed_size: 0
    .reqd_workgroup_size:
      - 256
      - 1
      - 1
    .sgpr_count:     79
    .sgpr_spill_count: 0
    .symbol:         Attn_p2.kd
    .unfolded_args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         8
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         16
        .size:           8
        .value_kind:     global_buffer
      - .address_space:  global
        .offset:         24
        .size:           8
        .value_kind:     global_buffer
    .uses_dynamic_stack: false
    .vgpr_count:     248
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx936
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
	.section	.debug_line,"",@progbits
.Lline_table_start0:
root@dcu1:/workspace/deepgengraph# 

[END] 2026/10/14 18:33:32
