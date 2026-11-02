v_mov_b32_e32 v171, v0
v_lshrrev_b32_e32 v172, 2, v171
v_and_b32_e32 v66, 0x7b, v171
v_and_b32_e32 v67, 4, v172
v_add_lshl_u32 v67, v67, v171, 7
v_lshlrev_b32_e32 v66, 7, v66
v_add_co_u32_e32 v69, vcc, s23, v171
v_and_or_b32 v66, v67, s10, v66
v_lshrrev_b32_e32 v67, 4, v171
v_addc_co_u32_e64 v70, s[12:13], 0, 0, vcc
v_lshlrev_b32_e32 v173, 4, v171
v_and_b32_e32 v67, 0xffffff8, v67
v_alignbit_b32 v69, v70, v69, 7
v_add_u32_e32 v67, v66, v67
s_add_i32 s29, s25, s8
v_or_b32_e32 v68, 0x1000, v173
v_lshl_add_u32 v66, v69, 3, v66
v_and_b32_e32 v65, 0xc00, v173
v_add_lshl_u32 v175, s29, v67, 1
v_and_b32_e32 v68, 0x1c00, v68
v_add_lshl_u32 v176, s29, v66, 1
v_add_u32_e32 v67, 0x100000, v175
v_add_u32_e32 v66, 0x100000, v176
v_or_b32_e32 v69, 0x2000, v65
v_add_u32_e32 v70, 0x100040, v175
v_or_b32_e32 v71, 0x2000, v68
v_readfirstlane_b32 s12, v65
s_mov_b32 m0, s12
buffer_load_dwordx4 v67, s[16:19], 0 offen lds
v_readfirstlane_b32 s13, v68
s_mov_b32 m0, s13
buffer_load_dwordx4 v66, s[16:19], 0 offen lds
s_waitcnt vmcnt(0)
s_barrier
v_readfirstlane_b32 s30, v69
s_mov_b32 m0, s30
buffer_load_dwordx4 v70, s[16:19], 0 offen lds
v_readfirstlane_b32 s31, v71
v_add_u32_e32 v65, 0x100040, v176
s_mov_b32 m0, s31
buffer_load_dwordx4 v65, s[16:19], 0 offen lds
v_mul_lo_u32 v65, v172, s11
v_lshlrev_b32_e32 v66, 30, v171
v_lshlrev_b32_e32 v68, 2, v171
v_ashrrev_i32_e32 v66, 25, v66
v_mad_u32_u24 v65, v171, s26, v65
v_and_b32_e32 v75, 48, v68
v_or_b32_e32 v65, v65, v75
v_and_b32_e32 v66, 0x7fffffc0, v66
v_bfe_u32 v67, v171, 5, 1
v_lshl_add_u32 v65, v66, 1, v65
v_lshrrev_b32_e32 v66, 1, v171
v_and_b32_e32 v76, 24, v66
v_mul_u32_u24_e32 v66, 0x3f8, v67
v_lshlrev_b32_e32 v77, 1, v66
v_add3_u32 v211, v65, v76, v77
ds_read2st64_b64 v[105:108], v211 offset1:8
v_mov_b64_e32 v[67:68], s[2:3]
v_mov_b64_e32 v[65:66], s[0:1]
s_nop 1
v_mov_b64_e32 v[71:72], v[67:68]
v_mov_b64_e32 v[69:70], v[65:66]
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[69:72], v[129:130], v[105:106], v[69:72]
v_mov_b64_e32 v[73:74], v[161:162]
v_bfe_i32 v74, v171, 1, 1
v_sub_u32_e32 v73, v74, v73
v_lshlrev_b32_e32 v74, 9, v171
v_lshlrev_b32_e32 v174, 6, v171
v_and_b32_e32 v74, 0x600, v74
v_and_b32_e32 v78, 0xc0, v174
v_or3_b32 v74, v74, v78, v75
v_lshl_add_u32 v73, v73, 7, v74
v_add3_u32 v212, v73, v76, v77
v_add_u32_e32 v213, 0x1000, v212
ds_read2_b64 v[115:118], v212 offset0:8 offset1:24
ds_read2_b64 v[119:122], v213 offset0:8 offset1:24
v_mmac_f32_16x16x16_f16 v[69:72], v[131:132], v[107:108], v[69:72]
ds_read2_b64 v[123:126], v211 offset0:16 offset1:128
v_mov_b64_e32 v[75:76], v[67:68]
v_mov_b64_e32 v[73:74], v[65:66]
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_f16 v[73:76], v[129:130], v[115:116], v[73:76]
v_add_u32_e32 v77, 0x80, v211
ds_read2st64_b64 v[167:170], v77 offset0:2 offset1:8
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_f16 v[73:76], v[131:132], v[119:120], v[73:76]
v_mov_b64_e32 v[79:80], v[67:68]
v_mov_b64_e32 v[77:78], v[65:66]
s_waitcnt lgkmcnt(1)
v_mmac_f32_16x16x16_f16 v[77:80], v[129:130], v[123:124], v[77:80]
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[77:80], v[131:132], v[169:170], v[77:80]
v_mov_b64_e32 v[83:84], v[67:68]
v_mov_b64_e32 v[81:82], v[65:66]
v_mmac_f32_16x16x16_f16 v[81:84], v[129:130], v[117:118], v[81:84]
v_add_u32_e32 v214, 0x1000, v211
ds_read2_b64 v[179:182], v214 offset0:128 offset1:144
v_mmac_f32_16x16x16_f16 v[81:84], v[131:132], v[121:122], v[81:84]
ds_read2_b64 v[183:186], v212 offset0:136 offset1:152
v_mov_b64_e32 v[87:88], v[67:68]
v_mov_b64_e32 v[85:86], v[65:66]
v_mmac_f32_16x16x16_f16 v[85:88], v[129:130], v[125:126], v[85:88]
ds_read2_b64 v[187:190], v213 offset0:136 offset1:152
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_f16 v[85:88], v[131:132], v[179:180], v[85:88]
v_mov_b64_e32 v[91:92], v[67:68]
v_mov_b64_e32 v[89:90], v[65:66]
s_waitcnt lgkmcnt(1)
v_mmac_f32_16x16x16_f16 v[89:92], v[129:130], v[183:184], v[89:92]
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[89:92], v[131:132], v[187:188], v[89:92]
v_mov_b64_e32 v[95:96], v[67:68]
v_mov_b64_e32 v[93:94], v[65:66]
v_mmac_f32_16x16x16_f16 v[93:96], v[129:130], v[167:168], v[93:96]
v_mmac_f32_16x16x16_f16 v[93:96], v[131:132], v[181:182], v[93:96]
v_mov_b64_e32 v[99:100], v[67:68]
v_mov_b64_e32 v[97:98], v[65:66]
v_mmac_f32_16x16x16_f16 v[97:100], v[129:130], v[185:186], v[97:100]
v_mmac_f32_16x16x16_f16 v[97:100], v[131:132], v[189:190], v[97:100]
v_mov_b64_e32 v[103:104], v[67:68]
v_mov_b64_e32 v[101:102], v[65:66]
v_mmac_f32_16x16x16_f16 v[101:104], v[133:134], v[105:106], v[101:104]
v_mmac_f32_16x16x16_f16 v[101:104], v[135:136], v[107:108], v[101:104]
v_mov_b64_e32 v[107:108], v[67:68]
v_mov_b64_e32 v[105:106], v[65:66]
v_mmac_f32_16x16x16_f16 v[105:108], v[133:134], v[115:116], v[105:108]
v_mmac_f32_16x16x16_f16 v[105:108], v[135:136], v[119:120], v[105:108]
v_mov_b64_e32 v[111:112], v[67:68]
v_mov_b64_e32 v[109:110], v[65:66]
v_mmac_f32_16x16x16_f16 v[109:112], v[133:134], v[123:124], v[109:112]
v_mmac_f32_16x16x16_f16 v[109:112], v[135:136], v[169:170], v[109:112]
v_mov_b64_e32 v[115:116], v[67:68]
v_mov_b64_e32 v[113:114], v[65:66]
v_mmac_f32_16x16x16_f16 v[113:116], v[133:134], v[117:118], v[113:116]
v_mmac_f32_16x16x16_f16 v[113:116], v[135:136], v[121:122], v[113:116]
v_mov_b64_e32 v[119:120], v[67:68]
v_mov_b64_e32 v[117:118], v[65:66]
v_mmac_f32_16x16x16_f16 v[117:120], v[133:134], v[125:126], v[117:120]
v_mmac_f32_16x16x16_f16 v[117:120], v[135:136], v[179:180], v[117:120]
v_mov_b64_e32 v[123:124], v[67:68]
v_mov_b64_e32 v[121:122], v[65:66]
v_mmac_f32_16x16x16_f16 v[121:124], v[133:134], v[183:184], v[121:124]
v_mmac_f32_16x16x16_f16 v[121:124], v[135:136], v[187:188], v[121:124]
v_mov_b64_e32 v[127:128], v[67:68]
v_mov_b64_e32 v[125:126], v[65:66]
v_mmac_f32_16x16x16_f16 v[125:128], v[133:134], v[167:168], v[125:128]
v_mmac_f32_16x16x16_f16 v[125:128], v[135:136], v[181:182], v[125:128]
v_mmac_f32_16x16x16_f16 v[65:68], v[133:134], v[185:186], v[65:68]
v_mmac_f32_16x16x16_f16 v[65:68], v[135:136], v[189:190], v[65:68]
s_waitcnt vmcnt(0)
v_add_u32_e32 v167, 0x100080, v175
s_barrier
s_mov_b32 m0, s12
buffer_load_dwordx4 v167, s[16:19], 0 offen lds
v_add_u32_e32 v167, 0x100080, v176
s_mov_b32 m0, s13
buffer_load_dwordx4 v167, s[16:19], 0 offen lds
v_add_u32_e32 v170, 0x2000, v211
v_add_u32_e32 v169, 0x3000, v211
v_add_u32_e32 v167, 0x2000, v212
ds_read2_b64 v[179:182], v170 offset1:16
ds_read2_b64 v[183:186], v169 offset1:16
ds_read2_b64 v[187:190], v167 offset0:8 offset1:24
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_f16 v[69:72], v[137:138], v[179:180], v[69:72]
v_add_u32_e32 v168, 0x3000, v212
ds_read2_b64 v[191:194], v168 offset0:8 offset1:24
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_f16 v[69:72], v[139:140], v[183:184], v[69:72]
s_waitcnt lgkmcnt(1)
v_mmac_f32_16x16x16_f16 v[73:76], v[137:138], v[187:188], v[73:76]
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[73:76], v[139:140], v[191:192], v[73:76]
v_mmac_f32_16x16x16_f16 v[77:80], v[137:138], v[181:182], v[77:80]
v_mmac_f32_16x16x16_f16 v[77:80], v[139:140], v[185:186], v[77:80]
ds_read2_b64 v[195:198], v170 offset0:128 offset1:144
v_mmac_f32_16x16x16_f16 v[81:84], v[137:138], v[189:190], v[81:84]
ds_read2_b64 v[199:202], v169 offset0:128 offset1:144
v_mmac_f32_16x16x16_f16 v[81:84], v[139:140], v[193:194], v[81:84]
ds_read2_b64 v[203:206], v167 offset0:136 offset1:152
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_f16 v[85:88], v[137:138], v[195:196], v[85:88]
ds_read2_b64 v[207:210], v168 offset0:136 offset1:152
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_f16 v[85:88], v[139:140], v[199:200], v[85:88]
s_waitcnt lgkmcnt(1)
v_mmac_f32_16x16x16_f16 v[89:92], v[137:138], v[203:204], v[89:92]
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[89:92], v[139:140], v[207:208], v[89:92]
v_mmac_f32_16x16x16_f16 v[93:96], v[137:138], v[197:198], v[93:96]
v_mmac_f32_16x16x16_f16 v[93:96], v[139:140], v[201:202], v[93:96]
v_mmac_f32_16x16x16_f16 v[97:100], v[137:138], v[205:206], v[97:100]
v_mmac_f32_16x16x16_f16 v[97:100], v[139:140], v[209:210], v[97:100]
v_mmac_f32_16x16x16_f16 v[101:104], v[141:142], v[179:180], v[101:104]
v_mmac_f32_16x16x16_f16 v[101:104], v[143:144], v[183:184], v[101:104]
v_mmac_f32_16x16x16_f16 v[105:108], v[141:142], v[187:188], v[105:108]
v_mmac_f32_16x16x16_f16 v[105:108], v[143:144], v[191:192], v[105:108]
v_mmac_f32_16x16x16_f16 v[109:112], v[141:142], v[181:182], v[109:112]
v_mmac_f32_16x16x16_f16 v[109:112], v[143:144], v[185:186], v[109:112]
v_mmac_f32_16x16x16_f16 v[113:116], v[141:142], v[189:190], v[113:116]
v_mmac_f32_16x16x16_f16 v[113:116], v[143:144], v[193:194], v[113:116]
v_mmac_f32_16x16x16_f16 v[117:120], v[141:142], v[195:196], v[117:120]
v_mmac_f32_16x16x16_f16 v[117:120], v[143:144], v[199:200], v[117:120]
v_mmac_f32_16x16x16_f16 v[121:124], v[141:142], v[203:204], v[121:124]
v_mmac_f32_16x16x16_f16 v[121:124], v[143:144], v[207:208], v[121:124]
v_mmac_f32_16x16x16_f16 v[125:128], v[141:142], v[197:198], v[125:128]
v_mmac_f32_16x16x16_f16 v[125:128], v[143:144], v[201:202], v[125:128]
v_mmac_f32_16x16x16_f16 v[65:68], v[141:142], v[205:206], v[65:68]
v_mmac_f32_16x16x16_f16 v[65:68], v[143:144], v[209:210], v[65:68]
s_waitcnt vmcnt(0)
s_barrier
v_add_u32_e32 v175, 0x1000c0, v175
s_mov_b32 m0, s30
buffer_load_dwordx4 v175, s[16:19], 0 offen lds
v_add_u32_e32 v175, 0x1000c0, v176
s_mov_b32 m0, s31
buffer_load_dwordx4 v175, s[16:19], 0 offen lds
ds_read2_b64 v[179:182], v211 offset1:16
ds_read2_b64 v[183:186], v214 offset1:16
ds_read2_b64 v[187:190], v212 offset0:8 offset1:24
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_f16 v[69:72], v[145:146], v[179:180], v[69:72]
ds_read2_b64 v[191:194], v213 offset0:8 offset1:24
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_f16 v[69:72], v[147:148], v[183:184], v[69:72]
s_waitcnt lgkmcnt(1)
v_mmac_f32_16x16x16_f16 v[73:76], v[145:146], v[187:188], v[73:76]
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[73:76], v[147:148], v[191:192], v[73:76]
v_mmac_f32_16x16x16_f16 v[77:80], v[145:146], v[181:182], v[77:80]
v_mmac_f32_16x16x16_f16 v[77:80], v[147:148], v[185:186], v[77:80]
ds_read2_b64 v[195:198], v211 offset0:128 offset1:144
v_mmac_f32_16x16x16_f16 v[81:84], v[145:146], v[189:190], v[81:84]
ds_read2_b64 v[199:202], v214 offset0:128 offset1:144
v_mmac_f32_16x16x16_f16 v[81:84], v[147:148], v[193:194], v[81:84]
ds_read2_b64 v[203:206], v212 offset0:136 offset1:152
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_f16 v[85:88], v[145:146], v[195:196], v[85:88]
ds_read2_b64 v[207:210], v213 offset0:136 offset1:152
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_f16 v[85:88], v[147:148], v[199:200], v[85:88]
s_waitcnt lgkmcnt(1)
v_mmac_f32_16x16x16_f16 v[89:92], v[145:146], v[203:204], v[89:92]
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[89:92], v[147:148], v[207:208], v[89:92]
v_mmac_f32_16x16x16_f16 v[93:96], v[145:146], v[197:198], v[93:96]
v_mmac_f32_16x16x16_f16 v[93:96], v[147:148], v[201:202], v[93:96]
v_mmac_f32_16x16x16_f16 v[97:100], v[145:146], v[205:206], v[97:100]
v_mmac_f32_16x16x16_f16 v[97:100], v[147:148], v[209:210], v[97:100]
v_mmac_f32_16x16x16_f16 v[101:104], v[149:150], v[179:180], v[101:104]
v_mmac_f32_16x16x16_f16 v[101:104], v[151:152], v[183:184], v[101:104]
v_mmac_f32_16x16x16_f16 v[105:108], v[149:150], v[187:188], v[105:108]
v_mmac_f32_16x16x16_f16 v[105:108], v[151:152], v[191:192], v[105:108]
v_mmac_f32_16x16x16_f16 v[109:112], v[149:150], v[181:182], v[109:112]
v_mmac_f32_16x16x16_f16 v[109:112], v[151:152], v[185:186], v[109:112]
v_mmac_f32_16x16x16_f16 v[113:116], v[149:150], v[189:190], v[113:116]
v_mmac_f32_16x16x16_f16 v[113:116], v[151:152], v[193:194], v[113:116]
v_mmac_f32_16x16x16_f16 v[117:120], v[149:150], v[195:196], v[117:120]
v_mmac_f32_16x16x16_f16 v[117:120], v[151:152], v[199:200], v[117:120]
v_mmac_f32_16x16x16_f16 v[121:124], v[149:150], v[203:204], v[121:124]
v_mmac_f32_16x16x16_f16 v[121:124], v[151:152], v[207:208], v[121:124]
v_mmac_f32_16x16x16_f16 v[125:128], v[149:150], v[197:198], v[125:128]
v_mmac_f32_16x16x16_f16 v[125:128], v[151:152], v[201:202], v[125:128]
v_mmac_f32_16x16x16_f16 v[65:68], v[149:150], v[205:206], v[65:68]
v_mmac_f32_16x16x16_f16 v[65:68], v[151:152], v[209:210], v[65:68]
s_waitcnt vmcnt(0)
v_lshlrev_b32_e32 v175, 3, v171
v_and_b32_e32 v176, 24, v175
v_lshl_or_b32 v175, v172, 7, v176
v_add_lshl_u32 v175, s29, v175, 1
s_barrier
v_add_u32_e32 v179, 0x100000, v175
s_mov_b32 m0, s12
buffer_load_dwordx4 v179, s[4:7], 0 offen lds
v_lshl_add_u32 v179, v171, 5, v164
v_and_or_b32 v176, v179, s27, v176
v_add_lshl_u32 v176, s29, v176, 1
v_add_u32_e32 v179, 0x100000, v176
s_mov_b32 m0, s13
buffer_load_dwordx4 v179, s[4:7], 0 offen lds
ds_read2_b64 v[179:182], v170 offset1:16
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[69:72], v[153:154], v[179:180], v[69:72]
ds_read2_b64 v[183:186], v169 offset1:16
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[69:72], v[155:156], v[183:184], v[69:72]
ds_read2_b64 v[187:190], v167 offset0:8 offset1:24
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[73:76], v[153:154], v[187:188], v[73:76]
ds_read2_b64 v[191:194], v168 offset0:8 offset1:24
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[73:76], v[155:156], v[191:192], v[73:76]
v_mmac_f32_16x16x16_f16 v[77:80], v[153:154], v[181:182], v[77:80]
v_max_f32_e32 v195, v74, v74
v_max_f32_e32 v196, v70, v70
v_max_f32_e32 v204, v196, v195
v_max_f32_e32 v195, v76, v76
v_max_f32_e32 v196, v72, v72
v_max3_f32 v203, v69, s28, v73
v_max_f32_e32 v205, v196, v195
v_mmac_f32_16x16x16_f16 v[77:80], v[155:156], v[185:186], v[77:80]
v_mmac_f32_16x16x16_f16 v[81:84], v[153:154], v[189:190], v[81:84]
v_max3_f32 v207, v71, v75, v79
v_mmac_f32_16x16x16_f16 v[81:84], v[155:156], v[193:194], v[81:84]
ds_read2_b64 v[195:198], v170 offset0:128 offset1:144
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[85:88], v[153:154], v[195:196], v[85:88]
ds_read2_b64 v[199:202], v169 offset0:128 offset1:144
v_max3_f32 v208, v203, v77, v81
v_max3_f32 v209, v204, v78, v82
v_max3_f32 v210, v205, v80, v84
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[85:88], v[155:156], v[199:200], v[85:88]
ds_read2_b64 v[203:206], v167 offset0:136 offset1:152
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[89:92], v[153:154], v[203:204], v[89:92]
ds_read2_b64 v[167:170], v168 offset0:136 offset1:152
v_max3_f32 v207, v207, v83, v87
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[89:92], v[155:156], v[167:168], v[89:92]
v_mmac_f32_16x16x16_f16 v[93:96], v[153:154], v[197:198], v[93:96]
v_max3_f32 v208, v208, v85, v89
v_max3_f32 v209, v209, v86, v90
v_max3_f32 v210, v210, v88, v92
v_mmac_f32_16x16x16_f16 v[93:96], v[155:156], v[201:202], v[93:96]
v_mmac_f32_16x16x16_f16 v[97:100], v[153:154], v[205:206], v[97:100]
v_max3_f32 v207, v207, v91, v95
v_mmac_f32_16x16x16_f16 v[97:100], v[155:156], v[169:170], v[97:100]
v_mmac_f32_16x16x16_f16 v[101:104], v[157:158], v[179:180], v[101:104]
v_max3_f32 v179, v208, v93, v97
v_max3_f32 v208, v210, v96, v100
v_max3_f32 v180, v209, v94, v98
v_max3_f32 v207, v207, v99, v208
v_max3_f32 v179, v179, v180, v207
v_mmac_f32_16x16x16_f16 v[101:104], v[159:160], v[183:184], v[101:104]
v_mmac_f32_16x16x16_f16 v[105:108], v[157:158], v[187:188], v[105:108]
v_mmac_f32_16x16x16_f16 v[105:108], v[159:160], v[191:192], v[105:108]
v_mmac_f32_16x16x16_f16 v[109:112], v[157:158], v[181:182], v[109:112]
v_mmac_f32_16x16x16_f16 v[109:112], v[159:160], v[185:186], v[109:112]
v_mmac_f32_16x16x16_f16 v[113:116], v[157:158], v[189:190], v[113:116]
v_mmac_f32_16x16x16_f16 v[113:116], v[159:160], v[193:194], v[113:116]
v_mmac_f32_16x16x16_f16 v[117:120], v[157:158], v[195:196], v[117:120]
v_mmac_f32_16x16x16_f16 v[117:120], v[159:160], v[199:200], v[117:120]
v_mmac_f32_16x16x16_f16 v[121:124], v[157:158], v[203:204], v[121:124]
v_mmac_f32_16x16x16_f16 v[121:124], v[159:160], v[167:168], v[121:124]
v_mmac_f32_16x16x16_f16 v[125:128], v[157:158], v[197:198], v[125:128]
v_mmac_f32_16x16x16_f16 v[125:128], v[159:160], v[201:202], v[125:128]
v_mmac_f32_16x16x16_f16 v[65:68], v[157:158], v[205:206], v[65:68]
v_mmac_f32_16x16x16_f16 v[65:68], v[159:160], v[169:170], v[65:68]
v_add_u32_e32 v167, 64, v163
v_xor_b32_e32 v168, 16, v163
v_and_b32_e32 v169, 0xc0, v167
v_cmp_lt_u32_e32 vcc, v168, v169
v_cndmask_b32_e32 v167, v163, v168, vcc
v_lshlrev_b32_e32 v167, 2, v167
ds_bpermute_b32 v168, v167, v179
v_max_f32_e32 v181, v102, v102
v_max_f32_e32 v182, v104, v104
v_max3_f32 v180, v101, s28, v105
v_max3_f32 v180, v180, v109, v113
s_waitcnt lgkmcnt(0)
v_max_f32_e32 v168, v168, v168
v_max_f32_e32 v179, v179, v168
v_max_f32_e32 v168, v106, v106
v_max_f32_e32 v168, v181, v168
v_max_f32_e32 v181, v108, v108
v_max_f32_e32 v181, v182, v181
v_max3_f32 v182, v103, v107, v111
v_max3_f32 v181, v181, v112, v116
v_max3_f32 v168, v168, v110, v114
v_max3_f32 v182, v182, v115, v119
v_max3_f32 v181, v181, v120, v124
v_max3_f32 v180, v180, v117, v121
v_max3_f32 v168, v168, v118, v122
v_max3_f32 v182, v182, v123, v127
v_max3_f32 v181, v181, v128, v68
v_max3_f32 v180, v180, v125, v65
v_max3_f32 v168, v168, v126, v66
v_max3_f32 v181, v182, v67, v181
v_max3_f32 v180, v180, v168, v181
ds_bpermute_b32 v181, v167, v180
v_xor_b32_e32 v170, 32, v163
v_cmp_lt_u32_e32 vcc, v170, v169
v_cndmask_b32_e32 v168, v163, v170, vcc
v_lshlrev_b32_e32 v168, 2, v168
s_waitcnt lgkmcnt(0)
v_max_f32_e32 v170, v181, v181
v_max_f32_e32 v170, v180, v170
ds_bpermute_b32 v169, v168, v179
ds_bpermute_b32 v180, v168, v170
s_waitcnt lgkmcnt(1)
v_max3_f32 v169, v179, v169, v178
s_waitcnt lgkmcnt(0)
v_max3_f32 v170, v170, v180, v177
v_sub_f32_e32 v192, v66, v170
v_sub_f32_e32 v66, v178, v169
v_sub_f32_e32 v191, v68, v170
v_exp_f32_e32 v68, v66
v_sub_f32_e32 v66, v177, v170
v_sub_f32_e32 v179, v72, v169
v_exp_f32_e32 v66, v66
v_sub_f32_e32 v71, v71, v169
v_sub_f32_e32 v70, v70, v169
v_sub_f32_e32 v69, v69, v169
v_mul_f32_e32 v166, v166, v66
v_sub_f32_e32 v180, v76, v169
v_sub_f32_e32 v181, v75, v169
v_sub_f32_e32 v182, v74, v169
v_sub_f32_e32 v183, v73, v169
v_sub_f32_e32 v184, v80, v169
v_sub_f32_e32 v185, v79, v169
v_sub_f32_e32 v186, v78, v169
v_sub_f32_e32 v187, v77, v169
v_sub_f32_e32 v84, v84, v169
v_sub_f32_e32 v188, v83, v169
v_sub_f32_e32 v189, v82, v169
v_sub_f32_e32 v190, v81, v169
v_sub_f32_e32 v88, v88, v169
v_sub_f32_e32 v87, v87, v169
v_sub_f32_e32 v86, v86, v169
v_sub_f32_e32 v85, v85, v169
v_sub_f32_e32 v92, v92, v169
v_sub_f32_e32 v91, v91, v169
v_sub_f32_e32 v90, v90, v169
v_sub_f32_e32 v89, v89, v169
v_sub_f32_e32 v96, v96, v169
v_sub_f32_e32 v95, v95, v169
v_sub_f32_e32 v94, v94, v169
v_sub_f32_e32 v93, v93, v169
v_sub_f32_e32 v100, v100, v169
v_sub_f32_e32 v99, v99, v169
v_sub_f32_e32 v98, v98, v169
v_sub_f32_e32 v97, v97, v169
v_sub_f32_e32 v104, v104, v170
v_sub_f32_e32 v103, v103, v170
v_sub_f32_e32 v102, v102, v170
v_sub_f32_e32 v101, v101, v170
v_sub_f32_e32 v108, v108, v170
v_sub_f32_e32 v107, v107, v170
v_sub_f32_e32 v106, v106, v170
v_sub_f32_e32 v105, v105, v170
v_sub_f32_e32 v112, v112, v170
v_sub_f32_e32 v111, v111, v170
v_sub_f32_e32 v110, v110, v170
v_sub_f32_e32 v109, v109, v170
v_sub_f32_e32 v116, v116, v170
v_sub_f32_e32 v115, v115, v170
v_sub_f32_e32 v114, v114, v170
v_sub_f32_e32 v113, v113, v170
v_sub_f32_e32 v120, v120, v170
v_sub_f32_e32 v119, v119, v170
v_sub_f32_e32 v118, v118, v170
v_sub_f32_e32 v117, v117, v170
v_sub_f32_e32 v124, v124, v170
v_sub_f32_e32 v123, v123, v170
v_sub_f32_e32 v122, v122, v170
v_sub_f32_e32 v121, v121, v170
v_sub_f32_e32 v128, v128, v170
v_sub_f32_e32 v127, v127, v170
v_sub_f32_e32 v126, v126, v170
v_sub_f32_e32 v125, v125, v170
v_sub_f32_e32 v67, v67, v170
v_exp_f32_e32 v72, v69
v_exp_f32_e32 v73, v70
v_exp_f32_e32 v74, v71
v_exp_f32_e32 v75, v179
v_sub_f32_e32 v65, v65, v170
v_mul_f32_e32 v69, v165, v68
v_mov_b64_e32 v[78:79], v[74:75]
v_mov_b64_e32 v[76:77], v[72:73]
v_cvt_f16_f32_e32 v70, v79
v_cvt_f16_f32_e32 v71, v78
v_cvt_f16_f32_e32 v77, v77
v_cvt_f16_f32_e32 v76, v76
v_pack_b32_f16 v71, v71, v70
v_pack_b32_f16 v70, v76, v77
v_exp_f32_e32 v76, v183
v_exp_f32_e32 v77, v182
v_exp_f32_e32 v78, v181
v_exp_f32_e32 v79, v180
v_add_f32_e32 v165, v72, v76
v_add_f32_e32 v177, v73, v77
v_add_f32_e32 v178, v74, v78
v_mov_b64_e32 v[82:83], v[78:79]
v_mov_b64_e32 v[80:81], v[76:77]
v_cvt_f16_f32_e32 v72, v83
v_cvt_f16_f32_e32 v73, v82
v_cvt_f16_f32_e32 v74, v81
v_cvt_f16_f32_e32 v76, v80
v_pack_b32_f16 v73, v73, v72
v_pack_b32_f16 v72, v76, v74
v_add_f32_e32 v82, v75, v79
v_exp_f32_e32 v74, v187
v_exp_f32_e32 v75, v186
v_exp_f32_e32 v76, v185
v_exp_f32_e32 v77, v184
v_add_f32_e32 v83, v74, v165
v_add_f32_e32 v165, v75, v177
v_add_f32_e32 v177, v76, v178
v_mov_b64_e32 v[80:81], v[76:77]
v_mov_b64_e32 v[78:79], v[74:75]
v_cvt_f16_f32_e32 v74, v81
v_cvt_f16_f32_e32 v75, v80
v_cvt_f16_f32_e32 v76, v79
v_cvt_f16_f32_e32 v78, v78
v_pack_b32_f16 v75, v75, v74
v_pack_b32_f16 v74, v78, v76
v_add_f32_e32 v178, v77, v82
v_exp_f32_e32 v77, v189
v_exp_f32_e32 v78, v188
v_exp_f32_e32 v76, v190
v_exp_f32_e32 v79, v84
v_add_f32_e32 v165, v77, v165
v_add_f32_e32 v177, v78, v177
v_add_f32_e32 v84, v76, v83
v_mov_b64_e32 v[82:83], v[78:79]
v_mov_b64_e32 v[80:81], v[76:77]
v_cvt_f16_f32_e32 v76, v83
v_cvt_f16_f32_e32 v77, v82
v_cvt_f16_f32_e32 v78, v81
v_cvt_f16_f32_e32 v80, v80
v_pack_b32_f16 v77, v77, v76
v_pack_b32_f16 v76, v80, v78
v_add_f32_e32 v178, v79, v178
v_exp_f32_e32 v78, v85
v_exp_f32_e32 v79, v86
v_exp_f32_e32 v80, v87
v_exp_f32_e32 v81, v88
v_add_f32_e32 v86, v78, v84
v_add_f32_e32 v87, v79, v165
v_add_f32_e32 v88, v80, v177
v_mov_b64_e32 v[84:85], v[80:81]
v_mov_b64_e32 v[82:83], v[78:79]
v_cvt_f16_f32_e32 v78, v85
v_cvt_f16_f32_e32 v79, v84
v_cvt_f16_f32_e32 v80, v83
v_cvt_f16_f32_e32 v82, v82
v_pack_b32_f16 v79, v79, v78
v_pack_b32_f16 v78, v82, v80
v_add_f32_e32 v165, v81, v178
v_exp_f32_e32 v80, v89
v_exp_f32_e32 v81, v90
v_exp_f32_e32 v82, v91
v_exp_f32_e32 v83, v92
v_add_f32_e32 v89, v80, v86
v_add_f32_e32 v90, v81, v87
v_add_f32_e32 v88, v82, v88
v_mov_b64_e32 v[86:87], v[82:83]
v_mov_b64_e32 v[84:85], v[80:81]
v_cvt_f16_f32_e32 v80, v87
v_cvt_f16_f32_e32 v81, v86
v_cvt_f16_f32_e32 v82, v85
v_cvt_f16_f32_e32 v84, v84
v_pack_b32_f16 v81, v81, v80
v_pack_b32_f16 v80, v84, v82
v_add_f32_e32 v91, v83, v165
v_exp_f32_e32 v82, v93
v_exp_f32_e32 v83, v94
v_exp_f32_e32 v84, v95
v_exp_f32_e32 v85, v96
v_add_f32_e32 v92, v82, v89
v_add_f32_e32 v90, v83, v90
v_add_f32_e32 v93, v84, v88
v_mov_b64_e32 v[88:89], v[84:85]
v_mov_b64_e32 v[86:87], v[82:83]
v_cvt_f16_f32_e32 v82, v89
v_cvt_f16_f32_e32 v83, v88
v_cvt_f16_f32_e32 v84, v87
v_cvt_f16_f32_e32 v86, v86
v_pack_b32_f16 v83, v83, v82
v_pack_b32_f16 v82, v86, v84
v_exp_f32_e32 v86, v97
v_exp_f32_e32 v87, v98
v_exp_f32_e32 v88, v99
v_add_f32_e32 v91, v85, v91
v_add_f32_e32 v84, v86, v92
v_add_f32_e32 v85, v87, v90
v_exp_f32_e32 v89, v100
v_add_f32_e32 v92, v84, v85
v_add_f32_e32 v93, v88, v93
v_mov_b64_e32 v[84:85], v[86:87]
v_mov_b64_e32 v[86:87], v[88:89]
v_cvt_f16_f32_e32 v87, v87
v_cvt_f16_f32_e32 v86, v86
v_cvt_f16_f32_e32 v85, v85
v_cvt_f16_f32_e32 v84, v84
v_pack_b32_f16 v87, v86, v87
v_pack_b32_f16 v86, v84, v85
v_add_f32_e32 v84, v89, v91
v_add_f32_e32 v84, v93, v84
v_add_f32_e32 v84, v92, v84
v_add_f32_e32 v165, v69, v84
v_exp_f32_e32 v88, v101
v_exp_f32_e32 v89, v102
v_exp_f32_e32 v90, v103
v_exp_f32_e32 v91, v104
s_nop 1
v_mov_b64_e32 v[94:95], v[90:91]
v_mov_b64_e32 v[92:93], v[88:89]
v_cvt_f16_f32_e32 v69, v95
v_cvt_f16_f32_e32 v84, v94
v_cvt_f16_f32_e32 v93, v93
v_cvt_f16_f32_e32 v92, v92
v_pack_b32_f16 v85, v84, v69
v_pack_b32_f16 v84, v92, v93
v_exp_f32_e32 v92, v105
v_exp_f32_e32 v93, v106
v_exp_f32_e32 v94, v107
v_exp_f32_e32 v95, v108
v_add_f32_e32 v69, v88, v92
v_add_f32_e32 v100, v89, v93
v_add_f32_e32 v101, v90, v94
v_mov_b64_e32 v[98:99], v[94:95]
v_mov_b64_e32 v[96:97], v[92:93]
v_cvt_f16_f32_e32 v88, v99
v_cvt_f16_f32_e32 v89, v98
v_cvt_f16_f32_e32 v90, v97
v_cvt_f16_f32_e32 v92, v96
v_pack_b32_f16 v89, v89, v88
v_pack_b32_f16 v88, v92, v90
v_add_f32_e32 v98, v91, v95
v_exp_f32_e32 v90, v109
v_exp_f32_e32 v91, v110
v_exp_f32_e32 v92, v111
v_exp_f32_e32 v93, v112
v_add_f32_e32 v69, v90, v69
v_add_f32_e32 v99, v91, v100
v_add_f32_e32 v100, v92, v101
v_mov_b64_e32 v[96:97], v[92:93]
v_mov_b64_e32 v[94:95], v[90:91]
v_cvt_f16_f32_e32 v90, v97
v_cvt_f16_f32_e32 v91, v96
v_cvt_f16_f32_e32 v92, v95
v_cvt_f16_f32_e32 v94, v94
v_pack_b32_f16 v91, v91, v90
v_pack_b32_f16 v90, v94, v92
v_add_f32_e32 v101, v93, v98
v_exp_f32_e32 v92, v113
v_exp_f32_e32 v93, v114
v_exp_f32_e32 v94, v115
v_exp_f32_e32 v95, v116
v_add_f32_e32 v69, v92, v69
v_add_f32_e32 v102, v93, v99
v_add_f32_e32 v100, v94, v100
v_mov_b64_e32 v[98:99], v[94:95]
v_mov_b64_e32 v[96:97], v[92:93]
v_cvt_f16_f32_e32 v92, v99
v_cvt_f16_f32_e32 v93, v98
v_cvt_f16_f32_e32 v94, v97
v_cvt_f16_f32_e32 v96, v96
v_pack_b32_f16 v93, v93, v92
v_pack_b32_f16 v92, v96, v94
v_add_f32_e32 v103, v95, v101
v_exp_f32_e32 v94, v117
v_exp_f32_e32 v95, v118
v_exp_f32_e32 v96, v119
v_exp_f32_e32 v97, v120
v_add_f32_e32 v69, v94, v69
v_add_f32_e32 v102, v95, v102
v_add_f32_e32 v104, v96, v100
v_mov_b64_e32 v[100:101], v[96:97]
v_mov_b64_e32 v[98:99], v[94:95]
v_cvt_f16_f32_e32 v94, v101
v_cvt_f16_f32_e32 v95, v100
v_cvt_f16_f32_e32 v96, v99
v_cvt_f16_f32_e32 v98, v98
v_pack_b32_f16 v95, v95, v94
v_pack_b32_f16 v94, v98, v96
v_add_f32_e32 v105, v97, v103
v_exp_f32_e32 v96, v121
v_exp_f32_e32 v97, v122
v_exp_f32_e32 v98, v123
v_exp_f32_e32 v99, v124
v_add_f32_e32 v69, v96, v69
v_add_f32_e32 v106, v97, v102
v_add_f32_e32 v104, v98, v104
v_mov_b64_e32 v[102:103], v[98:99]
v_mov_b64_e32 v[100:101], v[96:97]
v_cvt_f16_f32_e32 v96, v103
v_cvt_f16_f32_e32 v97, v102
v_cvt_f16_f32_e32 v98, v101
v_cvt_f16_f32_e32 v100, v100
v_pack_b32_f16 v97, v97, v96
v_pack_b32_f16 v96, v100, v98
v_add_f32_e32 v107, v99, v105
v_exp_f32_e32 v98, v125
v_exp_f32_e32 v99, v126
v_exp_f32_e32 v100, v127
v_exp_f32_e32 v101, v128
v_add_f32_e32 v69, v98, v69
v_add_f32_e32 v106, v99, v106
v_add_f32_e32 v108, v100, v104
v_mov_b64_e32 v[104:105], v[100:101]
v_mov_b64_e32 v[102:103], v[98:99]
v_cvt_f16_f32_e32 v98, v105
v_cvt_f16_f32_e32 v99, v104
v_cvt_f16_f32_e32 v100, v103
v_cvt_f16_f32_e32 v102, v102
v_pack_b32_f16 v99, v99, v98
v_pack_b32_f16 v98, v102, v100
v_exp_f32_e32 v100, v65
v_add_f32_e32 v65, v101, v107
v_exp_f32_e32 v101, v192
v_exp_f32_e32 v102, v67
v_add_f32_e32 v67, v100, v69
v_exp_f32_e32 v103, v191
v_add_f32_e32 v69, v101, v106
v_add_f32_e32 v67, v67, v69
v_add_f32_e32 v69, v102, v108
v_mov_b64_e32 v[106:107], v[102:103]
v_mov_b64_e32 v[104:105], v[100:101]
v_cvt_f16_f32_e32 v100, v107
v_cvt_f16_f32_e32 v101, v106
v_cvt_f16_f32_e32 v102, v105
v_cvt_f16_f32_e32 v104, v104
v_pack_b32_f16 v101, v101, v100
v_pack_b32_f16 v100, v104, v102
v_add_f32_e32 v65, v103, v65
v_add_f32_e32 v65, v69, v65
v_add_f32_e32 v65, v67, v65
v_add_f32_e32 v166, v166, v65
s_waitcnt vmcnt(0)
v_and_b32_e32 v67, 60, v171
v_bfe_u32 v69, v171, 4, 2
v_add_u32_e32 v67, v67, v171
v_mul_u32_u24_e32 v69, 0x300, v69
v_and_b32_e32 v65, 0xc00, v174
v_lshl_add_u32 v67, v67, 4, v69
v_lshlrev_b32_e32 v69, 6, v172
v_and_or_b32 v65, v173, s24, v65
v_sub_u32_e32 v67, v67, v69
v_add_u32_e32 v103, 0x200, v67
v_or_b32_e32 v105, 0x1000, v65
v_add_u32_e32 v107, 0x1200, v67
s_barrier
v_add_u32_e32 v102, 0x100, v67
ds_read_m32x16_b16 v[109:112], v65
ds_read_m32x16_b16 v[113:116], v102
v_add_u32_e32 v104, 0x300, v67
ds_read_m32x16_b16 v[117:120], v103
ds_read_m32x16_b16 v[121:124], v104
v_add_u32_e32 v106, 0x1100, v67
ds_read_m32x16_b16 v[125:128], v105
ds_read_m32x16_b16 v[171:174], v106
v_add_u32_e32 v108, 0x1300, v67
ds_read_m32x16_b16 v[177:180], v107
ds_read_m32x16_b16 v[181:184], v108
v_mov_b32_e32 v69, v68
s_nop 1
v_pk_mul_f32 v[35:36], v[35:36], v[68:69]
v_pk_mul_f32 v[33:34], v[33:34], v[68:69]
s_waitcnt lgkmcnt(7)
v_mmac_f32_16x16x16_f16 v[33:36], v[70:71], v[109:110], v[33:36]
s_waitcnt lgkmcnt(6)
v_mmac_f32_16x16x16_f16 v[33:36], v[72:73], v[113:114], v[33:36]
s_waitcnt lgkmcnt(5)
v_mmac_f32_16x16x16_f16 v[33:36], v[74:75], v[117:118], v[33:36]
s_waitcnt lgkmcnt(4)
v_mmac_f32_16x16x16_f16 v[33:36], v[76:77], v[121:122], v[33:36]
s_waitcnt lgkmcnt(0)
s_barrier
v_add_u32_e32 v67, 0x100040, v175
s_mov_b32 m0, s12
buffer_load_dwordx4 v67, s[4:7], 0 offen lds
v_add_u32_e32 v67, 0x100040, v176
s_mov_b32 m0, s13
buffer_load_dwordx4 v67, s[4:7], 0 offen lds
v_mmac_f32_16x16x16_f16 v[33:36], v[78:79], v[125:126], v[33:36]
v_mmac_f32_16x16x16_f16 v[33:36], v[80:81], v[171:172], v[33:36]
v_mmac_f32_16x16x16_f16 v[33:36], v[82:83], v[177:178], v[33:36]
v_mmac_f32_16x16x16_f16 v[33:36], v[86:87], v[181:182], v[33:36]
v_pk_mul_f32 v[39:40], v[39:40], v[68:69]
v_pk_mul_f32 v[37:38], v[37:38], v[68:69]
v_mmac_f32_16x16x16_f16 v[37:40], v[70:71], v[111:112], v[37:40]
v_mmac_f32_16x16x16_f16 v[37:40], v[72:73], v[115:116], v[37:40]
v_mmac_f32_16x16x16_f16 v[37:40], v[74:75], v[119:120], v[37:40]
v_mmac_f32_16x16x16_f16 v[37:40], v[76:77], v[123:124], v[37:40]
v_mmac_f32_16x16x16_f16 v[37:40], v[78:79], v[127:128], v[37:40]
v_mmac_f32_16x16x16_f16 v[37:40], v[80:81], v[173:174], v[37:40]
v_mmac_f32_16x16x16_f16 v[37:40], v[82:83], v[179:180], v[37:40]
v_mmac_f32_16x16x16_f16 v[37:40], v[86:87], v[183:184], v[37:40]
v_mov_b32_e32 v67, v66
s_nop 1
v_pk_mul_f32 v[3:4], v[3:4], v[66:67]
v_pk_mul_f32 v[1:2], v[1:2], v[66:67]
v_mmac_f32_16x16x16_f16 v[1:4], v[84:85], v[109:110], v[1:4]
v_mmac_f32_16x16x16_f16 v[1:4], v[88:89], v[113:114], v[1:4]
v_mmac_f32_16x16x16_f16 v[1:4], v[90:91], v[117:118], v[1:4]
v_mmac_f32_16x16x16_f16 v[1:4], v[92:93], v[121:122], v[1:4]
v_mmac_f32_16x16x16_f16 v[1:4], v[94:95], v[125:126], v[1:4]
v_mmac_f32_16x16x16_f16 v[1:4], v[96:97], v[171:172], v[1:4]
v_mmac_f32_16x16x16_f16 v[1:4], v[98:99], v[177:178], v[1:4]
v_mmac_f32_16x16x16_f16 v[1:4], v[100:101], v[181:182], v[1:4]
v_pk_mul_f32 v[7:8], v[7:8], v[66:67]
v_pk_mul_f32 v[5:6], v[5:6], v[66:67]
v_mmac_f32_16x16x16_f16 v[5:8], v[84:85], v[111:112], v[5:8]
v_mmac_f32_16x16x16_f16 v[5:8], v[88:89], v[115:116], v[5:8]
v_mmac_f32_16x16x16_f16 v[5:8], v[90:91], v[119:120], v[5:8]
v_mmac_f32_16x16x16_f16 v[5:8], v[92:93], v[123:124], v[5:8]
v_mmac_f32_16x16x16_f16 v[5:8], v[94:95], v[127:128], v[5:8]
v_mmac_f32_16x16x16_f16 v[5:8], v[96:97], v[173:174], v[5:8]
v_mmac_f32_16x16x16_f16 v[5:8], v[98:99], v[179:180], v[5:8]
v_mmac_f32_16x16x16_f16 v[5:8], v[100:101], v[183:184], v[5:8]
s_waitcnt vmcnt(0)
s_barrier
ds_read_m32x16_b16 v[109:112], v65
ds_read_m32x16_b16 v[113:116], v102
ds_read_m32x16_b16 v[117:120], v103
ds_read_m32x16_b16 v[121:124], v104
ds_read_m32x16_b16 v[125:128], v105
ds_read_m32x16_b16 v[171:174], v106
ds_read_m32x16_b16 v[177:180], v107
ds_read_m32x16_b16 v[181:184], v108
v_pk_mul_f32 v[43:44], v[43:44], v[68:69]
v_pk_mul_f32 v[41:42], v[41:42], v[68:69]
s_waitcnt lgkmcnt(7)
v_mmac_f32_16x16x16_f16 v[41:44], v[70:71], v[109:110], v[41:44]
s_waitcnt lgkmcnt(6)
v_mmac_f32_16x16x16_f16 v[41:44], v[72:73], v[113:114], v[41:44]
s_waitcnt lgkmcnt(5)
v_mmac_f32_16x16x16_f16 v[41:44], v[74:75], v[117:118], v[41:44]
s_waitcnt lgkmcnt(4)
v_mmac_f32_16x16x16_f16 v[41:44], v[76:77], v[121:122], v[41:44]
s_waitcnt lgkmcnt(0)
s_barrier
v_add_u32_e32 v185, 0x100080, v175
s_mov_b32 m0, s12
buffer_load_dwordx4 v185, s[4:7], 0 offen lds
v_add_u32_e32 v185, 0x100080, v176
s_mov_b32 m0, s13
buffer_load_dwordx4 v185, s[4:7], 0 offen lds
v_mmac_f32_16x16x16_f16 v[41:44], v[78:79], v[125:126], v[41:44]
v_mmac_f32_16x16x16_f16 v[41:44], v[80:81], v[171:172], v[41:44]
v_mmac_f32_16x16x16_f16 v[41:44], v[82:83], v[177:178], v[41:44]
v_mmac_f32_16x16x16_f16 v[41:44], v[86:87], v[181:182], v[41:44]
v_pk_mul_f32 v[47:48], v[47:48], v[68:69]
v_pk_mul_f32 v[45:46], v[45:46], v[68:69]
v_mmac_f32_16x16x16_f16 v[45:48], v[70:71], v[111:112], v[45:48]
v_mmac_f32_16x16x16_f16 v[45:48], v[72:73], v[115:116], v[45:48]
v_mmac_f32_16x16x16_f16 v[45:48], v[74:75], v[119:120], v[45:48]
v_mmac_f32_16x16x16_f16 v[45:48], v[76:77], v[123:124], v[45:48]
v_mmac_f32_16x16x16_f16 v[45:48], v[78:79], v[127:128], v[45:48]
v_mmac_f32_16x16x16_f16 v[45:48], v[80:81], v[173:174], v[45:48]
v_mmac_f32_16x16x16_f16 v[45:48], v[82:83], v[179:180], v[45:48]
v_mmac_f32_16x16x16_f16 v[45:48], v[86:87], v[183:184], v[45:48]
v_pk_mul_f32 v[11:12], v[11:12], v[66:67]
v_pk_mul_f32 v[9:10], v[9:10], v[66:67]
v_mmac_f32_16x16x16_f16 v[9:12], v[84:85], v[109:110], v[9:12]
v_mmac_f32_16x16x16_f16 v[9:12], v[88:89], v[113:114], v[9:12]
v_mmac_f32_16x16x16_f16 v[9:12], v[90:91], v[117:118], v[9:12]
v_mmac_f32_16x16x16_f16 v[9:12], v[92:93], v[121:122], v[9:12]
v_mmac_f32_16x16x16_f16 v[9:12], v[94:95], v[125:126], v[9:12]
v_mmac_f32_16x16x16_f16 v[9:12], v[96:97], v[171:172], v[9:12]
v_mmac_f32_16x16x16_f16 v[9:12], v[98:99], v[177:178], v[9:12]
v_mmac_f32_16x16x16_f16 v[9:12], v[100:101], v[181:182], v[9:12]
v_pk_mul_f32 v[15:16], v[15:16], v[66:67]
v_pk_mul_f32 v[13:14], v[13:14], v[66:67]
v_mmac_f32_16x16x16_f16 v[13:16], v[84:85], v[111:112], v[13:16]
v_mmac_f32_16x16x16_f16 v[13:16], v[88:89], v[115:116], v[13:16]
v_mmac_f32_16x16x16_f16 v[13:16], v[90:91], v[119:120], v[13:16]
v_mmac_f32_16x16x16_f16 v[13:16], v[92:93], v[123:124], v[13:16]
v_mmac_f32_16x16x16_f16 v[13:16], v[94:95], v[127:128], v[13:16]
v_mmac_f32_16x16x16_f16 v[13:16], v[96:97], v[173:174], v[13:16]
v_mmac_f32_16x16x16_f16 v[13:16], v[98:99], v[179:180], v[13:16]
v_mmac_f32_16x16x16_f16 v[13:16], v[100:101], v[183:184], v[13:16]
s_waitcnt vmcnt(0)
s_barrier
ds_read_m32x16_b16 v[109:112], v65
ds_read_m32x16_b16 v[113:116], v102
ds_read_m32x16_b16 v[117:120], v103
ds_read_m32x16_b16 v[121:124], v104
ds_read_m32x16_b16 v[125:128], v105
ds_read_m32x16_b16 v[171:174], v106
ds_read_m32x16_b16 v[177:180], v107
ds_read_m32x16_b16 v[181:184], v108
v_pk_mul_f32 v[59:60], v[59:60], v[68:69]
v_pk_mul_f32 v[57:58], v[57:58], v[68:69]
s_waitcnt lgkmcnt(7)
v_mmac_f32_16x16x16_f16 v[57:60], v[70:71], v[109:110], v[57:60]
s_waitcnt lgkmcnt(6)
v_mmac_f32_16x16x16_f16 v[57:60], v[72:73], v[113:114], v[57:60]
s_waitcnt lgkmcnt(5)
v_mmac_f32_16x16x16_f16 v[57:60], v[74:75], v[117:118], v[57:60]
s_waitcnt lgkmcnt(4)
v_mmac_f32_16x16x16_f16 v[57:60], v[76:77], v[121:122], v[57:60]
s_waitcnt lgkmcnt(0)
s_barrier
v_add_u32_e32 v175, 0x1000c0, v175
s_mov_b32 m0, s12
buffer_load_dwordx4 v175, s[4:7], 0 offen lds
v_add_u32_e32 v175, 0x1000c0, v176
s_mov_b32 m0, s13
buffer_load_dwordx4 v175, s[4:7], 0 offen lds
v_mmac_f32_16x16x16_f16 v[57:60], v[78:79], v[125:126], v[57:60]
v_mmac_f32_16x16x16_f16 v[57:60], v[80:81], v[171:172], v[57:60]
v_mmac_f32_16x16x16_f16 v[57:60], v[82:83], v[177:178], v[57:60]
v_mmac_f32_16x16x16_f16 v[57:60], v[86:87], v[181:182], v[57:60]
v_pk_mul_f32 v[63:64], v[63:64], v[68:69]
v_pk_mul_f32 v[61:62], v[61:62], v[68:69]
v_mmac_f32_16x16x16_f16 v[61:64], v[70:71], v[111:112], v[61:64]
v_mmac_f32_16x16x16_f16 v[61:64], v[72:73], v[115:116], v[61:64]
v_mmac_f32_16x16x16_f16 v[61:64], v[74:75], v[119:120], v[61:64]
v_mmac_f32_16x16x16_f16 v[61:64], v[76:77], v[123:124], v[61:64]
v_mmac_f32_16x16x16_f16 v[61:64], v[78:79], v[127:128], v[61:64]
v_mmac_f32_16x16x16_f16 v[61:64], v[80:81], v[173:174], v[61:64]
v_mmac_f32_16x16x16_f16 v[61:64], v[82:83], v[179:180], v[61:64]
v_mmac_f32_16x16x16_f16 v[61:64], v[86:87], v[183:184], v[61:64]
v_pk_mul_f32 v[19:20], v[19:20], v[66:67]
v_pk_mul_f32 v[17:18], v[17:18], v[66:67]
v_mmac_f32_16x16x16_f16 v[17:20], v[84:85], v[109:110], v[17:20]
v_mmac_f32_16x16x16_f16 v[17:20], v[88:89], v[113:114], v[17:20]
v_mmac_f32_16x16x16_f16 v[17:20], v[90:91], v[117:118], v[17:20]
v_mmac_f32_16x16x16_f16 v[17:20], v[92:93], v[121:122], v[17:20]
v_mmac_f32_16x16x16_f16 v[17:20], v[94:95], v[125:126], v[17:20]
v_mmac_f32_16x16x16_f16 v[17:20], v[96:97], v[171:172], v[17:20]
v_mmac_f32_16x16x16_f16 v[17:20], v[98:99], v[177:178], v[17:20]
v_mmac_f32_16x16x16_f16 v[17:20], v[100:101], v[181:182], v[17:20]
v_pk_mul_f32 v[23:24], v[23:24], v[66:67]
v_pk_mul_f32 v[21:22], v[21:22], v[66:67]
v_mmac_f32_16x16x16_f16 v[21:24], v[84:85], v[111:112], v[21:24]
v_mmac_f32_16x16x16_f16 v[21:24], v[88:89], v[115:116], v[21:24]
v_mmac_f32_16x16x16_f16 v[21:24], v[90:91], v[119:120], v[21:24]
v_mmac_f32_16x16x16_f16 v[21:24], v[92:93], v[123:124], v[21:24]
v_mmac_f32_16x16x16_f16 v[21:24], v[94:95], v[127:128], v[21:24]
v_mmac_f32_16x16x16_f16 v[21:24], v[96:97], v[173:174], v[21:24]
v_mmac_f32_16x16x16_f16 v[21:24], v[98:99], v[179:180], v[21:24]
v_mmac_f32_16x16x16_f16 v[21:24], v[100:101], v[183:184], v[21:24]
s_waitcnt vmcnt(0)
s_barrier
ds_read_m32x16_b16 v[109:112], v65
ds_read_m32x16_b16 v[113:116], v102
ds_read_m32x16_b16 v[117:120], v103
ds_read_m32x16_b16 v[121:124], v104
ds_read_m32x16_b16 v[102:105], v105
ds_read_m32x16_b16 v[125:128], v106
ds_read_m32x16_b16 v[171:174], v107
ds_read_m32x16_b16 v[175:178], v108
v_pk_mul_f32 v[51:52], v[51:52], v[68:69]
v_pk_mul_f32 v[49:50], v[49:50], v[68:69]
s_waitcnt lgkmcnt(7)
v_mmac_f32_16x16x16_f16 v[49:52], v[70:71], v[109:110], v[49:52]
s_waitcnt lgkmcnt(6)
v_mmac_f32_16x16x16_f16 v[49:52], v[72:73], v[113:114], v[49:52]
s_waitcnt lgkmcnt(5)
v_mmac_f32_16x16x16_f16 v[49:52], v[74:75], v[117:118], v[49:52]
s_waitcnt lgkmcnt(4)
v_mmac_f32_16x16x16_f16 v[49:52], v[76:77], v[121:122], v[49:52]
s_waitcnt lgkmcnt(3)
v_mmac_f32_16x16x16_f16 v[49:52], v[78:79], v[102:103], v[49:52]
s_waitcnt lgkmcnt(2)
v_mmac_f32_16x16x16_f16 v[49:52], v[80:81], v[125:126], v[49:52]
s_waitcnt lgkmcnt(1)
v_mmac_f32_16x16x16_f16 v[49:52], v[82:83], v[171:172], v[49:52]
s_waitcnt lgkmcnt(0)
v_mmac_f32_16x16x16_f16 v[49:52], v[86:87], v[175:176], v[49:52]
v_pk_mul_f32 v[55:56], v[55:56], v[68:69]
v_pk_mul_f32 v[53:54], v[53:54], v[68:69]
v_mmac_f32_16x16x16_f16 v[53:56], v[70:71], v[111:112], v[53:56]
v_mmac_f32_16x16x16_f16 v[53:56], v[72:73], v[115:116], v[53:56]
v_mmac_f32_16x16x16_f16 v[53:56], v[74:75], v[119:120], v[53:56]
v_mmac_f32_16x16x16_f16 v[53:56], v[76:77], v[123:124], v[53:56]
v_mmac_f32_16x16x16_f16 v[53:56], v[78:79], v[104:105], v[53:56]
v_mmac_f32_16x16x16_f16 v[53:56], v[80:81], v[127:128], v[53:56]
v_mmac_f32_16x16x16_f16 v[53:56], v[82:83], v[173:174], v[53:56]
v_mmac_f32_16x16x16_f16 v[53:56], v[86:87], v[177:178], v[53:56]
v_pk_mul_f32 v[27:28], v[27:28], v[66:67]
v_pk_mul_f32 v[25:26], v[25:26], v[66:67]
v_mmac_f32_16x16x16_f16 v[25:28], v[84:85], v[109:110], v[25:28]
v_mmac_f32_16x16x16_f16 v[25:28], v[88:89], v[113:114], v[25:28]
v_mmac_f32_16x16x16_f16 v[25:28], v[90:91], v[117:118], v[25:28]
v_mmac_f32_16x16x16_f16 v[25:28], v[92:93], v[121:122], v[25:28]
v_mmac_f32_16x16x16_f16 v[25:28], v[94:95], v[102:103], v[25:28]
v_mmac_f32_16x16x16_f16 v[25:28], v[96:97], v[125:126], v[25:28]
v_mmac_f32_16x16x16_f16 v[25:28], v[98:99], v[171:172], v[25:28]
v_mmac_f32_16x16x16_f16 v[25:28], v[100:101], v[175:176], v[25:28]
v_pk_mul_f32 v[31:32], v[31:32], v[66:67]
v_pk_mul_f32 v[29:30], v[29:30], v[66:67]
v_mmac_f32_16x16x16_f16 v[29:32], v[84:85], v[111:112], v[29:32]
v_mmac_f32_16x16x16_f16 v[29:32], v[88:89], v[115:116], v[29:32]
v_mmac_f32_16x16x16_f16 v[29:32], v[90:91], v[119:120], v[29:32]
v_mmac_f32_16x16x16_f16 v[29:32], v[92:93], v[123:124], v[29:32]
v_mmac_f32_16x16x16_f16 v[29:32], v[94:95], v[104:105], v[29:32]
v_mmac_f32_16x16x16_f16 v[29:32], v[96:97], v[127:128], v[29:32]
v_mmac_f32_16x16x16_f16 v[29:32], v[98:99], v[173:174], v[29:32]
v_mmac_f32_16x16x16_f16 v[29:32], v[100:101], v[177:178], v[29:32]
s_add_u32 s8, s8, 0x4000
s_addc_u32 s9, s9, 0
s_cmp_lg_u64 s[8:9], 0
v_mov_b32_e32 v178, v169
v_mov_b32_e32 v177, v170
s_barrier
s_cbranch_scc1 .LBB0_1
