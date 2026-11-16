target triple = "amdgcn-amd-amdhsa"
define i32 @probe(float %a, float %b) {
  %p = call i32 asm "v_cvt_f16_f32 $0, $1\0A\09v_cvt_f16_f32_sdwa $0, $2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD", "=&v,v,v"(float %a, float %b)
  ret i32 %p
}
