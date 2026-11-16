target triple = "amdgcn-amd-amdhsa"
define i32 @probe(float %a, float %b) {
  %p = call i32 asm "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v"(float %a, float %b)
  ret i32 %p
}
