# 1. 物理强行合并为一个文件
llfile=final.ll
ROCM_BC=/opt/dtk/amdgcn/bitcode

llvm-link $llfile \
    $ROCM_BC/ocml.bc \
    $ROCM_BC/ockl.bc \
    $ROCM_BC/oclc_isa_version_936.bc \
    $ROCM_BC/oclc_abi_version_400.bc \
    $ROCM_BC/oclc_wavefrontsize64_on.bc \
    $ROCM_BC/oclc_correctly_rounded_sqrt_off.bc \
    $ROCM_BC/oclc_finite_only_off.bc \
    $ROCM_BC/oclc_unsafe_math_off.bc \
    $ROCM_BC/oclc_daz_opt_off.bc \
    -o merged.bc

# ls /opt/dtk/amdgcn/bitcode/
# asanrtl.bc     hipdrt_928.bc             ockl.bc                             oclc_daz_opt_on.bc       oclc_isa_version_928.bc      ocml.bc                 softfloat_near_maxMag.bc
# cuda2gcn.bc    hipdrt_936.bc             oclc_abi_version_400.bc             oclc_finite_only_off.bc  oclc_isa_version_936.bc      opencl.bc
# func.bc        hyptxas-device-gfx906.bc  oclc_abi_version_500.bc             oclc_finite_only_on.bc   oclc_unsafe_math_off.bc      softfloat_max.bc
# hip.bc         hyptxas-device-gfx926.bc  oclc_correctly_rounded_sqrt_off.bc  oclc_isa_version_900.bc  oclc_unsafe_math_on.bc       softfloat_min.bc
# hipdrt_906.bc  hyptxas-device-gfx928.bc  oclc_correctly_rounded_sqrt_on.bc   oclc_isa_version_906.bc  oclc_wavefrontsize64_off.bc  softfloat_minMag.bc
# hipdrt_926.bc  hyptxas-device-gfx936.bc  oclc_daz_opt_off.bc                 oclc_isa_version_926.bc  oclc_wavefrontsize64_on.bc   softfloat_near_even.bc

# 2. 仅做常规 O3 内联优化（不清除外部配置符号）
opt -passes='default<O3>' merged.bc -o opt.bc

# 3. 编译为目标文件并链接
llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx936 -filetype=obj opt.bc -o kernel.o
ld.lld -shared kernel.o -o kernel.hsaco
