#!/usr/bin/env bash
set -euo pipefail

# 1. 物理强行合并为一个文件
# 这里的输入 ll 需要已经带有 !dbg / !DIFile / !DILocation。
# 例如 test_src.cpp 现在生成的 finalLLVMText.ll。
llfile=${1:-finalLLVMText.ll}
ROCM_BC=/opt/dtk/amdgcn/bitcode
GPU_ARCH=gfx936

# 调试 hsaco 时建议先用 O0，行号和指令对应关系最直观。
# 需要性能时可以这样跑：
#   IR_OPT_PIPELINE='default<O3>' LLVM_CODEGEN_OPT='-O3' ./GenHsaco.sh finalLLVMText.ll
IR_OPT_PIPELINE=${IR_OPT_PIPELINE:-default<O0>}
LLVM_CODEGEN_OPT=${LLVM_CODEGEN_OPT:--O0}
DWARF_VERSION=${DWARF_VERSION:-4}

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

# 2. 保留 debug metadata 做 IR 优化。
# 不要加 strip-debug / strip-named-metadata，否则 !dbg / !DIFile 会丢。
opt -passes="${IR_OPT_PIPELINE}" \
    -debugger-tune=gdb \
    merged.bc \
    -o opt.bc

# 可选：本地检查 bitcode 里是否还有 debug 信息。
# llvm-dis opt.bc -o - | grep -E '!dbg|DIFile|DILocation' | head

# # 1. 用 llc 将 IR 输出为 GCN 汇编文本 (kernel.s)
# llc -mtriple=amdgcn-amd-amdhsa -mcpu=$GPU_ARCH \
#     $LLVM_CODEGEN_OPT \
#     -debugger-tune=gdb \
#     -dwarf-version=$DWARF_VERSION \
#     opt.bc -filetype=asm -o kernel.s

# # 2. 如果走 asm 路线，用 clang 的内置汇编器重新生成 obj。
# # 注意：这条路线通常映射 kernel.s 行号；要保留 MLIR 原始行号，优先用下面 llc 直接出 obj。
# clang -triple=amdgcn-amd-amdhsa -mcpu=$GPU_ARCH -g -c kernel.s -o kernel.o


# 3. 编译为目标文件并链接
llc -mtriple=amdgcn-amd-amdhsa \
    -mcpu=$GPU_ARCH \
    $LLVM_CODEGEN_OPT \
    -debugger-tune=gdb \
    -dwarf-version=$DWARF_VERSION \
    -filetype=obj \
    opt.bc \
    -o kernel.o

# ld.lld 默认不会剥离 .debug_* section；不要加 -s/--strip-*。
ld.lld -shared --build-id=sha1 kernel.o -o kernel.hsaco

# 可选：检查 obj/hsaco 是否真的带了行号表。
# llvm-dwarfdump --debug-line kernel.o
# llvm-dwarfdump --debug-line kernel.hsaco
# llvm-objdump -d --line-numbers kernel.hsaco | less
echo "---- GenHsaco Done! -----"