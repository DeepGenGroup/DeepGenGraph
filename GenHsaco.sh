#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./GenHsaco.sh <input.ll> <opt_level> <output.hsaco>
#
# Examples:
#   ./GenHsaco.sh ./final.ll 0 ./kernel.hsaco
#   ./GenHsaco.sh ./final.ll 3 ./kernel.hsaco
#
# opt_level:
#   0 -> opt default<O0> + llc -O0
#   1 -> opt default<O1> + llc -O1
#   2 -> opt default<O2> + llc -O2
#   3 -> opt default<O3> + llc -O3

LL_FILE=${1:-finalLLVMText.ll}
OPT_LEVEL=${2:-0}
KERNEL_FILE=${3:-kernel.hsaco}

ROCM_BC=/opt/dtk/amdgcn/bitcode
GPU_ARCH=gfx936
DWARF_VERSION=${DWARF_VERSION:-4}

# ------------------------------------------------------------
# 参数检查
# ------------------------------------------------------------

if [[ ! -f "${LL_FILE}" ]]; then
    echo "Error: input LLVM IR file not found: ${LL_FILE}" >&2
    exit 1
fi

case "${OPT_LEVEL}" in
    0|1|2|3)
        ;;
    *)
        echo "Error: opt_level must be 0, 1, 2, or 3." >&2
        echo "Usage: $0 <input.ll> <opt_level> <output.hsaco>" >&2
        echo "Example: $0 ./final.ll 3 ./kernel.hsaco" >&2
        exit 1
        ;;
esac

IR_OPT_PIPELINE="default<O${OPT_LEVEL}>"
LLVM_CODEGEN_OPT="-O${OPT_LEVEL}"

echo "========================================"
echo " LLVM IR      : ${LL_FILE}"
echo " Optimization : O${OPT_LEVEL}"
echo " IR pipeline  : ${IR_OPT_PIPELINE}"
echo " LLC opt      : ${LLVM_CODEGEN_OPT}"
echo " GPU arch     : ${GPU_ARCH}"
echo " Output       : ${KERNEL_FILE}"
echo "========================================"

# ------------------------------------------------------------
# 1. Link LLVM IR + ROCm device bitcode
# ------------------------------------------------------------

llvm-link "${LL_FILE}" \
    "${ROCM_BC}/ocml.bc" \
    "${ROCM_BC}/ockl.bc" \
    "${ROCM_BC}/oclc_isa_version_936.bc" \
    "${ROCM_BC}/oclc_abi_version_400.bc" \
    "${ROCM_BC}/oclc_wavefrontsize64_on.bc" \
    "${ROCM_BC}/oclc_correctly_rounded_sqrt_off.bc" \
    "${ROCM_BC}/oclc_finite_only_off.bc" \
    "${ROCM_BC}/oclc_unsafe_math_off.bc" \
    "${ROCM_BC}/oclc_daz_opt_off.bc" \
    -o merged.bc

# ------------------------------------------------------------
# 2. LLVM IR optimization
#
# 保留 debug metadata。
# 不要使用 strip-debug / strip-named-metadata。
# ------------------------------------------------------------

opt \
    -passes="${IR_OPT_PIPELINE}" \
    -debugger-tune=gdb \
    merged.bc \
    -o opt.bc

# 可选检查：
# llvm-dis opt.bc -o - | grep -E '!dbg|DIFile|DILocation' | head

# ------------------------------------------------------------
# 3. LLVM IR -> AMDGPU object
# ------------------------------------------------------------

llc \
    -mtriple=amdgcn-amd-amdhsa \
    -mcpu="${GPU_ARCH}" \
    "${LLVM_CODEGEN_OPT}" \
    -debugger-tune=gdb \
    -dwarf-version="${DWARF_VERSION}" \
    -filetype=obj \
    opt.bc \
    -o kernel.o

# ------------------------------------------------------------
# 4. Object -> HSACO
# ------------------------------------------------------------

ld.lld \
    -shared \
    --build-id=sha1 \
    kernel.o \
    -o "${KERNEL_FILE}"

# 可选检查 debug line：
#
# llvm-dwarfdump --debug-line kernel.o
# llvm-dwarfdump --debug-line "${KERNEL_FILE}"
# llvm-objdump -d --line-numbers "${KERNEL_FILE}" | less

echo
echo "---- Generated ${KERNEL_FILE} from ${LL_FILE}, Opt=O${OPT_LEVEL} ----"