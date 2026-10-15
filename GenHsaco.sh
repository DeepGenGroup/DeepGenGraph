#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./GenHsaco.sh <input.ll> <opt_level> <output.hsaco>
#
# Examples:
#   ./GenHsaco.sh ./final.ll 0 ./kernel.hsaco
#   ./GenHsaco.sh ./final.ll 3 ./kernel.hsaco
# GPU_FEATURES is an explicit target capability override; use only features
# supported by the target hardware. It is forwarded to opt and both llc runs.
# The supplied gfx936 DTK still rejects HCU global DMA with
# +glb2lds-dword-x2x4; split32 exports need no feature override.
#
# opt_level:
#   0 -> opt default<O0> + llc -O0
#   1 -> opt default<O1> + llc -O1
#   2 -> opt default<O2> + llc -O2
#   3 -> opt default<O3> + llc -O3

LL_FILE=${1:-finalLLVMText.ll}
OPT_LEVEL=${2:-0}
KERNEL_FILE=${3:-kernel.hsaco}

ROCM_BC=${ROCM_BC:-/opt/dtk/amdgcn/bitcode}
GPU_ARCH=${GPU_ARCH:-gfx936}
GPU_FEATURES=${GPU_FEATURES:-}
GPU_FEATURE_ARGS=()
if [[ -n "${GPU_FEATURES}" ]]; then
    GPU_FEATURE_ARGS=("-mattr=${GPU_FEATURES}")
fi
DWARF_VERSION=${DWARF_VERSION:-4}
LLVM_DIR=${LLVM_DIR:-/workspace/deepgengraph/dtk/aillvm/bin}
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
echo " GPU features : ${GPU_FEATURES:-CPU defaults}"
echo " Output       : ${KERNEL_FILE}"
echo " Input SHA256 : $(sha256sum -- "${LL_FILE}" | cut -d ' ' -f 1)"
echo " LLVM tools   : ${LLVM_DIR}"
echo "========================================"
"${LLVM_DIR}/llc" --version

# ------------------------------------------------------------
# 1. Link LLVM IR + ROCm device bitcode
# ------------------------------------------------------------

${LLVM_DIR}/llvm-link "${LL_FILE}" \
    "${ROCM_BC}/ocml.bc" \
    "${ROCM_BC}/ockl.bc" \
    "${ROCM_BC}/oclc_isa_version_${GPU_ARCH#gfx}.bc" \
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

${LLVM_DIR}/opt \
    -mtriple=amdgcn-amd-amdhsa \
    -mcpu="${GPU_ARCH}" \
    "${GPU_FEATURE_ARGS[@]}" \
    -passes="${IR_OPT_PIPELINE}" \
    -debugger-tune=gdb \
    merged.bc \
    -o opt.bc

# 可选检查：
# llvm-dis opt.bc -o - | grep -E '!dbg|DIFile|DILocation' | head

# ------------------------------------------------------------
# 3. LLVM IR -> AMDGPU object
# ------------------------------------------------------------

# Keep ABI v4 for the older HIP runtime, including the assembly sidecar.
${LLVM_DIR}/llc \
    -mtriple=amdgcn-amd-amdhsa \
    -mcpu="${GPU_ARCH}" \
    "${GPU_FEATURE_ARGS[@]}" \
    "${LLVM_CODEGEN_OPT}" \
    -debugger-tune=gdb \
    -dwarf-version="${DWARF_VERSION}" \
    -filetype=obj \
    -amdhsa-code-object-version=4 \
    opt.bc \
    -o kernel.o

# ------------------------------------------------------------
# 4. Object -> HSACO
# ------------------------------------------------------------

${LLVM_DIR}/ld.lld \
    -shared \
    --build-id=sha1 \
    kernel.o \
    -o "${KERNEL_FILE}"

# 可选检查 debug line：
#
# llvm-dwarfdump --debug-line kernel.o
# llvm-dwarfdump --debug-line "${KERNEL_FILE}"
# llvm-objdump -d --line-numbers "${KERNEL_FILE}" | less

${LLVM_DIR}/llc -mtriple=amdgcn-amd-amdhsa -mcpu="${GPU_ARCH}" \
    "${GPU_FEATURE_ARGS[@]}" \
    "${LLVM_CODEGEN_OPT}" -debugger-tune=gdb \
    -dwarf-version="${DWARF_VERSION}" -filetype=asm \
    -print-after-all -filter-print-funcs=Attn_p2 \
    -amdhsa-code-object-version=4 \
    opt.bc -o "${KERNEL_FILE}.amdgcn" 2> passes.log
    
echo
echo "---- Generated ${KERNEL_FILE} and ${KERNEL_FILE}.amdgcn from ${LL_FILE}, Opt=O${OPT_LEVEL} ----"
sha256sum -- "${KERNEL_FILE}" "${KERNEL_FILE}.amdgcn"
