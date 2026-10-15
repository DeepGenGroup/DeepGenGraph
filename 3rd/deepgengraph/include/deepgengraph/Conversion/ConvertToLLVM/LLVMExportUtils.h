#ifndef DEEPGENGRAPH_CONVERSION_CONVERTTOLLVM_LLVMEXPORTUTILS_H
#define DEEPGENGRAPH_CONVERSION_CONVERTTOLLVM_LLVMEXPORTUTILS_H

#include <string>

namespace llvm {
class Module;
}

namespace mlir::frisk {
/// Emit LLVM 24 floating-point literals in LLVM 22 syntax without changing
/// their bit patterns. Preserve LLVM 22 memory effects, flags and attributes.
void legalizeLLVMTextForLLVM22(std::string &text);

/// Apply legalizeLLVMText.py's memory-attribute mappings first, then the
/// existing legacy text rewrites. Unmatched memory effects are not stripped.
void legalizeLLVMTextForLegacyLLVM(std::string &text);

/// Restore precise effects for the exact register MMA emitted by Frisk.
/// Keep sideeffect, convergence, operands and hardware padding intact. Run
/// immediately after MLIR translation, before LLVM optimizations or printing.
void prepareRegisterMMAForLLVM(llvm::Module &module);

/// Pin each complete global-to-LDS DMA source address before final export.
/// Prevent backend source-offset folding from also shifting the LDS address.
/// Retains DMA and synchronization; safe to call repeatedly.
void protectAsyncDMASourceAddresses(llvm::Module &module);

/// Final DTK export: 16-byte GM->LDS packets use the HCU dwordx4 intrinsic.
/// Its last operands are offset, aux, bps (not byte count, offset, aux).
/// Run after LLVM optimizations; upstream LLVM does not know HCU intrinsics.
void lowerWideAsyncDMAToHCU(llvm::Module &module);

/// Compatibility export for the packed attention fixture: replace each 16B
/// packet with four 4B async instructions, preserving the physical LDS layout.
/// REQUIRES complete active wave64 waves and a wave-uniform LDS base. Do not
/// use on partial-wave tails. Only source addresses cross lanes; data stays DMA.
void splitFullWaveAsyncDMAToDwords(llvm::Module &module);

/// Experimental gfx9 buffer DMA export, following example.cpp's offen asm.
/// Sources must resolve to uniform kernel pointer arguments, and every source
/// tensor has sourceBytes bytes (<4GiB). Retains per-lane physical addresses.
/// Replaces wait.asyncmark(0) with an explicit VMEM wait for opaque asm loads.
void lowerWideAsyncDMAToBufferAsm(llvm::Module &module, unsigned sourceBytes);

/// Run the opt-in MMAC accumulator fusion pipeline after LLVM translation.
/// Reassociates MMAC accumulation only; retains OCML exp2 and exact division.
void fuseMMACAccumulators(llvm::Module &module);

/// Shorten divergent address and math/narrowing live ranges on AMDGPU.
/// Does not reassociate arithmetic or replace the math library implementation.
void optimizeRegisterLifetimes(llvm::Module &module);

/// Explicit approximate mode: reassociate single-use MMA accumulator chains,
/// use native f32 exp2 (flushes subnormals), and share reciprocal divisions.
/// Never enabled by the strict export path; no nnan/ninf assumptions are made.
void optimizeApproximateRegisterMath(llvm::Module &module);
/// Independently selectable approximation experiments, both include MMAC fusion.
void optimizeRegisterMath(llvm::Module &module, bool nativeExp2, bool reciprocal);
/// Hoist a bounded batch of global loads across pure exp2 work, before lifetime scheduling.
void prefetchGlobalPackets(llvm::Module &module, unsigned maxWords = 8);
/// Reuse identical vector LDS reads within a synchronization-free block phase.
/// Budget overlapping retained packets; do not move loads, calls or barriers.
void reuseLDSFragments(llvm::Module &module, unsigned maxWords = 8);

/// LLVM 15 LDS lowering can miss shared pointers hidden inside constant memref
/// descriptors. Split aggregate selects/PHIs into scalar joins, then expose
/// constant LDS references as instructions. Scalar joins prevent downstream
/// optimization from rebuilding descriptor selects with hidden LDS references.
/// Call after optimization and before exporting LLVM IR to the legacy backend.
void prepareSharedMemoryForLegacyLLVM(llvm::Module &module);
} // namespace mlir::frisk

#endif
