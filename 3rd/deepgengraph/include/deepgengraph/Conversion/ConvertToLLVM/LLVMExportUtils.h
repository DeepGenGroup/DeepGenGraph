#ifndef DEEPGENGRAPH_CONVERSION_CONVERTTOLLVM_LLVMEXPORTUTILS_H
#define DEEPGENGRAPH_CONVERSION_CONVERTTOLLVM_LLVMEXPORTUTILS_H

#include <string>

namespace llvm {
class Module;
}
namespace mlir { class DialectRegistry; }

namespace mlir::frisk {
/// Translate validated register-MMA and convert_pack lifetime-anchor contracts
/// on LLVM inline asm at export time. Lifetime anchors retain sideeffect.
void registerFriskLLVMTranslation(mlir::DialectRegistry &registry);
/// Emit LLVM 24 floating-point literals in LLVM 22 syntax without changing
/// their bit patterns. Preserve LLVM 22 memory effects, flags and attributes.
void legalizeLLVMTextForLLVM22(std::string &text);

/// Apply legalizeLLVMText.py's memory-attribute mappings first, then the
/// existing legacy text rewrites. Unmatched memory effects are not stripped.
void legalizeLLVMTextForLegacyLLVM(std::string &text);

/// Complete export contracts without folding instructions or changing the CFG.
/// Validate legacy register MMA, remove its sideeffect and restore convergent,
/// memory(none), nounwind and willreturn. Also annotate empty tied-register
/// lifetime anchors, retaining their sideeffect. Compatibility for LLVM-only
/// input or external translators without the Frisk export hook. Idempotent;
/// registered MLIR export already prepares both MMA and lifetime anchors.
void prepareLLVMExportContracts(llvm::Module &module);
/// Compatibility alias for prepareLLVMExportContracts.
void prepareRegisterMMAForLLVM(llvm::Module &module);

/// Fold duplicate export-created DMA descriptors and address expressions.
/// EarlyCSE/InstCombine cleanup only; no fragment or LDS scheduling.
void canonicalizeLLVMExport(llvm::Module &module);

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
// With reuseLeaderRead, an LDS base that is already a readfirstlane is passed
// straight to M0 instead of being read a second time at the DMA.
void lowerWideAsyncDMAToBufferAsm(llvm::Module &module, unsigned sourceBytes, bool reuseLeaderRead = false);

/// Generic instruction cleanup; does not recognize MMA or restore contracts.
/// Runs two InstCombine passes, optionally preceded by SimplifyCFG after late
/// thread-bound/lifetime rewrites. Early cleanup preserves the existing CFG.
/// Requires contracts supplied by registered MLIR export or, for LLVM-only
/// inputs, a separate prepareLLVMExportContracts call before optimization.
void cleanupLLVMInstructions(llvm::Module &module, bool simplifyCFG = false);

/// Compatibility wrapper: prepareLLVMExportContracts + cleanupLLVMInstructions.
void cleanupRegisterMMA(llvm::Module &module, bool simplifyCFG = false);

/// Shorten address, integer-predicate and math/narrowing live ranges on AMDGPU.
/// Repeated i64 predicates with common upper words share those tests and use
/// local i32 low-word operands, preserving the complete signed/unsigned range.
/// Does not reassociate FP arithmetic or replace the math implementation.
void optimizeRegisterLifetimes(llvm::Module &module);

/// Expose only proven constant LDS pointer differences; does not force read2.
void canonicalizeLDSGEPs(llvm::Module &module);

/// Explicit approximate mode: share reciprocal divisions.
/// Native f32 exp2 is selected during MLIR lowering.
/// Never enabled by the strict export path; no nnan/ninf assumptions are made.
void optimizeApproximateRegisterMath(llvm::Module &module);
/// Optional reciprocal approximation, including register MMA cleanup.
void optimizeRegisterMath(llvm::Module &module, bool reciprocal);
/// Hoist a bounded batch of global loads across pure exp2 work, before lifetime scheduling.
void prefetchGlobalPackets(llvm::Module &module, unsigned maxWords = 8);
/// Reuse identical vector LDS reads within a synchronization-free block phase.
/// Budget overlapping retained packets; do not move loads, calls or barriers.
void reuseLDSFragments(llvm::Module &module, unsigned maxWords = 8);

/// Opt-in one-MMAC lookahead for complete A/B LDS operand packets. Run after
/// lifetime/reuse optimization. Preserve SSA reuse, arithmetic, MMAC order and
/// synchronization. Bound raw packet words and the local SSA peak increase
/// (including address temporaries); this is not a physical VGPR allocation cap.
/// Scheduling barriers are compiler hints, never waits. Machine wait insertion
/// remains responsible for dependencies after instruction selection/allocation.
/// Requires contracts from registered MLIR export or prepareLLVMExportContracts.
void prefetchMMACFragments(llvm::Module &module, unsigned maxExtraWords = 8);

/// LLVM 15 LDS lowering can miss shared pointers hidden inside constant memref
/// descriptors. Split aggregate selects/PHIs into scalar joins, then expose
/// constant LDS references as instructions. Scalar joins prevent downstream
/// optimization from rebuilding descriptor selects with hidden LDS references.
/// Call after optimization and before exporting LLVM IR to the legacy backend.
void prepareSharedMemoryForLegacyLLVM(llvm::Module &module);
} // namespace mlir::frisk

#endif
