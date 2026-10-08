#ifndef DEEPGENGRAPH_CONVERSION_CONVERTTOLLVM_LLVMEXPORTUTILS_H
#define DEEPGENGRAPH_CONVERSION_CONVERTTOLLVM_LLVMEXPORTUTILS_H

#include <string>

namespace llvm {
class Module;
}

namespace mlir::frisk {
/// Apply legalizeLLVMText.py's memory-attribute mappings first, then the
/// existing legacy text rewrites. Unmatched memory effects are not stripped.
void legalizeLLVMTextForLegacyLLVM(std::string &text);

/// Restore precise effects for the exact register MMA emitted by Frisk.
/// Keep sideeffect, convergence, operands and hardware padding intact. Run
/// immediately after MLIR translation, before LLVM optimizations or printing.
void prepareRegisterMMAForLLVM(llvm::Module &module);

/// LLVM 15 LDS lowering can miss shared pointers hidden inside constant memref
/// descriptors. Split aggregate selects/PHIs into scalar joins, then expose
/// constant LDS references as instructions. Scalar joins prevent downstream
/// optimization from rebuilding descriptor selects with hidden LDS references.
/// Call after optimization and before exporting LLVM IR to the legacy backend.
void prepareSharedMemoryForLegacyLLVM(llvm::Module &module);
} // namespace mlir::frisk

#endif
