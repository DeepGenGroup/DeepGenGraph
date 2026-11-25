// Coalesce redundant synchronization only after named buffer readiness has
// become target wait-all operations. A wait on buffer A at Frisk Base level
// must never be assumed to complete a different buffer B.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace {
struct CoalesceWaitsAndBarriers
    : PassWrapper<CoalesceWaitsAndBarriers, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CoalesceWaitsAndBarriers)
  StringRef getArgument() const final {
    return "frisk-coalesce-waits-and-barriers";
  }
  StringRef getDescription() const final {
    return "Coalesce local wait-all/barrier sequences without crossing memory or control flow";
  }
  void runOnOperation() final {
    // State is local to a block. In particular it never crosses a branch,
    // a loop backedge, an async launch, a call, a fence or a memory access.
    getOperation().walk([&](Block *block) {
      bool completed = false, published = false;
      for (Operation &op : llvm::make_early_inc_range(*block)) {
        if (auto wait = dyn_cast<ROCDL::WaitAsyncmarkOp>(op);
            wait && wait.getCount() == 0) {
          if (completed) {
            op.erase();
          } else {
            completed = true;
            // A previously retained barrier need not publish the writes that
            // this wait has just completed. Keep the next barrier.
            published = false;
          }
          continue;
        }
        if (isa<gpu::BarrierOp>(op)) {
          if (published)
            op.erase();
          else
            published = true;
          continue;
        }
        // Be deliberately narrower than "no memory effects": convergent GPU
        // operations and opaque target intrinsics may change synchronization
        // state even when their effects are not described.
        StringRef dialect = op.getName().getDialectNamespace();
        // DMA lowering has already expanded descriptor/address arithmetic to
        // LLVM here. Such pure setup must not hide a duplicate handoff barrier.
        // Keep calls, target intrinsics, inline asm and memory/fences excluded.
        if (isa<LLVM::ConstantOp, LLVM::AddressOfOp, LLVM::ZeroOp,
                LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::ShlOp,
                LLVM::LShrOp, LLVM::AShrOp, LLVM::AndOp, LLVM::OrOp,
                LLVM::XOrOp, LLVM::ZExtOp, LLVM::SExtOp, LLVM::TruncOp,
                LLVM::PtrToIntOp, LLVM::IntToPtrOp, LLVM::GEPOp,
                LLVM::ICmpOp, LLVM::SelectOp, LLVM::ExtractValueOp,
                LLVM::InsertValueOp, LLVM::UndefOp, LLVM::PoisonOp,
                UnrealizedConversionCastOp>(op))
          continue;
        if ((dialect == "arith" || dialect == "index" ||
             dialect == "affine" || dialect == "memref" ||
             dialect == "vector") &&
            !op.getNumRegions() && !op.getNumSuccessors() &&
            !op.hasTrait<OpTrait::IsTerminator>() &&
            isMemoryEffectFree(&op) && isSpeculatable(&op))
          continue;
        completed = published = false;
      }
    });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::frisk::createCoalesceWaitsAndBarriersPass() {
  return std::make_unique<CoalesceWaitsAndBarriers>();
}
