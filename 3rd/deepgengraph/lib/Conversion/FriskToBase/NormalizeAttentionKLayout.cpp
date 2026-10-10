#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::frisk {
#define GEN_PASS_DEF_NORMALIZEATTENTIONKLAYOUT
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

namespace {
class NormalizeAttentionKLayout
    : public impl::NormalizeAttentionKLayoutBase<NormalizeAttentionKLayout> {
  void runOnOperation() override {
    auto module = getOperation();
    auto func = module.lookupSymbol<func::FuncOp>("Attn_p2");
    if (!func || func->hasAttr("frisk.k_layout_bhsd"))
      return;
    auto reject = [&](const Twine &message) {
      func.emitError(message);
      signalPassFailure();
    };
    if (func.isDeclaration() || func.getNumArguments() != 4 ||
        func.getNumResults() != 0) {
      reject("K layout normalization expects the four-argument Attn_p2 kernel");
      return;
    }
    auto qType = dyn_cast<MemRefType>(func.getArgument(0).getType());
    if (!qType || qType.getRank() != 4 || !qType.hasStaticShape()) {
      reject("K layout normalization expects a static rank-four Q");
      return;
    }
    SmallVector<int64_t> bhds(qType.getShape());
    std::swap(bhds[2], bhds[3]);
    int index = keyArg;
    if (index == -1) {
      if (bhds[2] == bhds[3]) {
        reject("S == D is ambiguous; specify key-arg=1 or key-arg=2 for BHDS K");
        return;
      }
      for (int i : {1, 2}) {
        auto type = dyn_cast<MemRefType>(func.getArgument(i).getType());
        if (type && type.getShape() == ArrayRef<int64_t>(bhds)) {
          if (index != -1) {
            reject("multiple BHDS candidates; specify key-arg explicitly");
            return;
          }
          index = i;
        }
      }
      if (index == -1) {
        // Already BHSD is a no-op only for the exact supported signature.
        if (func.getArgument(1).getType() == qType &&
            func.getArgument(2).getType() == qType &&
            func.getArgument(3).getType() == qType)
          return;
        reject("cannot identify BHDS K in the Attn_p2 signature");
        return;
      }
    }
    if (index != 1 && index != 2) {
      reject("key-arg must be 1 or 2");
      return;
    }
    auto key = func.getArgument(index);
    auto oldType = dyn_cast<MemRefType>(key.getType());
    if (!oldType || oldType.getShape() != ArrayRef<int64_t>(bhds) ||
        !oldType.getLayout().isIdentity() ||
        oldType.getElementType() != qType.getElementType() ||
        oldType.getMemorySpace() != qType.getMemorySpace() ||
        func.getArgument(3 - index).getType() != qType ||
        func.getArgument(3).getType() != qType) {
      reject("expected contiguous BHDS K and matching BHSD Q/V/O");
      return;
    }
    // This pass changes an externally launched kernel ABI, not call sites.
    if (!SymbolTable::symbolKnownUseEmpty(func, module)) {
      reject("cannot change Attn_p2 ABI while symbol uses exist in the module");
      return;
    }
    for (Operation *user : key.getUsers()) {
      if (!isa<frisk::BufferViewOp>(user)) {
        reject("K layout normalization currently supports buffer_view users only");
        return;
      }
    }
    auto newType = MemRefType::get(qType.getShape(), oldType.getElementType(),
                                  AffineMap(), oldType.getMemorySpace());
    SmallVector<Type> inputs(func.getFunctionType().getInputs());
    inputs[index] = newType;
    key.setType(newType);
    func.setType(FunctionType::get(&getContext(), inputs,
                                  func.getFunctionType().getResults()));
    OpBuilder builder(&func.front(), func.front().begin());
    auto permutation = AffineMap::getPermutationMap(
        ArrayRef<unsigned>{0, 1, 3, 2}, &getContext());
    // Only a strided view: there is no allocation or full-tensor transpose.
    auto transposed = builder.create<memref::TransposeOp>(
        func.getLoc(), key, AffineMapAttr::get(permutation));
    key.replaceAllUsesExcept(transposed.getResult(), transposed);
    func->setAttr("frisk.k_layout_bhsd", builder.getUnitAttr());
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createNormalizeAttentionKLayoutPass() {
  return std::make_unique<NormalizeAttentionKLayout>();
}
} // namespace mlir::frisk
