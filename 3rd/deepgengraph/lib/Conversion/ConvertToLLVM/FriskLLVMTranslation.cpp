#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "LateIRPatterns.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"

namespace mlir::frisk {
namespace {
class FriskLLVMTranslation : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;
  LogicalResult amendOperation(Operation *op,
      ArrayRef<llvm::Instruction *> instructions, NamedAttribute attribute,
      LLVM::ModuleTranslation &translation) const override {
    bool isAnchor = attribute.getName() == "frisk.register_anchor";
    if (!isAnchor && attribute.getName() != "frisk.register_mma") return success();
    auto assembly = dyn_cast<LLVM::InlineAsmOp>(op);
    if (!assembly || !isa<UnitAttr>(attribute.getValue()) ||
        op->getNumResults() != 1)
      return op->emitError() << attribute.getName()
                             << " requires register inline asm with a unit marker";
    auto *call = llvm::dyn_cast_or_null<llvm::CallInst>(
        translation.lookupValue(op->getResult(0)));
    auto *code = call ? llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand()) : nullptr;
    auto &ctx = translation.getLLVMContext();
    auto *half4 = llvm::FixedVectorType::get(llvm::Type::getHalfTy(ctx), 4);
    auto *float4 = llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4);
    // A marker alone cannot justify erasing memory effects. Validate the
    // generated LLVM call, including its exact register ABI and constraints.
    if (!code || code->isAlignStack() || code->canThrow() ||
        call->getNumOperandBundles())
      return op->emitError() << "invalid " << attribute.getName() << " contract";
    if (isAnchor) {
      if (!code->hasSideEffects() || !code->getAsmString().empty() ||
          code->getConstraintString() != "=v,0" || call->arg_size() != 1 ||
          call->getType() != call->getArgOperand(0)->getType() ||
          (call->getType() != half4 && call->getType() != float4))
        return op->emitError("invalid frisk.register_anchor contract");
      // Keep sideeffect: this identity pins the two convert_pack boundaries.
      call->setMetadata(detail::RegisterLifetimesMarker,
                        llvm::MDNode::get(ctx, {}));
    } else {
      if (call->arg_size() != 3 ||
          !isRegisterMMA(code->getAsmString(), code->getConstraintString()) ||
          call->getType() != float4 || call->getArgOperand(0)->getType() != half4 ||
          call->getArgOperand(1)->getType() != half4 ||
          call->getArgOperand(2)->getType() != float4)
        return op->emitError("invalid frisk.register_mma contract");
      call->setCalledOperand(llvm::InlineAsm::get(code->getFunctionType(),
          code->getAsmString(), code->getConstraintString(), false, false,
          code->getDialect(), false));
    }
    call->setMemoryEffects(llvm::MemoryEffects::none());
    call->setDoesNotThrow();
    call->addFnAttr(llvm::Attribute::WillReturn);
    call->addFnAttr(llvm::Attribute::Convergent);
    return success();
  }
};
} // namespace

void registerFriskLLVMTranslation(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *, FriskDialect *dialect) {
    dialect->addInterfaces<FriskLLVMTranslation>();
  });
}
} // namespace mlir::frisk
