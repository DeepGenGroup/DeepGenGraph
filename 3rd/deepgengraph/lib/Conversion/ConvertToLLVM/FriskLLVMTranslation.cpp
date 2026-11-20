#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
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
    if (attribute.getName() != "frisk.register_mma") return success();
    auto assembly = dyn_cast<LLVM::InlineAsmOp>(op);
    if (!assembly || !isa<UnitAttr>(attribute.getValue()) ||
        op->getNumResults() != 1)
      return op->emitError("frisk.register_mma requires register MMA inline asm");
    auto *call = llvm::dyn_cast_or_null<llvm::CallInst>(
        translation.lookupValue(op->getResult(0)));
    auto *code = call ? llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand()) : nullptr;
    auto &ctx = translation.getLLVMContext();
    auto *half4 = llvm::FixedVectorType::get(llvm::Type::getHalfTy(ctx), 4);
    auto *float4 = llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4);
    // Never trust a free-form marker to erase arbitrary assembly effects.
    if (!code || code->isAlignStack() || code->canThrow() ||
        call->getNumOperandBundles() || call->arg_size() != 3 ||
        !isRegisterMMA(code->getAsmString(), code->getConstraintString()) ||
        call->getType() != float4 || call->getArgOperand(0)->getType() != half4 ||
        call->getArgOperand(1)->getType() != half4 ||
        call->getArgOperand(2)->getType() != float4)
      return op->emitError("invalid frisk.register_mma contract");
    call->setCalledOperand(llvm::InlineAsm::get(code->getFunctionType(),
        code->getAsmString(), code->getConstraintString(), false, false,
        code->getDialect(), false));
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
