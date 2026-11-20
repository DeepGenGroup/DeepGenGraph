#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  using namespace mlir;
  if (argc != 2 && argc != 3) return 2;
  DialectRegistry registry;
  registry.insert<frisk::FriskDialect, func::FuncDialect, arith::ArithDialect,
                  memref::MemRefDialect, LLVM::LLVMDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerROCDLDialectTranslation(registry);
  frisk::registerFriskLLVMTranslation(registry);
  MLIRContext context(registry);
  context.getOrLoadDialect<frisk::FriskDialect>();
  auto module = parseSourceFile<ModuleOp>(argv[1], &context);
  if (!module) return 1;
  if (argc == 3) {
    llvm::LLVMContext llvmContext;
    auto result = translateModuleToLLVMIR(*module, llvmContext);
    if (!result || llvm::verifyModule(*result, &llvm::errs())) return 1;
    result->print(llvm::outs(), nullptr);
    return 0;
  }
  module->walk([&](Operation *op) {
    auto id = op->getAttrOfType<StringAttr>("test.id");
    if (!id) return;
    auto s = frisk::getExecutionSemantics(op);
    llvm::outs() << id.getValue() << " scope=" << unsigned(s.scope)
                 << " kind=" << unsigned(s.kind)
                 << " registers=" << s.registerOnly
                 << " collective=" << s.collective
                 << " boundary=" << s.schedulingBoundary
                 << " memory_free=" << isMemoryEffectFree(op)
                 << " speculatable=" << isSpeculatable(op);
    if (auto interface = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> effects;
      interface.getEffects(effects);
      for (auto effect : effects) {
        llvm::outs() << " " << (isa<MemoryEffects::Read>(effect.getEffect()) ? "read" :
                               isa<MemoryEffects::Write>(effect.getEffect()) ? "write" : "other");
        if (Value value = effect.getValue()) {
          bool found = false;
          for (auto operand : llvm::enumerate(op->getOperands()))
            if (operand.value() == value) {
              llvm::outs() << "@" << operand.index(); found = true; break;
            }
          if (!found) llvm::outs() << "@result";
        } else llvm::outs() << "@unknown";
      }
    }
    llvm::outs() << '\n';
  });
  return 0;
}
