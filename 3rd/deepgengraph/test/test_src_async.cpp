#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

// Export just this standalone kernel; no Frisk/attention pipeline is needed.
int main(int argc, char **argv) {
  bool legacy = argc > 1 && llvm::StringRef(argv[1]) == "--legacy-lds";
  int first = legacy ? 2 : 1;
  if (argc != first + 2) {
    llvm::errs() << "usage: MyTestAsync [--legacy-lds] input.mlir output.ll\n";
    return 1;
  }
  if (!legacy && !llvm::Intrinsic::lookupIntrinsicID(
                     "llvm.amdgcn.global.load.async.lds")) {
    llvm::errs() << "This LLVM has no explicit async LDS intrinsics. Use a newer "
                    "LLVM or --legacy-lds (compiler-managed waits).\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::LLVM::LLVMDialect, mlir::ROCDL::ROCDLDialect>();
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerROCDLDialectTranslation(registry);
  mlir::MLIRContext context(registry);
  // Only legacy mode permits unknown ops; replace them before export.
  context.allowUnregisteredDialects(legacy);
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[first], &context);
  if (!module)
    return 1;

  if (legacy) {
    llvm::SmallVector<mlir::Operation *> ops;
    module->walk([&](mlir::Operation *op) {
      auto name = op->getName().getStringRef();
      if (name == "rocdl.global.load.async.lds" || name == "rocdl.asyncmark" ||
          name == "rocdl.wait.asyncmark")
        ops.push_back(op);
    });
    for (auto *op : ops) {
      mlir::OpBuilder builder(op);
      auto name = op->getName().getStringRef();
      if (name == "rocdl.global.load.async.lds") {
        if (op->getNumOperands() != 2 || op->getNumResults() != 0) {
          op->emitError("expected two pointers and no results");
          return 1;
        }
        llvm::SmallVector<mlir::Value> args(op->getOperands());
        for (auto key : {"size", "offset", "aux"}) {
          auto attr = op->getAttrOfType<mlir::IntegerAttr>(key);
          if (!attr || !attr.getType().isInteger(32)) {
            op->emitError("expected i32 immediate attribute: ") << key;
            return 1;
          }
          args.push_back(builder.create<mlir::LLVM::ConstantOp>(
              op->getLoc(), builder.getI32Type(), attr));
        }
        // Avoid the old/new ROCDL immediate operand API split.
        builder.create<mlir::LLVM::CallIntrinsicOp>(
            op->getLoc(), mlir::TypeRange{}, "llvm.amdgcn.global.load.lds",
            args, mlir::LLVM::FastmathFlags::none,
            llvm::ArrayRef<mlir::ValueRange>{}, nullptr, nullptr, nullptr);
      } else if (name == "rocdl.wait.asyncmark") {
        auto count = op->getAttrOfType<mlir::IntegerAttr>("count");
        if (!count || count.getInt() != 0) {
          op->emitError("legacy compatibility supports only wait.asyncmark 0");
          return 1;
        }
        builder.create<mlir::ROCDL::SWaitcntOp>(op->getLoc(), 0u);
      }
      op->erase();
    }
    llvm::errs() << "Using legacy global.load.lds with compiler-managed waits.\n";
  }
  if (mlir::failed(mlir::verify(*module)))
    return 1;
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule || llvm::verifyModule(*llvmModule, &llvm::errs()))
    return 1;
  std::error_code error;
  llvm::ToolOutputFile output(argv[first + 1], error, llvm::sys::fs::OF_Text);
  if (error) {
    llvm::errs() << error.message() << '\n';
    return 1;
  }
  llvmModule->print(output.os(), nullptr);
  output.keep();
  return 0;
}
