#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc == 3 && llvm::StringRef(argv[1]) == "--legalize-text") {
    auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(argv[2]);
    if (!buffer) {
      llvm::errs() << buffer.getError().message() << "\n";
      return 1;
    }
    std::string text = (*buffer)->getBuffer().str();
    mlir::frisk::legalizeLLVMTextForLegacyLLVM(text);
    llvm::outs() << text;
    return 0;
  }
  if (argc != 2) {
    llvm::errs() << "usage: LegacyLLVMExportTest [--legalize-text] <input.ll or ->\n";
    return 1;
  }
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseIRFile(argv[1], diagnostic, context);
  if (!module) {
    diagnostic.print(argv[0], llvm::errs());
    return 1;
  }
  mlir::frisk::prepareRegisterMMAForLLVM(*module);
  mlir::frisk::prepareSharedMemoryForLegacyLLVM(*module);
  if (llvm::verifyModule(*module, &llvm::errs()))
    return 1;
  module->print(llvm::outs(), nullptr);
  return 0;
}
