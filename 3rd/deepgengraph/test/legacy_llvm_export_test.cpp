#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  if (argc == 3 && (llvm::StringRef(argv[1]) == "--legalize-text" ||
                    llvm::StringRef(argv[1]) == "--llvm22-text")) {
    auto buffer = llvm::MemoryBuffer::getFileOrSTDIN(argv[2]);
    if (!buffer) {
      llvm::errs() << buffer.getError().message() << "\n";
      return 1;
    }
    std::string text = (*buffer)->getBuffer().str();
    if (llvm::StringRef(argv[1]) == "--llvm22-text")
      mlir::frisk::legalizeLLVMTextForLLVM22(text);
    else
      mlir::frisk::legalizeLLVMTextForLegacyLLVM(text);
    llvm::outs() << text;
    return 0;
  }
  bool approximate = argc == 3 && llvm::StringRef(argv[1]) == "--approx-register-math";
  bool fuseMMAC = argc == 3 && llvm::StringRef(argv[1]) == "--fuse-mmac-accumulator";
  bool lifetimes = argc == 3 && llvm::StringRef(argv[1]) == "--optimize-register-lifetimes";
  bool nativeExp2 = argc == 3 && llvm::StringRef(argv[1]) == "--native-exp2";
  bool reciprocal = argc == 3 && llvm::StringRef(argv[1]) == "--reciprocal";
  bool prefetch = argc == 3 && llvm::StringRef(argv[1]) == "--prefetch-global-packets";
  bool ldsReuse = argc == 3 && llvm::StringRef(argv[1]) == "--reuse-lds-fragments";
  bool guardDMA = argc == 3 && llvm::StringRef(argv[1]) == "--guard-dma-addresses";
  bool hcuDMA = argc == 3 && llvm::StringRef(argv[1]) == "--hcu-wide-dma";
  bool splitDMA = argc == 3 && llvm::StringRef(argv[1]) == "--split-full-wave-dma";
  bool asmDMA = argc == 3 && llvm::StringRef(argv[1]) == "--buffer-asm-dma";
  if (argc != 2 && !approximate && !fuseMMAC && !lifetimes && !nativeExp2 && !reciprocal && !prefetch && !ldsReuse && !guardDMA && !hcuDMA && !splitDMA && !asmDMA) {
    llvm::errs() << "usage: LegacyLLVMExportTest [--legalize-text|--approx-register-math|--fuse-mmac-accumulator|--optimize-register-lifetimes|--native-exp2|--reciprocal|--prefetch-global-packets|--reuse-lds-fragments|--guard-dma-addresses|--hcu-wide-dma] <input.ll or ->\n";
    return 1;
  }
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseIRFile(argv[argc == 3 ? 2 : 1], diagnostic, context);
  if (!module) {
    diagnostic.print(argv[0], llvm::errs());
    return 1;
  }
  mlir::frisk::prepareRegisterMMAForLLVM(*module);
  if (approximate)
    mlir::frisk::optimizeApproximateRegisterMath(*module);
  else if (fuseMMAC)
    mlir::frisk::fuseMMACAccumulators(*module);
  else if (lifetimes)
    mlir::frisk::optimizeRegisterLifetimes(*module);
  else if (nativeExp2 || reciprocal)
    mlir::frisk::optimizeRegisterMath(*module, nativeExp2, reciprocal);
  else if (prefetch)
    mlir::frisk::prefetchGlobalPackets(*module);
  else if (ldsReuse)
    mlir::frisk::reuseLDSFragments(*module);
  else if (guardDMA)
    mlir::frisk::protectAsyncDMASourceAddresses(*module);
  else if (splitDMA)
    mlir::frisk::splitFullWaveAsyncDMAToDwords(*module);
  else if (asmDMA)
    mlir::frisk::lowerWideAsyncDMAToBufferAsm(*module, 33554432);
  else if (hcuDMA) {
    mlir::frisk::protectAsyncDMASourceAddresses(*module);
    mlir::frisk::lowerWideAsyncDMAToHCU(*module);
  }
  mlir::frisk::prepareSharedMemoryForLegacyLLVM(*module);
  if (llvm::verifyModule(*module, &llvm::errs()))
    return 1;
  module->print(llvm::outs(), nullptr);
  return 0;
}
