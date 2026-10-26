#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LocalReduction.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  bool fragPrefetch = false;
  unsigned fragPrefetchWords = 8;
  llvm::StringRef maskAddBypass;
  llvm::StringRef localReduce;
  mlir::frisk::LocalReductionOptions localReductionOptions;
  llvm::SmallVector<char *> positional{argv[0]};
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef option(argv[i]);
    bool consumed = false;
    if (!mlir::frisk::parseLocalReductionOption(option, localReduce, localReductionOptions, consumed)) {
      llvm::errs() << "invalid local-reduce option: " << option << "\n";
      return 1;
    }
    if (consumed) continue;
    if (option.consume_front("--mask-add-bypass=")) {
      if (option != "off" && option != "legal" && option != "approx") {
        llvm::errs() << "mask-add-bypass must be off, legal, or approx\n";
        return 1;
      }
      maskAddBypass = option;
    } else if (option.starts_with("--mask-add-bypass")) {
      llvm::errs() << "use --mask-add-bypass=off|legal|approx\n";
      return 1;
    } else if (option.consume_front("--frag-prefetch=")) {
      if (option != "0" && option != "1") {
        llvm::errs() << "frag-prefetch must be 0 or 1\n";
        return 1;
      }
      fragPrefetch = option == "1";
    } else if (option.consume_front("--frag-prefetch-words=")) {
      if (option.getAsInteger(10, fragPrefetchWords) || !fragPrefetchWords) {
        llvm::errs() << "frag-prefetch-words must be a positive integer\n";
        return 1;
      }
    } else if (option.starts_with("--frag-prefetch")) {
      llvm::errs() << "use --frag-prefetch=0|1 and --frag-prefetch-words=N\n";
      return 1;
    } else {
      positional.push_back(argv[i]);
    }
  }
  argc = positional.size();
  argv = positional.data();
  if (!localReductionOptions.valid()) {
    llvm::errs() << "local-reduce requires chains=2..16, min-inputs>=2*chains and <=2048\n";
    return 1;
  }
  // Exact standalone A/B path: do not prepare MMA attributes, change LDS
  // descriptors or run unrelated optimizations on an already-exported IR.
  if (!maskAddBypass.empty() || !localReduce.empty()) {
    if (argc != 2 || fragPrefetch) {
      llvm::errs() << "usage: LegacyLLVMExportTest [--mask-add-bypass=off|legal|approx] [--local-reduce=off|legal|approx] <input.ll>\n";
      return 1;
    }
    llvm::LLVMContext context;
    llvm::SMDiagnostic diagnostic;
    auto module = llvm::parseIRFile(argv[1], diagnostic, context);
    if (!module) {
      diagnostic.print(argv[0], llvm::errs());
      return 1;
    }
    if (!maskAddBypass.empty() && maskAddBypass != "off")
      mlir::frisk::bypassMaskAdds(*module, maskAddBypass == "approx");
    if (!localReduce.empty() && localReduce != "off")
      mlir::frisk::optimizeLocalReductions(*module, localReductionOptions);
    if (llvm::verifyModule(*module, &llvm::errs())) return 1;
    module->print(llvm::outs(), nullptr);
    return 0;
  }
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
  bool ldsGeps = argc == 3 && llvm::StringRef(argv[1]) == "--canonicalize-lds-geps";
  bool ldsGepsOnly = argc == 3 && llvm::StringRef(argv[1]) == "--canonicalize-lds-geps-only";
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
  if (argc != 2 && !ldsGeps && !ldsGepsOnly && !approximate && !fuseMMAC && !lifetimes && !nativeExp2 && !reciprocal && !prefetch && !ldsReuse && !guardDMA && !hcuDMA && !splitDMA && !asmDMA) {
    llvm::errs() << "usage: LegacyLLVMExportTest [--canonicalize-lds-geps|--canonicalize-lds-geps-only|--legalize-text|--approx-register-math|--fuse-mmac-accumulator|--optimize-register-lifetimes|--native-exp2|--reciprocal|--prefetch-global-packets|--reuse-lds-fragments|--guard-dma-addresses|--hcu-wide-dma] [--frag-prefetch=0|1] [--frag-prefetch-words=8] <input.ll or ->\n";
    return 1;
  }
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseIRFile(argv[argc == 3 ? 2 : 1], diagnostic, context);
  if (!module) {
    diagnostic.print(argv[0], llvm::errs());
    return 1;
  }
  // A standalone A/B entry point: do not alter MMAC attributes, shared-memory
  // constants or fragment scheduling as part of an LDS address experiment.
  if (ldsGepsOnly) {
    if (fragPrefetch) {
      llvm::errs() << "--canonicalize-lds-geps-only cannot enable fragment prefetch\n";
      return 1;
    }
    mlir::frisk::canonicalizeLDSGEPs(*module);
    if (llvm::verifyModule(*module, &llvm::errs())) return 1;
    module->print(llvm::outs(), nullptr);
    return 0;
  }
  mlir::frisk::prepareRegisterMMAForLLVM(*module);
  if (ldsGeps)
    mlir::frisk::canonicalizeLDSGEPs(*module);
  else if (approximate)
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
  if (fragPrefetch)
    mlir::frisk::prefetchMMACFragments(*module, fragPrefetchWords);
  mlir::frisk::prepareSharedMemoryForLegacyLLVM(*module);
  if (llvm::verifyModule(*module, &llvm::errs()))
    return 1;
  module->print(llvm::outs(), nullptr);
  return 0;
}
