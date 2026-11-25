#include "deepgengraph/Conversion/ConvertToLLVM/SoftmaxCvtPack.h"
#include "deepgengraph/Conversion/ConvertToLLVM/PrefetchNextK.h"
#include "deepgengraph/Conversion/ConvertToLLVM/ScheduleSoftmaxFragments.h"
#include "deepgengraph/Conversion/ConvertToLLVM/PrefetchPairedLDS.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LocalReduction.h"
#include "deepgengraph/Conversion/ConvertToLLVM/DeferLaneSum.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

// Compatibility for old .ll experiments only. Production exp2 selection happens
// in MLIR lowering, before translation to LLVM IR.
static void replaceLegacyExp2(llvm::Module &module) {
  for (llvm::Function &function : module) {
    if (function.isDeclaration() || function.hasFnAttribute(llvm::Attribute::StrictFP) ||
        function.hasFnAttribute(llvm::Attribute::OptimizeNone)) continue;
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &inst : llvm::make_early_inc_range(block)) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        if (!call) continue;
        auto *callee = call->getCalledFunction();
        if (!callee || callee->getName() != "__ocml_exp2_f32" ||
            call->arg_size() != 1 || !call->getType()->isFloatTy() ||
            call->getNumOperandBundles()) continue;
        llvm::IRBuilder<> b(call);
        auto *native = llvm::Intrinsic::getOrInsertDeclaration(
            &module, llvm::Intrinsic::amdgcn_exp2, {b.getFloatTy()});
        call->replaceAllUsesWith(b.CreateCall(native, {call->getArgOperand(0)}));
        call->eraseFromParent();
      }
  }
}

int main(int argc, char **argv) {
  bool fragPrefetch = false;
  unsigned fragPrefetchWords = 8;
  bool crossBNK0 = false, crossBNK0Specified = false;
  bool softmaxCvtPack = false;
  llvm::StringRef localReduce;
  mlir::frisk::LocalReductionOptions localReductionOptions;
  llvm::StringRef deferLaneSum;
  mlir::frisk::DeferLaneSumOptions deferLaneSumOptions;
  llvm::SmallVector<char *> positional{argv[0]};
  for (int i = 1; i < argc; ++i) {
    llvm::StringRef option(argv[i]);
    bool consumed = false;
    if (option.starts_with("--softmax-fragments") ||
        option.starts_with("--k-lds-prefetch") ||
        option.starts_with("--k-tail-prefetch")) {
      llvm::errs() << "paired LDS/softmax scheduling moved to ThreadIR; use MyTest or ThreadTilingTest\n";
      return 1;
    }
    if (!mlir::frisk::parseCrossBNK0Option(option, crossBNK0, consumed)) {
      llvm::errs() << "use --cross-bn-k0=0|1\n"; return 1;
    }
    if (consumed) { crossBNK0Specified = true; continue; }
    if (option.starts_with("--elide-backedge-barrier")) {
      llvm::errs() << "backedge barrier elision moved to ThreadIR; use MyTest or ThreadTilingTest\n";
      return 1;
    }
    if (option.starts_with("--lazy-softmax-base")) {
      llvm::errs() << "lazy softmax base moved to ThreadIR; use MyTest or ThreadTilingTest\n";
      return 1;
    }
    if (!mlir::frisk::parseSoftmaxCvtPackOption(option, softmaxCvtPack, consumed)) {
      llvm::errs() << "use --softmax-cvt-pack=off|rtz (rtz changes rounding)\n"; return 1;
    }
    if (consumed) {
      llvm::errs() << "softmax cvt pack moved to ThreadIR; use MyTest or ThreadTilingTest\n";
      return 1;
    }
    if (option.starts_with("--k-dma")) {
      llvm::errs() << "DMA/layout planning moved to Frisk Base IR; use MyTest --shared-copy-constraints\n";
      return 1;
    }
    if (option.starts_with("--v-double-buffer")) {
      llvm::errs() << "V double buffering moved to Thread IR; use MyTest --v-double-buffer=1\n";
      return 1;
    }
    if (option.starts_with("--pv-interleave")) {
      llvm::errs() << "PV interleaving moved to Thread IR; use MyTest --pv-interleave=0|1|2|4\n";
      return 1;
    }
    if (!mlir::frisk::parseDeferLaneSumOption(option, deferLaneSum, deferLaneSumOptions, consumed)) {
      llvm::errs() << "invalid defer-lane-sum option: " << option << "\n";
      return 1;
    }
    if (consumed) continue;
    if (!mlir::frisk::parseLocalReductionOption(option, localReduce, localReductionOptions, consumed)) {
      llvm::errs() << "invalid local-reduce option: " << option << "\n";
      return 1;
    }
    if (consumed) continue;
    if (option.starts_with("--mask-add-bypass")) {
      llvm::errs() << "mask-add-bypass moved to ThreadIR; use MyTest --mask-add-bypass=off|approx or ThreadTilingTest --frisk-bypass-mask-adds\n";
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
  if (!localReduce.empty() || !deferLaneSum.empty() || crossBNK0Specified) {
    if (argc != 2 || fragPrefetch) {
      llvm::errs() << "usage: LegacyLLVMExportTest [--local-reduce=off|legal|approx] [--defer-lane-sum=off|legal|approx] [--defer-lane-sum-wave-size=0|32|64] [--cross-bn-k0=0|1] <input.ll>\n";
      return 1;
    }
    llvm::LLVMContext context;
    llvm::SMDiagnostic diagnostic;
    auto module = llvm::parseIRFile(argv[1], diagnostic, context);
    if (!module) {
      diagnostic.print(argv[0], llvm::errs());
      return 1;
    }
    if (!localReduce.empty() && localReduce != "off")
      mlir::frisk::optimizeLocalReductions(*module, localReductionOptions);
    if (!deferLaneSum.empty() && deferLaneSum != "off")
      mlir::frisk::deferLaneSums(*module, deferLaneSumOptions);
    mlir::frisk::prefetchNextK(*module, crossBNK0);
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
  bool contractsOnly = argc == 3 && llvm::StringRef(argv[1]) == "--export-contracts-only";
  bool cleanupOnly = argc == 3 && llvm::StringRef(argv[1]) == "--cleanup-llvm-only";
  bool cleanupMMA = argc == 3 && llvm::StringRef(argv[1]) == "--cleanup-register-mma";
  bool lifetimes = argc == 3 && llvm::StringRef(argv[1]) == "--optimize-register-lifetimes";
  bool scheduledLifetimes = argc == 3 && llvm::StringRef(argv[1]) == "--scheduled-register-lifetimes";
  lifetimes |= scheduledLifetimes;
  bool nativeExp2 = argc == 3 && llvm::StringRef(argv[1]) == "--native-exp2";
  bool reciprocal = argc == 3 && llvm::StringRef(argv[1]) == "--reciprocal";
  bool prefetch = argc == 3 && llvm::StringRef(argv[1]) == "--prefetch-global-packets";
  bool ldsReuse = argc == 3 && llvm::StringRef(argv[1]) == "--reuse-lds-fragments";
  bool guardDMA = argc == 3 && llvm::StringRef(argv[1]) == "--guard-dma-addresses";
  bool hcuDMA = argc == 3 && llvm::StringRef(argv[1]) == "--hcu-wide-dma";
  bool splitDMA = argc == 3 && llvm::StringRef(argv[1]) == "--split-full-wave-dma";
  bool asmDMA = argc == 3 && llvm::StringRef(argv[1]) == "--buffer-asm-dma";
  if (argc != 2 && !contractsOnly && !cleanupOnly && !ldsGeps && !ldsGepsOnly && !approximate && !cleanupMMA && !lifetimes && !nativeExp2 && !reciprocal && !prefetch && !ldsReuse && !guardDMA && !hcuDMA && !splitDMA && !asmDMA) {
    llvm::errs() << "usage: LegacyLLVMExportTest [--export-contracts-only|--cleanup-llvm-only|--canonicalize-lds-geps|--canonicalize-lds-geps-only|--legalize-text|--approx-register-math|--cleanup-register-mma|--optimize-register-lifetimes|--native-exp2|--reciprocal|--prefetch-global-packets|--reuse-lds-fragments|--guard-dma-addresses|--hcu-wide-dma] [--frag-prefetch=0|1] [--frag-prefetch-words=8] [--cross-bn-k0=0|1] <input.ll or ->\n";
    return 1;
  }
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  auto module = llvm::parseIRFile(argv[argc == 3 ? 2 : 1], diagnostic, context);
  if (!module) {
    diagnostic.print(argv[0], llvm::errs());
    return 1;
  }
  if (contractsOnly || cleanupOnly) {
    if (contractsOnly) mlir::frisk::prepareLLVMExportContracts(*module);
    else mlir::frisk::cleanupLLVMInstructions(*module, /*simplifyCFG=*/true);
    if (llvm::verifyModule(*module, &llvm::errs())) return 1;
    module->print(llvm::outs(), nullptr);
    return 0;
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
  mlir::frisk::prepareLLVMExportContracts(*module);
  if (approximate || nativeExp2)
    replaceLegacyExp2(*module);
  if (ldsGeps)
    mlir::frisk::canonicalizeLDSGEPs(*module);
  else if (approximate)
    mlir::frisk::optimizeApproximateRegisterMath(*module);
  else if (cleanupMMA)
    mlir::frisk::cleanupLLVMInstructions(*module, /*simplifyCFG=*/true);
  else if (lifetimes)
    mlir::frisk::optimizeRegisterLifetimes(*module, scheduledLifetimes);
  else if (nativeExp2 || reciprocal)
    mlir::frisk::optimizeRegisterMath(*module, reciprocal);
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
