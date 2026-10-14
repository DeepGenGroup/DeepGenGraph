#include "deepgengraph/Common.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "llvm/IR/InlineAsm.h"
#undef TID
#include <cassert>
#include <cctype>
#include <dlfcn.h>
#include <filesystem>
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include <optional>
#include <regex>
#include <string>
#include <sstream>
#include <fstream>
#include <initializer_list>
#include <climits>
#include <cfloat>
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::frisk{
using namespace llvm;

namespace {
bool containsSharedPointer(llvm::Type *type) {
  if (auto *pointer = dyn_cast<PointerType>(type))
    return pointer->getAddressSpace() == 3;
  if (auto *structure = dyn_cast<StructType>(type))
    return llvm::any_of(structure->elements(), containsSharedPointer);
  if (auto *array = dyn_cast<ArrayType>(type))
    return containsSharedPointer(array->getElementType());
  return false;
}

unsigned aggregateSize(llvm::Type *type) {
  if (auto *structure = dyn_cast<StructType>(type))
    return structure->getNumElements();
  return cast<ArrayType>(type)->getNumElements();
}

llvm::Type *aggregateElement(llvm::Type *type, unsigned index) {
  if (auto *structure = dyn_cast<StructType>(type))
    return structure->getElementType(index);
  return cast<ArrayType>(type)->getElementType();
}

llvm::Value *splitDescriptorSelect(llvm::IRBuilder<> &builder,
                                   llvm::Value *condition, llvm::Value *lhs,
                                   llvm::Value *rhs) {
  llvm::Type *type = lhs->getType();
  if (!type->isAggregateType())
    return builder.CreateSelect(condition, lhs, rhs, "lds.select");
  llvm::Value *result = PoisonValue::get(type);
  for (unsigned i = 0, e = aggregateSize(type); i != e; ++i) {
    llvm::Value *selected = splitDescriptorSelect(
        builder, condition, builder.CreateExtractValue(lhs, {i}),
        builder.CreateExtractValue(rhs, {i}));
    result = builder.CreateInsertValue(result, selected, {i});
  }
  return result;
}

llvm::Value *splitDescriptorPhi(PHINode *phi, llvm::Type *type,
                                SmallVector<unsigned> &indices,
                                llvm::IRBuilder<> &builder) {
  if (type->isAggregateType()) {
    llvm::Value *result = PoisonValue::get(type);
    for (unsigned i = 0, e = aggregateSize(type); i != e; ++i) {
      indices.push_back(i);
      llvm::Value *field = splitDescriptorPhi(
          phi, aggregateElement(type, i), indices, builder);
      indices.pop_back();
      result = builder.CreateInsertValue(result, field, {i});
    }
    return result;
  }
  auto *field = PHINode::Create(type, phi->getNumIncomingValues(), "lds.phi",
                                phi->getIterator());
  field->setDebugLoc(phi->getDebugLoc());
  for (unsigned i = 0, e = phi->getNumIncomingValues(); i != e; ++i) {
    BasicBlock *predecessor = phi->getIncomingBlock(i);
    llvm::IRBuilder<> edge(predecessor->getTerminator());
    edge.SetCurrentDebugLocation(phi->getDebugLoc());
    field->addIncoming(edge.CreateExtractValue(phi->getIncomingValue(i), indices),
                       predecessor);
  }
  return field;
}
} // namespace

void prepareRegisterMMAForLLVM(llvm::Module &module) {
  auto *half4 = llvm::FixedVectorType::get(llvm::Type::getHalfTy(module.getContext()), 4);
  auto *float4 = llvm::FixedVectorType::get(llvm::Type::getFloatTy(module.getContext()), 4);
  for (llvm::Function &function : module)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &inst : block) {
        auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
        if (!call || call->getNumOperandBundles() || call->arg_size() != 3)
          continue;
        auto *assembly = llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand());
        if (!assembly || !assembly->hasSideEffects() || assembly->isAlignStack() ||
            !isPaddedRegisterMMA(assembly->getAsmString(), assembly->getConstraintString()) ||
            call->getType() != float4 || call->getArgOperand(0)->getType() != half4 ||
            call->getArgOperand(1)->getType() != half4 ||
            call->getArgOperand(2)->getType() != float4)
          continue;
        call->setMemoryEffects(llvm::MemoryEffects::none());
        // All lanes participate; no speculative motion across control flow.
        call->addFnAttr(llvm::Attribute::Convergent);
      }
}

void protectAsyncDMASourceAddresses(llvm::Module &module) {
  // The global_load_* lds immediate applies to BOTH global and LDS addresses.
  // Source-only constant-offset folding in affected backends leaves M0
  // uncompensated. Pin the final source, rather than an intermediate base.
  // The tied empty asm preserves all pointer bits and keeps the DMA visible
  // to the backend's memory/synchronization analysis.
  for (llvm::Function &function : module)
    for (llvm::BasicBlock &block : function)
      for (llvm::Instruction &inst : block) {
        auto *dma = llvm::dyn_cast<llvm::CallInst>(&inst);
        if (!dma || !dma->getCalledFunction() ||
            dma->getCalledFunction()->getName() != "llvm.amdgcn.global.load.async.lds")
          continue;
        llvm::Value *source = dma->getArgOperand(0);
        // Recognize the structural identity as well as our own exports, so
        // importing a previously guarded diagnostic does not add another one.
        if (auto *pointer = llvm::dyn_cast<llvm::IntToPtrInst>(source))
          if (auto *call = llvm::dyn_cast<llvm::CallInst>(pointer->getOperand(0)))
            if (auto *assembly = llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand());
                assembly && assembly->hasSideEffects() &&
                assembly->getAsmString().empty() &&
                assembly->getConstraintString() == "=v,0" &&
                call->getType()->isIntegerTy(64) && call->arg_size() == 1)
              continue;
        llvm::IRBuilder<> builder(dma);
        builder.SetCurrentDebugLocation(dma->getDebugLoc());
        auto *type = builder.getInt64Ty();
        auto *bits = builder.CreatePtrToInt(source, type, "dma.source.bits");
        auto *signature = llvm::FunctionType::get(type, {type}, false);
        auto *identity = llvm::InlineAsm::get(signature, "", "=v,0", true);
        auto *guard = builder.CreateCall(signature, identity, {bits}, "dma.source.opaque");
        guard->setDoesNotAccessMemory();
        guard->setDoesNotThrow();
        guard->addFnAttr(llvm::Attribute::Convergent);
        dma->setArgOperand(0, builder.CreateIntToPtr(guard, source->getType(), "dma.source.ptr"));
      }
}

void prepareSharedMemoryForLegacyLLVM(llvm::Module &module) {
  SmallVector<Instruction *> joins;
  for (Function &function : module)
    for (BasicBlock &block : function)
      for (Instruction &inst : block)
        if (isa<SelectInst, PHINode>(inst) && inst.getType()->isAggregateType() &&
            containsSharedPointer(inst.getType()))
          joins.push_back(&inst);
  for (Instruction *join : joins) {
    llvm::IRBuilder<> builder(join);
    builder.SetCurrentDebugLocation(join->getDebugLoc());
    llvm::Value *replacement;
    if (auto *select = dyn_cast<SelectInst>(join)) {
      replacement = splitDescriptorSelect(builder, select->getCondition(),
                                           select->getTrueValue(), select->getFalseValue());
    } else {
      auto *phi = cast<PHINode>(join);
      // An invoke result is only available on its normal edge; splitting that
      // edge needs an EH-aware transform. Frisk emits ordinary branch edges.
      if (!llvm::all_of(phi->blocks(), [](BasicBlock *block) {
            return isa<UncondBrInst, CondBrInst, SwitchInst>(block->getTerminator());
          }))
        continue;
      builder.SetInsertPoint(phi->getParent(), phi->getParent()->getFirstInsertionPt());
      SmallVector<unsigned> indices;
      replacement = splitDescriptorPhi(phi, phi->getType(), indices, builder);
    }
    join->replaceAllUsesWith(replacement);
    join->eraseFromParent();
  }
  SmallVector<Constant *> sharedGlobals;
  for (GlobalVariable &global : module.globals())
    if (global.getAddressSpace() == 3)
      sharedGlobals.push_back(&global);
  convertUsersOfConstantsToInstructions(sharedGlobals);
}

static void replaceAll(std::string &text, StringRef from, StringRef to) {
  size_t pos = 0;
  while ((pos = text.find(from.str(), pos)) != std::string::npos) {
    text.replace(pos, from.size(), to.str());
    pos += to.size();
  }
}

void legalizeLLVMTextForLLVM22(std::string &text) {
  // LLVM 24 prints decimal literals rounded at the declared precision, plus
  // f0x/inf/nan spellings. LLVM 22 requires exact hexadecimal constants.
  // Track the last scalar FP type on each line, including vector element
  // types, since binary instructions do not repeat the second operand's type.
  // Skip strings/comments so assembly, names and debug info remain unchanged.
  static const std::regex literal(
      R"re(("(?:[^"\\]|\\.)*"|;[^\n]*|\n)|\b(half|bfloat|float|double)\b|(f0x[0-9A-Fa-f]+|[+-]?(?:[0-9]+\.[0-9]*(?:[eE][+-]?[0-9]+)?|inf|qnan|s?nan\(0x[0-9A-Fa-f]+\))))re");
  llvm::LLVMContext context;
  llvm::Module module("llvm22-float-constants", context);
  std::string result;
  std::string type;
  size_t position = 0;
  for (std::sregex_iterator it(text.begin(), text.end(), literal), end;
       it != end; ++it) {
    const auto &match = *it;
    if (match[1].matched) {
      if (match.str() == "\n")
        type.clear();
      continue;
    }
    auto isNameChar = [](unsigned char c) {
      return std::isalnum(c) || c == '_' || c == '.' || c == '$' ||
             c == '-' || c == '%' || c == '@' || c == '!';
    };
    size_t begin = match.position(), tokenEnd = begin + match.length();
    if ((begin && isNameChar(text[begin - 1])) ||
        (tokenEnd < text.size() && isNameChar(text[tokenEnd])))
      continue;
    if (match[2].matched) {
      type = match.str();
      continue;
    }
    if (type.empty())
      continue;
    llvm::SMDiagnostic diagnostic;
    auto *constant = llvm::dyn_cast_or_null<llvm::ConstantFP>(
        llvm::parseConstantValue(type + " " + match.str(), diagnostic, module));
    if (!constant)
      continue;
    llvm::APFloat value = constant->getValueAPF();
    llvm::APInt bits = value.bitcastToAPInt();
    if (type == "float") {
      if (value.isNaN()) {
        // APFloat::convert quiets signaling NaNs. Widen the encoding directly
        // so the sign, signaling bit and payload survive the textual roundtrip.
        uint64_t raw = bits.getZExtValue();
        bits = llvm::APInt(64, ((raw & 0x80000000ULL) << 32) |
                                  0x7ff0000000000000ULL |
                                  ((raw & 0x007fffffULL) << 29));
      } else {
        bool losesInfo;
        value.convert(llvm::APFloat::IEEEdouble(),
                      llvm::APFloat::rmNearestTiesToEven, &losesInfo);
        bits = value.bitcastToAPInt();
      }
    }
    llvm::SmallString<32> hex;
    bits.toStringUnsigned(hex, 16);
    result.append(text, position, match.position() - position);
    result += "0x";
    if (type == "half") result += 'H';
    if (type == "bfloat") result += 'R';
    result.append((bits.getBitWidth() + 3) / 4 - hex.size(), '0');
    result.append(hex.data(), hex.size());
    position = match.position() + match.length();
  }
  result.append(text, position, std::string::npos);
  text = std::move(result);
}

void legalizeLLVMTextForLegacyLLVM(std::string &text) {
  // Keep these ordered mappings in sync with legalizeLLVMText.py. Preserve
  // representable memory effects before applying the historical fallback.
  static const std::pair<std::regex, const char *> mappings[] = {
      {std::regex(R"(memory\(none\))"), "readnone"},
      {std::regex(R"(memory\(read\))"), "readonly"},
      {std::regex(R"(memory\(write\))"), "writeonly"},
      {std::regex(R"(memory\(argmem:\s*readwrite\))"), "argmemonly"},
      {std::regex(R"(memory\(argmem:\s*read\))"), "argmemonly readonly"},
      {std::regex(R"(memory\(argmem:\s*write\))"), "argmemonly writeonly"},
      {std::regex(R"(memory\(inaccessiblemem:\s*readwrite\))"), "inaccessiblememonly"},
      {std::regex(R"(memory\(inaccessiblemem:\s*read\))"), "inaccessiblememonly readonly"},
      {std::regex(R"(memory\(inaccessiblemem:\s*write\))"), "inaccessiblememonly writeonly"},
      {std::regex(R"(memory\(argmem:\s*readwrite,\s*inaccessiblemem:\s*readwrite\))"),
       "inaccessiblemem_or_argmemonly"},
  };
  for (const auto &[pattern, replacement] : mappings)
    text = std::regex_replace(text, pattern, replacement);

  // Keep the textual IR parseable by older llvm-link builds used downstream.
  // Shared-memory views can fold into constant-expression GEPs. Older parsers
  // accept inbounds, but not the newer GEP nuw/nusw flags. Drop only GEP flags;
  // integer arithmetic no-wrap flags and the address calculation stay intact.
  replaceAll(text, "getelementptr inbounds nuw ", "getelementptr inbounds ");
  replaceAll(text, "getelementptr nusw ", "getelementptr ");
  replaceAll(text, "getelementptr nuw ", "getelementptr ");
  // InstCombine after MMAC fusion may infer disjoint OR operands. The flag
  // is newer than LLVM 15; dropping the promise preserves the OR result.
  replaceAll(text, "or disjoint ", "or ");
  replaceAll(text, "zext nneg ", "zext ");
  replaceAll(text, "icmp samesign ", "icmp ");
  replaceAll(text, " captures(none)", "");
  replaceAll(text, " memory(none)", " readnone");
  replaceAll(text, " memory(argmem: read)", "");
  replaceAll(text, " memory(argmem: write)", "");
  replaceAll(text, " memory(argmem: readwrite)", "");
}

static void printLLVM22CompatibleIR(Module &module, raw_ostream &os) {
  prepareRegisterMMAForLLVM(module);
  protectAsyncDMASourceAddresses(module);
  std::string text;
  raw_string_ostream buffer(text);
  module.print(buffer, nullptr);
  buffer.flush();
  legalizeLLVMTextForLLVM22(text);

  os << text;
}

static std::optional<OptimizationLevel> mapToLevel(unsigned optLevel) {
  switch (optLevel) {
  case 0:
    return OptimizationLevel::O0;
  case 1:
    return OptimizationLevel::O1;
  case 2:
    return OptimizationLevel::O2;
  case 3:
    return OptimizationLevel::O3;
  }
  return std::nullopt;
}

static std::function<Error(Module *)> makeOptimizingPipeline(unsigned optLevel, TargetMachine *targetMachine) {
  return [optLevel, targetMachine](Module *m) -> Error {
    std::optional<OptimizationLevel> ol = mapToLevel(optLevel);
    if (!ol) {
      assert(false);
    }
    LoopAnalysisManager lam;
    FunctionAnalysisManager fam;
    CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;

    PipelineTuningOptions tuningOptions;
    tuningOptions.LoopUnrolling = true;
    tuningOptions.LoopInterleaving = true;
    tuningOptions.LoopVectorization = true;
    tuningOptions.SLPVectorization = true;

    PassBuilder pb(targetMachine, tuningOptions);

    // std::string pluginFile = KernelCodeGen::tools::getenv("AMDGCN_INSTRUMENTATION_LIB");
    // if (!pluginFile.empty()) {
    //     llvm::errs() << "Adding AMDGCN instrumentation pass to pipeline" << "\n";
    //     auto passPlugin = llvm::PassPlugin::Load(pluginFile);
    //     if (!passPlugin) {
    //             llvm::Error Err = passPlugin.takeError();
    //             llvm::errs() << "ERROR: " << Err << "\n";
    //             consumeError(std::move(Err));
    //     }
    //     passPlugin->registerPassBuilderCallbacks(pb);
    // }

    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    ModulePassManager mpm;
    pb.registerVectorizerStartEPCallback(
        [&](llvm::FunctionPassManager &fpm, llvm::OptimizationLevel level) {
          fpm.addPass(InstCombinePass());
        });
    mpm.addPass(pb.buildPerModuleDefaultPipeline(*ol));
    mpm.run(*m, mam);
    return Error::success();
  };
}

std::string translateMLIRToLLVMIR(mlir::ModuleOp module, Target target, const int wavesPerEU) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::DLTIDialect, mlir::func::FuncDialect>();
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerNVVMDialectTranslation(registry);
  registerROCDLDialectTranslation(registry);
  module.getContext()->appendDialectRegistry(registry);

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);

  if (!llvmModule) {
    llvm::errs() << "Failed to emit LLVM IR\n";
    return "";
  }
  
//   auto externLibs = getExternLibs(module);
//   for (auto &lib : externLibs) {
//     if (linkExternLib(*llvmModule, lib.first, lib.second, target))
//       return nullptr;
//   }

  prepareRegisterMMAForLLVM(*llvmModule);
  auto optPipeline = makeOptimizingPipeline(/*optLevel=*/3, /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return "";
  }  
  std::string str;
  llvm::raw_string_ostream os(str);
  printLLVM22CompatibleIR(*llvmModule, os);
  os.flush();
  return str;
}


std::string translate(mlir::ModuleOp& mod, mlir::frisk::Target target) {
#if 1
  if (target == mlir::frisk::Target::ROCm) {
    const int wavesPerEU = 0;
    return translateMLIRToLLVMIR(mod, target, wavesPerEU);
  }
  if(target == mlir::frisk::Target::CUDA){
    return translateMLIRToLLVMIR(mod, target, /*wavesPerEU=*/0);
  }
  return "-";
#endif

#if 0  // 外部导入 mlir llvm
  mlir::MLIRContext testContext;
  testContext.loadDialect<
    func::FuncDialect,memref::MemRefDialect,scf::SCFDialect,gpu::GPUDialect, NVVM::NVVMDialect, 
    arith::ArithDialect,cf::ControlFlowDialect,LLVM::LLVMDialect,ROCDL::ROCDLDialect
  >();
  const char* llvmdialectfileName = "/home/xiebaokang/projects/mymlir/DeepGen/_tmp/our.mlir";
  auto temp = mlir::parseSourceFile<ModuleOp>(llvmdialectfileName,&testContext);
  auto testmod = temp.get();
  std::string llvmIR = std::move(translateMLIRToLLVMIR(testmod, target, 0));
  llvm::outs() << "======================llvm ir\n" << llvmIR << "\n";
  // const int capability = 80;
  // const int version = 81;
  // auto paths = generatePTXAndCubinFromLLIRFile(llvmIR, capability, version);
  // return paths.second;
#endif

#if 0  // 外部导入 llvm ir
  std::ifstream ifs("/home/xiebaokang/projects/mymlir/DeepGen/_tmp/our.llvm");
  std::stringstream buffer;
  if(ifs.is_open()){
    buffer << ifs.rdbuf();
    ifs.close();
  }
  auto llvmIR = buffer.str();
  // llvm::outs() << "======================llvm ir\n" << llvmIR << "\n";
  const int capability = 80;
  const int version = 81;
  auto paths = generatePTXAndCubinFromLLIRFile(llvmIR, capability, version);
  return paths.second;
#endif

}

} // namespace mlir::frisk
