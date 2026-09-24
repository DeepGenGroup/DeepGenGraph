#include "deepgengraph/Common.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#undef TID
#include <cassert>
#include <dlfcn.h>
#include <filesystem>
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include <optional>
#include <string>
#include <sstream>
#include <fstream>
#include <initializer_list>
#include <climits>
#include <cfloat>
#include "llvm/ADT/StringRef.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/MC/TargetRegistry.h"

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
            return isa<BranchInst, SwitchInst>(block->getTerminator());
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

static llvm::Value *materializeSplatConstant(llvm::Constant *constant,
                                             llvm::Instruction *insertBefore,
                                             llvm::Constant *zero,
                                             llvm::DebugLoc debugLoc) {
  if (!constant ||
      llvm::isa<llvm::ConstantAggregateZero, llvm::UndefValue,
                llvm::PoisonValue>(constant)) {
    return constant;
  }

  if (auto *vectorType =
          llvm::dyn_cast<llvm::FixedVectorType>(constant->getType())) {
    llvm::Constant *splatValue = constant->getSplatValue();
    if (!splatValue ||
        llvm::isa<llvm::UndefValue, llvm::PoisonValue>(splatValue)) {
      return constant;
    }

    auto *insert = llvm::InsertElementInst::Create(
        llvm::PoisonValue::get(vectorType), splatValue, zero,
        "compat.splat.insert", insertBefore);
    insert->setDebugLoc(debugLoc);
    llvm::SmallVector<int, 16> mask(vectorType->getNumElements(), 0);
    auto *shuffle = new llvm::ShuffleVectorInst(
        insert, llvm::PoisonValue::get(vectorType), mask, "compat.splat",
        insertBefore);
    shuffle->setDebugLoc(debugLoc);
    return shuffle;
  }

  if (!constant->getType()->isAggregateType() || constant->getNumOperands() == 0) {
    return constant;
  }

  bool changed = false;
  llvm::SmallVector<llvm::Value *, 8> elements;
  elements.reserve(constant->getNumOperands());
  for (llvm::Use &operand : constant->operands()) {
    auto *elementConstant = llvm::dyn_cast<llvm::Constant>(operand.get());
    if (!elementConstant) {
      return constant;
    }
    llvm::Value *element =
        materializeSplatConstant(elementConstant, insertBefore, zero, debugLoc);
    changed |= element != elementConstant;
    elements.push_back(element);
  }
  if (!changed) {
    return constant;
  }

  llvm::Value *aggregate = llvm::PoisonValue::get(constant->getType());
  for (auto [idx, element] : llvm::enumerate(elements)) {
    auto *insert = llvm::InsertValueInst::Create(
        aggregate, element, {static_cast<unsigned>(idx)},
        "compat.splat.aggregate", insertBefore);
    insert->setDebugLoc(debugLoc);
    aggregate = insert;
  }
  return aggregate;
}

static void materializeSplatVectorConstants(Module &module) {
  LLVMContext &context = module.getContext();
  llvm::Type *i32Type = llvm::Type::getInt32Ty(context);
  llvm::Constant *zero = llvm::ConstantInt::get(i32Type, 0);

  for (llvm::Function &function : module) {
    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &inst : llvm::make_early_inc_range(block)) {
        if (auto *phi = llvm::dyn_cast<llvm::PHINode>(&inst)) {
          for (unsigned idx = 0, e = phi->getNumIncomingValues(); idx < e;
               ++idx) {
            auto *constant =
                llvm::dyn_cast<llvm::Constant>(phi->getIncomingValue(idx));
            if (!constant) {
              continue;
            }
            llvm::Instruction *insertBefore =
                phi->getIncomingBlock(idx)->getTerminator();
            llvm::Value *materialized = materializeSplatConstant(
                constant, insertBefore, zero, phi->getDebugLoc());
            if (materialized != constant) {
              phi->setIncomingValue(idx, materialized);
            }
          }
          continue;
        }

        for (unsigned idx = 0, e = inst.getNumOperands(); idx < e; ++idx) {
          auto *constant =
              llvm::dyn_cast<llvm::Constant>(inst.getOperand(idx));
          if (!constant) {
            continue;
          }
          llvm::Value *materialized =
              materializeSplatConstant(constant, &inst, zero, inst.getDebugLoc());
          if (materialized != constant) {
            inst.setOperand(idx, materialized);
          }
        }
      }
    }
  }
}

static void replaceAll(std::string &text, StringRef from, StringRef to) {
  size_t pos = 0;
  while ((pos = text.find(from.str(), pos)) != std::string::npos) {
    text.replace(pos, from.size(), to.str());
    pos += to.size();
  }
}

static void printLegacyCompatibleLLVMIR(Module &module, raw_ostream &os) {
  prepareSharedMemoryForLegacyLLVM(module);

  std::string text;
  raw_string_ostream buffer(text);
  module.print(buffer, nullptr);
  buffer.flush();

  // Keep the textual IR parseable by older llvm-link builds used downstream.
  // Shared-memory views can fold into constant-expression GEPs. Older parsers
  // accept inbounds, but not the newer GEP nuw/nusw flags. Drop only GEP flags;
  // integer arithmetic no-wrap flags and the address calculation stay intact.
  replaceAll(text, "getelementptr inbounds nuw ", "getelementptr inbounds ");
  replaceAll(text, "getelementptr nusw ", "getelementptr ");
  replaceAll(text, "getelementptr nuw ", "getelementptr ");
  replaceAll(text, " captures(none)", "");
  replaceAll(text, " memory(none)", "");
  replaceAll(text, " memory(argmem: read)", "");
  replaceAll(text, " memory(argmem: write)", "");
  replaceAll(text, " memory(argmem: readwrite)", "");

  os << text;
}

static std::optional<OptimizationLevel> mapToLevel(unsigned optLevel, unsigned sizeLevel) {
  switch (optLevel) {
  case 0:
    return OptimizationLevel::O0;
  case 1:
    return OptimizationLevel::O1;
  case 2:
    switch (sizeLevel) {
    case 0:
      return OptimizationLevel::O2;
    case 1:
      return OptimizationLevel::Os;
    case 2:
      return OptimizationLevel::Oz;
    }
    break;
  case 3:
    return OptimizationLevel::O3;
  }
  return std::nullopt;
}

static std::function<Error(Module *)> makeOptimizingPipeline(unsigned optLevel, unsigned sizeLevel, TargetMachine *targetMachine) {
  return [optLevel, sizeLevel, targetMachine](Module *m) -> Error {
    std::optional<OptimizationLevel> ol = mapToLevel(optLevel, sizeLevel);
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

  auto optPipeline = makeOptimizingPipeline(/*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return "";
  }  
  materializeSplatVectorConstants(*llvmModule);
  std::string str;
  llvm::raw_string_ostream os(str);
  printLegacyCompatibleLLVMIR(*llvmModule, os);
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
