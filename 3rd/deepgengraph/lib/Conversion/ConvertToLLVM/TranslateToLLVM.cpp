#include "deepgengraph/Common.h"
#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Analysis/ValueTracking.h"
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
            (dma->getCalledFunction()->getName() != "llvm.amdgcn.global.load.async.lds" &&
             dma->getCalledFunction()->getName() != "llvm.hcu.global.load.dwordx4.lds"))
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

void lowerWideAsyncDMAToHCU(llvm::Module &module) {
  auto *generic = module.getFunction("llvm.amdgcn.global.load.async.lds");
  if (!generic) return;
  SmallVector<llvm::CallInst *> wide;
  for (llvm::User *user : generic->users())
    if (auto *call = llvm::dyn_cast<llvm::CallInst>(user))
      if (auto *bytes = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2));
          bytes && bytes->equalsInt(16))
        wide.push_back(call);
  if (wide.empty()) return;
  // IntrinsicsHCU.td / HCUGlobalLoadLDS: (global AS1, LDS AS3,
  // imm offset, aux/cpol, bps). Width is encoded in the intrinsic name.
  // bps=0 selects the ordinary M0-based, non-swizzled LDS packet layout.
  auto hcu = module.getOrInsertFunction("llvm.hcu.global.load.dwordx4.lds",
                                       generic->getFunctionType());
  auto *declaration = llvm::cast<llvm::Function>(hcu.getCallee());
  declaration->addFnAttr(llvm::Attribute::Convergent);
  declaration->addFnAttr(llvm::Attribute::NoUnwind);
  declaration->addFnAttr(llvm::Attribute::WillReturn);
  for (auto *call : wide) {
    llvm::Value *offset = call->getArgOperand(3);
    llvm::Value *aux = call->getArgOperand(4);
    call->setCalledFunction(hcu);
    // Do not inherit upstream intrinsic memory attributes: HCU performs a
    // global read AND an LDS write, even in an upstream-only verifier build.
    call->setAttributes(llvm::AttributeList{});
    call->addFnAttr(llvm::Attribute::Convergent);
    call->setArgOperand(2, offset);
    call->setArgOperand(3, aux);
    call->setArgOperand(4, llvm::ConstantInt::get(offset->getType(), 0));
  }
  if (generic->use_empty()) generic->eraseFromParent();
}

void lowerWideAsyncDMAToBufferAsm(llvm::Module &module, unsigned sourceBytes) {
  auto *generic = module.getFunction("llvm.amdgcn.global.load.async.lds");
  if (!generic) return;
  if (!sourceBytes) llvm::report_fatal_error("buffer DMA requires a nonzero tensor byte range");
  SmallVector<llvm::CallInst *> copies;
  for (llvm::User *user : generic->users())
    if (auto *call = llvm::dyn_cast<llvm::CallInst>(user)) copies.push_back(call);
  SmallVector<llvm::Function *> functions;
  for (auto *call : copies) {
    auto *width = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2));
    auto *offset = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(3));
    auto *aux = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(4));
    auto *source = call->getArgOperand(0);
    auto *base = llvm::dyn_cast<llvm::Argument>(llvm::getUnderlyingObject(source, 32));
    if (!width || !width->equalsInt(16) || !offset || !offset->isZero() ||
        !aux || !aux->isZero() || !base ||
        call->getFunction()->getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL)
      llvm::report_fatal_error("asm128 requires 16B/zero-control DMA from a kernel tensor argument");
    if (!llvm::is_contained(functions, call->getFunction())) functions.push_back(call->getFunction());
    llvm::IRBuilder<> b(call);
    b.SetCurrentDebugLocation(call->getDebugLoc());
    auto *i32 = b.getInt32Ty();
    auto *v4i32 = llvm::FixedVectorType::get(i32, 4);
    auto *baseBits = b.CreatePtrToInt(base, b.getInt64Ty());
    auto *srcBits = b.CreatePtrToInt(source, b.getInt64Ty());
    auto *byteOffset = b.CreateTrunc(b.CreateSub(srcBits, baseBits), i32, "dma.buffer.offset");
    llvm::Value *rsrc = llvm::PoisonValue::get(v4i32);
    rsrc = b.CreateInsertElement(rsrc, b.CreateTrunc(baseBits, i32), uint64_t(0));
    rsrc = b.CreateInsertElement(rsrc, b.CreateTrunc(b.CreateLShr(baseBits, 32), i32), 1);
    rsrc = b.CreateInsertElement(rsrc, b.getInt32(sourceBytes), 2);
    // CK's gfx9 raw byte-addressed buffer descriptor: stride=0, no swizzle.
    rsrc = b.CreateInsertElement(rsrc, b.getInt32(0x00020000), 3, "dma.buffer.rsrc");
    auto *readFirst = llvm::Intrinsic::getOrInsertDeclaration(&module,
        llvm::Intrinsic::amdgcn_readfirstlane, {i32});
    auto *lds = b.CreateCall(readFirst,
        {b.CreatePtrToInt(call->getArgOperand(1), i32)}, "dma.buffer.lds");
    // M0 is an explicit output (its post-asm value is the LDS base), not an
    // opaque reserved-register clobber. The unused result still describes the
    // register definition to codegen and avoids the reserved clobber warning.
    auto *type = llvm::FunctionType::get(i32, {i32, i32, v4i32}, false);
    auto *assembly = llvm::InlineAsm::get(type,
        "s_mov_b32 $0, $1\n\tbuffer_load_dwordx4 $2, $3, 0 offen lds",
        "={m0},s,v,s,~{memory}", true);
    auto *issue = b.CreateCall(type, assembly, {lds, byteOffset, rsrc});
    issue->addFnAttr(llvm::Attribute::Convergent);
    issue->setDoesNotThrow();
    call->eraseFromParent();
  }
  // Inline asm is opaque to asyncmark bookkeeping. This path only accepts
  // drain-all waits, and emits an explicit hardware wait before publication.
  for (auto *function : functions) {
    SmallVector<llvm::CallInst *> markers;
    for (auto &block : *function)
      for (auto &inst : block)
        if (auto *call = llvm::dyn_cast<llvm::CallInst>(&inst))
          if (auto *callee = call->getCalledFunction(); callee &&
              (callee->getName() == "llvm.amdgcn.asyncmark" ||
               callee->getName() == "llvm.amdgcn.wait.asyncmark")) markers.push_back(call);
    for (auto *call : markers) {
      if (call->arg_size()) {
        auto *count = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(0));
        if (!count || !count->isZero()) llvm::report_fatal_error("asm128 requires wait.asyncmark(0)");
        llvm::IRBuilder<> b(call);
        b.SetCurrentDebugLocation(call->getDebugLoc());
        auto *type = llvm::FunctionType::get(b.getVoidTy(), false);
        auto *wait = b.CreateCall(type, llvm::InlineAsm::get(type,
            "s_waitcnt vmcnt(0)", "~{memory}", true));
        wait->addFnAttr(llvm::Attribute::Convergent);
        wait->setDoesNotThrow();
      }
      call->eraseFromParent();
    }
  }
  if (generic->use_empty()) generic->eraseFromParent();
}

void splitFullWaveAsyncDMAToDwords(llvm::Module &module) {
  auto *generic = module.getFunction("llvm.amdgcn.global.load.async.lds");
  if (!generic) return;
  SmallVector<llvm::CallInst *> wide;
  for (llvm::User *user : generic->users())
    if (auto *call = llvm::dyn_cast<llvm::CallInst>(user))
      if (auto *bytes = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(2));
          bytes && bytes->equalsInt(16))
        wide.push_back(call);
  if (wide.empty()) return;
  auto *loCount = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::amdgcn_mbcnt_lo);
  auto *hiCount = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::amdgcn_mbcnt_hi);
  auto *permute = llvm::Intrinsic::getOrInsertDeclaration(&module, llvm::Intrinsic::amdgcn_ds_bpermute);
  for (auto *call : wide) {
    // The generated attention copies have zero immediate offset. Fold any
    // source displacement into the address before using this transformation.
    auto *offset = llvm::dyn_cast<llvm::ConstantInt>(call->getArgOperand(3));
    if (!offset || !offset->isZero())
      llvm::report_fatal_error("split DMA requires zero immediate offset");
    llvm::IRBuilder<> b(call);
    b.SetCurrentDebugLocation(call->getDebugLoc());
    auto *lowLane = b.CreateCall(loCount, {b.getInt32(-1), b.getInt32(0)});
    auto *lane = b.CreateCall(hiCount, {b.getInt32(-1), lowLane}, "dma.lane");
    auto *bits = b.CreatePtrToInt(call->getArgOperand(0), b.getInt64Ty());
    auto *low = b.CreateTrunc(bits, b.getInt32Ty());
    auto *high = b.CreateTrunc(b.CreateLShr(bits, 32), b.getInt32Ty());
    auto *word = b.CreateShl(b.CreateAnd(lane, b.getInt32(3)), 2);
    for (unsigned part = 0; part < 4; ++part) {
      // Original destination = base + 16*sourceLane + 4*word.
      // Instruction part stores at base + 256*part + 4*lane, hence
      // sourceLane = 16*part + lane/4, word = lane%4.
      auto *index = b.CreateAdd(b.CreateAnd(lane, b.getInt32(60)), b.getInt32(64 * part));
      auto *srcLow = b.CreateCall(permute, {index, low});
      auto *srcHigh = b.CreateCall(permute, {index, high});
      auto *address = b.CreateOr(b.CreateZExt(srcLow, b.getInt64Ty()),
          b.CreateShl(b.CreateZExt(srcHigh, b.getInt64Ty()), 32));
      address = b.CreateAdd(address, b.CreateZExt(word, b.getInt64Ty()));
      auto *src = b.CreateIntToPtr(address, call->getArgOperand(0)->getType());
      auto *dst = b.CreateGEP(b.getInt8Ty(), call->getArgOperand(1), b.getInt32(part * 256));
      b.CreateCall(generic, {src, dst, b.getInt32(4), b.getInt32(0), call->getArgOperand(4)});
    }
    call->eraseFromParent();
  }
  // Guard the final shuffled source plus subword offset, not just the original
  // wide pointer. Otherwise backend offset folding can shift LDS as well.
  protectAsyncDMASourceAddresses(module);
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
  lowerWideAsyncDMAToHCU(module);
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
