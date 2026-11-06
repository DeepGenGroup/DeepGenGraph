// Materialize the measured noncausal_cvt_pack/lazy_t8_pk_rtz experiment.
// Two v_cvt_pkrtz_f16_f32 instructions replace a four-float fptrunc + pack.
// This is an APPROXIMATE transformation (RNE -> RTZ), enabled explicitly by
// --softmax-cvt-pack=rtz. The denominator and final output conversion stay RNE.
// Keep both lifetime anchors, all scheduling hints and the MMAC operands/order.
// Use the measured inline asm form, not a new intrinsic/backend selection path.
#include "deepgengraph/Conversion/ConvertToLLVM/SoftmaxCvtPack.h"
#include "LateIRPatterns.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::frisk {
namespace {
constexpr llvm::StringLiteral Marker = "frisk.softmax_cvt_pack";

llvm::CallInst *identity(llvm::Value *value) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(value);
  if (!call || call->arg_size() != 1 ||
      !call->getMetadata(detail::RegisterLifetimesMarker) ||
      !call->doesNotAccessMemory() || call->getType() != call->getArgOperand(0)->getType())
    return nullptr;
  auto *asmOp = detail::getInlineAsm(call);
  return asmOp && asmOp->hasSideEffects() && asmOp->getAsmString().empty() &&
                 asmOp->getConstraintString() == "=v,0" ? call : nullptr;
}

bool match(llvm::FPTruncInst &cast) {
  auto &ctx = cast.getContext();
  if (cast.getSrcTy() != llvm::FixedVectorType::get(llvm::Type::getFloatTy(ctx), 4) ||
      cast.getDestTy() != llvm::FixedVectorType::get(llvm::Type::getHalfTy(ctx), 4) ||
      !cast.hasOneUse())
    return false;
  auto *result = identity(cast.getOperand(0));
  auto *packed = identity(*cast.user_begin());
  if (!result || !packed || packed->use_empty()) return false;
  for (llvm::Use &use : packed->uses()) {
    auto *mmac = llvm::dyn_cast<llvm::CallInst>(use.getUser());
    if (!detail::isTypedRegisterMMACall(mmac) || &use != &mmac->getArgOperandUse(0))
      return false;
  }
  // Prove this is the exp2 softmax fragment, independently of SSA names and
  // lazy-base subtraction/phi structure. Arbitrary half stores/Q/K/V casts
  // and the epilogue do not qualify for the rounding change.
  llvm::Value *value = result->getArgOperand(0);
  for (unsigned lane = 4; lane-- > 0;) {
    auto *insert = llvm::dyn_cast<llvm::InsertElementInst>(value);
    if (!insert) return false;
    auto *index = llvm::dyn_cast<llvm::ConstantInt>(insert->getOperand(2));
    auto *exp = llvm::dyn_cast<llvm::IntrinsicInst>(insert->getOperand(1));
    if (!index || index->getZExtValue() != lane || !exp ||
        exp->getIntrinsicID() != llvm::Intrinsic::amdgcn_exp2 || !exp->getType()->isFloatTy())
      return false;
    value = insert->getOperand(0);
  }
  return llvm::isa<llvm::UndefValue>(value); // Includes poison.
}

void rewrite(llvm::FPTruncInst &cast) {
  llvm::IRBuilder<> builder(&cast);
  builder.SetCurrentDebugLocation(cast.getDebugLoc());
  std::string stem = (cast.getName() + ".cvtpack").str();
  llvm::Value *elements[4];
  for (unsigned lane = 0; lane < 4; ++lane)
    elements[lane] = builder.CreateExtractElement(cast.getOperand(0), builder.getInt32(lane),
                                                  stem + ".f" + std::to_string(lane));
  auto *ty = llvm::FunctionType::get(builder.getInt32Ty(),
                                    {builder.getFloatTy(), builder.getFloatTy()}, false);
  auto *asmOp = llvm::InlineAsm::get(ty, "v_cvt_pkrtz_f16_f32 $0, $1, $2", "=v,v,v", false);
  llvm::Value *pairs[2];
  for (unsigned pair = 0; pair < 2; ++pair)
    pairs[pair] = builder.CreateCall(ty, asmOp, {elements[2*pair], elements[2*pair+1]},
                                     stem + ".p" + std::to_string(pair));
  llvm::Value *packed = llvm::PoisonValue::get(llvm::FixedVectorType::get(builder.getInt32Ty(), 2));
  packed = builder.CreateInsertElement(packed, pairs[0], builder.getInt32(0), stem + ".v0");
  packed = builder.CreateInsertElement(packed, pairs[1], builder.getInt32(1), stem + ".v1");
  auto *replacement = llvm::cast<llvm::Instruction>(builder.CreateBitCast(packed, cast.getDestTy()));
  replacement->takeName(&cast);
  replacement->copyMetadata(cast);
  cast.replaceAllUsesWith(replacement);
  cast.eraseFromParent();
}
} // namespace

bool parseSoftmaxCvtPackOption(llvm::StringRef arg, bool &enabled, bool &consumed) {
  consumed = arg.starts_with("--softmax-cvt-pack");
  if (!consumed) return true;
  if (!arg.consume_front("--softmax-cvt-pack=")) return false;
  if (arg != "off" && arg != "rtz") return false;
  enabled = arg == "rtz";
  return true;
}

llvm::PreservedAnalyses SoftmaxCvtPackPass::run(llvm::Function &function,
                                              llvm::FunctionAnalysisManager &) {
  if (function.isDeclaration() || function.hasFnAttribute(Marker) ||
      function.getCallingConv() != llvm::CallingConv::AMDGPU_KERNEL ||
      function.hasFnAttribute(llvm::Attribute::OptimizeNone) ||
      function.hasFnAttribute(llvm::Attribute::StrictFP))
    return llvm::PreservedAnalyses::all();
  llvm::SmallVector<llvm::FPTruncInst *, 16> casts;
  for (auto &block : function)
    for (auto &inst : block)
      if (auto *cast = llvm::dyn_cast<llvm::FPTruncInst>(&inst); cast && match(*cast))
        casts.push_back(cast);
  if (casts.empty()) return llvm::PreservedAnalyses::all();
  for (auto *cast : casts) rewrite(*cast);
  function.addFnAttr(Marker, "rtz");
  llvm::errs() << "Softmax cvt pack: fragments=" << casts.size()
               << ", packed-pairs=" << 2*casts.size() << ", rounding=rtz, function="
               << function.getName() << "\n";
  return llvm::PreservedAnalyses::none();
}

void softmaxCvtPack(llvm::Module &module, bool enabled) {
  if (!enabled) return;
  llvm::FunctionAnalysisManager analyses;
  SoftmaxCvtPackPass pass;
  for (auto &function : module) pass.run(function, analyses);
}
} // namespace mlir::frisk
