#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"

namespace mlir::frisk {
namespace {
bool isExp2(llvm::Instruction &inst) {
  auto *call = llvm::dyn_cast<llvm::CallInst>(&inst);
  auto *fn = call ? call->getCalledFunction() : nullptr;
  return fn && call->arg_size() == 1 && call->getType()->isFloatTy() &&
      call->getArgOperand(0)->getType()->isFloatTy() && !call->getNumOperandBundles() &&
      (fn->getName() == "__ocml_exp2_f32" || fn->getName() == "llvm.amdgcn.exp2.f32" ||
       fn->getName() == "llvm.exp2.f32");
}

bool isRegisterExchange(llvm::CallBase *call) {
  auto *intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(call);
  // bpermute exchanges registers, not LDS memory. The call itself stays in
  // place and the CFG/active lanes do not change when a global load crosses it.
  return intrinsic && intrinsic->getIntrinsicID() == llvm::Intrinsic::amdgcn_ds_bpermute;
}

bool collectAddress(llvm::Value *value, llvm::Instruction *point,
                    llvm::SmallPtrSetImpl<llvm::Instruction *> &seen,
                    llvm::SmallVectorImpl<llvm::Instruction *> &ordered) {
  auto *inst = llvm::dyn_cast<llvm::Instruction>(value);
  if (!inst || inst->getParent() != point->getParent() || inst->comesBefore(point)) return true;
  if (inst == point || seen.size() >= 96) return false;
  if (!seen.insert(inst).second) return true;
  if ((!inst->getType()->isIntegerTy() && !inst->getType()->isPointerTy()) ||
      !llvm::isa<llvm::BinaryOperator, llvm::CastInst, llvm::GetElementPtrInst,
                 llvm::SelectInst, llvm::ICmpInst>(inst) ||
      !llvm::isSafeToSpeculativelyExecute(inst)) return false;
  for (auto *operand : inst->operand_values())
    if (!collectAddress(operand, point, seen, ordered)) return false;
  ordered.push_back(inst);
  return true;
}
} // namespace

// Software prefetch into ordinary registers. Stay in one basic block and stop
// at synchronization/unknown effects. LDS writes cannot alias global packets.
// The first publication barrier bounds the batch to one shared-memory phase;
// no double-buffer allocation or cross-iteration speculation is introduced.
void prefetchGlobalPackets(llvm::Module &module, unsigned maxWords) {
  unsigned packets = 0;
  for (auto &function : module) {
    if (function.isDeclaration() || function.hasFnAttribute(llvm::Attribute::StrictFP) ||
        function.hasFnAttribute(llvm::Attribute::OptimizeNone) ||
        function.hasFnAttribute("frisk.global_prefetch")) continue;
    const auto &triple = module.getTargetTriple();
    if (!triple.isAMDGPU() && !(triple.str().empty() &&
        function.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL)) continue;
    for (auto &block : function) {
      llvm::Instruction *point = nullptr;
      llvm::SmallVector<llvm::LoadInst *> loads;
      unsigned words = 0;
      for (auto &inst : block) {
        if (!point) { if (isExp2(inst)) point = &inst; continue; }
        if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst)) {
          if (!load->isSimple()) break;
          if (load->getPointerAddressSpace() != 1) continue;
          auto bits = function.getDataLayout().getTypeSizeInBits(load->getType());
          if (bits.isScalable() || bits > 128) break;
          unsigned size = (bits.getFixedValue() + 31) / 32;
          if (words + size > maxWords) break;
          llvm::SmallPtrSet<llvm::Instruction *, 32> seen;
          llvm::SmallVector<llvm::Instruction *> address;
          if (!collectAddress(load->getPointerOperand(), point, seen, address)) break;
          loads.push_back(load);
          words += size;
        } else if (auto *store = llvm::dyn_cast<llvm::StoreInst>(&inst)) {
          if (!store->isSimple() || store->getPointerAddressSpace() != 3) break;
        } else if (llvm::isa<llvm::FenceInst, llvm::AtomicRMWInst, llvm::AtomicCmpXchgInst>(inst)) {
          break;
        } else if (auto *call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
          if (!isExp2(inst) && !isRegisterExchange(call) && (!call->doesNotAccessMemory() || !call->doesNotThrow() ||
              !call->hasFnAttr(llvm::Attribute::WillReturn) || call->isConvergent())) break;
        } else if (inst.mayHaveSideEffects()) {
          break;
        }
      }
      for (auto *load : loads) {
        llvm::SmallPtrSet<llvm::Instruction *, 32> seen;
        llvm::SmallVector<llvm::Instruction *> address;
        if (!collectAddress(load->getPointerOperand(), point, seen, address)) continue;
        for (auto *inst : address) inst->moveBefore(point->getIterator());
        load->moveBefore(point->getIterator());
        ++packets;
      }
    }
    function.addFnAttr("frisk.global_prefetch", std::to_string(maxWords));
  }
  llvm::errs() << "PrefetchGlobalPackets: " << packets << " packets, at most "
               << maxWords << " register words per block phase\n";
}
} // namespace mlir::frisk
