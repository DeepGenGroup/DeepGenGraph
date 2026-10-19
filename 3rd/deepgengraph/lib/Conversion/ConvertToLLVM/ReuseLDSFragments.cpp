#include "deepgengraph/Conversion/ConvertToLLVM/LLVMExportUtils.h"
#include "deepgengraph/Conversion/ConvertToLLVM/RegisterMMAUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::frisk {
namespace {
// Reuse never crosses a synchronization or unknown call, even if an opaque
// call happens to be annotated readonly. Only known register operations may
// separate the loads. No instruction or convergent operation is moved.
bool endsPhase(llvm::Instruction &inst) {
  if (auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst))
    return !load->isSimple();
  if (llvm::isa<llvm::FenceInst, llvm::AtomicRMWInst,
                llvm::AtomicCmpXchgInst>(inst)) return true;
  if (auto *call = llvm::dyn_cast<llvm::CallBase>(&inst)) {
    if (call->getNumOperandBundles()) return true;
    if (auto *assembly = llvm::dyn_cast<llvm::InlineAsm>(call->getCalledOperand())) {
      if (assembly->getAsmString().empty() &&
          assembly->getConstraintString() == "=v,0") return false;
      return !call->doesNotAccessMemory() ||
          !isRegisterMMA(assembly->getAsmString(), assembly->getConstraintString());
    }
    if (auto *intrinsic = llvm::dyn_cast<llvm::IntrinsicInst>(call)) {
      if (intrinsic->getIntrinsicID() == llvm::Intrinsic::amdgcn_ds_bpermute)
        return false;
      return !call->doesNotAccessMemory() || call->isConvergent();
    }
    return true;
  }
  return inst.mayWriteToMemory() || inst.mayHaveSideEffects();
}
} // namespace

void reuseLDSFragments(llvm::Module &module, unsigned maxWords) {
  if (!maxWords) return;
  unsigned removed = 0;
  for (auto &function : module) {
    const auto &triple = module.getTargetTriple();
    if (function.isDeclaration() || function.hasFnAttribute(llvm::Attribute::OptimizeNone) ||
        function.hasFnAttribute("frisk.lds_reuse") ||
        (!triple.isAMDGPU() && !(triple.str().empty() &&
          function.getCallingConv() == llvm::CallingConv::AMDGPU_KERNEL))) continue;
    for (auto &block : function) {
      llvm::DenseMap<llvm::Instruction *, unsigned> positions;
      unsigned position = 0;
      for (auto &inst : block) positions[&inst] = position++;
      // Conservatively charge the entire retained packet from its first load
      // through all replaced loads' direct uses. Intervals, not a cache size,
      // bound overlapping retention. Cross-block users and PHIs are rejected.
      llvm::SmallVector<unsigned> reserved(position, 0);
      llvm::DenseMap<llvm::LoadInst *, unsigned> retainedEnd;
      llvm::SmallVector<llvm::LoadInst *> available;
      for (auto &inst : llvm::make_early_inc_range(block)) {
        if (endsPhase(inst)) { available.clear(); continue; }
        auto *load = llvm::dyn_cast<llvm::LoadInst>(&inst);
        if (!load || load->getPointerAddressSpace() != 3 || !load->isSimple()) continue;
        auto *type = llvm::dyn_cast<llvm::FixedVectorType>(load->getType());
        if (!type) continue;
        auto bits = module.getDataLayout().getTypeSizeInBits(type).getFixedValue();
        unsigned words = (bits + 31) / 32;
        if (!words || words > maxWords) continue;
        llvm::LoadInst *previous = nullptr;
        for (auto *candidate : llvm::reverse(available))
          if (candidate->getType() == type &&
              candidate->getPointerOperand() == load->getPointerOperand()) {
            previous = candidate;
            break;
          }
        bool reused = false;
        if (previous) {
          unsigned end = positions[load];
          bool local = true;
          for (llvm::User *user : load->users()) {
            auto *use = llvm::dyn_cast<llvm::Instruction>(user);
            if (!use || use->getParent() != &block || llvm::isa<llvm::PHINode>(use)) {
              local = false;
              break;
            }
            end = std::max(end, positions[use]);
          }
          unsigned start = retainedEnd.contains(previous)
              ? retainedEnd[previous] + 1 : positions[previous];
          bool fits = local;
          for (unsigned i = start; fits && i <= end; ++i)
            fits = reserved[i] <= maxWords - words;
          if (fits) {
            for (unsigned i = start; i <= end; ++i) reserved[i] += words;
            retainedEnd[previous] = std::max(retainedEnd.lookup(previous), end);
            load->replaceAllUsesWith(previous);
            load->eraseFromParent();
            ++removed;
            reused = true;
          }
        }
        if (!reused) available.push_back(load);
      }
    }
    function.addFnAttr("frisk.lds_reuse", std::to_string(maxWords));
  }
  llvm::errs() << "ReuseLDSFragments: " << removed << " loads removed, at most "
               << maxWords << " retained register words per thread\n";
}
} // namespace mlir::frisk
