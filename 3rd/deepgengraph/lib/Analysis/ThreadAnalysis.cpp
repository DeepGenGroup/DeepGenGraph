#include "deepgengraph/Analysis/ThreadAnalysis.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace analyze {
using namespace mlir::deepgengraph::triton;

int BlockThreads(){
  return 128;
}

PtrInfoMapType PointerTracer::m_map = {};

BlockPointerOfOp PointerTracer::findBlockPointerOfBlockLoadOp(BlockLoadOp op) {
  using namespace mlir;
  deepgengraph::triton::BlockPointerType srcPointerType = op.getSrcPointer().getType();
  mlir::RankedTensorType tensorType = srcPointerType.getPointeeType();
  deepgengraph::triton::BlockPointerOfOp blockPtrOfOp =
      op.getSrcPointer().getDefiningOp<mlir::deepgengraph::triton::BlockPointerOfOp>();
  if (blockPtrOfOp != nullptr) { // cast ok
    return blockPtrOfOp;
  } else if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(op.getSrcPointer())) { // op.getSrcPointer 来自 blockArg
    // 1. 获取该参数属于哪个 Block
    mlir::Block *ownerBlock = blockArg.getOwner();

    // 2. 获取是这个 Block 的第几个参数 (Index)
    unsigned argNumber = blockArg.getArgNumber();

    // 3. 获取拥有这个 Block 的上层 Operation (Parent Op)
    mlir::Operation *parentOp = ownerBlock->getParentOp();
    if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(parentOp)) {
      // scf.for 的第 0 个参数是 induction variable (比如 i)
      if (argNumber > 0) {
        // 计算它是第几个 iter_arg
        unsigned iterArgIdx = argNumber - 1;

        // 获取这个 iter_arg 在循环外部传入的初始值 (即 IR 中的 %17)
        mlir::Value initValue = forOp.getInitArgs()[iterArgIdx];

        // 现在你可以对 initValue 再次调用 getDefiningOp()
        // 此时 initValue 是 %17，它是由 block_ptr_of 产生的 OpResult！
        auto ptrOfOp = initValue.getDefiningOp<mlir::deepgengraph::triton::BlockPointerOfOp>();
        if (ptrOfOp) {
          return ptrOfOp;
        }
      }
    }
  }
  return nullptr;
}


PointerOfOp PointerTracer::findPointerOfOp(mlir::deepgengraph::triton::BlockPointerOfOp op){
  using namespace mlir;
  auto defOp = op.getBasePointer().getDefiningOp<deepgengraph::triton::PointerOfOp>();
  return defOp;
}

void PointerTracer::getPointerInfo(mlir::ModuleOp mod) {
  mod->walk([&](mlir::deepgengraph::triton::BlockLoadOp op) { 
    auto blockPtrOfOp = findBlockPointerOfBlockLoadOp(op); 
    auto ptrOfOp = findPointerOfOp(blockPtrOfOp);
    PtrInfo info {blockPtrOfOp, ptrOfOp};
    m_map[op] = info;
  });
}

const PtrInfoMapType& PointerTracer::getMap() {
  return m_map;
}

}