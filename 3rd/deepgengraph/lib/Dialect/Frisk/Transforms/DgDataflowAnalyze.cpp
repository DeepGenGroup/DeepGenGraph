#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskOps.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"

#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenACC/OpenACC.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace mlir::frisk {

#define GEN_PASS_DEF_CALCULATEOPTOFRISK

#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h.inc"

namespace {

namespace dg = deepgengraph;
namespace dgt = deepgengraph::triton;

using friskMs = frisk::attr::MemorySpace;


static Operation* GetEffectiveDefOp(mlir::Value operand, int maxRecursive = 5){

  if(maxRecursive < 0){
    llvm::outs() << operand << "\n";llvm::outs().flush();
    assert(false);
  }

  auto defOp = operand.getDefiningOp();
  // 递归地获取有效defOp
  while(defOp == nullptr){
    // operand is a blockargument.追踪其所在的block，得到block的主人op
    auto blockArg = mlir::dyn_cast<BlockArgument>(operand);
    auto id = blockArg.getArgNumber();
    auto arg = operand.getParentBlock()->getArgument(id);
    auto parentOp = arg.getOwner()->getParentOp();
    
    if(auto forOp = mlir::dyn_cast<affine::AffineForOp>(parentOp)){
      auto val = forOp.getInits()[id-1];
      return GetEffectiveDefOp(val, maxRecursive-1);
    }
    defOp = arg.getDefiningOp();
  }
  return defOp;
}

static int GetOperandMemspace(mlir::Value operand){
  // 对于非memref 或者 tensor 类型，不存在 memspace概念。用 None 占位
  if(!mlir::isa<MemRefType, TensorType>(operand.getType())){
    return int(frisk::attr::MemorySpace::None);
  }
  auto defOp = GetEffectiveDefOp(operand);
  // operand 源于 block_load ： 标记为 shm
  if(mlir::isa<dgt::BlockLoadOp>(defOp)){
    return int(frisk::attr::MemorySpace::Shared);
  }
  // operand 源于 reduce_op 的结果 ： memspace和 init 参数的memspace相同
  if(auto reduceOp = mlir::dyn_cast<dg::ReduceOp>(defOp)){
    return GetOperandMemspace(reduceOp.getInit());
  }
  // operand 源于 affineFor的返回（affineYield 更新iterArgs） ： 等于initArg中的对应value的memspace
  if(auto forOp = mlir::dyn_cast<affine::AffineForOp>(defOp)){
    int id = 0;
    for(int i=0;i < forOp->getNumResults();++i){
      if(forOp->getResults()[i] == operand){
        auto initValue = forOp.getInits()[i];
        return GetOperandMemspace(initValue);
      }
    }
  }
  // operand 源于 mamtul的结果 ：标记为 local
  if(mlir::isa<dg::PreciseDotOp>(defOp)){
    return int(frisk::attr::MemorySpace::Local);
  }
  // operand 源于mask、zero的结果 ： 标记为 local （可以在local里直接处理）
  if(mlir::isa<dg::MaskOp, dg::ZeroOp>(defOp)){
    return int(frisk::attr::MemorySpace::Local);
  }
  // operand 源于 convert 的结果 ：标记为 和convert 输入value的memspace相同
  if(auto convertOp = mlir::dyn_cast<deepgengraph::ConvertOp>(defOp)){
    return GetOperandMemspace(convertOp.getOperand());
  }
  // operand 源于 arith.constant 结果 ： 标记为 local
  if(mlir::isa<arith::ConstantOp>(defOp)){
    return int(frisk::attr::MemorySpace::Local);
  }
  // 源于其他op（计算类、dg.zero等），不能准确得知
  return int(frisk::attr::MemorySpace::Unknown);
}


// 首次推断 : 得到operands的ms。将result的ms标为未知
static void FirstAnalyze(mlir::Operation* op, OpBuilder& rewriter, std::vector<Operation*>& pendingOps){
  bool isValid = false;
  for(auto ty : op->getOperandTypes()){
    if(mlir::isa<TensorType, MemRefType>(ty)){
      isValid = true; break;
    }
  }
  
  for(auto ty : op->getResultTypes()){
    if(mlir::isa<TensorType, MemRefType>(ty)){
      isValid = true; break;
    }
  }
  if(!isValid){
    return;
  }

  std::vector<int> in_ms;
  std::vector<int> out_ms;
  int ms_shared = int(frisk::attr::MemorySpace::Shared);
  int ms_local = int(frisk::attr::MemorySpace::Local);
  bool hasUnkownOperand = false;
  if(mlir::isa<dg::PreciseDotOp>(op)){
    // 对于 gemm， 直接将 AB 标记为 shm
    in_ms = {ms_shared, ms_shared};
    out_ms = {ms_local};
    op->setAttr(OUT_MEMSPACE, rewriter.getDenseI32ArrayAttr(out_ms)); 
  }
  else if(mlir::isa<dg::BroadcastableBinaryOpInterface, dg::ExpOp, dg::Exp2Op, dg::ReduceOp, dg::ConvertOp>(op)){
    // llvm::outs() << "[analyze] " << op->getName().getStringRef() << "\n"; llvm::outs().flush();
    
    for(auto operand : op->getOperands()){
      auto ms = GetOperandMemspace(operand);
      in_ms.push_back(ms);
      if(ms == int(friskMs::Unknown)){
        hasUnkownOperand = true;
      }
    }
  }
  op->setAttr(IN_MEMSPACE, rewriter.getDenseI32ArrayAttr(in_ms)); 
  if(hasUnkownOperand){
    pendingOps.push_back(op);
  }
}

// 二次推断: 根据 userOp 的operand ms，推断result的ms
static void SecondAnalyze(mlir::Operation* op, OpBuilder& rewriter) {
  if(op->hasAttr(OUT_MEMSPACE)){
    return;
  }
  bool isValid = false;
  for(auto ty : op->getResultTypes()){
    if(mlir::isa<MemRefType, TensorType>(ty)){
      isValid = true;break;
    }
  }
  if(!isValid){
    return;
  }
  auto inMsAttr = op->getAttrOfType<DenseI32ArrayAttr>(IN_MEMSPACE);
  std::vector<int> out_ms;
  for (auto resultVal : op->getResults()) {
    int retMs = int(friskMs::Local);
    for (auto &use : resultVal.getUses()) {
      mlir::Operation* userOp = use.getOwner(); // 获取使用者 Operation
      
      if (userOp->hasAttr(IN_MEMSPACE)) {
        auto userInputMs = userOp->getAttrOfType<DenseI32ArrayAttr>(IN_MEMSPACE).asArrayRef();
        
        // 【地道写法】直接获取当前 Use 对应的操作数索引（0-based）
        unsigned operandIdx = use.getOperandNumber();
        
        // 安全地获取对应的 memory space 属性
        if (operandIdx < userInputMs.size()) {
          auto userMs = userInputMs[operandIdx];
          auto cmp = compareMemspace(frisk::attr::MemorySpace(retMs), frisk::attr::MemorySpace(userMs));
          if(cmp < 0){  // userMs 比 retMs更慢，取更慢的 (排除 unkonw和none)
            retMs = userMs;
          }
        }
      }
    }
    out_ms.push_back(retMs);
  }
  op->setAttr(OUT_MEMSPACE, rewriter.getDenseI32ArrayAttr(out_ms));
}

// 对每个计算op，进行memspace推断
// First：推断operands的 memspace
// Second ： 推断 result的memspace

struct DataflowAnalyzePass : public PassWrapper<DataflowAnalyzePass, OperationPass<deepgengraph::KernelOp>> {

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DataflowAnalyzePass)

  StringRef getArgument() const final { return "add-memspace"; }
  StringRef getDescription() const final { return "Add memspace"; }

  void runOnOperation() override {
    auto kernelOp = getOperation();
    std::vector<Operation*> pendingOps;
    std::vector<Operation*> _pendingOps;
    
    OpBuilder b{kernelOp->getContext()};
    kernelOp->walk([&](mlir::Operation* op){
      FirstAnalyze(op, b, pendingOps);
    });
    kernelOp->walk([&](mlir::Operation* op){
      SecondAnalyze(op, b);
    });
    // 得到 outMs inMs 后，对inMs中的待定operand再做推断。依据为defOp的outMs
    int maxIter = 10;
    while(!pendingOps.empty()){
      maxIter--;
      if(maxIter < 0){
        assert(false);
      }
      _pendingOps.clear();
      for(auto pendingOp : pendingOps){
        if(pendingOp->hasAttr(IN_MEMSPACE)){
          auto inMsAttr = pendingOp->getAttrOfType<mlir::DenseI32ArrayAttr>(IN_MEMSPACE);
          auto inMs = inMsAttr.asArrayRef();

          // ArrayRef 是只读的。如果后续需要修改 IN_MEMSPACE 并写回 IR，需用 vector 暂存
          std::vector<int32_t> newInMs(inMs.begin(), inMs.end());
          bool hasUpdated = false;

          for (int i = 0; i < inMs.size(); ++i) {
            // 若 inMs[i]==Unknown, 则查找其defineOp的 outMs.
            if (inMs[i] == int(friskMs::Unknown)) {
              mlir::Value operandVal = pendingOp->getOperand(i);
              auto defOp = GetEffectiveDefOp(operandVal);
              if (!defOp){
                continue; // 防御性判断：可能没有 defOp（例如是 BlockArgument）
              }
              unsigned resultIdx = 0;
              // 获取 pendingOp->getOperand(i) 是 defOp results中的第几个
              // 在 MLIR 中，Operation 产生的结果属于 OpResult 类型
              if (auto opResult = llvm::dyn_cast<mlir::OpResult>(operandVal)) {
                // 情况 1: operand 直接就是 defOp 产生的结果
                if (opResult.getOwner() == defOp) {
                  resultIdx = opResult.getResultNumber(); // O(1) 获取索引
                }
                // 情况 2: GetEffectiveDefOp 穿透了某些操作 (例如 Reshape/Cast)
                // 此时 operandVal 的直接 owner 并不是 defOp。
                // 如果 defOp 通常只有一个返回值，默认 0 即可；否则需要根据你的业务逻辑追踪。
                else {
                  resultIdx = 0;
                }
              } else {
                // 如果 cast 失败，说明它是 BlockArgument 等，非 Operation 结果
                continue;
              }

              // 获取 defOp 的 OUT_MEMSPACE 属性并赋值给当前的 inMs
              if (defOp->hasAttr(OUT_MEMSPACE)) {
                auto outMs = defOp->getAttrOfType<mlir::DenseI32ArrayAttr>(OUT_MEMSPACE).asArrayRef();
                if (resultIdx < outMs.size()) {
                  // inMs推断为和defOp的outMs一致。如果仍为unknwon，继续
                  newInMs[i] = outMs[resultIdx];
                  if(newInMs[i] != int(friskMs::Unknown)){
                    hasUpdated = true;
                  }
                }
              }
            }
          }

          // 如果数据发生了更改，重新设置 pendingOp 的 Attribute
          if (hasUpdated) {
            pendingOp->setAttr(IN_MEMSPACE, b.getDenseI32ArrayAttr(newInMs));
          }
          else{
            _pendingOps.push_back(pendingOp);
          }
        }
      }
      std::swap(pendingOps, _pendingOps);
    }
  }
};
} // namespace

std::unique_ptr<Pass> createDataflowAnalyzePass() {
  return std::make_unique<DataflowAnalyzePass>();
}



}