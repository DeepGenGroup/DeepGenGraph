#ifndef FRISK_UTILS_H_
#define FRISK_UTILS_H_

#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.h"
#include "mlir/Support/LLVM.h"
#include <cstdint>

namespace mlir::frisk {

namespace {

class OpBuilderWithLoc {   // 封装了 OpBuilder IR 构建器
public:
  OpBuilderWithLoc(MLIRContext *context) {
    builder = std::make_unique<OpBuilder>(context);
    lastLoc = std::make_unique<Location>(builder->getUnknownLoc());
  }

  OpBuilder &getBuilder() { return *builder; }

  void setLastLoc(Location loc) { lastLoc = std::make_unique<Location>(loc); }

  void setLastLoc(const std::string &fileName, int line, int column) {
    auto context = builder->getContext();
    setLastLoc(FileLineColLoc::get(context, fileName, line, column));
  }

  Location getLastLoc() {
    assert(lastLoc);
    return *lastLoc;
  }

  void setInsertionPointToStart(Block &block) {
    if (!block.empty())
      setLastLoc(block.begin()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToStart(&block);
  }

  void setInsertionPointToEnd(Block &block) {
    if (!block.empty())
      setLastLoc(block.back().getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->setInsertionPointToEnd(&block);
  }

  void setInsertionPointAfter(Operation &op) {
    setLastLoc(op.getLoc());
    builder->setInsertionPointAfter(&op);
  }

  void restoreInsertionPoint(OpBuilder::InsertPoint pt) {
    if (pt.isSet() && pt.getPoint() != pt.getBlock()->end())
      setLastLoc(pt.getPoint()->getLoc());
    else
      setLastLoc(builder->getUnknownLoc());
    builder->restoreInsertionPoint(pt);
  }

  Operation *clone(Operation &op) { return builder->clone(op); }

  template <typename OpTy, typename... Args>
  OpTy create(Args &&...args) {
    auto loc = getLastLoc();
    return builder->create<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a single result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::OneResult>(), Value> createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

  // Overload to create or fold a zero result operation.
  template <typename OpTy, typename... Args>
  std::enable_if_t<OpTy::template hasTrait<OpTrait::ZeroResults>(), OpTy> createOrFold(Args &&...args) {
    auto loc = getLastLoc();
    return builder->createOrFold<OpTy>(loc, std::forward<Args>(args)...);
  }

private:
  std::unique_ptr<OpBuilder> builder;
  std::unique_ptr<Location> lastLoc;
};

} // namespace

using friskMs = ::mlir::frisk::attr::MemorySpace;

enum class FriskDType : int {
  unknown = 0,
  f32,
  f16,
  i32,
  i16
};

inline std::string FriskDTypeToString(FriskDType d){
  switch (d) {
    case mlir::frisk::FriskDType::f16: return "f16";
    case mlir::frisk::FriskDType::f32 : return "f32";
    case mlir::frisk::FriskDType::i16 : return "i16";
    case mlir::frisk::FriskDType::i32 : return "i32";
    default: break;
  }
  return "";
}


// 比较memspace层级大小（）
static inline int compareMemspace(frisk::attr::MemorySpace lhs, frisk::attr::MemorySpace rhs){
  auto toInt = [](frisk::attr::MemorySpace ms){
    switch (ms) {
      case mlir::frisk::attr::MemorySpace::Global: return 2; 
      case mlir::frisk::attr::MemorySpace::Shared: return 1; 
      case mlir::frisk::attr::MemorySpace::Local: return 0;
      default: return -1; 
    }
  };
  // 数字越大，层级越高，速度越慢
  if(toInt(lhs) > toInt(rhs)){
    return 1;
  }
  if(toInt(lhs) < toInt(rhs)){
    return -1;
  }
  return 0;
}

static inline void AppendMemspaceToMemrefValue(Value& v, int ms){
  if(mlir::isa<MemRefType>(v.getType())){
    auto _ty = mlir::cast<MemRefType>(v.getType());
    auto tA = MemRefType::get(_ty.getShape(), _ty.getElementType(), AffineMap{}, int(ms));
    v.setType(tA);
  }
}

template<typename OpTy>
Operation* getOuterMostOp(mlir::Operation* op){
  mlir::Operation* currOp = op;
  while (true) {
    auto parentForOp = currOp->getParentOfType<OpTy>();
    if(parentForOp == nullptr){
      break;
    }
    else{
      currOp = parentForOp;
    }
  }
  return currOp;
}

static inline Operation* getOuterMostOpWithName(mlir::Operation* op, const char* name){
  mlir::Operation* currOp = op;
  while (true) {
    auto parentOp = currOp->getParentOp();
    if(parentOp == nullptr){
      return nullptr;
    }
    else{
      if(parentOp->getName().getStringRef() == name){
        return parentOp;
      }
      currOp = parentOp;
    }
  }
  return currOp;
}

#define IN_MEMSPACE "inMs"
#define OUT_MEMSPACE "outMs"

static inline DenseI32ArrayAttr getOpInputMemspaceAttr(mlir::Operation* op){
  return op->getAttrOfType<DenseI32ArrayAttr>(IN_MEMSPACE);
}

static inline DenseI32ArrayAttr getOpOutputMemspaceAttr(mlir::Operation* op){
  return op->getAttrOfType<DenseI32ArrayAttr>(OUT_MEMSPACE);
}

struct WgmmaConfig {
  static int mma_m ;
  static int mma_k_bytes ;
};

enum class VendorKind : int32_t {
  DCU = 1,
  NVIDIA = 2,
  AMD = 3
};

}

#endif