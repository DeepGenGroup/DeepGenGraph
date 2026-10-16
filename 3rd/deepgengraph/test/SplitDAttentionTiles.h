#ifndef DEEPGENGRAPH_TEST_SPLIT_D_ATTENTION_TILES_H
#define DEEPGENGRAPH_TEST_SPLIT_D_ATTENTION_TILES_H

#include "deepgengraph/Analysis/HardwareSpecification.h"
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace {
namespace tileDG = mlir::deepgengraph;
namespace tileDGT = mlir::deepgengraph::triton;
constexpr llvm::StringLiteral outputTileMarker = "frisk.split_d_output";
constexpr llvm::StringLiteral qkTileMarker = "frisk.split_d_qk";
constexpr llvm::StringLiteral expTileMarker = "frisk.split_d_exp";

// Driver-level retile of a split-D tensor program. Track dimension identities
// through SSA, not equal numeric shapes: Q, V, O and even the score may have
// identical types while their axes have different meanings.
class SplitDTileDimensions {
  using Axis = std::pair<mlir::Value, unsigned>;
  llvm::DenseMap<Axis, llvm::SmallVector<Axis>> edges;
  llvm::DenseMap<Axis, int64_t> replacements;
  llvm::SmallPtrSet<mlir::Operation *, 32> supported;
public:
  static mlir::RankedTensorType tensor(mlir::Value value) {
    if (auto pointer = mlir::dyn_cast<tileDGT::BlockPointerType>(value.getType()))
      return mlir::dyn_cast<mlir::RankedTensorType>(pointer.getPointeeType());
    return mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
  }
  static bool tile(mlir::Value value) {
    auto type = tensor(value);
    return type && type.hasStaticShape() && type.getRank() == 2;
  }
  void connect(mlir::Value a, unsigned x, mlir::Value b, unsigned y) {
    if (!tile(a) || !tile(b) || tensor(a).getDimSize(x) != tensor(b).getDimSize(y)) return;
    edges[{a, x}].push_back({b, y});
    edges[{b, y}].push_back({a, x});
  }
  void connect(mlir::Value a, mlir::Value b) {
    for (unsigned axis = 0; axis < 2; ++axis) connect(a, axis, b, axis);
  }
  explicit SplitDTileDimensions(mlir::Operation *scope) {
    scope->walk([&](mlir::Operation *op) {
      if (auto dot = mlir::dyn_cast<tileDG::PreciseDotOp>(op)) {
        connect(dot.getLhs(), 1, dot.getRhs(), 0);
        connect(dot.getLhs(), 0, dot.getResult(), 0);
        connect(dot.getRhs(), 1, dot.getResult(), 1);
      } else if (auto load = mlir::dyn_cast<tileDGT::BlockLoadOp>(op)) {
        connect(load.getSrcPointer(), load.getResult());
      } else if (auto store = mlir::dyn_cast<tileDGT::BlockStoreOp>(op)) {
        connect(store.getDstPointer(), store.getValue());
      } else if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
        auto yield = mlir::cast<mlir::scf::YieldOp>(loop.getBody()->getTerminator());
        for (unsigned i = 0; i < loop.getNumResults(); ++i) {
          connect(loop.getInitArgs()[i], loop.getRegionIterArgs()[i]);
          connect(loop.getInitArgs()[i], loop.getResult(i));
          connect(loop.getInitArgs()[i], yield.getOperand(i));
        }
      } else if (mlir::isa<tileDG::AddOp, tileDG::MulOp, tileDG::DivOp,
                           tileDG::ConvertOp, tileDG::Exp2Op>(op)) {
        // Broadcast dimensions of size one do not follow a wider tile axis.
        for (mlir::Value result : op->getResults())
          for (mlir::Value operand : op->getOperands()) connect(result, operand);
      } else if (!mlir::isa<tileDG::ZeroOp, tileDGT::BlockPointerOfOp,
                            mlir::arith::ConstantOp, mlir::scf::YieldOp>(op)) {
        return;
      }
      supported.insert(op);
    });
  }
  mlir::LogicalResult resize(mlir::Value value, unsigned axis, int64_t extent) {
    llvm::SmallVector<Axis> work{{value, axis}};
    while (!work.empty()) {
      Axis key = work.pop_back_val();
      if (!tile(key.first)) return mlir::failure();
      auto [it, fresh] = replacements.try_emplace(key, extent);
      if (!fresh) {
        if (it->second != extent) return mlir::failure();
        continue;
      }
      for (Axis next : edges[key]) work.push_back(next);
    }
    return mlir::success();
  }
  bool unchanged(mlir::Value value, unsigned axis) const {
    auto found = replacements.find({value, axis});
    return found == replacements.end() || found->second == tensor(value).getDimSize(axis);
  }
  mlir::LogicalResult apply(mlir::Operation *scope) {
    llvm::DenseMap<mlir::Value, mlir::RankedTensorType> types;
    for (auto [axis, extent] : replacements) {
      mlir::Value value = axis.first;
      if (tensor(value).getDimSize(axis.second) == extent) continue;
      auto *definition = value.getDefiningOp();
      if (!definition) {
        auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
        definition = arg ? arg.getOwner()->getParentOp() : nullptr;
      }
      if (!supported.contains(definition) ||
          llvm::any_of(value.getUsers(), [&](mlir::Operation *user) { return !supported.contains(user); }))
        return scope->emitError("split-D tile dimension escapes supported tensor dataflow");
      auto old = tensor(value);
      llvm::SmallVector<int64_t> shape(old.getShape());
      for (unsigned i = 0; i < 2; ++i)
        if (auto found = replacements.find({value, i}); found != replacements.end()) shape[i] = found->second;
      auto type = mlir::RankedTensorType::get(shape, old.getElementType(), old.getEncoding());
      if (auto constant = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(value.getDefiningOp())) {
        auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue());
        if (!dense || !dense.isSplat()) return constant.emitError("cannot resize a non-splat tile initializer");
      }
      types[value] = type;
    }
    // All checks precede mutation. Global tensor/pointer types and global
    // strides are deliberately outside the block-pointer dimension graph.
    for (auto [value, type] : types) {
      if (mlir::isa<tileDGT::BlockPointerType>(value.getType()))
        value.setType(tileDGT::BlockPointerType::get(type));
      else value.setType(type);
      if (auto pointer = value.getDefiningOp<tileDGT::BlockPointerOfOp>())
        pointer.setBlockShape(type.getShape());
      if (auto zero = value.getDefiningOp<tileDG::ZeroOp>()) zero.setShape(type.getShape());
      if (auto constant = value.getDefiningOp<mlir::arith::ConstantOp>()) {
        auto dense = mlir::cast<mlir::DenseElementsAttr>(constant.getValue());
        constant.setValueAttr(mlir::DenseElementsAttr::get(type, dense.getSplatValue<mlir::Attribute>()));
      }
    }
    return mlir::success();
  }
};

mlir::LogicalResult configureSplitDAttention(mlir::ModuleOp module, bool kpack,
                                            int64_t requestedQK, int64_t requestedVO) {
  auto kernel = module.lookupSymbol<tileDG::KernelOp>("Attn_p2");
  if (!kernel) return module.emitError("split-D tiling requires an Attn_p2 tensor kernel");
  llvm::SmallVector<mlir::affine::AffineForOp> outputs;
  kernel.walk([&](mlir::affine::AffineForOp loop) {
    if (llvm::any_of(loop.getBody()->without_terminator(), [](mlir::Operation &op) {
          return mlir::isa<tileDGT::BlockStoreOp>(op);
        })) outputs.push_back(loop);
  });
  if (outputs.size() != 1) return kernel.emitError("split-D tiling expects one output-D store loop");
  auto output = outputs.front();
  if (!output.hasConstantLowerBound() || !output.hasConstantUpperBound() ||
      output.getConstantLowerBound() != 0 || output.getConstantUpperBound() <= 0)
    return output.emitError("output-D loop requires static bounds 0:D");
  llvm::SmallVector<tileDG::PreciseDotOp> dots;
  output.walk([&](tileDG::PreciseDotOp dot) { dots.push_back(dot); });
  if (dots.size() != 2) return output.emitError("split-D tiling expects QK and PV tensor dots");
  auto qk = dots[0], pv = dots[1];
  auto reduction = mlir::dyn_cast<mlir::scf::ForOp>(qk->getParentOp());
  auto keyLoop = mlir::dyn_cast<mlir::scf::ForOp>(pv->getParentOp());
  if (!reduction || !keyLoop || reduction->getParentOp() != keyLoop ||
      !output->isAncestor(keyLoop))
    return output.emitError("split-D tiling requires QK reduction and PV in the same key loop");
  auto lower = reduction.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto upper = reduction.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
  auto step = reduction.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
  if (!lower || lower.value() || !upper || upper.value() <= 0 || !step || step.value() <= 0)
    return reduction.emitError("QK reduction requires static bounds 0:D and a positive constant step");
  for (mlir::Value value : {qk.getLhs(), qk.getRhs(), pv.getLhs(), pv.getRhs()})
    if (!SplitDTileDimensions::tile(value) || !SplitDTileDimensions::tensor(value).getElementType().isF16())
      return output.emitError("split-D tiling requires static rank-two FP16 dot operands");
  int64_t oldQK = step.value(), oldVO = output.getStepAsInt();
  int64_t dk = upper.value(), dv = output.getConstantUpperBound();
  if (oldVO <= 0 || dk % oldQK || dv % oldVO)
    return output.emitError("input QK/VO steps must divide their respective D extents");
  auto lhs = SplitDTileDimensions::tensor(qk.getLhs());
  auto rhs = SplitDTileDimensions::tensor(qk.getRhs());
  auto v = SplitDTileDimensions::tensor(pv.getRhs());
  auto out = SplitDTileDimensions::tensor(pv.getResult());
  if (lhs.getDimSize(1) != oldQK || rhs.getDimSize(0) != oldQK ||
      v.getDimSize(1) != oldVO || !out || out.getDimSize(1) != oldVO)
    return output.emitError("QK/VO operand tile extents must match their own loop steps");

  auto *hw = GetHWSpecification(HW_KIND_DCU, HW_VERSION_DCU_BW1000, module.getContext());
  const auto *dma = hw->dataCopyInfo.getMaxDMATransfer();
  int64_t fragmentBytes = 0;
  for (const auto &mma : hw->gemmInfo.validInsts)
    if (mma.asm_str.find("16x16x16_f16") != std::string::npos)
      fragmentBytes = mma.desc_a.thread_creg[0] * mma.desc_a.thread_creg[1] * 2;
  if (!dma || !fragmentBytes || dma->bytesPerLane % fragmentBytes)
    return kernel.emitError("no full-width FP16 DMA/MMAC kpack ratio");
  int64_t factor = kpack ? dma->bytesPerLane / fragmentBytes : 1;
  auto choose = [&](int64_t requested, int64_t old, int64_t bound) -> int64_t {
    if (requested) return requested;
    return factor > 0 && old <= bound / factor ? old * factor : 0;
  };
  int64_t newQK = choose(requestedQK, oldQK, dk), newVO = choose(requestedVO, oldVO, dv);
  if (newQK <= 0 || newQK % 16 || newQK > dk || dk % newQK ||
      newVO <= 0 || newVO % 16 || newVO > dv || dv % newVO)
    return output.emitError("QK/VO tiles must be positive multiples of 16 dividing their respective D extents; requested/effective QK=")
           << newQK << ", VO=" << newVO << ", Dqk=" << dk << ", Dvo=" << dv;
  SplitDTileDimensions dimensions(kernel);
  if (mlir::failed(dimensions.resize(qk.getLhs(), 1, newQK)) ||
      mlir::failed(dimensions.resize(pv.getRhs(), 1, newVO)))
    return output.emitError("QK reduction and V/O output dimensions are not independent");
  if (!dimensions.unchanged(qk.getLhs(), 0) || !dimensions.unchanged(qk.getRhs(), 1) ||
      !dimensions.unchanged(pv.getLhs(), 0) || !dimensions.unchanged(pv.getLhs(), 1) ||
      !dimensions.unchanged(qk.getResult(), 0) || !dimensions.unchanged(qk.getResult(), 1))
    return output.emitError("split-D retile would also change BM/BN or the score/probability tile");
  if (mlir::failed(dimensions.apply(kernel))) return mlir::failure();
  output.setStep(newVO);
  mlir::OpBuilder builder(reduction);
  reduction.getStepMutable().assign(builder.create<mlir::arith::ConstantIndexOp>(kernel.getLoc(), newQK));
  auto marker = builder.getUnitAttr();
  output->setAttr(outputTileMarker, marker);
  qk->setAttr(qkTileMarker, marker);
  keyLoop.walk([&](tileDG::Exp2Op exp) { exp->setAttr(expTileMarker, marker); });
  if (kpack && oldQK == oldVO && newQK == newVO)
    llvm::errs() << "useKPack: DMA=" << dma->bytesPerLane << "B, MMAC input=" << fragmentBytes
                 << "B, kpack=" << factor << ", BK=" << oldQK << " -> " << newQK << "\n";
  llvm::errs() << "split-D tiles: BM=" << lhs.getDimSize(0) << ", BN=" << rhs.getDimSize(1)
               << ", Dqk=" << dk << ", Dvo=" << dv << ", QK=" << oldQK << " -> " << newQK
               << ", VO=" << oldVO << " -> " << newVO << ", kpack=" << factor << "\n";
  return mlir::verify(module);
}

// Count copies AND surviving output-loop iterations. Counting just one exp2
// operation would miss a partially jammed loop that recomputes it per slice.
mlir::LogicalResult checkSplitDSharing(mlir::ModuleOp module, bool requireShared) {
  uint64_t qk = 0, exp = 0;
  module.walk([&](mlir::Operation *op) {
    if (!op->hasAttr(qkTileMarker) && !op->hasAttr(expTileMarker)) return;
    uint64_t groups = 1;
    for (auto *parent = op->getParentOp(); parent; parent = parent->getParentOp())
      if (parent->hasAttr(outputTileMarker)) {
        auto loop = mlir::dyn_cast<mlir::affine::AffineForOp>(parent);
        auto trips = loop ? mlir::affine::getConstantTripCount(loop) : std::nullopt;
        if (!trips || *trips > UINT64_MAX / groups) { groups = UINT64_MAX; break; }
        groups *= *trips;
      }
    auto &count = op->hasAttr(qkTileMarker) ? qk : exp;
    count = count > UINT64_MAX - groups ? UINT64_MAX : count + groups;
  });
  module.walk([&](mlir::Operation *op) {
    op->removeAttr(outputTileMarker); op->removeAttr(qkTileMarker); op->removeAttr(expTileMarker);
  });
  if (!qk && !exp && !requireShared) return mlir::success();
  llvm::errs() << "split-D sharing: QK copies=" << qk << ", softmax copies=" << exp << " per key tile\n";
  if (requireShared && (qk != 1 || exp != 1))
    return module.emitError("explicit split-D tiles would repeat QK/softmax across output slices; increase carrier-words or VO tile (tensor jam limit is 8)");
  return mlir::success();
}
} // namespace
#endif
