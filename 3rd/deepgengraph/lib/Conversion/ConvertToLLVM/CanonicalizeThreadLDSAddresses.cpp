// Canonicalize physical LDS reads after packing, storage planning and unrolling.
// Symbolic affine differences cover dynamic common terms. Exact evaluation over
// the launch contract additionally proves nonlinear thread layouts. Unknown
// inputs are never sampled or assigned a guessed value.
#include "deepgengraph/Analysis/ThreadBounds.h"
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/Alignment.h"
#include <map>

using namespace mlir;

namespace {
class Addresses {
  using Table = SmallVector<APInt>;
  func::FuncOp function;
  std::optional<frisk::ThreadIdRange> threads;
  SmallVector<Value> symbols;
  DenseMap<Value, AffineExpr> expressions;
  std::map<void *, Table> tables;
  DenseSet<Value> unknown;

  // Evaluate affine arithmetic at the 64-bit index width used by LowerPipe.
  std::optional<APInt> evaluate(AffineExpr expr, unsigned thread) {
    if (auto c = dyn_cast<AffineConstantExpr>(expr)) return APInt(64, c.getValue());
    if (auto s = dyn_cast<AffineSymbolExpr>(expr)) {
      auto *values = table(symbols[s.getPosition()]);
      if (!values) return std::nullopt;
      return (*values)[thread].sextOrTrunc(64);
    }
    auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!binary) return std::nullopt;
    auto a = evaluate(binary.getLHS(), thread), b = evaluate(binary.getRHS(), thread);
    if (!a || !b) return std::nullopt;
    switch (expr.getKind()) {
    case AffineExprKind::Add: return *a + *b;
    case AffineExprKind::Mul: return *a * *b;
    case AffineExprKind::FloorDiv:
    case AffineExprKind::CeilDiv:
    case AffineExprKind::Mod: {
      if (!b->isStrictlyPositive()) return std::nullopt;
      APInt q = a->sdiv(*b), rem = a->srem(*b);
      if (expr.getKind() == AffineExprKind::Mod)
        return rem.isNegative() ? rem + *b : rem;
      if (expr.getKind() == AffineExprKind::FloorDiv && rem.isNegative()) --q;
      if (expr.getKind() == AffineExprKind::CeilDiv && rem.isStrictlyPositive()) ++q;
      return q;
    }
    default: return std::nullopt;
    }
  }

  const Table *table(Value value, unsigned depth = 0) {
    if (!threads || depth > 128) return nullptr;
    if (auto it = tables.find(value.getAsOpaquePointer()); it != tables.end()) return &it->second;
    if (!unknown.insert(value).second) return nullptr;
    Type type = value.getType();
    if (!type.isIndex() && (!isa<IntegerType>(type) || type.getIntOrFloatBitWidth() > 64))
      return nullptr;
    unsigned count = threads->upper - threads->lower;
    Table result;
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant))) {
      result.assign(count, constant);
    } else if (auto id = value.getDefiningOp<gpu::ThreadIdOp>()) {
      auto range = frisk::getThreadIdRange(id, id.getDimension());
      if (!range) return nullptr;
      for (unsigned t = 0; t < count; ++t) {
        if (id.getDimension() == gpu::Dimension::x)
          result.emplace_back(64, threads->lower + t);
        else if (range->upper == range->lower + 1)
          result.emplace_back(64, range->lower);
        else return nullptr;
      }
    } else if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
      auto expr = expression(value);
      for (unsigned t = 0; t < count; ++t) {
        auto evaluated = evaluate(expr, t);
        if (!evaluated) return nullptr;
        result.push_back(*evaluated);
      }
    } else {
      Operation *op = value.getDefiningOp();
      // Only scalar arith expressions. No memory, calls, region control flow,
      // poison/undef, loop-carried values or workgroup IDs enter this proof.
      if (!op || op->getName().getDialectNamespace() != "arith" ||
          op->getNumResults() != 1 || op->getNumRegions()) return nullptr;
      SmallVector<const Table *> inputs;
      for (Value operand : op->getOperands()) {
        auto *input = table(operand, depth + 1);
        if (!input) return nullptr;
        inputs.push_back(input);
      }
      // Fold a detached clone: some folders can update their operation in
      // place. Such folds are rejected, and never mutate the analyzed IR.
      Operation *clone = op->clone();
      auto cleanup = llvm::scope_exit([&] { clone->destroy(); });
      for (unsigned t = 0; t < count; ++t) {
        SmallVector<Attribute> operands;
        for (auto [i, input] : llvm::enumerate(inputs))
          operands.push_back(IntegerAttr::get(op->getOperand(i).getType(), (*input)[t]));
        SmallVector<OpFoldResult> folded;
        if (failed(clone->fold(operands, folded)) || folded.size() != 1) return nullptr;
        if (auto attr = dyn_cast<Attribute>(folded[0])) {
          auto integer = dyn_cast<IntegerAttr>(attr);
          if (!integer) return nullptr;
          result.push_back(integer.getValue());
        } else {
          Value replacement = cast<Value>(folded[0]);
          auto it = llvm::find(op->getOperands(), replacement);
          if (it == op->operand_end()) return nullptr;
          result.push_back((*inputs[it - op->operand_begin()])[t]);
        }
      }
    }
    return &tables.emplace(value.getAsOpaquePointer(), std::move(result)).first->second;
  }

public:
  explicit Addresses(func::FuncOp f)
      : function(f), threads(frisk::getThreadIdRange(f, gpu::Dimension::x)) {}

  AffineExpr expression(Value value, unsigned depth = 0) {
    if (auto it = expressions.find(value); it != expressions.end()) return it->second;
    auto *ctx = function.getContext();
    // Opaque leaves are shared by SSA identity. They may cancel symbolically
    // even when neither leaf can be evaluated (e.g. a common loop IV).
    AffineExpr result;
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant)) && constant.getBitWidth() <= 64)
      result = getAffineConstantExpr(constant.getSExtValue(), ctx);
    else if (depth < 128) {
      if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
        SmallVector<AffineExpr> dims, syms;
        for (Value v : apply.getDimOperands()) dims.push_back(expression(v, depth + 1));
        for (Value v : apply.getSymbolOperands()) syms.push_back(expression(v, depth + 1));
        result = apply.getAffineMap().getResult(0).replaceDimsAndSymbols(dims, syms);
      } else if (value.getType().isIndex()) {
        if (auto add = value.getDefiningOp<arith::AddIOp>())
          result = expression(add.getLhs(), depth + 1) + expression(add.getRhs(), depth + 1);
        else if (auto sub = value.getDefiningOp<arith::SubIOp>())
          result = expression(sub.getLhs(), depth + 1) - expression(sub.getRhs(), depth + 1);
        else if (auto mul = value.getDefiningOp<arith::MulIOp>()) {
          if (matchPattern(mul.getRhs(), m_ConstantInt(&constant)))
            result = expression(mul.getLhs(), depth + 1) * constant.getSExtValue();
          else if (matchPattern(mul.getLhs(), m_ConstantInt(&constant)))
            result = expression(mul.getRhs(), depth + 1) * constant.getSExtValue();
        }
      }
    }
    if (!result) {
      result = getAffineSymbolExpr(symbols.size(), ctx);
      symbols.push_back(value);
    }
    expressions[value] = result;
    return result;
  }

  AffineExpr simplify(AffineExpr expr) {
    return simplifyAffineExpr(expr, 0, symbols.size());
  }

  std::optional<int64_t> difference(AffineExpr a, AffineExpr b) {
    auto delta = simplify(a - b);
    if (auto c = dyn_cast<AffineConstantExpr>(delta)) return c.getValue();
    if (!threads) return std::nullopt;
    std::optional<APInt> first;
    for (unsigned t = 0, n = threads->upper - threads->lower; t < n; ++t) {
      auto value = evaluate(delta, t);
      if (!value || (first && *first != *value)) return std::nullopt;
      first = value;
    }
    return first ? std::optional<int64_t>(first->getSExtValue()) : std::nullopt;
  }

  uint64_t offsetAlignment(uint64_t base, AffineExpr offset) {
    offset = simplify(offset);
    auto symbolic = llvm::commonAlignment(llvm::Align(base), offset.getLargestKnownDivisor());
    if (!threads) return symbolic.value();
    auto exact = llvm::Align(base);
    for (unsigned t = 0, n = threads->upper - threads->lower; t < n; ++t) {
      auto value = evaluate(offset, t);
      if (!value) return symbolic.value();
      exact = llvm::commonAlignment(exact, value->getZExtValue());
    }
    return std::max(symbolic.value(), exact.value());
  }

  // Alignment of the descriptor's aligned pointer, before its element offset.
  // Views shift that pointer; subviews/reinterpret_casts only change metadata.
  uint64_t baseAlignment(Value buffer, unsigned depth = 0) {
    if (depth > 128) return 1;
    if (auto alloc = buffer.getDefiningOp<memref::AllocOp>())
      return alloc.getAlignment().value_or(1);
    if (auto alloc = buffer.getDefiningOp<memref::AllocaOp>())
      return alloc.getAlignment().value_or(1);
    if (auto get = buffer.getDefiningOp<memref::GetGlobalOp>()) {
      auto global = SymbolTable::lookupNearestSymbolFrom<memref::GlobalOp>(get, get.getNameAttr());
      return global ? global.getAlignment().value_or(1) : 1;
    }
    if (auto cast = buffer.getDefiningOp<memref::CastOp>())
      return baseAlignment(cast.getSource(), depth + 1);
    if (auto sub = buffer.getDefiningOp<memref::SubViewOp>())
      return baseAlignment(sub.getSource(), depth + 1);
    if (auto cast = buffer.getDefiningOp<memref::ReinterpretCastOp>())
      return baseAlignment(cast.getSource(), depth + 1);
    if (auto view = buffer.getDefiningOp<memref::ViewOp>())
      return offsetAlignment(baseAlignment(view.getSource(), depth + 1),
                             expression(view.getByteShift()));
    return 1;
  }
};

struct Access {
  Operation *op;
  Value buffer;
  SmallVector<Value> indices;
  AffineMap map;
  AffineExpr offset;
  uint64_t alignment = 1;
  bool nontemporal = false;
};

struct CanonicalizeThreadLDSAddressesPass
    : PassWrapper<CanonicalizeThreadLDSAddressesPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeThreadLDSAddressesPass)
  StringRef getArgument() const final { return "frisk-canonicalize-lds-addresses"; }
  StringRef getDescription() const final { return "Canonicalize physical shared load addresses in ThreadIR"; }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<frisk::FriskDialect, affine::AffineDialect>();
  }
  void runOnOperation() final {
    Addresses addresses(getOperation());
    DominanceInfo dominance(getOperation());
    SmallVector<Access> accesses;
    getOperation().walk([&](Operation *op) {
      Access access{op, {}, {}, {}, {}};
      if (auto load = dyn_cast<memref::LoadOp>(op)) {
        access.buffer = load.getMemref();
        access.indices.assign(load.getIndices().begin(), load.getIndices().end());
      } else if (auto load = dyn_cast<vector::LoadOp>(op)) {
        if (load.getVectorType().getRank() != 1 || load.getVectorType().isScalable()) return;
        access.buffer = load.getBase();
        access.indices.assign(load.getIndices().begin(), load.getIndices().end());
      } else if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
        access.buffer = load.getMemRef();
        access.indices.assign(load.getMapOperands().begin(), load.getMapOperands().end());
        access.map = load.getAffineMap();
      } else return; // Includes atomics, matrix reads and already-canonical loads.
      // Do not discard a caller's unrecognized memory contract.
      for (NamedAttribute attr : op->getAttrs())
        if (attr.getName() != "alignment" && attr.getName() != "nontemporal" &&
            attr.getName() != "map") return;
      auto type = cast<MemRefType>(access.buffer.getType());
      auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
      if (!space || space.getInt() != 3 || !type.getElementType().isIntOrFloat()) return;
      unsigned bits = type.getElementTypeBitWidth();
      // For these element widths the physical allocation stride equals the
      // byte width. Do not guess DataLayout padding for e.g. i24 or f80.
      if (bits < 8 || !llvm::isPowerOf2_64(bits)) return;
      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(type.getStridesAndOffset(strides, offset)) || ShapedType::isDynamic(offset) ||
          llvm::any_of(strides, ShapedType::isDynamic)) return;
      if (isa<vector::LoadOp>(op) && !strides.empty() && strides.back() != 1) return;
      SmallVector<AffineExpr> indices;
      for (Value index : access.indices) indices.push_back(addresses.expression(index));
      if (access.map) {
        SmallVector<AffineExpr> mapped;
        for (AffineExpr expr : access.map.getResults())
          mapped.push_back(expr.replaceDimsAndSymbols(
              ArrayRef(indices).take_front(access.map.getNumDims()),
              ArrayRef(indices).drop_front(access.map.getNumDims())));
        indices = std::move(mapped);
      }
      access.offset = getAffineConstantExpr(offset, op->getContext());
      for (auto [index, stride] : llvm::zip(indices, strides))
        access.offset = access.offset + index * stride;
      access.offset = addresses.simplify(access.offset * (bits / 8));
      if (auto alignment = op->getAttrOfType<IntegerAttr>("alignment"))
        access.alignment = alignment.getInt();
      if (auto nt = op->getAttrOfType<BoolAttr>("nontemporal")) access.nontemporal = nt.getValue();
      accesses.push_back(std::move(access));
    });

    struct Rewrite { unsigned load, anchor; int64_t offset; uint64_t alignment; };
    SmallVector<Rewrite> rewrites;
    SmallVector<unsigned> anchors;
    for (auto [i, access] : llvm::enumerate(accesses)) {
      uint64_t alignment = std::max(access.alignment,
          addresses.offsetAlignment(addresses.baseAlignment(access.buffer), access.offset));
      unsigned anchorIndex = i;
      int64_t offset = 0;
      for (unsigned candidate : anchors) {
        auto &anchor = accesses[candidate];
        if (anchor.buffer != access.buffer || !dominance.dominates(anchor.op, access.op)) continue;
        auto delta = addresses.difference(access.offset, anchor.offset);
        if (!delta) continue;
        anchorIndex = candidate;
        offset = *delta;
        break;
      }
      if (anchorIndex == i && anchors.size() < 128) anchors.push_back(i);
      if (anchorIndex != i || alignment > access.alignment)
        rewrites.push_back({unsigned(i), anchorIndex, offset, alignment});
    }

    // Materialize affine anchors before replacing any load, at their original
    // positions. No load is speculated, widened or moved across a barrier.
    DenseMap<unsigned, SmallVector<Value>> anchorIndices;
    for (auto rewrite : rewrites) {
      if (anchorIndices.contains(rewrite.anchor)) continue;
      auto &anchor = accesses[rewrite.anchor];
      auto &indices = anchorIndices[rewrite.anchor];
      OpBuilder b(anchor.op);
      if (anchor.map) {
        for (unsigned i = 0; i < anchor.map.getNumResults(); ++i)
          indices.push_back(b.create<affine::AffineApplyOp>(anchor.op->getLoc(),
              anchor.map.getSubMap({i}), anchor.indices));
      } else indices = anchor.indices;
    }
    for (auto rewrite : rewrites) {
      auto &access = accesses[rewrite.load];
      auto &anchor = accesses[rewrite.anchor];
      OpBuilder b(access.op);
      auto load = b.create<frisk::LDSLoadOp>(access.op->getLoc(), access.op->getResult(0).getType(),
          anchor.buffer, anchorIndices[rewrite.anchor], b.getI64IntegerAttr(rewrite.offset),
          b.getI64IntegerAttr(rewrite.alignment), b.getBoolAttr(access.nontemporal));
      access.op->getResult(0).replaceAllUsesWith(load.getResult());
      access.op->erase();
    }
    if (!rewrites.empty())
      llvm::errs() << "CanonicalizeThreadLDSAddresses: " << getOperation().getName()
                   << ": " << rewrites.size() << " canonical loads\n";
  }
};
} // namespace

std::unique_ptr<Pass> mlir::frisk::createCanonicalizeThreadLDSAddressesPass() {
  return std::make_unique<CanonicalizeThreadLDSAddressesPass>();
}
