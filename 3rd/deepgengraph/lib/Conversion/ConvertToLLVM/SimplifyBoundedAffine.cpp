// ===========================================================================
// Pass: SimplifyBoundedAffine（frisk-simplify-bounded-affine）   优化汇总.py 第 17 项，thread 阶段，未实测
//
// 作用：利用循环上下界证明 affine.apply 里的 floordiv / mod 可以消掉，去掉有符号除法，并让更多
// 地址计算变成循环不变量，由 secondLowering 的 LICM 外提。
//
// 只改 affine.apply 的 map。下面是原文件头的恒等式说明。
// ===========================================================================
// Remove affine floordiv/mod whose result is decided by constant loop bounds.
//
// Tile arithmetic produces maps such as
//   ((iv + s0 * 4096 + s1 * 131072 + 1024) floordiv 4096) mod 32
// with iv in [0, 32). Upstream simplification does not use loop bounds, so the
// signed division survives lowering. Exact integer identities used here:
//   (c * E + R) floordiv c == E + k    when R is in [k * c, (k + 1) * c)
//   (m * E + R) mod m      == R - k * m when R is in [k * m, (k + 1) * m)
// Both hold for any integer E, so unbounded symbols are fine when their
// coefficients are multiples of the divisor. The remainder must be bounded.
#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <optional>

using namespace mlir;

namespace {
// 闭区间 [low, high]。std::optional 为空表示无界或未知。
struct Range {
  int64_t low, high;
};

std::optional<Range> add(std::optional<Range> a, std::optional<Range> b) {
  int64_t low, high;
  if (!a || !b || llvm::AddOverflow(a->low, b->low, low) || llvm::AddOverflow(a->high, b->high, high))
    return std::nullopt;
  return Range{low, high};
}

std::optional<Range> multiply(std::optional<Range> a, std::optional<Range> b) {
  if (!a || !b) return std::nullopt;
  int64_t values[4];
  if (llvm::MulOverflow(a->low, b->low, values[0]) || llvm::MulOverflow(a->low, b->high, values[1]) ||
      llvm::MulOverflow(a->high, b->low, values[2]) || llvm::MulOverflow(a->high, b->high, values[3]))
    return std::nullopt;
  return Range{*std::min_element(values, values + 4), *std::max_element(values, values + 4)};
}

// dims / symbols：affine map 每个 dim、symbol 的范围。只有操作数是常量边界的循环 iv 时才有值。
class BoundedSimplifier {
  ArrayRef<std::optional<Range>> dims, symbols;

  std::optional<Range> range(AffineExpr expr) const {
    if (auto constant = dyn_cast<AffineConstantExpr>(expr))
      return Range{constant.getValue(), constant.getValue()};
    if (auto dim = dyn_cast<AffineDimExpr>(expr)) return dims[dim.getPosition()];
    if (auto symbol = dyn_cast<AffineSymbolExpr>(expr)) return symbols[symbol.getPosition()];
    auto binary = cast<AffineBinaryOpExpr>(expr);
    auto left = range(binary.getLHS()), right = range(binary.getRHS());
    switch (expr.getKind()) {
    case AffineExprKind::Add: return add(left, right);
    case AffineExprKind::Mul: return multiply(left, right);
    case AffineExprKind::FloorDiv:
      if (left && right && right->low == right->high && right->low > 0)
        return Range{llvm::divideFloorSigned(left->low, right->low),
                     llvm::divideFloorSigned(left->high, right->low)};
      return std::nullopt;
    case AffineExprKind::CeilDiv:
      if (left && right && right->low == right->high && right->low > 0)
        return Range{llvm::divideCeilSigned(left->low, right->low),
                     llvm::divideCeilSigned(left->high, right->low)};
      return std::nullopt;
    case AffineExprKind::Mod:
      // An affine mod by a positive constant is Euclidean, whatever the LHS is.
      if (right && right->low == right->high && right->low > 0)
        return Range{0, right->low - 1};
      return std::nullopt;
    default: return std::nullopt;
    }
  }

  // 表达式是否用到循环 iv。没有用到就不简化，这正是“线程坐标表达式不折叠”的实现方式。
  bool usesBoundedLoop(AffineExpr expr) const {
    bool found = false;
    expr.walk([&](AffineExpr leaf) {
      if (auto dim = dyn_cast<AffineDimExpr>(leaf)) found |= dims[dim.getPosition()].has_value();
      if (auto symbol = dyn_cast<AffineSymbolExpr>(leaf)) found |= symbols[symbol.getPosition()].has_value();
    });
    return found;
  }

  // Split a sum into the part that is a multiple of `divisor` and the rest.
  // 拆分 expr = divisor * quotient + rest。系数能被 divisor 整除的项进 quotient，其它进 rest。
  // 符号项的系数被整除时也进 quotient，所以符号可以无界。
  static void split(AffineExpr expr, int64_t divisor, AffineExpr &quotient, AffineExpr &rest) {
    if (expr.getKind() == AffineExprKind::Add) {
      auto sum = cast<AffineBinaryOpExpr>(expr);
      split(sum.getLHS(), divisor, quotient, rest);
      split(sum.getRHS(), divisor, quotient, rest);
      return;
    }
    if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
      if (constant.getValue() % divisor == 0) quotient = quotient + constant.getValue() / divisor;
      else rest = rest + expr;
      return;
    }
    if (expr.getKind() == AffineExprKind::Mul) {
      auto product = cast<AffineBinaryOpExpr>(expr);
      auto coefficient = dyn_cast<AffineConstantExpr>(product.getRHS());
      if (coefficient && coefficient.getValue() % divisor == 0) {
        quotient = quotient + product.getLHS() * (coefficient.getValue() / divisor);
        return;
      }
    }
    rest = rest + expr;
  }

public:
  BoundedSimplifier(ArrayRef<std::optional<Range>> dims, ArrayRef<std::optional<Range>> symbols)
      : dims(dims), symbols(symbols) {}

  // Returns `expr` itself unless a loop bound proves a simplification. Thread
  // expressions have no loop operand and are deliberately left alone, even when
  // their own mod bounds would allow folding: the LLVM stage proves LDS addresses
  // from those exact forms, and folding them here makes its V and K analysis fail.
  AffineExpr simplify(AffineExpr expr) const {
    auto binary = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!binary) return expr;
    AffineExpr left = simplify(binary.getLHS()), right = simplify(binary.getRHS());
    bool childChanged = left != binary.getLHS() || right != binary.getRHS();
    auto kind = expr.getKind();
    auto constant = dyn_cast<AffineConstantExpr>(right);
    if ((kind == AffineExprKind::FloorDiv || kind == AffineExprKind::Mod) && constant && constant.getValue() > 0) {
      int64_t divisor = constant.getValue();
      AffineExpr zero = getAffineConstantExpr(0, expr.getContext()), quotient = zero, rest = zero;
      split(left, divisor, quotient, rest);
      // bounds：rest 的范围。
      // period：如果 rest 的整个范围都落在同一个 divisor 周期内，就是这个周期号 k，对应文件头公式的 k。
      std::optional<Range> bounds;
      if (usesBoundedLoop(rest)) bounds = range(rest);
      std::optional<int64_t> period;
      if (bounds) {
        int64_t low = llvm::divideFloorSigned(bounds->low, divisor);
        if (low == llvm::divideFloorSigned(bounds->high, divisor)) period = low;
      }
      if (kind == AffineExprKind::FloorDiv && period) return quotient + *period;
      if (kind == AffineExprKind::Mod && period) return rest - *period * divisor;
      // Multiples exposed by an inner proof can be dropped from an outer mod.
      // 内层化简后暴露出的倍数项，可以直接从外层 mod 中去掉，不需要 rest 有界。
      if (kind == AffineExprKind::Mod && childChanged) return rest % divisor;
    }
    return childChanged ? getAffineBinaryOpExpr(kind, left, right) : expr;
  }
};

// value 是常量边界 affine.for 的 iv 时返回 [lb, ub-1]。忽略 step，只会让范围更宽，仍然安全。
std::optional<Range> loopRange(Value value) {
  auto loop = affine::getForInductionVarOwner(value);
  if (!loop || !loop.hasConstantBounds()) return std::nullopt;
  int64_t low = loop.getConstantLowerBound(), high = loop.getConstantUpperBound();
  if (high <= low) return std::nullopt;
  return Range{low, high - 1};
}

struct SimplifyBoundedAffinePass
    : PassWrapper<SimplifyBoundedAffinePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplifyBoundedAffinePass)
  StringRef getArgument() const final { return "frisk-simplify-bounded-affine"; }
  StringRef getDescription() const final {
    return "Fold affine floordiv/mod decided by constant affine.for bounds";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<affine::AffineDialect>();
  }
  void runOnOperation() final {
    unsigned changed = 0;
    getOperation().walk([&](affine::AffineApplyOp op) {
      AffineMap map = op.getAffineMap();
      SmallVector<std::optional<Range>> bounds;
      for (Value operand : op.getMapOperands()) bounds.push_back(loopRange(operand));
      ArrayRef<std::optional<Range>> all(bounds);
      BoundedSimplifier simplifier(all.take_front(map.getNumDims()), all.drop_front(map.getNumDims()));
      SmallVector<AffineExpr> results;
      for (AffineExpr result : map.getResults()) results.push_back(simplifier.simplify(result));
      if (llvm::equal(results, map.getResults())) return;
      op.setMap(simplifyAffineMap(
          AffineMap::get(map.getNumDims(), map.getNumSymbols(), results, op.getContext())));
      ++changed;
    });
    if (changed)
      llvm::errs() << "SimplifyBoundedAffine: " << getOperation().getName() << ": "
                   << changed << " affine.apply maps\n";
  }
};
} // namespace

std::unique_ptr<Pass> mlir::frisk::createSimplifyBoundedAffinePass() {
  return std::make_unique<SimplifyBoundedAffinePass>();
}
