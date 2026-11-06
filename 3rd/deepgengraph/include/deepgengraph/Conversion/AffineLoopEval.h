// 在给定的循环迭代上求值 index 表达式。
// FuseFragmentAccumulator 和 FriskFragIRReorder 靠它做“有界证明”：循环边界是小常量时，
// 逐次迭代算出坐标再比较。求值不了就返回空，调用方放弃优化。
#ifndef DEEPGENGRAPH_CONVERSION_AFFINELOOPEVAL_H
#define DEEPGENGRAPH_CONVERSION_AFFINELOOPEVAL_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Matchers.h"
#include <optional>

namespace mlir::frisk {

// value 在 loop 的循环变量等于 iv 时的值。支持循环变量、整型常量、affine.apply。
inline std::optional<int64_t> evaluateAtIteration(Value value, affine::AffineForOp loop,
                                                  int64_t iv, unsigned depth = 0) {
  if (depth > 64) return std::nullopt;
  if (value == loop.getInductionVar()) return iv;
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant))) return constant.getSExtValue();
  auto apply = value.getDefiningOp<affine::AffineApplyOp>();
  if (!apply) return std::nullopt;
  SmallVector<Attribute> inputs, results;
  for (Value operand : apply.getMapOperands()) {
    auto result = evaluateAtIteration(operand, loop, iv, depth + 1);
    if (!result) return std::nullopt;
    inputs.push_back(IntegerAttr::get(IndexType::get(value.getContext()), *result));
  }
  if (failed(apply.getAffineMap().constantFold(inputs, results))) return std::nullopt;
  return cast<IntegerAttr>(results.front()).getInt();
}

} // namespace mlir::frisk

#endif // DEEPGENGRAPH_CONVERSION_AFFINELOOPEVAL_H
