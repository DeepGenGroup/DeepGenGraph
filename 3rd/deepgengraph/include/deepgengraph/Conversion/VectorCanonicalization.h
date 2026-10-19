#ifndef DEEPGENGRAPH_CONVERSION_VECTORCANONICALIZATION_H
#define DEEPGENGRAPH_CONVERSION_VECTORCANONICALIZATION_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::frisk {

// The LLVM 24 FromElementsToShapeCast pattern compares getStaticPosition()
// and constructs an extract from those integers alone. For extracts such as
// source[row, 0..3], this loses the dynamic row operand but retains kDynamic.
// Different dynamic rows also compare equal, so repairing only the malformed
// result would not suffice. Filter this one pattern at every place where we
// collect canonicalizations, including the post-unroll greedy driver. Keep
// other vector folds, canonicalization and verification enabled. This is a
// project-local compatibility workaround; no installed LLVM changes required.
inline llvm::ArrayRef<std::string> unsafeVectorCanonicalizations() {
  static const std::string names[] = {"FromElementsToShapeCast"};
  return names;
}

inline std::unique_ptr<Pass> createVectorSafeCanonicalizerPass() {
  // Preserve the canonicalizer's own defaults. GreedyRewriteConfig defaults
  // differ (bottom-up/aggressive versus top-down/normal region simplification).
  CanonicalizerPassOptions options;
  auto disabled = unsafeVectorCanonicalizations();
  options.disabledPatterns.append(disabled.begin(), disabled.end());
  return mlir::createCanonicalizerPass(options);
}

} // namespace mlir::frisk
#endif
