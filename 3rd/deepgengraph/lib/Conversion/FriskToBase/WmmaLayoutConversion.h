#ifndef DEEPGENGRAPH_WMMA_LAYOUT_CONVERSION_H
#define DEEPGENGRAPH_WMMA_LAYOUT_CONVERSION_H

#include "deepgengraph/Analysis/WmmaLayout.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::frisk {
// Serializable metadata lets the planning pass operate independently of the
// ephemeral LowerInfoMap. The lowering revalidates the plan before using it.
void annotateWmmaLayoutConversion(ConvertLayoutOp op, const LowerInfo &from,
                                  const LowerInfo &to);
void planWmmaLayoutConversions(func::FuncOp kernel, bool contiguousOutputs);
// Select only GEMM producers on an elementwise/loop-carried path to another
// GEMM's A input (the P boundary in attention), never terminal GEMM outputs.
void selectGemmAInputLayoutMethod(func::FuncOp kernel, StringRef method);
DictionaryAttr getWmmaShufflePlan(ConvertLayoutOp op);
} // namespace mlir::frisk
#endif
