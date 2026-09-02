#ifndef DEEPGENGRAPH_COMMON_H
#define DEEPGENGRAPH_COMMON_H

#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonDialect.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/Transforms/Passes.h"
#include "deepgengraph/Dialect/TL/IR/TilelangDialect.h"
#include "deepgengraph/Dialect/TL/Transforms/Passes.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <memory>
#include <vector>
#include "deepgengraph/Conversion/DeepgengraphToLinalgOnTensor/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/InitAllExtensions.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include <cassert>
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllPasses.h"

#include "deepgengraph/Analysis/ThreadAnalysis.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Analysis/LowerInfo.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"

#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"

namespace mlir::frisk {
enum class Target : int32_t { ROCm = 1, CUDA = 2 };

struct TargetInfo {
  Target target;
  std::string arch;
};

constexpr char* GRID_DIM =   "grid_dims" ;
constexpr char* THREAD_NUM =   "thread_num" ;

struct KernelConfig {
  std::array<int,3> gridDimXYZ = {0,0,0};
  int num_threads = 0;
};

KernelConfig* GetKernelConfig();
void AppendNameToLoc( mlir::Operation* targetOp);

} // namespace mlir::firsk

#endif  // DEEPGENGRAPH_COMMON_H
