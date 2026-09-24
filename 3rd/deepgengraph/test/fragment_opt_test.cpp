#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::vector::VectorDialect>();
  registry.insert<mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect>();
  mlir::registerPass([] { return mlir::frisk::createFuseFragmentAccumulatorPass(); });
  mlir::registerPass([] { return mlir::frisk::createIRDeepOptimizePass(); });
  mlir::registerPass([] { return mlir::frisk::createBarrierOptimizePass(); });
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Fragment optimization tests\n", registry));
}
