#include "deepgengraph/Conversion/ConvertToLLVM/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::vector::VectorDialect>();
  mlir::registerPass([] { return mlir::frisk::createFuseFragmentAccumulatorPass(); });
  mlir::registerPass([] { return mlir::frisk::createIRDeepOptimizePass(); });
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Fragment optimization tests\n", registry));
}
