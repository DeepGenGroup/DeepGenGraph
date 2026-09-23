#include "deepgengraph/Common.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::frisk::FriskDialect>();
  mlir::registerPass([] {
    return mlir::frisk::createConvertFriskBaseToThreadLevelIRPass();
  });
  mlir::registerPass([] {
    return mlir::frisk::createFinalizeThreadTilingPass();
  });
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Thread tiling tests\n", registry));
}
