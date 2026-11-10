// Standalone bridge for testing Base IR without rerunning the frontend.
#include "deepgengraph/Pipeline/ScheduleJSON.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::frisk::FriskDialect, mlir::deepgengraph::DeepgengraphDialect>();
  mlir::MLIRContext context(registry);
  mlir::pipeline::ScheduleJSONOptions options;
  if (argc < 2) { llvm::errs() << "usage: FriskScheduleJSONTest input.mlir --export=file [--import=file] [--baseline=file] [--loop=function:id] [--ir=file]\n"; return 1; }
  for (int i = 2; i < argc; ++i) {
    llvm::StringRef arg(argv[i]);
    if (arg.consume_front("--export=")) options.exportPath = arg.str();
    else if (arg.consume_front("--import=")) options.importPath = arg.str();
    else if (arg.consume_front("--baseline=")) options.baselinePath = arg.str();
    else if (arg.consume_front("--loop=")) options.loopSelector = arg.str();
    else if (arg.consume_front("--ir=")) options.dumpIRPath = arg.str();
    else { llvm::errs() << "unknown argument: " << arg << '\n'; return 1; }
  }
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(argv[1], &context);
  return !module || mlir::failed(mlir::pipeline::processScheduleJSON(*module, options));
}
