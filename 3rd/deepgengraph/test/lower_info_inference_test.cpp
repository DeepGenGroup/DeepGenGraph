#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include <cstdlib>

using namespace mlir;
using namespace mlir::frisk;

static void check(bool condition, const char *message) {
  if (!condition) {
    llvm::errs() << "FAIL: " << message << '\n';
    std::exit(1);
  }
}

// The QK row is known before the output tile. Its inherited column repeat
// describes 128 columns, while O has only 32. The multiply must wait until
// the add propagates the PV layout backwards, then complete on a later sweep.
static constexpr char fixture[] = R"mlir(
module {
  func.func @broadcast(%q: memref<128x32xf16, 3>,
                       %k: memref<32x128xf16, 3>,
                       %p: memref<128x128xf16, 3>,
                       %v: memref<128x32xf16, 3>,
                       %old: vector<128x32xf32>)
      attributes {thread_num = 256 : i32,
                  frisk.block_warps = array<i64: 4, 1>} {
    %scores = frisk.gemm(%q, %k) {transA = false, transB = false} : memref<128x32xf16, 3>, memref<32x128xf16, 3> -> memref<128x128xf32>
    %shared = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x128xf32, 3>
    %copied = frisk.copy %scores to %shared [affine_map<() -> (2)>] : memref<128x128xf32>, memref<128x128xf32, 3> -> memref<128x128xf32, 3>
    %row = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
    frisk.reduce %copied, %row {dim = 1 : i64, kind = "max"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
    %scaled = frisk.mul %old, %row : vector<128x32xf32>, memref<128x1xf32, 3> -> memref<128x32xf32>
    %pv = frisk.gemm(%p, %v) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3> -> memref<128x32xf32>
    %sum = frisk.add %scaled, %pv : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
    return
  }
}
)mlir";

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<FriskDialect, arith::ArithDialect, func::FuncDialect>();
  MLIRContext context(registry);
  StringRef mode = argc > 1 ? argv[1] : "";
  if (mode == "--no-anchor") {
    auto empty = parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func @no_anchor() attributes {thread_num = 256 : i32} {
          return
        }
      }
    )mlir", &context);
    check(bool(empty), "parse missing-anchor fixture");
    LowerInfoAnalysis::run(*empty->getOps<func::FuncOp>().begin());
    check(false, "missing anchor must fail");
  }
  auto module = parseSourceString<ModuleOp>(fixture, &context);
  check(bool(module), "parse inference fixture");
  auto func = *module->getOps<func::FuncOp>().begin();
  MulOp mul;
  AddOp add;
  SmallVector<GemmOp> gemms;
  func.walk([&](MulOp op) { mul = op; });
  func.walk([&](AddOp op) { add = op; });
  func.walk([&](GemmOp op) { gemms.push_back(op); });

  // Exercise genuine inference failures as subprocesses: neither may hang or
  // report success when built with NDEBUG.
  if (mode == "--unresolved")
    add.erase();

  auto *layouts = LowerInfoAnalysis::run(func);
  check(mode.empty(), "invalid inference must fail");
  auto *pv = layouts->getLowerInfo(gemms.back().getResult(), gemms.back());
  auto *scores = layouts->getLowerInfo(gemms.front().getResult(), gemms.front());
  auto *scaled = layouts->getLowerInfo(mul.getResult(), mul);
  auto *row = layouts->getLowerInfo(mul->getOperand(1), mul);
  check(pv && scores && scaled && row, "all layouts resolved");
  check(pv->block_repeat[1] != scores->block_repeat[1],
        "fixture must distinguish score and output column coverage");
  check(scaled->block_repeat == pv->block_repeat &&
            scaled->get_thread_own_data_size() == pv->get_thread_own_data_size(),
        "broadcast multiply must inherit the full PV output layout");
  check(row->ignoreDim == 1 && row->thread_own_data_size[1] == 1,
        "broadcast row must retain singleton storage");
  check(scaled->ignoreDim == -1, "full output must not inherit the reduced axis");
  llvm::outs() << "LowerInfo deferred broadcast inference tests passed\n";
}
