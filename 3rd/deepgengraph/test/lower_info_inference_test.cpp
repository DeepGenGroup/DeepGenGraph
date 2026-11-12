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
    %scores_init = frisk.zero : memref<128x128xf32>
    %scores = frisk.gemm(%q, %k, %scores_init) {transA = false, transB = false} : memref<128x32xf16, 3>, memref<32x128xf16, 3>, memref<128x128xf32> -> memref<128x128xf32>
    %shared = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x128xf32, 3>
    %copied = frisk.copy %scores to %shared [affine_map<() -> (2)>] : memref<128x128xf32>, memref<128x128xf32, 3> -> memref<128x128xf32, 3>
    %row = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
    frisk.reduce %copied, %row {dim = 1 : i64, kind = "max"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
    %scaled = frisk.mul %old, %row : vector<128x32xf32>, memref<128x1xf32, 3> -> memref<128x32xf32>
    %pv_init = frisk.zero : memref<128x32xf32>
    %pv = frisk.gemm(%p, %v, %pv_init) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3>, memref<128x32xf32> -> memref<128x32xf32>
    %sum = frisk.add %scaled, %pv : memref<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
    return
  }
}
)mlir";

static void testIndependentLoopAxes(MLIRContext &context) {
  using Attr = BufferAxisCalcuateAttr;
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @loop(%a: memref<32x32xf16, 3>,
                      %b: memref<32x32xf16, 3>)
          attributes {thread_num = 64 : i32} {
        %out:2 = affine.for %i = 0 to 2
            iter_args(%lhs = %a, %rhs = %b)
            -> (memref<32x32xf16, 3>, memref<32x32xf16, 3>) {
          %c_init = frisk.zero : memref<32x32xf32>
          %c = frisk.gemm(%lhs, %rhs, %c_init) {transA = false, transB = false} : memref<32x32xf16, 3>, memref<32x32xf16, 3>, memref<32x32xf32> -> memref<32x32xf32>
          affine.yield %lhs, %rhs : memref<32x32xf16, 3>, memref<32x32xf16, 3>
        }
        return
      }
    }
  )mlir", &context);
  check(bool(module), "parse independent loop axes fixture");
  auto func = *module->getOps<func::FuncOp>().begin();
  affine::AffineForOp loop;
  func.walk([&](affine::AffineForOp op) { loop = op; });
  auto *layouts = LowerInfoAnalysis::run(func);
  for (unsigned slot = 0; slot < 2; ++slot) {
    std::array<Attr, 2> expected{Attr::Normal, Attr::Normal};
    expected[1 - slot] = Attr::InnerWarpReduce;
    SmallVector<Value, 3> values{loop.getInits()[slot],
                                loop.getRegionIterArgs()[slot],
                                loop.getResult(slot)};
    for (Value value : values) {
      auto *info = layouts->getLowerInfo(value, loop);
      check(info && info->calculateAttr == expected,
            "same-shaped loop carriers must keep independent reduction axes");
    }
    auto *yield = loop.getBody()->getTerminator();
    auto *info = layouts->getLowerInfo(yield->getOperand(slot), yield);
    check(info && info->calculateAttr == expected,
          "loop yields retain their own slot's reduction requirements");
  }
}

static void testChainedGemm(MLIRContext &context) {
  using Attr = BufferAxisCalcuateAttr;
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      func.func @chain(%a: memref<128x32xf16, 3>,
                       %b: memref<32x128xf16, 3>,
                       %d: memref<128x32xf16, 3>)
          attributes {thread_num = 256 : i32,
                      frisk.block_warps = array<i64: 2, 2>} {
        %c_init = frisk.zero : memref<128x128xf32>
        %c = frisk.gemm(%a, %b, %c_init) {transA = false, transB = false} : memref<128x32xf16, 3>, memref<32x128xf16, 3>, memref<128x128xf32> -> memref<128x128xf32>
        %half = frisk.cast %c : memref<128x128xf32> -> memref<128x128xf16>
        %e_init = frisk.zero : memref<128x32xf32>
        %e = frisk.gemm(%half, %d, %e_init) {transA = false, transB = false} : memref<128x128xf16>, memref<128x32xf16, 3>, memref<128x32xf32> -> memref<128x32xf32>
        return
      }
    }
  )mlir", &context);
  check(bool(module), "parse chained GEMM fixture");
  auto func = *module->getOps<func::FuncOp>().begin();
  SmallVector<GemmOp> gemms;
  CastOp castOp;
  func.walk([&](GemmOp op) { gemms.push_back(op); });
  func.walk([&](CastOp op) { castOp = op; });
  auto *layouts = LowerInfoAnalysis::run(func);
  for (auto gemm : gemms) {
    auto *seed = layouts->getLowerInfo(gemm.getAcc(), gemm);
    auto *result = layouts->getLowerInfo(gemm.getC(), gemm);
    check(seed && result && seed->pos == LowerInfo::BufPos::In &&
              seed->get_thread_own_data_size() == result->get_thread_own_data_size() &&
              seed->get_thread_widths() == result->get_thread_widths() &&
              seed->get_warp_repeat() == result->get_warp_repeat() &&
              seed->get_warp_layout() == result->get_warp_layout(),
          "GEMM accumulator inherits C's register layout");
  }
  auto *c = layouts->getLowerInfo(gemms.front().getResult(), gemms.front());
  auto *castInput = layouts->getLowerInfo(castOp->getOperand(0), castOp);
  auto *castOutput = layouts->getLowerInfo(castOp->getResult(0), castOp);
  auto *a = layouts->getLowerInfo(gemms.back().getMatrixA(), gemms.back());
  auto *b = layouts->getLowerInfo(gemms.back().getMatrixB(), gemms.back());
  auto *e = layouts->getLowerInfo(gemms.back().getResult(), gemms.back());
  check(c && castInput && castOutput && a && b && e,
        "resolve all chained GEMM uses");
  for (auto *info : {c, castInput, castOutput, a})
    check(info->calculateAttr == std::array<Attr, 2>{Attr::Normal,
                                                    Attr::CrossWarpReduce},
          "propagate the strictest K requirement across a cast and conversion");
  check(b->calculateAttr == std::array<Attr, 2>{Attr::InnerWarpReduce,
                                               Attr::Normal},
        "independent B operand keeps its own reduction scope");
  check(e->calculateAttr == std::array<Attr, 2>{Attr::Normal, Attr::Normal},
        "GEMM output does not inherit its operands' K reduction");
  bool hasConversion = false;
  for (auto &entry : *layouts) {
    auto &info = entry.second;
    if (info.convertFrom) {
      hasConversion = true;
      check(info.calculateAttr == info.convertFrom->calculateAttr,
            "conversion source candidates receive merged requirements");
    }
  }
  check(hasConversion, "fixture must exercise conflicting GEMM layouts");

  // Re-inserting an identical physical candidate must merge semantic
  // requirements without introducing a layout conversion.
  LowerInfoMap candidates;
  candidates.getOpsOrder(func);
  LowerInfo normal = *e;
  normal.convertFrom = nullptr;
  candidates.addLowerInfo(gemms.back(), normal);
  LowerInfo stricter = normal;
  stricter.calculateAttr[1] = Attr::UnknownReduce;
  candidates.addLowerInfo(gemms.back(), stricter);
  candidates.conflictResolve();
  auto *merged = candidates.getLowerInfo(normal.buffer, gemms.back());
  check(merged && merged->calculateAttr[1] == Attr::CrossWarpReduce &&
            !merged->convertFrom,
        "merge duplicate metadata and resolve a single remaining candidate");
  candidates.conflictResolve();
  check(candidates.getLowerInfo(normal.buffer, gemms.back())->calculateAttr ==
            std::array<Attr, 2>{Attr::Normal, Attr::CrossWarpReduce},
        "calculation attribute resolution is idempotent");
}

// The acc edge must carry the public distribution. Native C is an explicit
// instruction constraint, including any logical B permutation, not another
// competing inference candidate for the same SSA value.
static void testGemmLayoutContracts(MLIRContext &context) {
  for (int mode = 0; mode != 5; ++mode) {
    std::string source = R"mlir(
      module {
        func.func @contracts(%a: memref<32x32xf16, 3>,
                             %b: memref<32x64xf16, 3>,
                             %acc: memref<32x64xf32>)
            attributes {thread_num = 64 : i32} {
          %c = frisk.gemm(%a, %b, %acc) {transA = false, transB = false}
              : memref<32x32xf16, 3>, memref<32x64xf16, 3>,
                memref<32x64xf32> -> memref<32x64xf32>
          return
        }
      }
    )mlir";
    // Inter-instruction B permutation requires 64 columns. Unsupported N
    // still needs distinct native/result layouts for its shuffle fallback.
    if (mode == 4)
      for (size_t pos = 0; (pos = source.find("x64", pos)) != std::string::npos;)
        source.replace(pos, 3, "x32");
    auto module = parseSourceString<ModuleOp>(source, &context);
    check(bool(module), "parse GEMM layout contracts fixture");
    auto func = *module->getOps<func::FuncOp>().begin();
    GemmOp gemm;
    func.walk([&](GemmOp op) { gemm = op; });
    if (mode)
      gemm->setAttr("frisk.contiguous_c", UnitAttr::get(&context));
    if (mode == 2 || mode == 4)
      gemm->setAttr("frisk.b_permute", UnitAttr::get(&context));
    if (mode == 3)
      gemm->setAttr("frisk.inner_inst_b_permute", UnitAttr::get(&context));
    auto *infos = LowerInfoAnalysis::run(func);
    auto *layouts = infos->getGemmLayouts(gemm);
    check(layouts, "every inferred GEMM records both layout contracts");
    auto computation = layouts->computation;
    auto result = layouts->result;
    auto *acc = infos->getLowerInfo(gemm.getAcc(), gemm);
    auto *output = infos->getLowerInfo(gemm.getC(), gemm);
    check(acc && output && acc->getAffineMap() == result.getAffineMap() &&
              output->getAffineMap() == result.getAffineMap(),
          "acc and downstream inference use the public result contract");
    check(computation.base_layout.thread_creg == computation.mmaInst->desc_c.thread_creg &&
              computation.base_layout.warp_repeat == computation.mmaInst->desc_c.warp_repeat,
          "computation C retains the hardware register distribution");
    bool permuted = mode == 2 || mode == 3;
    check(bool(computation.logicalTransform) == permuted &&
              computation.logicalTransform == result.logicalTransform,
          "both contracts preserve the selected B permutation's logical columns");
    check((computation.getAffineMap() != result.getAffineMap()) ==
              (mode == 1 || mode == 4),
          "only contiguous C/fallback require an explicit layout boundary");
  }
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<FriskDialect, arith::ArithDialect, func::FuncDialect,
                  affine::AffineDialect>();
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
  using Attr = BufferAxisCalcuateAttr;
  ReduceOp reduce;
  CopyOp copy;
  func.walk([&](ReduceOp op) { reduce = op; });
  func.walk([&](CopyOp op) { copy = op; });
  auto *reduceSrc = layouts->getLowerInfo(reduce.getSrc(), reduce);
  auto *reduceDst = layouts->getLowerInfo(reduce.getDst(), reduce);
  check(reduceSrc && reduceSrc->pos == LowerInfo::BufPos::In && reduceDst,
        "reduction registers both input and output candidates");
  check(reduceSrc->calculateAttr[1] == Attr::InnerWarpReduce &&
            scores->calculateAttr[1] == Attr::InnerWarpReduce &&
            layouts->getLowerInfo(copy.getDstMemRef(), copy)->calculateAttr[1] ==
                Attr::InnerWarpReduce,
        "propagate a late reduction requirement through every copy value");
  check(reduceDst->calculateAttr[1] == Attr::Normal &&
            row->calculateAttr[1] == Attr::Normal &&
            scaled->calculateAttr[1] == Attr::Normal,
        "the reduced/broadcast axis must stop reduction propagation");
  for (auto &entry : *layouts)
    for (Attr attr : entry.second.calculateAttr)
      check(attr != Attr::UnknownReduce, "resolve every reduction scope");
  func->setAttr("frisk.block_warps",
                DenseI64ArrayAttr::get(&context, {2, 2}));
  layouts = LowerInfoAnalysis::run(func);
  check(layouts->getLowerInfo(reduce.getSrc(), reduce)->calculateAttr[1] ==
            Attr::CrossWarpReduce &&
            layouts->getLowerInfo(reduce.getDst(), reduce)->calculateAttr[1] ==
                Attr::Normal,
        "a reduction across column waves is not classified as warp-local");
  testGemmLayoutContracts(context);
  testChainedGemm(context);
  testIndependentLoopAxes(context);
  llvm::outs() << "LowerInfo deferred broadcast inference tests passed\n";
}
