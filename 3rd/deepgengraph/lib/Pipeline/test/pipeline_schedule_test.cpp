//===- pipeline_schedule_test.cpp -----------------------------------------===//
//
// Structural test for the frisk-pipeline-schedule pass.
//
// Usage:
//   FriskPipelineTest <input.mlir> [--idempotence] [--bad-annotation]
//                                    [--sync-check [base.mlir]]
//
//===----------------------------------------------------------------------===//

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Pipeline/PipelineSchedule.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <string>

using namespace mlir;

namespace {

int failures = 0;

void check(bool condition, const std::string &message) {
  if (condition) {
    llvm::outs() << "[ ok ] " << message << "\n";
  } else {
    llvm::errs() << "[FAIL] " << message << "\n";
    ++failures;
  }
}

// --- small helpers ---------------------------------------------------------

int64_t slotOf(Value value) {
  auto alloc = value.getDefiningOp<frisk::AllocBufferOp>();
  if (!alloc)
    return -1;
  auto attr = alloc->getAttrOfType<IntegerAttr>("pipeline_slot");
  return attr ? attr.getInt() : -1;
}

/// Slots the two branches of the runtime slot selector yield for `handle`, i.e.
/// `{slot used for an even tile, slot used for an odd tile}`.  `{-1, -1}` when
/// the value is not a result of such a selector.
std::pair<int64_t, int64_t> slotPairOf(Value handle) {
  auto ifOp = handle.getDefiningOp<scf::IfOp>();
  auto result = dyn_cast<OpResult>(handle);
  if (!ifOp || !result)
    return {-1, -1};
  unsigned index = result.getResultNumber();
  auto slotFromYield = [&](Region &region) {
    auto yield = cast<scf::YieldOp>(region.front().getTerminator());
    return index < yield.getNumOperands() ? slotOf(yield.getOperand(index)) : -1;
  };
  return {slotFromYield(ifOp.getThenRegion()),
          slotFromYield(ifOp.getElseRegion())};
}

/// Is `handle` the slot `even` / `odd` on an even / odd tile?
bool selectsSlots(Value handle, int64_t even, int64_t odd) {
  return slotPairOf(handle) == std::make_pair(even, odd);
}

std::string slotsOf(Value handle) {
  auto slots = slotPairOf(handle);
  return "{" + std::to_string(slots.first) + " even, " +
         std::to_string(slots.second) + " odd}";
}

/// The function argument a `frisk.copy` loads from (through a buffer_view).
Value copySourceArgument(frisk::CopyOp copy) {
  auto view = copy.getSrcMemRef().getDefiningOp<frisk::BufferViewOp>();
  return view ? view.getSource() : Value();
}

SmallVector<Operation *> topLevelOps(Region &region) {
  SmallVector<Operation *> ops;
  for (Operation &op : region.front())
    if (!op.hasTrait<OpTrait::IsTerminator>())
      ops.push_back(&op);
  return ops;
}

SmallVector<Operation *> topLevelOps(Block &block) {
  SmallVector<Operation *> ops;
  for (Operation &op : block)
    if (!op.hasTrait<OpTrait::IsTerminator>())
      ops.push_back(&op);
  return ops;
}

template <typename T> unsigned countOf(ArrayRef<Operation *> ops) {
  unsigned count = 0;
  for (Operation *op : ops)
    if (isa<T>(op))
      ++count;
  return count;
}

template <typename T> unsigned countIn(Region &region) {
  unsigned count = 0;
  region.walk([&](T) { ++count; });
  return count;
}

affine::AffineForOp findScheduledLoop(func::FuncOp func) {
  affine::AffineForOp result;
  func.walk([&](affine::AffineForOp loop) {
    if (!result && loop->hasAttr("pipeline.scheduled"))
      result = loop;
  });
  return result;
}

// --- sync check against the frozen input IR --------------------------------

std::string printModule(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream os(text);
  module.print(os);
  return text;
}

/// The first line the two dumps disagree on, as a message.
std::string firstLineDifference(StringRef lhs, StringRef rhs) {
  SmallVector<StringRef> left, right;
  lhs.split(left, '\n');
  rhs.split(right, '\n');
  for (size_t i = 0; i < std::max(left.size(), right.size()); ++i) {
    StringRef a = i < left.size() ? left[i] : StringRef("<end of file>");
    StringRef b = i < right.size() ? right[i] : StringRef("<end of file>");
    if (a != b)
      return "line " + std::to_string(i + 1) + "\n       input: " + a.str() +
             "\n       base : " + b.str();
  }
  return "";
}

/// Looks for the IR the input was copied from, i.e. `test/test_friskBase*.mlir`
/// in one of the ancestors of the input file.  The debug snapshot of the
/// attention kernel is the current base; the older snapshot is still looked up
/// so the test keeps working if either of the two is renamed.
std::string findBaseFile(StringRef inputPath) {
  static const char *const names[] = {"test_friskBaseDebug.mlir",
                                      "test_friskBase.mlir"};
  llvm::SmallString<256> dir(llvm::sys::path::parent_path(inputPath));
  for (;;) {
    for (const char *name : names) {
      llvm::SmallString<256> candidate(dir);
      llvm::sys::path::append(candidate, "test", name);
      if (llvm::sys::fs::exists(candidate))
        return candidate.str().str();
    }
    if (dir.empty())
      return "";
    // `parent_path` returns a reference into `dir`, so go through a copy.
    llvm::SmallString<256> parent(llvm::sys::path::parent_path(dir));
    if (parent == dir)
      return "";
    dir = parent;
  }
}

/// The scheme is only allowed to add `pipeline.stage`/`pipeline.order` to the
/// loop of the frozen IR: everything else of the input has to stay verbatim.
/// Both files are parsed and compared as printed modules, so comments and
/// formatting do not matter and only real IR changes are reported.
void checkSyncedWithBase(MLIRContext &context, const std::string &path,
                         const std::string &basePath) {
  auto base = parseSourceFile<ModuleOp>(basePath, &context);
  check(static_cast<bool>(base), "the sync check parses " + basePath);
  if (!base)
    return;

  auto input = parseSourceFile<ModuleOp>(path, &context);
  check(static_cast<bool>(input), "the sync check parses " + path);
  if (!input)
    return;

  unsigned annotated = 0;
  input->walk([&](affine::AffineForOp loop) {
    if (!loop->hasAttr("pipeline.stage"))
      return;
    ++annotated;
    loop->removeAttr("pipeline.stage");
    loop->removeAttr("pipeline.order");
  });
  check(annotated == 1, "the input annotates exactly one loop");

  std::string got = printModule(*input);
  std::string want = printModule(*base);
  if (got != want)
    llvm::errs() << "       first difference: " << firstLineDifference(got, want)
                 << "\n";
  check(got == want,
        "the input is " + llvm::sys::path::filename(basePath).str() +
            " verbatim, plus pipeline.stage/order on its loop");
}

/// The gemm that computes QK (its A operand is the scaled Q buffer).
frisk::GemmOp findQkGemm(Region &region) {
  frisk::GemmOp result;
  region.walk([&](frisk::GemmOp gemm) {
    if (!result && gemm.getA().getDefiningOp<frisk::MulOp>())
      result = gemm;
  });
  return result;
}

/// The gemm that computes PV: its A operand is a runtime selected P slot.
frisk::GemmOp findPvGemm(Region &region) {
  frisk::GemmOp result;
  region.walk([&](frisk::GemmOp gemm) {
    if (!result && gemm.getA().getDefiningOp<scf::IfOp>())
      result = gemm;
  });
  return result;
}

/// The PV gemm of the epilogue: the last PV gemm of the function.
frisk::GemmOp findEpiloguePvGemm(func::FuncOp func) {
  frisk::GemmOp result;
  func.walk([&](frisk::GemmOp gemm) {
    if (gemm.getA().getDefiningOp<scf::IfOp>())
      result = gemm;
  });
  return result;
}

frisk::CopyOp findCopyFrom(Region &region, Value argument) {
  frisk::CopyOp result;
  region.walk([&](frisk::CopyOp copy) {
    if (!result && copySourceArgument(copy) == argument)
      result = copy;
  });
  return result;
}

/// The copy that lands the softmax result in a buffer slot.  The `frisk.exp2`
/// result does not reach the slot directly: this IR keeps the f32 softmax in
/// shared memory first (`%qkme = frisk.copy %qkmeLocal to %qkmeShm`) and takes
/// the f16 P tile from there, so the search follows the memrefs the exp2 leaves
/// through and reports the copy that writes one of the selected slots.
frisk::CopyOp findSlotWriteOfSoftmax(ArrayRef<Operation *> bodyOps) {
  frisk::CopyOp result;
  SmallVector<Value> worklist;
  for (Operation *op : bodyOps)
    if (isa<frisk::Exp2Op>(op))
      llvm::append_range(worklist, op->getResults());
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (auto copy = dyn_cast<frisk::CopyOp>(user))
        if (!result && copy.getDstMemRef().getDefiningOp<scf::IfOp>())
          result = copy;
      for (Value nested : user->getResults())
        if (isa<MemRefType>(nested.getType()))
          worklist.push_back(nested);
    }
  }
  return result;
}

/// What the checks need to know about the IR the schedule was derived from.
/// Measuring the input instead of hard coding the numbers keeps them
/// meaningful: the pass has to preserve what the input carries (its barriers
/// and its copies) and to shorten exactly the iteration space it is given.
struct InputFacts {
  unsigned loopBarriers = 0; ///< barriers inside the annotated loop
  unsigned tailBarriers = 0; ///< barriers after it, they stay in the epilogue
  unsigned copies = 0;       ///< `frisk.copy` ops inside the annotated loop
  int64_t step = 0;          ///< step of the annotated loop
  AffineMap upperBound;      ///< upper bound of the annotated loop
};

InputFacts measureInput(func::FuncOp func) {
  InputFacts facts;
  affine::AffineForOp loop;
  func.walk([&](affine::AffineForOp candidate) {
    if (!loop && candidate->hasAttr("pipeline.stage"))
      loop = candidate;
  });
  if (!loop)
    return facts;

  facts.upperBound = loop.getUpperBoundMap();
  facts.step = loop.getStepAsInt();
  facts.loopBarriers = countIn<frisk::SyncThreadsInBlockOp>(loop.getRegion());
  facts.copies = countIn<frisk::CopyOp>(loop.getRegion());
  bool afterLoop = false;
  for (Operation &op : func.getBody().front()) {
    if (&op == loop.getOperation()) {
      afterLoop = true;
      continue;
    }
    if (afterLoop && isa<frisk::SyncThreadsInBlockOp>(&op))
      ++facts.tailBarriers;
  }
  return facts;
}

// --- the actual checks -----------------------------------------------------

void checkPrologue(func::FuncOp func, affine::AffineForOp scheduled,
                   const InputFacts &facts) {
  SmallVector<Operation *> prologue;
  for (Operation &op : func.getBody().front()) {
    if (&op == scheduled.getOperation())
      break;
    prologue.push_back(&op);
  }

  Value k = func.getArgument(2);
  Value v = func.getArgument(1);
  SmallVector<frisk::CopyOp> kCopies;
  SmallVector<frisk::CopyOp> vCopies;
  SmallVector<frisk::CopyToRegOp> toReg;
  for (Operation *op : prologue) {
    // The prologue guards the tiles that may not exist (tile 1 on a single
    // tile problem), so the copies can be nested in an `scf.if`.
    op->walk([&](Operation *nested) {
      if (auto copy = dyn_cast<frisk::CopyOp>(nested)) {
        if (copySourceArgument(copy) == k)
          kCopies.push_back(copy);
        if (copySourceArgument(copy) == v)
          vCopies.push_back(copy);
      }
      if (auto reg = dyn_cast<frisk::CopyToRegOp>(nested))
        toReg.push_back(reg);
    });
  }

  check(countOf<frisk::SyncThreadsInBlockOp>(prologue) ==
            1 + facts.loopBarriers,
        "prologue has 1 inserted barrier plus the " +
            std::to_string(facts.loopBarriers) + " of its statements");
  check(countOf<frisk::GemmOp>(prologue) == 1,
        "prologue contains exactly one gemm (QK of tile 0)");
  check(kCopies.size() == 2, "prologue issues 2 K loads (tiles 0 and 1)");
  check(vCopies.size() == 1, "prologue issues 1 V load (tile 0)");
  check(toReg.size() == 1, "prologue materialises the rowsum into a register");
  if (kCopies.size() == 2) {
    check(slotOf(kCopies[0].getDstMemRef()) == 0, "prologue K(tile 0) -> slot 0");
    check(slotOf(kCopies[1].getDstMemRef()) == 1, "prologue K(tile 1) -> slot 1");
  }
  if (vCopies.size() == 1)
    check(slotOf(vCopies[0].getDstMemRef()) == 0, "prologue V(tile 0) -> slot 0");
  if (toReg.size() == 1)
    check(scheduled.getInits()[1] == toReg[0].getResult(),
          "rowsum register is the init operand of the steady loop");
}

void checkSteadyLoop(func::FuncOp func, affine::AffineForOp scheduled,
                     const InputFacts &facts) {
  MLIRContext *context = func.getContext();
  Value k = func.getArgument(2);
  Value v = func.getArgument(1);

  check(scheduled.getNumIterOperands() == 2, "steady loop keeps 2 iter_args");
  check(scheduled.getNumResults() == 2, "steady loop keeps 2 results");

  check(scheduled.getStepAsInt() == 32, "steady loop keeps the tile step");

  AffineMap expected =
      AffineMap::get(facts.upperBound.getNumDims(),
                     facts.upperBound.getNumSymbols(),
                     {facts.upperBound.getResult(0) -
                      getAffineConstantExpr(facts.step, context)},
                     context);
  check(scheduled.getUpperBoundMap() == expected,
        "steady loop bound is Ub - step (the last tile is peeled)");

  Block &body = *scheduled.getBody();
  SmallVector<Operation *> bodyOps = topLevelOps(body);
  check(countOf<frisk::SyncThreadsInBlockOp>(bodyOps) == 1 + facts.loopBarriers,
        "steady body has 1 inserted barrier (top of the body) plus the " +
            std::to_string(facts.loopBarriers) + " of its statements");

  // The parity is only used to *select* the buffers: one small `scf.if` per
  // (buffer, offset) pair the body touches, and every statement is emitted
  // exactly once against those handles.
  unsigned selectors = 0, guards = 0, selectedSlots = 0;
  for (Operation *op : bodyOps) {
    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp)
      continue;
    if (ifOp.getNumResults() == 0) {
      ++guards;
      continue;
    }
    ++selectors;
    selectedSlots = ifOp.getNumResults();
  }
  check(selectors == 1 && selectedSlots == 6,
        "one scf.if selects the 6 slots of a steady step (3 buffers x 2 offsets)");
  check(guards == 1, "steady body guards the stage-0 statement (K load)");
  check(countOf<frisk::GemmOp>(bodyOps) == 2,
        "steady body has QK and PV gemms, each emitted once");
  check(countOf<frisk::MaskOp>(bodyOps) == 1,
        "steady body has the softmax, emitted once");
  check(countIn<frisk::CopyOp>(scheduled.getRegion()) == facts.copies,
        "steady body emits each of its " + std::to_string(facts.copies) +
            " copies once");

  auto kCopy = findCopyFrom(scheduled.getRegion(), k);
  auto vCopy = findCopyFrom(scheduled.getRegion(), v);
  auto qk = findQkGemm(scheduled.getRegion());
  auto pv = findPvGemm(scheduled.getRegion());
  check(kCopy && vCopy && qk && pv,
        "steady body has the 4 statements, each emitted once");
  if (kCopy && vCopy && qk && pv) {
    // slot(statement) = (tileParity + offset) % 2, so the even branch of a
    // selector yields the slot to use for an even tile.
    check(selectsSlots(kCopy.getDstMemRef(), 0, 1),
          "K load (offset 2) writes slot " + slotsOf(kCopy.getDstMemRef()));
    check(selectsSlots(qk.getB(), 1, 0),
          "QK (offset 1) reads K slot " + slotsOf(qk.getB()));
    check(selectsSlots(vCopy.getDstMemRef(), 1, 0),
          "V load (offset 1) writes slot " + slotsOf(vCopy.getDstMemRef()));
    check(selectsSlots(pv.getB(), 0, 1),
          "PV (offset 0) reads V slot " + slotsOf(pv.getB()));
    check(selectsSlots(pv.getA(), 0, 1),
          "PV (offset 0) reads P slot " + slotsOf(pv.getA()));

    // The softmax output must land in the other slot than the one PV reads.
    frisk::CopyOp pCopy = findSlotWriteOfSoftmax(bodyOps);
    check(pCopy != nullptr, "steady body writes the softmax result into a slot");
    if (pCopy)
      check(selectsSlots(pCopy.getDstMemRef(), 1, 0),
            "softmax (offset 1) writes P slot " +
                slotsOf(pCopy.getDstMemRef()));

    // Emission order: QK, then PV (so that PV overlaps the softmax), then the
    // softmax itself which consumes the QK result of this very step.
    unsigned qkPos = bodyOps.size(), pvPos = bodyOps.size();
    unsigned maskPos = bodyOps.size();
    for (auto [index, op] : llvm::enumerate(bodyOps)) {
      if (op == qk.getOperation())
        qkPos = index;
      if (op == pv.getOperation())
        pvPos = index;
      if (isa<frisk::MaskOp>(op))
        maskPos = index;
    }
    check(qkPos < pvPos && pvPos < maskPos,
          "emission order is QK -> PV -> softmax");
  }
}

void checkEpilogue(func::FuncOp func, affine::AffineForOp scheduled,
                   const InputFacts &facts) {
  frisk::DivOp div;
  bool afterLoop = false;
  for (Operation &op : func.getBody().front()) {
    if (&op == scheduled.getOperation()) {
      afterLoop = true;
      continue;
    }
    if (afterLoop)
      if (auto candidate = dyn_cast<frisk::DivOp>(op))
        if (!div) {
          div = candidate;
          break;
        }
  }
  check(div != nullptr, "the original `frisk.div` tail is still there");
  if (!div)
    return;

  unsigned barriers = 0;
  afterLoop = false;
  for (Operation &op : func.getBody().front()) {
    if (&op == scheduled.getOperation())
      afterLoop = true;
    if (afterLoop && isa<frisk::SyncThreadsInBlockOp>(&op))
      ++barriers;
  }
  check(barriers == 1 + facts.tailBarriers,
        "the epilogue is separated by 1 inserted barrier plus the " +
            std::to_string(facts.tailBarriers) + " that follow the loop");

  // The epilogue finishes the peeled tile: the last PV, accumulated into the
  // accumulator the steady loop produced, and materialised for the `div`.  Like
  // in the steady body the last parity only selects the slots.
  unsigned copiesToReg = 0, selectors = 0, selectedSlots = 0;
  afterLoop = false;
  for (Operation &op : func.getBody().front()) {
    if (&op == scheduled.getOperation()) {
      afterLoop = true;
      continue;
    }
    if (!afterLoop)
      continue;
    op.walk([&](Operation *nested) {
      copiesToReg += isa<frisk::CopyToRegOp>(nested);
      if (auto ifOp = dyn_cast<scf::IfOp>(nested)) {
        if (ifOp.getNumResults() == 0)
          return;
        ++selectors;
        selectedSlots = ifOp.getNumResults();
      }
    });
  }
  check(copiesToReg == 1, "the epilogue materialises the acc into a register");
  check(selectors == 1 && selectedSlots == 2,
        "one scf.if selects the V and P slot of the peeled tile");

  auto copyToReg = div.getLhs().getDefiningOp<frisk::CopyToRegOp>();
  check(copyToReg != nullptr, "the divide consumes the epilogue result");
  check(div.getRhs() == scheduled.getResult(1),
        "the rowsum of the divide still comes from the scheduled loop");

  auto pv = findEpiloguePvGemm(func);
  check(pv != nullptr && !scheduled->isAncestor(pv.getOperation()),
        "the epilogue computes the peeled PV, outside of the steady loop");
  if (!pv)
    return;
  check(selectsSlots(pv.getA(), 0, 1),
        "epilogue PV reads P slot " + slotsOf(pv.getA()));
  check(selectsSlots(pv.getB(), 0, 1),
        "epilogue PV reads V slot " + slotsOf(pv.getB()));
  if (!copyToReg)
    return;

  auto add = copyToReg.getSrc().getDefiningOp<frisk::AddOp>();
  check(add != nullptr && add->getOperand(0) == scheduled.getResult(0),
        "the peeled PV is accumulated into the steady state accumulator");
  if (add)
    check(add->getOperand(1) == pv.getC(),
          "the accumulator is updated with the peeled PV");
}

void checkSlots(func::FuncOp func) {
  llvm::DenseMap<int64_t, unsigned> histogram;
  func.walk([&](frisk::AllocBufferOp alloc) {
    auto attr = alloc->getAttrOfType<IntegerAttr>("pipeline_slot");
    if (attr)
      ++histogram[attr.getInt()];
  });
  check(histogram[0] == 3 && histogram[1] == 3,
        "3 double buffered tensors (K, V, P), 2 slots each");
  check(histogram.size() == 2, "no buffer needs more than 2 slots");
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "usage: " << argv[0] << " <input.mlir> [--idempotence]"
                 << " [--bad-annotation] [--sync-check [base.mlir]]\n";
    return 1;
  }
  std::string mode = argc > 2 ? argv[2] : "";
  std::string path = argv[1];

  DialectRegistry registry;
  registry.insert<affine::AffineDialect, arith::ArithDialect,
                  func::FuncDialect, gpu::GPUDialect, math::MathDialect,
                  memref::MemRefDialect, scf::SCFDialect, vector::VectorDialect,
                  frisk::FriskDialect>();
  MLIRContext context(registry);

  // `--sync-check` only compares the input with the frozen IR it was copied
  // from; a normal run does the same check automatically, as long as that file
  // can be found next to the input.
  if (mode == "--sync-check") {
    std::string basePath = argc > 3 ? argv[3] : findBaseFile(path);
    if (basePath.empty()) {
      llvm::errs() << "[FAIL] no test/test_friskBase*.mlir found above " << path
                   << "; pass the base file explicitly\n";
      return 1;
    }
    checkSyncedWithBase(context, path, basePath);
    llvm::outs() << (failures ? "TEST FAILED\n" : "TEST PASSED\n");
    return failures ? 1 : 0;
  }

  auto module = parseSourceFile<ModuleOp>(path, &context);
  if (!module) {
    llvm::errs() << "failed to parse " << path << "\n";
    return 1;
  }
  func::FuncOp func = *module->getOps<func::FuncOp>().begin();

  if (mode != "--bad-annotation")
    if (std::string basePath = findBaseFile(path); !basePath.empty())
      checkSyncedWithBase(context, path, basePath);

  if (mode == "--bad-annotation") {
    // Duplicate a `pipeline.order` entry onto a non adjacent one (entries that
    // are repeated *and* adjacent would just merge into one statement); the
    // pass has to reject it.
    bool patched = false;
    func.walk([&](affine::AffineForOp loop) {
      if (patched)
        return;
      auto orders = loop->getAttrOfType<DenseI64ArrayAttr>("pipeline.order");
      if (!orders || orders.empty())
        return;
      SmallVector<int64_t> copy(orders.asArrayRef());
      copy.back() = copy.front();
      loop->setAttr("pipeline.order", DenseI64ArrayAttr::get(&context, copy));
      patched = true;
    });
    if (!patched) {
      llvm::errs() << "[FAIL] nothing to patch\n";
      return 1;
    }
    PassManager pm(&context);
    pm.addNestedPass<func::FuncOp>(pipeline::createPipelineSchedulePass());
    bool rejected = failed(pm.run(module.get()));
    check(rejected, "duplicated pipeline.order is rejected");
    llvm::outs() << (failures ? "TEST FAILED\n" : "TEST PASSED\n");
    return failures ? 1 : 0;
  }

  std::string before;
  {
    llvm::raw_string_ostream os(before);
    module->print(os);
  }

  // Measure the annotated loop before the pass rewrites it: the checks below
  // compare the scheduled IR against what the input carries.
  InputFacts facts = measureInput(func);

  PassManager pm(&context);
  pm.addNestedPass<func::FuncOp>(pipeline::createPipelineSchedulePass());
  if (failed(pm.run(module.get()))) {
    llvm::errs() << "[FAIL] the pipeline pass failed\n";
    return 1;
  }

  llvm::outs() << "\n===== after frisk-pipeline-schedule =====\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  check(succeeded(verify(module.get())), "the transformed module verifies");
  check(func->getAttr("thread_num") != nullptr,
        "func attributes (thread_num) are preserved");

  affine::AffineForOp scheduled = findScheduledLoop(func);
  check(scheduled != nullptr, "a scheduled loop (pipeline.scheduled) exists");
  if (scheduled) {
    check(scheduled->getAttrOfType<IntegerAttr>("pipeline.num_stages") &&
              scheduled->getAttrOfType<IntegerAttr>("pipeline.num_stages")
                      .getInt() == 3,
          "pipeline.num_stages == 3");
    check(scheduled->getAttrOfType<IntegerAttr>("pipeline.slots") &&
              scheduled->getAttrOfType<IntegerAttr>("pipeline.slots").getInt() ==
                  2,
          "pipeline.slots == 2");
    checkPrologue(func, scheduled, facts);
    checkSteadyLoop(func, scheduled, facts);
    checkEpilogue(func, scheduled, facts);
    checkSlots(func);
  }

  // The scheduled module is dumped by the mainline and may be read back, so it
  // has to round-trip: printing it again after a re-parse must reproduce it.
  {
    std::string scheduledText = printModule(*module);
    auto reparsed = parseSourceString<ModuleOp>(scheduledText, &context);
    check(static_cast<bool>(reparsed), "the scheduled IR re-parses");
    if (reparsed)
      check(printModule(*reparsed) == scheduledText,
            "re-parsing and printing the scheduled IR is a no-op");
  }

  if (mode == "--idempotence") {
    std::string afterFirst;
    {
      llvm::raw_string_ostream os(afterFirst);
      module->print(os);
    }
    PassManager again(&context);
    again.addNestedPass<func::FuncOp>(pipeline::createPipelineSchedulePass());
    if (failed(again.run(module.get()))) {
      llvm::errs() << "[FAIL] the second run failed\n";
      return 1;
    }
    std::string afterSecond;
    {
      llvm::raw_string_ostream os(afterSecond);
      module->print(os);
    }
    check(afterFirst == afterSecond, "running the pass twice is a no-op");
    (void)before;
  }

  llvm::outs() << (failures ? "\nTEST FAILED\n" : "\nTEST PASSED\n");
  return failures ? 1 : 0;
}
