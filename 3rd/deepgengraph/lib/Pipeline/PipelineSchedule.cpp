//===- PipelineSchedule.cpp -----------------------------------------------===//
//
// FA3 style software pipelining for annotated frisk `affine.for` loops.
//
// The scheme is attached to the loop itself, mirroring tilelang's
// `T.Pipelined(..., stage = [...], order = [...])`:
//
//   pipeline.stage : pipeline level of a statement.  Level 0 is the most
//                    advanced one (its tile index is the largest one), a
//                    bigger number is closer to the current iteration.
//   pipeline.order : emission order inside one steady state iteration,
//                    smaller values are emitted first.
//
// Both are `array<i64: ...>` with one entry per top level op of the loop body,
// lined up with the ops in source order.  Consecutive ops sharing a
// `(stage, order)` pair form one statement, so the ops of one statement simply
// repeat the pair.
//
// Given `numStages = max(stage) + 1` a statement of stage `s` handles tile
// `i + offset(s)` in steady state iteration `i`, where
// `offset(s) = numStages - 1 - s`.
//
// A buffer that is written by one statement and read by another one keeps
// `slots = 1 + (writer offset - reader offset)` copies, which is the classic
// double buffering rule for a one step distance.  Buffers that stay inside a
// single statement (`slots == 1`) are left alone, so e.g. the QK result never
// has to be materialised.
//
// The generated code follows FA3's structure:
//
//   prologue : steps -numStages+1 .. -1, one barrier between two steps
//   steady   : `Ub - step` (i.e. all tiles but the peeled ones); the only
//              synchronisation point is a barrier at the top of the body,
//              the parity of the tile selects which slot is used
//   epilogue : barrier, the peeled tile(s), then the original tail
//
//===----------------------------------------------------------------------===//

#include "deepgengraph/Pipeline/PipelineSchedule.h"

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace mlir::pipeline {

#define GEN_PASS_DEF_PIPELINESCHEDULE
#include "deepgengraph/Pipeline/Passes.h.inc"

} // namespace mlir::pipeline

using namespace mlir;

namespace {

// pipeline.stage/pipeline.order 是用户方案；
// pipeline.scheduled/pipeline.num_stages/pipeline.slots 是变换后打的标记；
// pipeline_slot 给每个克隆出来的槽编号
constexpr llvm::StringLiteral kSlotAttr("pipeline_slot");
constexpr llvm::StringLiteral kScheduledAttr("pipeline.scheduled");
constexpr llvm::StringLiteral kNumStagesAttr("pipeline.num_stages");
constexpr llvm::StringLiteral kSlotsAttr("pipeline.slots");
constexpr llvm::StringLiteral kSchemeStageAttr("pipeline.stage");
constexpr llvm::StringLiteral kSchemeOrderAttr("pipeline.order");

/// `array<i64: ...>` shows up as a `DenseI64ArrayAttr` or, when it was built
/// programmatically, as a `DenseIntElementsAttr`.
// 从 op 上读一个 i64 数组。文本解析 array<i64: ...> 得到 DenseI64ArrayAttr，
// 程序构造可能是 DenseIntElementsAttr，两种都接。
// 属性不存在时 present=false 并返回成功（不算错误)
LogicalResult readI64Array(Operation *op, StringRef name, bool &present,
                           SmallVectorImpl<int64_t> &out) {
  present = false;
  Attribute attr = op->getAttr(name);
  if (!attr)
    return success();
  present = true;
  if (auto array = dyn_cast<DenseI64ArrayAttr>(attr)) {
    llvm::append_range(out, array.asArrayRef());
    return success();
  }
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (!dense.getElementType().isSignlessInteger(64))
      return op->emitError() << "`" << name << "` has to be an i64 array";
    llvm::append_range(out, dense.getValues<int64_t>());
    return success();
  }
  return op->emitError() << "`" << name << "` has to be an array of integers";
}

int64_t modulo(int64_t value, int64_t modulus) {
  int64_t result = value % modulus;
  return result < 0 ? result + modulus : result;
}

/// A maximal run of consecutive loop body ops sharing (stage, order).
// 一条语句。stage/order 来自标注，offset 是算出来的，ops 是组成它的 op，
// written 是它写出的值（用来推依赖)
struct Statement {
  int64_t stage = 0;
  int64_t order = 0;
  int64_t offset = 0;
  SmallVector<Operation *> ops;
  SmallVector<Value> written;
};

/// A loop local buffer that is alive across two pipeline stages and therefore
/// has to be split into several slots.
// 一个跨级 buffer。alloc 是原始分配，slots 需要几份，slotValues 是实际克隆出来的每份
struct RotatedBuffer {
  Value original;
  frisk::AllocBufferOp alloc;
  int64_t slots = 1;
  SmallVector<Value> slotValues;
  /// Every value that denotes this buffer: `original` plus the handle of each
  /// `frisk.copy ... to %original` writing into it.  The Frisk IR keeps using
  /// those handles instead of the buffer, so the dependency scan, the slot
  /// selection and the substitution all have to follow them as well.
  SmallVector<Value, 2> handles;
};

/// SSA handle of the slot a statement has to touch, used where the tile parity
/// is only known at runtime (steady state and epilogue).
///
/// The key is `(rotated buffer index, statement offset)`: which slot a
/// statement touches depends on its own offset *and* on the parity of the tile
/// it handles, so the two cannot be collapsed into one map (the writer of a
/// buffer and its reader usually have different offsets).
using SlotHandleMap = DenseMap<std::pair<unsigned, int64_t>, Value>;

/// The buffer a value denotes.  `frisk.copy` can hand back the handle of the
/// buffer it writes into (`%tile = frisk.copy %src to %slot`), and the Frisk IR
/// passes that handle around instead of the buffer itself; the handle therefore
/// denotes the same buffer as its destination.
Value bufferOf(Value value) {
  while (auto copy = value.getDefiningOp<frisk::CopyOp>()) {
    Value dst = copy.getDstMemRef();
    if (dst == value)
      break;
    value = dst;
  }
  return value;
}

/// Every value the statements use to denote `buffer`: the buffer plus the
/// handle of each `frisk.copy ... to <handle>` writing into it.  A dependency
/// found through a handle is a dependency on the buffer, and a statement that
/// touches a handle touches the buffer.
SmallVector<Value, 2> bufferHandles(Value buffer, ArrayRef<Statement> stmts) {
  SmallVector<Value, 2> handles{buffer};
  for (const Statement &st : stmts)
    for (Operation *op : st.ops)
      if (auto copy = dyn_cast<frisk::CopyOp>(op))
        if (bufferOf(copy.getDstMemRef()) == buffer)
          llvm::append_range(handles, copy.getResults());
  return handles;
}

/// Values that are written by the statements of the loop body.
// 这个 op「写」了什么。frisk.copy 取 dst（真的写进 buffer，句柄只是同一个
// buffer 的别名），frisk.reduce 取 dst，
// 其它 op 取所有 memref/vector 结果（比如 gemm 产出的 memref)
void collectWrittenValues(Operation *op, SmallVectorImpl<Value> &out) {
  if (auto copy = dyn_cast<frisk::CopyOp>(op)) {
    out.push_back(bufferOf(copy.getDstMemRef()));
    return;
  }
  if (auto reduce = dyn_cast<frisk::ReduceOp>(op)) {
    out.push_back(reduce.getDst());
    return;
  }
  for (Value result : op->getResults()) {
    Type type = result.getType();
    if (isa<MemRefType>(type) || isa<VectorType>(type))
      out.push_back(result);
  }
}

/// Does this op write into one of the loop carried values?
// 是不是在写循环携带值。只有 frisk.copy 的目标正好是某个 iter_arg 才算。
bool writesLoopCarriedValue(Operation *op, ValueRange iterArgs) {
  auto copy = dyn_cast<frisk::CopyOp>(op);
  if (!copy)
    return false;
  return llvm::is_contained(iterArgs, copy.getDstMemRef());
}

class PipelineScheduler {
public:
  PipelineScheduler(affine::AffineForOp loop,
                    const pipeline::PipelineScheduleOptions &options)
      : loop(loop), options(options) {}

  LogicalResult run();

private:
  // --- analysis -----------------------------------------------------------
  LogicalResult collectStatements();
  LogicalResult readLoopScheme(
      ArrayRef<Operation *> bodyOps,
      SmallVectorImpl<std::pair<int64_t, int64_t>> &annotations, bool &found);
  LogicalResult collectRotatedBuffers();

  // --- building -----------------------------------------------------------
  LogicalResult build();

  void emitStatement(OpBuilder &builder, const Statement &st, int64_t parity,
                    Value ivValue, ValueRange iterArgValues,
                    bool dropLoopCarriedWrites, Value guardCond, IRMapping &map,
                     DenseMap<Value, Value> *writtenIterArgs,
                     const SlotHandleMap *slotHandles = nullptr);

  /// Materialises the slot selectors of the statements in `indices`: for every
  /// `(buffer, offset)` pair those statements touch, one `scf.if` that yields
  /// the slot to use for an even tile and the one for an odd tile.
  SlotHandleMap materializeSlotHandles(OpBuilder &builder, Value evenCond,
                                      ArrayRef<unsigned> indices);

  /// Does `st` read or write `buffer`, either directly or through one of its
  /// `frisk.copy` handles?
  static bool statementTouches(const Statement &st, const RotatedBuffer &buffer);

  /// The slot handle a statement with `offset` uses for `buffer`.  Keyed on
  /// `offset % slots` because `(parity + offset) % slots` only depends on it, so
  /// statements whose offsets differ by a multiple of the slot count share one
  /// handle.
  static std::pair<unsigned, int64_t> slotKey(unsigned bufferIndex,
                                             const RotatedBuffer &buffer,
                                             int64_t offset);

  Value materializeBound(OpBuilder &builder, AffineMap map, ValueRange operands);
  Value tileIv(OpBuilder &builder, int64_t tile, int64_t step);
  Value offsetIv(OpBuilder &builder, Value iv, int64_t offset, int64_t step);

  affine::AffineForOp loop;
  pipeline::PipelineScheduleOptions options;

  SmallVector<Statement> stmts;
  SmallVector<unsigned> emitOrder; ///< statement indices sorted by order
  DenseMap<Operation *, unsigned> opToStmt;
  SmallVector<RotatedBuffer> rotated;
  int64_t numStages = 0;

  /// Lower/upper bound of the original loop and its upper bound minus one
  /// step, i.e. the induction value of the peeled tile.
  Value lbValue;
  Value ubValue;
  Value ubMinusStep;
};

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

/// Reads the tilelang style scheme attached to the loop.  `found` is false
/// when the loop carries no scheme at all, in which case there is nothing to
/// schedule.
LogicalResult PipelineScheduler::readLoopScheme(
    ArrayRef<Operation *> bodyOps,
    SmallVectorImpl<std::pair<int64_t, int64_t>> &annotations, bool &found) {
  found = false;
  SmallVector<int64_t> stages, orders;
  bool hasStages = false, hasOrders = false;
  // 读两个数组；一个都没有 → found=false 直接返回成功。这是幂等的来源：跑过一遍的新循环不再带方案
  if (failed(readI64Array(loop, kSchemeStageAttr, hasStages, stages)) ||
      failed(readI64Array(loop, kSchemeOrderAttr, hasOrders, orders)))
    return failure();
  if (!hasStages && !hasOrders)
    return success();
  found = true;
  // 只给了一个 → 报错
  if (hasStages != hasOrders)
    return loop.emitError()
           << "`" << kSchemeStageAttr << "` and `" << kSchemeOrderAttr
           << "` have to be given together";
  // pipeline.stage 长度必须等于循环体顶层 op 数
  if (stages.size() != bodyOps.size())
    return loop.emitError()
           << "`" << kSchemeStageAttr << "` has " << stages.size()
           << " entries but the loop body has " << bodyOps.size()
           << " top level ops; give one entry per op, repeating the "
              "`(stage, order)` pair for the ops of one statement";
  // 两个数组长度必须一致
  if (orders.size() != stages.size())
    return loop.emitError()
           << "`" << kSchemeOrderAttr << "` has " << orders.size()
           << " entries but `" << kSchemeStageAttr << "` has " << stages.size();

  // Consecutive ops sharing a `(stage, order)` pair are one statement.
  // 逐项写进 annotations，任何一项为负报错。注意这里还不管 order 是否全序，那是下一步的事。
  annotations.resize(bodyOps.size());
  for (auto [index, op] : llvm::enumerate(bodyOps)) {
    if (stages[index] < 0 || orders[index] < 0)
      return op->emitError() << "`" << kSchemeStageAttr << "` and `"
                             << kSchemeOrderAttr << "` must not be negative";
    annotations[index] = std::make_pair(stages[index], orders[index]);
  }
  return success();
}
// 切语句、算 offset
LogicalResult PipelineScheduler::collectStatements() {
  Block &body = loop.getRegion().front();

  // Round 1: gather the top level ops of the loop body.
  SmallVector<Operation *> bodyOps;
  // 收集循环体顶层 op（遇到 terminator 停）
  for (Operation &op : body) {
    if (op.hasTrait<OpTrait::IsTerminator>())
      break;
    bodyOps.push_back(&op);
  }
  if (bodyOps.empty())
    return success();

  // Round 2: read the scheme carried by the loop itself.
  SmallVector<std::pair<int64_t, int64_t>> annotations;
  bool found = false;
  // 调 readLoopScheme；没有方案就直接返回成功
  if (failed(readLoopScheme(bodyOps, annotations, found)))
    return failure();
  if (!found)
    return success();

  // Round 3: group maximal runs with the same (stage, order).
  // 合并同 pair 的相邻 op。相
  // 邻且 (stage, order) 相同就归到同一个 Statement，不同就开新的一条。
  for (size_t i = 0; i < bodyOps.size(); ++i) {
    int64_t stage = annotations[i].first;
    int64_t order = annotations[i].second;
    if (stmts.empty() || stmts.back().stage != stage ||
        stmts.back().order != order) {
      Statement st;
      st.stage = stage;
      st.order = order;
      stmts.push_back(st);
    }
    stmts.back().ops.push_back(bodyOps[i]);
  }

  // Round 4: validate the annotation.
  // 校验 stage/order 非负，order 两两不同（全序），并算出 numStages = maxStage + 1
  int64_t maxStage = 0;
  DenseSet<int64_t> orders;
  for (Statement &st : stmts) {
    if (st.stage < 0)
      return loop.emitError()
             << "`" << kSchemeStageAttr << "` must not be negative";
    if (st.order < 0)
      return loop.emitError()
             << "`" << kSchemeOrderAttr << "` must not be negative";
    if (!orders.insert(st.order).second)
      return loop.emitError()
             << "`" << kSchemeOrderAttr << "` = " << st.order
             << " is used by more than one statement; the emission order has "
                "to be a total order";
    maxStage = std::max(maxStage, st.stage);
  }
  numStages = maxStage + 1;
  // 它把「逻辑流水级」翻译成「提前多少块」：
  // stage 越小 offset 越大，越早处理靠后的 tile（FA3 里 stage0 处理 tile i+2）。
  // 同时建 opToStmt 映射并收集 written
  for (unsigned i = 0; i < stmts.size(); ++i) {
    stmts[i].offset = numStages - 1 - stmts[i].stage;
    for (Operation *op : stmts[i].ops)
      opToStmt[op] = i;
    collectWrittenValues(stmts[i].ops.front(), stmts[i].written);
    for (Operation *op : llvm::drop_begin(stmts[i].ops))
      collectWrittenValues(op, stmts[i].written);
  }

  // Emission order.
  // emitOrder = 语句下标按 order 稳定排序。order 只在这里起作用
  emitOrder.resize(stmts.size());
  for (unsigned i = 0; i < stmts.size(); ++i)
    emitOrder[i] = i;
  llvm::stable_sort(emitOrder, [&](unsigned lhs, unsigned rhs) {
    return stmts[lhs].order < stmts[rhs].order;
  });

  if (stmts.empty())
    return success();

  // The steady loop only runs while there are tiles left for the *last*
  // stage, i.e. we peel `options.peeledTiles` tiles into the epilogue.
  // 校验 peeledTiles 合法
  if (options.peeledTiles < 0 || options.peeledTiles >= numStages)
    return loop.emitError() << "invalid number of peeled tiles";

  return success();
}
// 算几条槽
LogicalResult PipelineScheduler::collectRotatedBuffers() {
  DenseSet<Value> seen;
  for (Statement &st : stmts) {
    // 对每条语句写出的每个值
    for (Value written : st.written) {
      // 跳过 iter_arg 和已处理过的
      if (isa<BlockArgument>(written))
        continue; // loop carried value, handled separately
      if (!seen.insert(written).second)
        continue;

      int64_t maxLead = 0;
      // 遍历它的所有 use（含 frisk.copy 给出的句柄，它们指向同一个 buffer）
      SmallVector<Value, 2> handles = bufferHandles(written, stmts);
      for (Value handle : handles) {
        for (OpOperand &use : handle.getUses()) {
          Operation *user = use.getOwner();
          auto it = opToStmt.find(user);
          // user 不在 opToStmt 里 → 报「写在内、用在外，不支持」
          if (it == opToStmt.end()) {
            return loop.emitError()
                   << "value " << written
                   << " is written inside the pipelined loop but used outside "
                      "of it; this is not supported";
          }
          Statement &consumer = stmts[it->second];
          // The dependency rules of the annotation, see tilelang: a producer
          // must not be placed after its consumer.
          // 消费者 offset 大于生产者 → 报错。
          // 因为 offset 越大越超前，生产者不能比消费者还超前，否则消费者要用一个还没算出来的值。
          if (consumer.offset > st.offset)
            return loop.emitError()
                   << "the statement with `" << kSchemeStageAttr
                   << "` = " << st.stage
                   << " produces a value that the statement with `"
                   << kSchemeStageAttr << "` = " << consumer.stage
                   << " consumes, but a lower stage runs earlier in the "
                      "pipeline; the producer stage has to be less than or "
                      "equal to the consumer stage";
          // 同 offset（同 stage）时，消费者的 order 必须不小于生产者
          if (consumer.offset == st.offset && consumer.order < st.order)
            return loop.emitError()
                   << "the statement with `" << kSchemeOrderAttr
                   << "` = " << st.order
                   << " produces a value that the statement with `"
                   << kSchemeOrderAttr << "` = " << consumer.order
                   << " consumes; within one stage the producer has to be "
                      "emitted first";
          // maxLead = max(生产者 offset − 消费者 offset)
          maxLead = std::max(maxLead, st.offset - consumer.offset);
        }
      }
      // slots = max(1, maxLead + 1) —— 双缓冲公式，offset 差 1 就是 2 槽
      int64_t slots = std::max<int64_t>(1, maxLead + 1);
      // slots == 1 直接跳过，buffer 原地不动。这就是为什么 QK 的结果 S 不必物化
      if (slots == 1)
        continue;
      // 超过 maxSlots 报错
      if (slots > options.maxSlots)
        return loop.emitError()
               << "buffer " << written << " needs " << slots
               << " slots but only " << options.maxSlots
               << " are supported (reduce the stage distance)";
      // 必须是 frisk.alloc_buffer 的结果，否则报错
      auto alloc = written.getDefiningOp<frisk::AllocBufferOp>();
      if (!alloc)
        return loop.emitError()
               << "value " << written
               << " crosses pipeline stages but is not the result of a "
                  "`frisk.alloc_buffer`; materialising it is not supported yet";
      // 记录成 RotatedBuffer
      RotatedBuffer buffer;
      buffer.original = written;
      buffer.alloc = alloc;
      buffer.slots = slots;
      buffer.handles = std::move(handles);
      rotated.push_back(buffer);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Building
//===----------------------------------------------------------------------===//

// 把 loop 的 lb/ub affine map 变成值。常量 map 直接用 arith.constant
// 否则 affine.apply
// 这样后面能直接拿 ubValue 做比较。
Value PipelineScheduler::materializeBound(OpBuilder &builder, AffineMap map,
                                          ValueRange operands) {
  Location loc = loop.getLoc();
  if (map.getNumInputs() == 0 && map.getNumResults() == 1)
    if (auto constant = dyn_cast<AffineConstantExpr>(map.getResult(0)))
      return builder.create<arith::ConstantIndexOp>(loc, constant.getValue());
  return builder.create<affine::AffineApplyOp>(loc, map, operands);
}
// 把「第 tile 块」换算成 induction value lb + tile*step
Value PipelineScheduler::tileIv(OpBuilder &builder, int64_t tile,
                                int64_t step) {
  Location loc = loop.getLoc();
  Value base = builder.create<arith::ConstantIndexOp>(loc, tile * step);
  return builder.createOrFold<arith::AddIOp>(loc, base, lbValue);
}
// 稳态循环里把 iv 加 offset*step，得到该语句处理的那块 tile 的 iv；offset==0 就直接用 iv
Value PipelineScheduler::offsetIv(OpBuilder &builder, Value iv, int64_t offset,
                                  int64_t step) {
  Location loc = loop.getLoc();
  if (offset == 0)
    return iv;
  // Keep shifted loop indices affine: buffer_view/copy lowering composes
  // them into affine.load maps, where an arith.addi of an IV is not a valid
  // dimension (unlike a top-level value outside the affine loop).
  AffineMap map = AffineMap::get(
      1, 0, builder.getAffineDimExpr(0) + offset * step, builder.getContext());
  return builder.create<affine::AffineApplyOp>(loc, map, ValueRange{iv});
}

// 这条语句有没有读/写这个 buffer：buffer 本身和它的 frisk.copy 句柄都算
// （copy 的 src/dst、gemm 的 A/B、mul 的操作数…都是操作数，查操作数就够）
bool PipelineScheduler::statementTouches(const Statement &st,
                                         const RotatedBuffer &buffer) {
  for (Operation *op : st.ops)
    for (Value handle : buffer.handles)
      if (llvm::is_contained(op->getOperands(), handle))
        return true;
  return false;
}

std::pair<unsigned, int64_t>
PipelineScheduler::slotKey(unsigned bufferIndex, const RotatedBuffer &buffer,
                           int64_t offset) {
  return {bufferIndex, modulo(offset, buffer.slots)};
}

// 稳态/epilogue 里 tile 的奇偶是运行期才知道的，没法在编译期定点槽。
// 这里不去复制整段语句（那是奇偶大分支，IR 直接翻倍），而是只对 buffer 做选择：
// 整段只发一个 scf.if，一次产出这一步所有要用的槽，后序 op 只发射一遍、
// 直接复用这些槽句柄。
//
// Steady state / epilogue: the parity of the tile is only known at runtime, so
// the slot a statement touches cannot be picked while building the IR.  Instead
// of cloning every statement into a then branch and an else branch -- which
// duplicates the whole body and therefore every op in it -- we only branch on
// the buffers, and only once per entry: a single `scf.if` yields the slot of
// every (buffer, offset) pair the emitted statements touch, and every statement
// is then emitted exactly once against those handles.
SlotHandleMap PipelineScheduler::materializeSlotHandles(
    OpBuilder &builder, Value evenCond, ArrayRef<unsigned> indices) {
  Location loc = loop.getLoc();

  // Collect the slots that are actually needed, in the order the statements use
  // them (see `slotKey`).
  //
  // 先收集真正要用的槽，按语句碰到的先后排序，offset 相差整数倍槽数的语句共用
  // 同一个句柄（见 slotKey）。
  SmallVector<std::pair<unsigned, int64_t>> keys;
  DenseSet<std::pair<unsigned, int64_t>> seen;
  for (unsigned index : indices) {
    const Statement &st = stmts[index];
    for (auto [bufferIndex, buffer] : llvm::enumerate(rotated)) {
      // 这条语句不碰这个 buffer，就不用给它选槽
      if (!statementTouches(st, buffer))
        continue;
      auto key = slotKey(unsigned(bufferIndex), buffer, st.offset);
      if (seen.insert(key).second)
        keys.push_back(key);
    }
  }

  SlotHandleMap handles;
  if (keys.empty())
    return handles;

  // One branch for the whole step instead of one per slot: every handle is a
  // result of this single `scf.if`, so the slot table is written out in exactly
  // two places (the two yields).
  //
  // 整个 step 只发一个 scf.if，所有槽句柄都是它的结果，槽表就只出现两次
  // （两个 yield），比每个槽单发一个 if 干净
  SmallVector<Type> types;
  for (auto [bufferIndex, offset] : keys) {
    (void)offset;
    types.push_back(rotated[bufferIndex].original.getType());
  }
  auto select = builder.create<scf::IfOp>(loc, TypeRange(types), evenCond,
                                          /*withElseRegion=*/true);
  for (int64_t parity : {int64_t(0), int64_t(1)}) {
    Region &region =
        parity == 0 ? select.getThenRegion() : select.getElseRegion();
    builder.setInsertionPointToStart(&region.front());
    SmallVector<Value> slots;
    for (auto [bufferIndex, offset] : keys) {
      const RotatedBuffer &buffer = rotated[bufferIndex];
      slots.push_back(buffer.slotValues[modulo(parity + offset, buffer.slots)]);
    }
    builder.create<scf::YieldOp>(loc, slots);
  }
  builder.setInsertionPointAfter(select);
  for (auto [index, key] : llvm::enumerate(keys))
    handles[key] = select.getResult(index);
  return handles;
}
// 克隆的核心
void PipelineScheduler::emitStatement(OpBuilder &builder, const Statement &st,
                                      int64_t parity, Value ivValue,
                                      ValueRange iterArgValues,
                                      bool dropLoopCarriedWrites,
                                      Value guardCond, IRMapping &map,
                                      DenseMap<Value, Value> *writtenIterArgs,
                                      const SlotHandleMap *slotHandles) {
  Location loc = loop.getLoc();

  // Per statement substitutions.
  // 设置 IRMapping（克隆时的值替换表
  // 原循环 iv → 传入的 ivValue
  map.map(loop.getInductionVar(), ivValue);
  // iter_arg → 传入的 iterArgValues
  for (auto [index, iterArg] : llvm::enumerate(loop.getRegionIterArgs()))
    map.map(iterArg, iterArgValues[index]);
  // 轮转 buffer 的绑定
  for (auto [index, buffer] : llvm::enumerate(rotated)) {
    // 写者和读者的 offset 不同，所以同一拍里它们绑到不同槽，实现双缓冲。
    // parity 是编译期常量时（prologue）直接算槽；
    // 是运行期值时（稳态/epilogue）用前面选出来的槽句柄
    Value slot;
    if (slotHandles) {
      slot = slotHandles->lookup(slotKey(unsigned(index), buffer, st.offset));
      if (!slot)
        continue; // 这条语句不碰这个 buffer
    } else {
      slot = buffer.slotValues[modulo(parity + st.offset, buffer.slots)];
    }
    // 句柄（`%tile = frisk.copy %src to %slot`）和 buffer 本身指向同一份存储，
    // 一起绑到选出来的槽，读句柄的语句才能读到正确的槽
    for (Value handle : buffer.handles)
      map.map(handle, slot);
  }

  scf::IfOp guardOp;
  if (guardCond) {
    guardOp = builder.create<scf::IfOp>(loc, guardCond, /*withElseRegion=*/false);
    builder.setInsertionPointToStart(&guardOp.getThenRegion().front());
  }
  // 逐 op 克隆
  for (Operation *op : st.ops) {
    // Allocations are hoisted / rotated beforehand.
    // alloc 跳过（前面统一处理过）
    if (isa<frisk::AllocBufferOp>(op))
      continue;
    // 如果是写 iter_arg 的 copy 且在「丢弃写回」模式（prologue/epilogue），
    // 不克隆，改成记录 writtenIterArgs[dst] = 映射后的 src
    // 这就是 prologue 里 rowsum 写回被截获、转成稳态循环初值的机制
    if (dropLoopCarriedWrites && writesLoopCarriedValue(op, loop.getRegionIterArgs())) {
      auto copy = cast<frisk::CopyOp>(op);
      if (writtenIterArgs)
        (*writtenIterArgs)[copy.getDstMemRef()] =
            map.lookupOrDefault(copy.getSrcMemRef());
      continue;
    }
    // 其余 op 直接 clone，IRMapping 自动完成 iv / iter_arg / buffer 的替换
    builder.clone(*op, map);
  }

  if (guardOp)
    builder.setInsertionPointAfter(guardOp);
}
// 七步拼出新结构
LogicalResult PipelineScheduler::build() {
  // OpBuilder builder(loop)
  // 把插入点放在旧循环之前，所以之后创建的东西都在旧循环前面
  // 物化 lbValue/ubValue/step/ubMinusStep
  OpBuilder builder(loop);
  Location loc = loop.getLoc();
  MLIRContext *context = builder.getContext();
  int64_t step = loop.getStepAsInt();

  ValueRange lbOperands = loop.getLowerBoundOperands();
  ValueRange ubOperands = loop.getUpperBoundOperands();

  // Values describing the iteration space, used for guards and for the parity.
  lbValue = materializeBound(builder, loop.getLowerBoundMap(), lbOperands);
  ubValue = materializeBound(builder, loop.getUpperBoundMap(), ubOperands);
  Value stepCst = builder.create<arith::ConstantIndexOp>(loc, step);
  ubMinusStep = builder.create<arith::SubIOp>(loc, ubValue, stepCst);

  // 1) Allocate the extra slots of every rotated buffer.
  // 给每个轮转 buffer clone 出 slots 个 alloc，打上 pipeline_slot 编号
  // 存进 slotValues。因为插入点在循环前，这些 alloc 都在循环外
  for (RotatedBuffer &buffer : rotated) {
    for (int64_t slot = 0; slot < buffer.slots; ++slot) {
      Operation *slotAlloc = builder.clone(*buffer.alloc.getOperation());
      slotAlloc->setAttr(kSlotAttr, builder.getI64IntegerAttr(slot));
      buffer.slotValues.push_back(slotAlloc->getResult(0));
    }
  }

  // 2) Hoist the single slot allocations out of the loop, they are shared by
  //    all the clones.
  // 把不轮转的 alloc 从循环体提到循环外
  // 它们在原循环里每次迭代都重新分配，语义上是同一个 buffer，提出来后所有克隆共享，也避免重复分配
  SmallVector<Operation *> hoist;
  for (Operation &op : loop.getBody()->getOperations()) {
    if (!isa<frisk::AllocBufferOp>(&op))
      continue;
    bool rotates = llvm::any_of(rotated, [&](const RotatedBuffer &buffer) {
      return buffer.original == op.getResult(0);
    });
    if (!rotates)
      hoist.push_back(&op);
  }
  for (Operation *op : hoist)
    op->moveBefore(loop);

  // 3) Prologue: steps -numStages+1 .. -1, with a barrier between two steps.
  SmallVector<Value> oldInits(loop.getInits().begin(), loop.getInits().end());
  DenseMap<Value, Value> prologueWritten;
  // k 从 -(numStages-1) 到 -1，共 numStages-1 步。FA3 里 numStages=3，就是 k=-2,-1
  for (int64_t k = -(numStages - 1); k < 0; ++k) {
    // 每一步一个新 IRMapping
    IRMapping map;
    // 按 emitOrder 遍历语句
    for (unsigned index : emitOrder) {
      Statement &st = stmts[index];
      // 语句 s 在第 k 步处理 tile k+offset(s)」
      int64_t tile = k + st.offset;
      // tile < 0 说明该语句这一步还没活干，跳过（比如 offset 0 的 PV 在 prologue 从不执行）
      if (tile < 0)
        continue; // this statement has no work in this step
      // 算出这块 tile 的 iv
      Value ivTile = tileIv(builder, tile, step);
      Value guard;
      // tile > 0 时加 ivTile < ub 守卫，因为循环可能不足 numStages-1 块；
      // tile 0 一定存在所以不用守卫
      if (tile > 0) // tile 0 always exists, check the others at runtime
        guard = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                              ivTile, ubValue);
      // 发射。parity 用 modulo(k, 2)，iterArgValues 用原始 init，dropLoopCarriedWrites=true 
      // 并把结果收进 prologueWritten
      emitStatement(builder, st, modulo(k, 2), ivTile, oldInits,
                    /*dropLoopCarriedWrites=*/true, guard, map, &prologueWritten);
    }
    if (k < -1)
      builder.create<frisk::SyncThreadsInBlockOp>(loc);
  }

  // 4) Entry values of the steady loop: whatever the prologue produced for a
  //    loop carried value, the original init operand otherwise.
  SmallVector<Value> initValues;
  // 算稳态循环的初值。默认用原 init；
  // 若某 iter_arg 在 prologue 被写过就用 prologue 产出的值
  // 类型不符（memref → vector）时补一个 copy_to_reg（:555–:556）
  // FA3 里就是 rowsum：prologue 已算完 tile 0 的 rowsum，稳态循环从它继续。
  for (auto [index, iterArg] : llvm::enumerate(loop.getRegionIterArgs())) {
    Value init = oldInits[index];
    auto it = prologueWritten.find(iterArg);
    if (it != prologueWritten.end()) {
      Value produced = it->second;
      if (produced.getType() != iterArg.getType()) {
        if (!isa<MemRefType>(produced.getType()) ||
            !isa<VectorType>(iterArg.getType()))
          return loop.emitError()
                 << "cannot materialise the prologue value of loop carried "
                    "value #" << index;
        produced = builder.create<frisk::CopyToRegOp>(loc, iterArg.getType(),
                                                      produced);
      }
      init = produced;
    }
    initValues.push_back(init);
  }

  // 5) Steady state loop.  Every step handles the tile `i + offset` of each
  //    statement, the parity of the tile decides which slot is touched.
  // 新上界 = 旧 ub - step，也就是少跑一块（peeledTiles=1），最后一块留给 epilogue。
  AffineMap oldUbMap = loop.getUpperBoundMap();
  AffineExpr ubExpr = oldUbMap.getResult(0) -
                      getAffineConstantExpr(step, context);
  AffineMap newUbMap =
      AffineMap::get(oldUbMap.getNumDims(), oldUbMap.getNumSymbols(), {ubExpr},
                     context);

  auto bodyBuilder = [&](OpBuilder &nb, Location bodyLoc, Value iv,
                         ValueRange args) {
    // The single synchronisation point of a steady state step: it makes sure
    // that the slot written by this step is no longer read by the step before.
    // 循环体顶部唯一一个 barrier。它保证本步要写的槽在上一步已经读完（距离 2 的依赖）
    nb.create<frisk::SyncThreadsInBlockOp>(bodyLoc);
    // 算 k = iv - lb、tile = k/step、parity = tile % 2、isEven
    Value zeroIdx = nb.create<arith::ConstantIndexOp>(bodyLoc, 0);
    Value twoIdx = nb.create<arith::ConstantIndexOp>(bodyLoc, 2);
    Value k = nb.create<arith::SubIOp>(bodyLoc, iv, lbValue);
    Value tile = nb.create<arith::DivSIOp>(bodyLoc, k, stepCst);
    Value parity = nb.create<arith::RemSIOp>(bodyLoc, tile, twoIdx);
    Value isEven = nb.create<arith::CmpIOp>(bodyLoc, arith::CmpIPredicate::eq,
                                            parity, zeroIdx);

    // Tile index (as an induction value) of every statement, plus the guards
    // of the statements whose tile can run past the last one.
    // 对每个出现过的 offset 预生成 iv（offsetIv）和守卫。
    // 守卫条件是 offset > peeledTiles
    // 该语句的 tile 是 i+offset，稳态只到 N-1-peeledTiles，
    // 所以 offset 更大的语句可能越界。peeledTiles=1 时正好只有 offset 2 的 K 需要守卫。
    DenseMap<int64_t, Value> ivForOffset;
    DenseMap<int64_t, Value> guardForOffset;
    for (unsigned index : emitOrder) {
      Statement &st = stmts[index];
      if (!ivForOffset.count(st.offset))
        ivForOffset[st.offset] = offsetIv(nb, iv, st.offset, step);
      if (st.offset > options.peeledTiles && !guardForOffset.count(st.offset))
        guardForOffset[st.offset] =
            nb.create<arith::CmpIOp>(bodyLoc, arith::CmpIPredicate::slt,
                                     ivForOffset[st.offset], ubValue);
    }
    // 奇偶只用来选 buffer：每个 (buffer, offset) 组合发一个小 scf.if 选出这拍
    // 要用的槽（见 materializeSlotHandles），后序语句照常只发射一遍、直接引用
    // 这些槽句柄，不再整段复制成两个奇偶大分支
    SlotHandleMap slotHandles = materializeSlotHandles(nb, isEven, emitOrder);
    // 整步共用一套 IRMapping；每个语句发射前会重设自己那份 buffer 绑定
    IRMapping map;
    for (unsigned index : emitOrder) {
      Statement &st = stmts[index];
      // slotHandles 非空时 parity 参数不再使用，槽全部来自选出来的句柄
      emitStatement(nb, st, /*parity=*/0, ivForOffset[st.offset], args,
                    /*dropLoopCarriedWrites=*/false,
                    guardForOffset.lookup(st.offset), map, nullptr,
                    &slotHandles);
    }
    // 对 iter_arg 的「更新」是由克隆出来的 frisk.copy ... to %iter_arg 原地写
    // 寄存器完成的，SSA 值本身不变，所以直接转发——和原循环 affine.yield
    // %arg5, %arg6 的写法一致
    nb.create<affine::AffineYieldOp>(bodyLoc, args);
  };
  // 创建新循环
  auto newLoop = builder.create<affine::AffineForOp>(
      loc, lbOperands, loop.getLowerBoundMap(), ubOperands, newUbMap, step,
      initValues, bodyBuilder);

  // 6) Epilogue: the peeled tiles (only the statements that consume the tile
  //    itself), then the original tail of the function.
  // 插入点挪到新循环之后，插一个 barrier，把稳态循环最后写的槽和 epilogue 的读隔开
  builder.setInsertionPointAfter(newLoop);
  builder.create<frisk::SyncThreadsInBlockOp>(loc);
  // 算最后一块（被 peel 的）的 tile 与 parity：
  // lastK = ub - step - lb、lastTile = lastK/step、lastParity = lastTile % 2。
  Value twoIdx = builder.create<arith::ConstantIndexOp>(loc, 2);
  Value lastK = builder.create<arith::SubIOp>(loc, ubMinusStep, lbValue);
  Value lastTile = builder.create<arith::DivSIOp>(loc, lastK, stepCst);
  Value lastParity = builder.create<arith::RemSIOp>(loc, lastTile, twoIdx);
  Value zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value isEvenLast = builder.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::eq, lastParity, zeroIdx);
  // 只有处理当前 tile 的那一级（offset == 0）在 epilogue 还有活干：
  // FA3 里就是最后一个 tile 的 PV + 累加。
  SmallVector<unsigned> peeled;
  for (unsigned index : emitOrder)
    if (stmts[index].offset == 0)
      peeled.push_back(index);
  // 和稳态一样，只按最后一块的奇偶选 buffer 槽，语句本身不复制
  SlotHandleMap slotHandles =
      materializeSlotHandles(builder, isEvenLast, peeled);
  DenseMap<Value, Value> epilogueWritten;
  IRMapping map;
  for (unsigned index : peeled) {
    // iv 固定为 ubMinusStep（最后一块），iterArgValues 用稳态循环的最终结果，
    // dropLoopCarriedWrites=true 并收集 epilogueWritten
    emitStatement(builder, stmts[index], /*parity=*/0, ubMinusStep,
                  newLoop.getResults(), /*dropLoopCarriedWrites=*/true, Value(),
                  map, &epilogueWritten, &slotHandles);
  }
  // 取出第一个 iter_arg 的 epilogue 结果，类型不符时补 copy_to_reg
  auto it = epilogueWritten.find(loop.getRegionIterArgs()[0]);
  if (it == epilogueWritten.end())
    return loop.emitError()
           << "the epilogue does not update the first loop carried value";
  Value epilogueResult = it->second;
  Type want = loop.getRegionIterArgs()[0].getType();
  if (epilogueResult.getType() != want) {
    if (!isa<MemRefType>(epilogueResult.getType()) || !isa<VectorType>(want))
      return loop.emitError()
             << "cannot materialise the epilogue value of the loop carried "
                "value #0";
    epilogueResult =
        builder.create<frisk::CopyToRegOp>(loc, want, epilogueResult);
  }

  // 7) Republish the results of the original loop and drop it.
  SmallVector<Operation *> deadAllocs;
  // 记录不再用的旧单槽 alloc
  for (RotatedBuffer &buffer : rotated)
    if (buffer.alloc && !loop->isAncestor(buffer.alloc))
      deadAllocs.push_back(buffer.alloc.getOperation());

  SmallVector<Value> replacements;
  // 重定向原循环结果的 uses：
  // 结果 #0（acc）用 epilogue 的结果，其余用新循环的结果。这就是「epilogue 收尾 O」的落地
  for (auto [index, result] : llvm::enumerate(loop.getResults())) {
    (void)result;
    replacements.push_back(index == 0 ? epilogueResult
                                      : newLoop.getResult(index));
  }
  for (auto [index, result] : llvm::enumerate(loop.getResults()))
    result.replaceAllUsesWith(replacements[index]);

  // Carry over the annotations of the original loop, but never its structural
  // attributes: the new loop has a different upper bound.
  // 把旧循环的非结构性属性搬到新循环，
  // 跳过 operandSegmentSizes/lowerBoundMap/upperBoundMap/step 和两个方案属性
  for (NamedAttribute attr : loop->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name == "operandSegmentSizes" || name == "lowerBoundMap" ||
        name == "upperBoundMap" || name == "step" ||
        name == kSchemeStageAttr || name == kSchemeOrderAttr)
      continue;
    newLoop->setAttr(attr.getName(), attr.getValue());
  }
  // 打上 pipeline.scheduled/num_stages/slots
  newLoop->setAttr(kScheduledAttr, builder.getUnitAttr());
  newLoop->setAttr(kNumStagesAttr, builder.getI64IntegerAttr(numStages));
  newLoop->setAttr(kSlotsAttr, builder.getI64IntegerAttr(options.maxSlots));

  loop.erase();

  for (Operation *alloc : deadAllocs)
    if (alloc->use_empty())
      alloc->erase();

  return success();
}

LogicalResult PipelineScheduler::run() {
  if (loop.getNumIterOperands() == 0)
    return loop.emitError() << "the pipelined loop has no iter_args";
  if (failed(collectStatements()))
    return failure();
  if (stmts.empty())
    return success();
  if (failed(collectRotatedBuffers()))
    return failure();
  return build();
}

/// Returns the annotated loop of `func`, if any.
affine::AffineForOp findAnnotatedLoop(func::FuncOp func) {
  affine::AffineForOp result;
  func.walk([&](affine::AffineForOp candidate) {
    if (result)
      return;
    if (candidate->hasAttr(kSchemeStageAttr)) {
      result = candidate;
      return;
    }
  });
  return result;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public entry points
//===----------------------------------------------------------------------===//

LogicalResult pipeline::applyPipelineSchedule(
    func::FuncOp func, const PipelineScheduleOptions &options) {
  affine::AffineForOp loop = findAnnotatedLoop(func);
  if (!loop)
    return success(); // already scheduled, or nothing to schedule

  PipelineScheduler scheduler(loop, options);
  return scheduler.run();
}

namespace {
struct PipelineSchedulePass
    : public pipeline::impl::PipelineScheduleBase<PipelineSchedulePass> {
  void runOnOperation() override {
    if (failed(pipeline::applyPipelineSchedule(getOperation(), options)))
      signalPassFailure();
  }

  pipeline::PipelineScheduleOptions options;
};
} // namespace

std::unique_ptr<Pass> pipeline::createPipelineSchedulePass() {
  return std::make_unique<PipelineSchedulePass>();
}
