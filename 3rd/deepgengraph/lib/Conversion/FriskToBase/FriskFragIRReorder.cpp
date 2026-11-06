//===----------------------------------------------------------------------===//
// Fuse affine fragment loops without losing SSA vector/memref carriers.
// Scheduling A(all), B(all) as (A(i), B(i)) reverses B(i) / A(j), j > i.
// Check precisely those dependences; never infer independence from a marker.
//===----------------------------------------------------------------------===//
// ===========================================================================
// 本文件包含 4 个 Pass（文件末尾注册）：
//
// FriskFragIRReorder（frisk-frag-ir-reorder）   优化汇总.py 第 8 项，fragment 阶段
//   1. tryFuse：把相邻的同边界 fragment 循环融合成一个，直到不动点。
//   2. pipelineFragments：只读 fragment 循环做一级 lookahead，让“下一片的 load”排在“当前片的
//      计算”之前，用计算掩盖 load 延迟。
//
// ScheduleAsyncCopies（frisk-schedule-async-copies）   第 11 项，fragment 阶段
//   scheduleAsyncCopies：在基本块内把 DMA 发起尽量上提、wait 尽量下沉。
//
// AnalyzeShmLiveness（frisk-analyze-shm-liveness）   第 12 项
//   只打印 LDS 生命周期报告，不改 IR。
//
// LowerFriskFragments（lower-frisk-fragments）   第 15 项，thread 阶段
//   LDS 打包 -> 回边同步 -> 再次调度 DMA -> LDS 物理复用 -> 展开 fragment 循环。
//
// 文件内的证明方法：循环边界都是小常量，所以把每次迭代的坐标直接求值出来比较，是
// “有界证明”而不是启发式。求值不出来或超预算就放弃优化。
// ===========================================================================
#include "deepgengraph/Conversion/AffineLoopEval.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "ScheduleAttrs.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Analysis/LowerInfo.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SetVector.h"
#include <functional>
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <optional>
#include <set>

namespace mlir::frisk {
namespace {
#define GEN_PASS_DEF_FRISKFRAGIRREORDER
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"
using Loop = affine::AffineForOp;
// Coordinate：一个多维坐标。Footprint：一组坐标，也就是一次访问覆盖的逻辑寄存器元素或内存位置。
using Coordinate = SmallVector<int64_t>;
using Footprint = std::set<Coordinate>;

// Evaluation is a bounded proof, not a heuristic. Unknown/dynamic coordinates
// cause rejection. Small static fragment loops are the intended workload.
// 可枚举的最大迭代次数，超过就放弃证明。
static constexpr int64_t maxIterations = 256;

// 把一组索引（常量或 SSA 值）求值成坐标。
static std::optional<Coordinate> position(ArrayRef<OpFoldResult> indices,
                                           Loop loop, int64_t iv) {
  Coordinate result;
  for (OpFoldResult index : indices) {
    if (auto attr = dyn_cast<Attribute>(index))
      result.push_back(cast<IntegerAttr>(attr).getInt());
    else if (auto number = evaluateAtIteration(cast<Value>(index), loop, iv))
      result.push_back(*number);
    else
      return std::nullopt;
  }
  return result;
}

// 枚举循环 iv 的所有取值。返回空表示非常量边界、超限或循环本身没有迭代，调用方都按“无法证明”处理。
static SmallVector<int64_t> iterations(Loop loop) {
  SmallVector<int64_t> values;
  if (!loop.hasConstantBounds())
    return values;
  for (int64_t i = loop.getConstantLowerBound(); i < loop.getConstantUpperBound();) {
    if (values.size() == maxIterations)
      return {};
    values.push_back(i);
    if (__builtin_add_overflow(i, loop.getStepAsInt(), &i))
      return {};
  }
  return values;
}

// A memref yield transports a descriptor, not a snapshot of the contents.
// Only invariant descriptors are resolved; swapped/selected aliases stay unknown.
// 穿过 iter_arg / 循环结果，找到 memref 真正的来源。只有 yield 原样传递时才穿透，
// 轮转交换（swap）的 buffer 保持未知。
static Value invariantMemref(Value value) {
  for (unsigned depth = 0; depth < 32; ++depth) {
    Loop loop;
    unsigned number;
    if (auto result = dyn_cast<OpResult>(value)) {
      loop = dyn_cast<Loop>(result.getOwner());
      number = result.getResultNumber();
    } else if (auto arg = dyn_cast<BlockArgument>(value)) {
      loop = dyn_cast_or_null<Loop>(arg.getOwner()->getParentOp());
      if (!loop || arg.getArgNumber() == 0)
        return value;
      number = arg.getArgNumber() - 1;
    } else {
      return value;
    }
    if (!loop || !isa<MemRefType>(value.getType()))
      return value;
    Value yielded = loop.getBody()->getTerminator()->getOperand(number);
    if (yielded != loop.getRegionIterArgs()[number] &&
        yielded != loop.getInits()[number])
      return value;
    value = loop.getInits()[number];
  }
  return value;
}

static bool belongsToIteration(Operation *op, Loop loop) {
  return op->getParentOp() == loop;
}

// Enumerate logical register elements, not fragment grid points. Different
// fragment shapes can overlap even when their grid coordinates differ.
// 计算 insert/extract 在第 iv 次迭代访问了哪些逻辑寄存器元素。
// InsertFrag/ExtractFrag 的索引是 fragment 网格坐标，要乘以 fragment 形状展开到元素，
// 因为形状不同的 fragment 即使网格坐标不同也可能重叠。
static std::optional<Footprint> registerFootprint(Operation *op, Loop loop, int64_t iv) {
  SmallVector<OpFoldResult> indices;
  SmallVector<int64_t> shape;
  if (auto insert = dyn_cast<InsertFragOp>(op)) {
    indices.append(insert.getIndices().begin(), insert.getIndices().end());
    shape = llvm::to_vector(cast<FragmentType>(insert.getValue().getType()).getRegisterType().getShape());
  } else if (auto extract = dyn_cast<ExtractFragOp>(op)) {
    indices.append(extract.getIndices().begin(), extract.getIndices().end());
    shape = llvm::to_vector(cast<FragmentType>(extract.getResult().getType()).getRegisterType().getShape());
  } else if (auto insert = dyn_cast<vector::InsertOp>(op)) {
    if (isa<VectorType>(insert.getValueToStore().getType())) return {};
    indices = insert.getMixedPosition();
  } else if (auto extract = dyn_cast<vector::ExtractOp>(op)) {
    if (isa<VectorType>(extract.getType())) return {};
    indices = extract.getMixedPosition();
  } else return {};
  auto start = position(indices, loop, iv);
  if (!start) return {};
  if (shape.empty()) return Footprint{*start};
  int64_t size = 1;
  for (auto [d, extent] : llvm::enumerate(shape)) {
    if (extent <= 0 || size > 65536 / extent ||
        __builtin_mul_overflow((*start)[d], extent, &(*start)[d])) return {};
    size *= extent;
  }
  Footprint result;
  for (int64_t n = 0; n < size; ++n) {
    auto coordinate = *start;
    int64_t rest = n;
    for (int d = shape.size() - 1; d >= 0; --d) {
      if (__builtin_add_overflow(coordinate[d], rest % shape[d], &coordinate[d])) return {};
      rest /= shape[d];
    }
    result.insert(coordinate);
  }
  return result;
}
static Value insertDestination(Operation *op) {
  if (auto insert = dyn_cast<InsertFragOp>(op)) return insert.getDestination();
  if (auto insert = dyn_cast<vector::InsertOp>(op)) return insert.getDest();
  return {};
}

// Prove an insert chain writes distinct scalar positions on each iteration.
// Restrict carrier uses to this chain and own-position scalar extracts; opaque
// vector arithmetic/reductions on the whole carrier are not pointwise updates.
// 证明第 number 个 vector carrier 是“每轮写不同位置”的 insert 链。
// 输出 writes[n] = 第 n 次迭代写入的足迹。
// versions：carrier 在 insert 链上的每个 SSA 版本。
// chain：链上的 insert 集合。
// 所有 version 的使用者只能是 yield、链上下一个 insert，或只读本轮已写位置的 extract。
static bool vectorWrites(Loop loop, unsigned number,
                         SmallVectorImpl<Footprint> &writes) {
  Value arg = loop.getRegionIterArgs()[number];
  Value value = loop.getBody()->getTerminator()->getOperand(number);
  SmallVector<Operation *> inserts;
  llvm::SmallPtrSet<Operation *, 16> chain;
  SmallVector<Value> versions{value};
  while (value != arg) {
    auto *insert = value.getDefiningOp();
    if (!insert || !insertDestination(insert) || insert->getBlock() != loop.getBody())
      return false;
    inserts.push_back(insert);
    chain.insert(insert);
    value = insertDestination(insert);
    versions.push_back(value);
  }
  auto ivs = iterations(loop);
  if (ivs.empty() || ivs.size() * inserts.size() > 65536)
    return false;
  Footprint all;
  for (int64_t iv : ivs) {
    Footprint current;
    for (auto insert : inserts) {
      auto footprint = registerFootprint(insert, loop, iv);
      if (!footprint || all.size() + footprint->size() > 65536) return false;
      for (const auto &index : *footprint)
        if (!current.insert(index).second || !all.insert(index).second) return false;
    }
    writes.push_back(std::move(current));
  }
  for (Value version : versions) {
    for (OpOperand &use : version.getUses()) {
      if (use.getOwner() == loop.getBody()->getTerminator() ||
          (chain.contains(use.getOwner()) && use.getOperandNumber() == 1))
        continue;
      auto *extract = use.getOwner();
      if (!isa<ExtractFragOp, vector::ExtractOp>(extract) ||
          !belongsToIteration(extract, loop) || use.getOperandNumber() != 0)
        return false;
      for (auto [n, iv] : llvm::enumerate(ivs)) {
        auto footprint = registerFootprint(extract, loop, iv);
        if (!footprint || !llvm::all_of(*footprint, [&](const auto &p) { return writes[n].count(p); }))
          return false;
      }
    }
  }
  return true;
}

// 证明 B 对 A 结果（vector）的使用可以融合：B 第 n 轮只读 A 第 n 轮写过的位置。
// 如果 result 是 B 的 init，还要求 B 写的位置与 A 完全相同。
static bool checkVectorUse(Value result, Loop a, Loop b, unsigned number) {
  SmallVector<Footprint> writes;
  if (!vectorWrites(a, number, writes))
    return false;
  auto ivs = iterations(b);
  if (ivs.size() != writes.size())
    return false;
  for (OpOperand &use : result.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == b) {
      // Bounds cannot depend on a vector. Replace this init by A's init only
      // when B overwrites exactly the same positions and reads its own slice.
      unsigned init = use.getOperandNumber() - b.getNumControlOperands();
      SmallVector<Footprint> consumerWrites;
      if (init >= b.getNumIterOperands() || !vectorWrites(b, init, consumerWrites) ||
          writes != consumerWrites)
        return false;
    } else if (b->isAncestor(owner)) {
      if (!isa<ExtractFragOp, vector::ExtractOp>(owner) || !belongsToIteration(owner, b) ||
          use.getOperandNumber() != 0)
        return false;
      for (auto [n, iv] : llvm::enumerate(ivs)) {
        auto footprint = registerFootprint(owner, b, iv);
        if (!footprint || !llvm::all_of(*footprint, [&](const auto &p) { return writes[n].count(p); }))
          return false;
      }
    }
  }
  return true;
}

// 一次内存访问的描述。
//   buffer    经 invariantMemref 解析后的 memref。
//   write     是否写。
//   indices / map  访问索引和 affine map（map 可为空）。
//   fragment  是否 LoadFrag/StoreFrag。它的坐标是寄存器坐标，不是内存索引，不能直接比较。
struct Access {
  Value buffer;
  bool write;
  SmallVector<OpFoldResult> indices;
  AffineMap map;
  bool fragment = false;
};
// 收集循环内所有内存访问。遇到 reduce、布局转换、mask、barrier、调用等一律返回 false，放弃融合。
static bool collectAccesses(Loop loop, SmallVectorImpl<Access> &accesses) {
  auto result = loop.walk([&](Operation *op) {
    Value buffer;
    bool write = false;
    ValueRange indices;
    AffineMap map;
    bool fragment = false;
    if (isa<ReduceFragOp, ConvertLayoutFragOp, MaskFragOp>(op))
      return WalkResult::interrupt();
    if (auto load = dyn_cast<LoadFragOp>(op)) {
      buffer = load.getSource(); fragment = true;
    } else if (auto store = dyn_cast<StoreFragOp>(op)) {
      buffer = store.getDestination(); write = true; fragment = true;
    } else if (isa<ToThreadTileOp, FromThreadTileOp>(op) &&
               llvm::any_of(op->getOperandTypes(), [](Type t) { return isa<MemRefType>(t); })) {
      return WalkResult::interrupt(); // Hidden memory access, not a descriptor.
    } else if (auto load = dyn_cast<affine::AffineLoadOp>(op)) {
      buffer = load.getMemref(); indices = load.getMapOperands(); map = load.getAffineMap();
    } else if (auto store = dyn_cast<affine::AffineStoreOp>(op)) {
      buffer = store.getMemref(); indices = store.getMapOperands(); map = store.getAffineMap(); write = true;
    } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
      buffer = load.getMemref(); indices = load.getIndices();
    } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
      buffer = store.getMemref(); indices = store.getIndices(); write = true;
    } else if (isa<Loop, affine::AffineYieldOp>(op)) {
      return WalkResult::advance();
    } else {
      // Includes barriers, calls, allocation/free, atomics and unknown effects.
      return isMemoryEffectFree(op) ? WalkResult::advance() : WalkResult::interrupt();
    }
    accesses.push_back({invariantMemref(buffer), write,
                       SmallVector<OpFoldResult>(indices.begin(), indices.end()), map, fragment});
    return WalkResult::advance();
  });
  return !result.wasInterrupted();
}

// 访问在第 iv 次迭代的具体地址。
static std::optional<Coordinate> address(const Access &access, Loop loop, int64_t iv) {
  auto operands = position(access.indices, loop, iv);
  if (!operands || !access.map)
    return operands;
  SmallVector<Attribute> attrs, results;
  for (int64_t value : *operands)
    attrs.push_back(IntegerAttr::get(IndexType::get(loop.getContext()), value));
  if (failed(access.map.constantFold(attrs, results)))
    return std::nullopt;
  Coordinate coord;
  for (Attribute attr : results)
    coord.push_back(cast<IntegerAttr>(attr).getInt());
  return coord;
}

// 融合 A(all), B(all) -> (A(i), B(i)) 只会颠倒“B(i) 与 A(j), j > i”的先后。
// 所以只需检查这些配对：至少一方是写、可能别名时，地址必须不同。
// 这里 earlier = B(i)，later = A(j)。
static bool memorySafe(Loop a, Loop b, AliasAnalysis &aliases) {
  SmallVector<Access> first, second;
  if (!collectAccesses(a, first) || !collectAccesses(b, second))
    return false;
  auto ivs = iterations(a);
  // Limit quadratic proof cost; exceeding the budget leaves the loops intact.
  if (first.size() * second.size() * ivs.size() * ivs.size() > 1048576)
    return false;
  for (const Access &x : first) {
    for (const Access &y : second) {
      if ((!x.write && !y.write) || aliases.alias(x.buffer, y.buffer).isNo())
        continue;
      // A distributed fragment's register coordinates are not memory indices.
      // Until the physical mapping is proved, reject aliasing writes.
      if (x.fragment || y.fragment) return false;
      if (x.buffer != y.buffer || ivs.empty() ||
          !cast<MemRefType>(x.buffer.getType()).getLayout().isIdentity())
        return false;
      for (unsigned i = 0; i < ivs.size(); ++i) {
        auto earlier = address(y, b, ivs[i]);
        if (!earlier)
          return false;
        for (unsigned j = i + 1; j < ivs.size(); ++j) {
          auto later = address(x, a, ivs[j]);
          if (!later || *earlier == *later)
            return false;
        }
      }
    }
  }
  return true;
}

// 融合相邻循环 A、B。合法性检查分 4 步：边界相同；中间的 op 纯净且不用 A 的结果；
// A 的结果被 B 使用时可以逐迭代对应（checkVectorUse）；memorySafe。
static bool tryFuse(Loop a, Loop b, AliasAnalysis &aliases) {
  if (a->hasAttr("frisk.pipelined") || b->hasAttr("frisk.pipelined"))
    return false;
  if (a.getLowerBoundMap() != b.getLowerBoundMap() ||
      a.getUpperBoundMap() != b.getUpperBoundMap() || a.getStep() != b.getStep() ||
      !llvm::equal(a.getLowerBoundOperands(), b.getLowerBoundOperands()) ||
      !llvm::equal(a.getUpperBoundOperands(), b.getUpperBoundOperands()))
    return false;
  // Crossing pure, speculatable setup is safe only if it does not consume A.
  for (Operation *op = a->getNextNode(); op != b; op = op->getNextNode()) {
    if (!isMemoryEffectFree(op) || !isSpeculatable(op) || op->getNumRegions())
      return false;
    for (Value operand : op->getOperands())
      if (operand.getDefiningOp() == a)
        return false;
  }
  for (auto [n, result] : llvm::enumerate(a.getResults())) {
    bool consumed = llvm::any_of(result.getUsers(), [&](Operation *op) {
      return op == b || b->isAncestor(op);
    });
    if (!consumed)
      continue;
    if (isa<MemRefType>(result.getType())) {
      if (invariantMemref(result) == result)
        return false;
    } else if (!isa<VectorType>(result.getType()) || !checkVectorUse(result, a, b, n)) {
      return false;
    }
  }
  if (!memorySafe(a, b, aliases))
    return false;

  // inits：融合循环的初值 = A 的 inits + B 的 inits。B 的 init 如果来自 A 的结果，改用 A 的 init，
  // 因为 A 的结果在融合循环内部才产生。下面的 overlay 负责补回 A 的本轮写入。
  OpBuilder builder(b);
  SmallVector<Value> inits(a.getInits());
  for (Value init : b.getInits()) {
    if (auto result = dyn_cast<OpResult>(init); result && result.getOwner() == a)
      init = a.getInits()[result.getResultNumber()];
    inits.push_back(init);
  }
  auto fused = builder.create<Loop>(a.getLoc(), a.getLowerBoundOperands(),
      a.getLowerBoundMap(), a.getUpperBoundOperands(), a.getUpperBoundMap(),
      a.getStepAsInt(), inits);
  // Keep common scheduling metadata only; a mismatching fragment shape must
  // not become a false promise for a later scheduler.
  for (NamedAttribute attr : a->getAttrs())
    if (b->getAttr(attr.getName()) == attr.getValue() &&
        attr.getName() != "operandSegmentSizes")
      fused->setAttr(attr.getName(), attr.getValue());
  fused->setAttr("frisk.fused", builder.getUnitAttr());
  builder.setInsertionPointToStart(fused.getBody());
  IRMapping mapping;
  mapping.map(a.getInductionVar(), fused.getInductionVar());
  mapping.map(b.getInductionVar(), fused.getInductionVar());
  // firstCount：融合循环中属于 A 的 carrier 数，前 firstCount 个属于 A，其余属于 B。
  unsigned firstCount = a.getNumResults();
  for (auto [old, replacement] : llvm::zip(a.getRegionIterArgs(),
                                         fused.getRegionIterArgs().take_front(firstCount)))
    mapping.map(old, replacement);
  for (auto [old, replacement] : llvm::zip(b.getRegionIterArgs(),
                                         fused.getRegionIterArgs().drop_front(firstCount)))
    mapping.map(old, replacement);
  for (Operation &op : a.getBody()->without_terminator())
    builder.clone(op, mapping);
  SmallVector<Value> yields;
  for (auto [result, value] : llvm::zip(a.getResults(), a.getBody()->getTerminator()->getOperands())) {
    Value yielded = mapping.lookupOrDefault(value);
    yields.push_back(yielded);
    mapping.map(result, yielded);
  }
  // A vector used as B's initial value was complete before fusion. Overlay
  // A's current slice on B's carried value before B reads it, retaining all
  // previously computed B slices. Merely substituting A's init is incorrect.
  for (auto [n, init] : llvm::enumerate(b.getInits())) {
    auto result = dyn_cast<OpResult>(init);
    if (!result || result.getOwner() != a || !isa<VectorType>(init.getType()))
      continue;
    Value value = a.getBody()->getTerminator()->getOperand(result.getResultNumber());
    SmallVector<Operation *> inserts;
    while (auto *insert = value.getDefiningOp()) {
      if (!insertDestination(insert)) break;
      inserts.push_back(insert);
      value = insertDestination(insert);
    }
    Value updated = fused.getRegionIterArgs()[firstCount + n];
    for (auto *op : llvm::reverse(inserts)) {
      if (auto insert = dyn_cast<InsertFragOp>(op)) {
        SmallVector<Value> indices;
        for (Value index : insert.getIndices()) indices.push_back(mapping.lookupOrDefault(index));
        updated = builder.create<InsertFragOp>(insert.getLoc(), updated.getType(),
            mapping.lookupOrDefault(insert.getValue()), updated, indices);
        continue;
      }
      auto insert = cast<vector::InsertOp>(op);
      SmallVector<OpFoldResult> indices;
      for (OpFoldResult index : insert.getMixedPosition())
        indices.push_back(isa<Value>(index) ? OpFoldResult(mapping.lookupOrDefault(cast<Value>(index))) : index);
      updated = builder.create<vector::InsertOp>(insert.getLoc(),
          mapping.lookupOrDefault(insert.getValueToStore()), updated, indices);
    }
    mapping.map(b.getRegionIterArgs()[n], updated);
  }
  for (Operation &op : b.getBody()->without_terminator())
    builder.clone(op, mapping);
  for (Value value : b.getBody()->getTerminator()->getOperands())
    yields.push_back(mapping.lookupOrDefault(value));
  if (!inits.empty())
    builder.create<affine::AffineYieldOp>(b.getLoc(), yields);
  a.replaceAllUsesWith(fused.getResults().take_front(firstCount));
  b.replaceAllUsesWith(fused.getResults().drop_front(firstCount));
  b.erase();
  a.erase();
  return true;
}

// A one-iteration lookahead schedule for read-only fragment loops:
//   prologue: load(0)
//   steady:   next = load(i+1); current = compute(i, carriedLoad)
//   epilogue: compute(last, carriedLoad)
// No speculative last load, no reordered accumulator updates, no movement
// across a barrier or a write. Dependencies come from ordinary SSA operands.
// 把循环拆成 prologue + steady 循环 + epilogue。
// 要求循环带 frisk.fragment_loop、至少 2 次迭代、至少有一个 load，其它 op 都是纯计算。
static bool pipelineFragments(Loop loop) {
  if (!loop->hasAttr("frisk.fragment_loop") || loop->hasAttr("frisk.pipelined"))
    return false;
  auto ivs = iterations(loop);
  if (ivs.size() < 2)
    return false;
  // loads：循环体内所有读操作，都要被提前一轮。
  SmallVector<Operation *> loads;
  for (Operation &op : loop.getBody()->without_terminator())
    if (isa<LoadFragOp, affine::AffineLoadOp, memref::LoadOp, vector::LoadOp>(op))
      loads.push_back(&op);
  if (loads.empty())
    return false;
  // Validate the entire iteration. Unknown effects, collectives, nested
  // control flow and writes are scheduling boundaries.
  for (Operation &op : loop.getBody()->without_terminator()) {
    if (isa<LoadFragOp, affine::AffineLoadOp, memref::LoadOp, vector::LoadOp>(op))
      continue;
    if (op.getNumRegions() || !isMemoryEffectFree(&op) ||
        isa<gpu::ShuffleOp, ReduceFragOp, ConvertLayoutFragOp, ToThreadTileOp, FromThreadTileOp>(op))
      return false;
  }

  // groups：load 集合。
  // slice：load 及其地址计算（反向依赖），提前时一起克隆。
  // addDependency：把 value 的定义加入 slice。依赖另一条 load（间接读）就失败。
  llvm::SmallPtrSet<Operation *, 16> groups, slice;
  groups.insert(loads.begin(), loads.end());
  std::function<bool(Value)> addDependency = [&](Value value) {
    if (value == loop.getInductionVar())
      return true;
    if (auto arg = dyn_cast<BlockArgument>(value))
      return arg.getOwner() != loop.getBody();
    Operation *def = value.getDefiningOp();
    if (!def || def->getBlock() != loop.getBody())
      return true;
    if (slice.contains(def))
      return true;
    // A dependent memory read is not a pure address calculation. Reject
    // indirect loads rather than duplicating or moving them into the prologue.
    if (groups.contains(def) || def->getNumRegions() ||
        !isMemoryEffectFree(def) || !isSpeculatable(def))
      return false;
    if (!llvm::all_of(def->getOperands(), addDependency))
      return false;
    slice.insert(def);
    return true;
  };
  for (Operation *load : loads) {
    if (!llvm::all_of(load->getOperands(), addDependency))
      return false;
  }
  slice.insert(loads.begin(), loads.end());

  OpBuilder b(loop);
  Location loc = loop.getLoc();
  // cloneLoads：在给定 iv 下克隆 load 切片，返回 load 结果。stage 记录 prologue / prefetch。
  // cloneCompute：在给定 iv 下克隆其余计算，load 结果用已预取的值替换，返回 yield 值。
  auto cloneLoads = [&](Value iv, StringRef stage) {
    IRMapping mapping;
    mapping.map(loop.getInductionVar(), iv);
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (!slice.contains(&op))
        continue;
      Operation *copy = b.clone(op, mapping);
      if (groups.contains(&op))
        copy->setAttr("frisk.pipeline_stage", b.getStringAttr(stage));
    }
    SmallVector<Value> values;
    for (auto load : loads)
      for (Value value : load->getResults())
        values.push_back(mapping.lookup(value));
    return values;
  };
  auto cloneCompute = [&](Value iv, ValueRange carriers, ValueRange prefetched) {
    IRMapping mapping;
    mapping.map(loop.getInductionVar(), iv);
    mapping.map(loop.getRegionIterArgs(), carriers);
    unsigned position = 0;
    for (auto load : loads)
      for (Value value : load->getResults())
        mapping.map(value, prefetched[position++]);
    for (Operation &op : loop.getBody()->without_terminator())
      if (!groups.contains(&op))
        b.clone(op, mapping);
    SmallVector<Value> values;
    for (Value value : loop.getBody()->getTerminator()->getOperands())
      values.push_back(mapping.lookupOrDefault(value));
    return values;
  };

  // 生成：
  //   prefetched = load(first)                         prologue
  //   steady for i in [first, last): next = load(i+1); compute(i, carried); carried = next
  //   compute(last, carried)                           epilogue
  // steady 的 iter_args = 原 carrier + 预取值。边界到 last 为止，不会越界多读。
  Value first = b.create<arith::ConstantIndexOp>(loc, ivs.front());
  SmallVector<Value> prefetched = cloneLoads(first, "prologue");
  SmallVector<Value> inits(loop.getInits());
  inits.append(prefetched);
  auto steady = b.create<Loop>(loc, ivs.front(), ivs.back(), loop.getStepAsInt(), inits);
  // Bounds/step belong to the newly constructed loop, not the old upper bound.
  for (NamedAttribute attr : loop->getAttrs())
    if (attr.getName().strref().starts_with("frisk.") || attr.getName() == "iterLabel")
      steady->setAttr(attr.getName(), attr.getValue());
  steady->setAttr("frisk.pipelined", b.getUnitAttr());
  b.setInsertionPointToStart(steady.getBody());
  Value next = b.create<affine::AffineApplyOp>(loc,
      AffineMap::get(1, 0, b.getAffineDimExpr(0) + loop.getStepAsInt()), steady.getInductionVar());
  SmallVector<Value> nextLoads = cloneLoads(next, "prefetch");
  unsigned carriers = loop.getNumResults();
  auto values = cloneCompute(steady.getInductionVar(),
      steady.getRegionIterArgs().take_front(carriers),
      steady.getRegionIterArgs().drop_front(carriers));
  values.append(nextLoads);
  b.create<affine::AffineYieldOp>(loc, values);
  b.setInsertionPointAfter(steady);
  Value last = b.create<arith::ConstantIndexOp>(loc, ivs.back());
  auto results = cloneCompute(last, steady.getResults().take_front(carriers),
                              steady.getResults().drop_front(carriers));
  loop.replaceAllUsesWith(results);
  loop.erase();
  return true;
}

#undef GEN_PASS_DEF_FRISKFRAGIRREORDER
#define GEN_PASS_DEF_LOWERFRISKFRAGMENTS
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

// 判断 wait 能否越过下一条 DMA（next）。wait lowering 后带 barrier，越过 DMA 相当于把 barrier 也
// 移到 DMA 之后。要证明两件事：两条都是可直发的 DMA 包；这个 barrier 不是在为 next 的目的地
// 释放旧读者。只在物理阶段使用（logical 时 scheduleAsyncCopies 直接不跨 DMA）。
// Crossing a DMA also moves a block barrier. Prove both direct packet legality
// and that this barrier is not releasing old readers of the crossed DMA's dst.
// Keep this separate from ordinary independence: issue hoisting must NOT gain
// permission to cross launches or waits from the completion-only relaxation.
class DirectDMAWaitMotion {
public:
  explicit DirectDMAWaitMotion(AliasAnalysis &aliases) : aliases(aliases) {}

  bool canCross(WaitBufferReadyOp wait, AsyncCopyGmToShmOp issue,
                AsyncCopyGmToShmOp next) {
    if (!isDirect(issue) || !isDirect(next) ||
        scheduled(wait) || scheduled(issue) || scheduled(next) ||
        !isInBlockUniformControlFlow(wait) ||
        !disjoint(issue.getDst(), next.getDst()) ||
        !disjoint(issue.getDst(), next.getSrc()) ||
        !disjoint(issue.getSrc(), next.getDst()))
      return false;

    // Walk back from the barrier being moved, not from next: descriptors and
    // intervening independent work were already checked by the ordinary sink.
    // Only an explicit barrier can discharge previous readers/writers. Named
    // waits may themselves move later, so do not rely on them here.
    // 从 wait 往前逐层回溯，直到遇到显式 barrier：路径上所有 op 都不能碰 next 的目的地。
    // 回溯到循环体开头时，还要处理上一轮迭代的旧读者，所以要求循环尾已有显式 barrier。
    Operation *cursor = wait;
    while (!isa<func::FuncOp>(cursor)) {
      for (Operation *prev = cursor->getPrevNode(); prev;
           prev = prev->getPrevNode()) {
        if (explicitBarrier(prev))
          return true;
        if (!safeBeforeBarrier(prev, next))
          return false;
      }
      Operation *parent = cursor->getParentOp();
      if (!isa<func::FuncOp>(parent)) {
        // The current block's previous iteration also contains old readers.
        // Require an existing explicit tail barrier; publication after the
        // next iteration's DMA would be too late.
        if (!isa<affine::AffineForOp, scf::ForOp>(parent) ||
            !explicitBarrier(cursor->getBlock()->getTerminator()->getPrevNode()))
          return false;
      }
      cursor = parent;
    }
    return true;
  }

private:
  static bool explicitBarrier(Operation *op) {
    return op && isa<gpu::BarrierOp, SyncThreadsInBlockOp>(op);
  }

  // op 或其祖先已有手动/流水调度，不能再动。
  static bool scheduled(Operation *op) {
    for (; op; op = op->getParentOp())
      if (hasAnyScheduleAttr(op))
        return true;
    return false;
  }

  // 穿过 view / cast 找到 memref 的根。
  static Value origin(Value value) {
    for (unsigned depth = 0; depth < 32; ++depth) {
      Operation *def = value.getDefiningOp();
      if (auto view = dyn_cast_or_null<ViewLikeOpInterface>(def))
        value = view.getViewSource();
      else if (auto cast = dyn_cast_or_null<memref::CastOp>(def))
        value = cast.getSource();
      else
        return value;
    }
    return {};
  }

  bool disjoint(Value a, Value b) {
    return aliases.alias(a, b).isNo();
  }

  // 能否作为“直发 DMA”：thread 数合法（64 的倍数，不超过 1024）、无 buffer_view、目的是未进入复用/打包
  // 计划的 alloc，且有可用的硬件传输宽度。
  bool isDirect(AsyncCopyGmToShmOp op) {
    auto function = op->getParentOfType<func::FuncOp>();
    auto threads = function->getAttrOfType<IntegerAttr>("thread_num");
    if (!threads || threads.getInt() <= 0 || threads.getInt() > 1024 ||
        threads.getInt() % 64 || op->hasAttr("frisk.force_sync") ||
        op.getSrc().getDefiningOp<BufferViewOp>() ||
        op.getDst().getDefiningOp<BufferViewOp>())
      return false;
    // Do not reschedule against an already materialized storage-reuse plan,
    // external storage, or an unresolved select/loop-carried descriptor.
    Value root = origin(op.getDst());
    auto alloc = root ? root.getDefiningOp<memref::AllocOp>() : memref::AllocOp{};
    if (!alloc || alloc->hasAttr("frisk.shm_pool") ||
        alloc->hasAttr("frisk.shared_pack"))
      return false;
    auto width = op->getAttrOfType<IntegerAttr>("frisk.dma_bytes");
    return selectAsyncDMATransferBytes(op, width ? width.getInt() : 0) != 0;
  }

  // op 位于被移动的 barrier 之前。判断它是否可能是 next 目的地的旧读者/写者，或是 next 源的写者。
  bool safeBeforeBarrier(Operation *op, AsyncCopyGmToShmOp next) {
    // These earlier synchronous regions are not being moved across the wait.
    // Inspect their accesses so an unrelated initialization/read loop does not
    // hide either an old reader or a source write. A nested barrier is not a
    // proof of a block-wide handoff on all paths.
    if (isa<affine::AffineForOp, affine::AffineIfOp, scf::ForOp, scf::IfOp>(op)) {
      for (Region &region : op->getRegions())
        for (Block &block : region)
          for (Operation &nested : block)
            if (!safeBeforeBarrier(&nested, next))
              return false;
      return true;
    }
    if (op->getNumRegions() || isa<ToThreadTileOp, FromThreadTileOp>(op))
      return false;
    if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(op))
      return disjoint(copy.getDst(), next.getDst()) &&
             disjoint(copy.getSrc(), next.getDst()) &&
             disjoint(copy.getDst(), next.getSrc());
    if (isa<WaitBufferReadyOp>(op))
      return false; // May include a staged repack or protect unrelated data.
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      return true;
    auto effects = dyn_cast<MemoryEffectOpInterface>(op);
    if (!effects)
      return false;
    SmallVector<MemoryEffects::EffectInstance> instances;
    effects.getEffects(instances);
    for (const auto &effect : instances) {
      Value value = effect.getValue();
      if (!value ||
          !isa<MemoryEffects::Read, MemoryEffects::Write, MemoryEffects::Free>(
              effect.getEffect()) ||
          !disjoint(value, next.getDst()) ||
          (!isa<MemoryEffects::Read>(effect.getEffect()) &&
           !disjoint(value, next.getSrc())))
        return false;
    }
    return true;
  }

  AliasAnalysis &aliases;
};

// op 或其祖先带手动流水标注。注意它不包含 frisk.pipelined，比 DirectDMAWaitMotion::scheduled 宽松。
static bool hasExplicitSchedule(Operation *op) {
  for (; op; op = op->getParentOp())
    if (hasPipelineScheduleAttr(op))
      return true;
  return false;
}

// Schedule within a block, at the logical fragment boundary and again after
// physical packing, always BEFORE reuse. Logical scheduling uses the storage
// analysis to require owned, understood destinations and proven task completion.
// It never crosses DMA launches with a wait: that needs the final packet proof.
// Physical reuse recomputes all lifetimes after the final schedule.
// 核心调度函数，ScheduleAsyncCopies（fragment 阶段）和 LowerFriskFragments（物理阶段）都调用。
//   logical  IR 仍含 fragment 或 to/from_thread_tile，说明还在逻辑阶段，规则更保守。
//   pooled   已有 frisk.shm_pool，物理复用已定，不能再调度。
// 对每条 DMA 做两件事：
//   1. 把 DMA（及其描述符切片）尽量上提。
//   2. 把对应 wait 尽量下沉到第一次真正依赖它的位置。
static void scheduleAsyncCopies(func::FuncOp kernel) {
  bool logical = false, pooled = false;
  kernel.walk([&](Operation *op) {
    logical |= isa<ToThreadTileOp, FromThreadTileOp>(op) ||
        llvm::any_of(op->getOperandTypes(), [](Type t) { return isa<FragmentType>(t); }) ||
        llvm::any_of(op->getResultTypes(), [](Type t) { return isa<FragmentType>(t); });
    pooled |= op->hasAttr("frisk.shm_pool");
  });
  if (pooled)
    return; // Never invalidate an already materialized storage-reuse plan.
  // 逻辑阶段只调度目的 LDS 可被生命周期分析完全理解（reusable）的 DMA。
  llvm::DenseSet<Value> understoodDestinations;
  if (logical) {
    auto liveness = analyzeShmLiveness(kernel);
    for (const auto &buffer : liveness.buffers)
      if (buffer.reusable)
        understoodDestinations.insert(buffer.aliases.begin(), buffer.aliases.end());
  }
  AliasAnalysis aliases(kernel);
  DirectDMAWaitMotion directWaitMotion(aliases);
  // op 是否与这条 DMA 无关，即 DMA / wait 可以越过它：
  // 遇到 barrier、wait、其它 DMA、to/from_thread_tile、手动调度就算相关；
  // 读写目的地算相关；写源算相关；副作用未知算相关。
  auto independent = [&](Operation *op, AsyncCopyGmToShmOp issue) {
    return !op->walk([&](Operation *nested) {
      if (isa<gpu::BarrierOp, SyncThreadsInBlockOp, WaitBufferReadyOp,
              AsyncCopyGmToShmOp, ToThreadTileOp, FromThreadTileOp>(nested) ||
          hasExplicitSchedule(nested)) return WalkResult::interrupt();
      if (isa<affine::AffineForOp, scf::ForOp, scf::IfOp>(nested))
        return WalkResult::advance(); // Inspect their bodies, not just operands.
      auto effects = dyn_cast<MemoryEffectOpInterface>(nested);
      if (!effects) return WalkResult::interrupt();
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (const auto &effect : instances) {
        Value value = effect.getValue();
        if (!value || !isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect()))
          return WalkResult::interrupt();
        if (!aliases.alias(value, issue.getDst()).isNo() ||
            (isa<MemoryEffects::Write>(effect.getEffect()) &&
             !aliases.alias(value, issue.getSrc()).isNo()))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }).wasInterrupted();
  };
  SmallVector<AsyncCopyGmToShmOp> issues;
  kernel.walk([&](AsyncCopyGmToShmOp op) { issues.push_back(op); });
  for (auto issue : issues) {
    if (hasExplicitSchedule(issue) ||
        (logical && (!understoodDestinations.contains(issue.getDst()) ||
                     !isInBlockUniformControlFlow(issue))))
      continue;
    // Do not reschedule manually staged copies before their slot selection or
    // across iterations. Descriptor dependencies bound the motion naturally.
    // setup：DMA 的 src/dst 描述符计算（纯计算或静态 LDS alloc），要和 DMA 一起上提。
    // anchors：描述符依赖中不能移动的 op，上提不能越过它。
    llvm::SetVector<Operation *> setup;
    llvm::SmallPtrSet<Operation *, 16> anchors;
    std::function<void(Value)> collect = [&](Value value) {
      Operation *def = value.getDefiningOp();
      if (!def || def->getBlock() != issue->getBlock() || setup.contains(def) || anchors.contains(def))
        return;
      auto allocation = dyn_cast<memref::AllocOp>(def);
      bool staticShared = allocation && allocation.getType().hasStaticShape() &&
          allocation.getType().getMemorySpaceAsInt() == 3 && allocation->getNumOperands() == 0;
      if (setup.size() >= 128 || def->getNumRegions() ||
          isa<ToThreadTileOp, FromThreadTileOp>(def) || hasExplicitSchedule(def) ||
          (!isPure(def) && !staticShared)) {
        anchors.insert(def);
        return;
      }
      for (Value operand : def->getOperands()) collect(operand);
      setup.insert(def);
    };
    collect(issue.getSrc());
    collect(issue.getDst());
    // 第 1 步：向上扫描，first 是 DMA 能到达的最早位置。
    Operation *first = issue;
    for (Operation *prev = issue->getPrevNode(); prev; prev = prev->getPrevNode()) {
      if (anchors.contains(prev)) break;
      if (!setup.contains(prev) && !independent(prev, issue)) break;
      first = prev;
    }
    while (first != issue.getOperation() && setup.contains(first))
      first = first->getNextNode();
    if (first != issue.getOperation()) {
      // Preserve the original topological order of the descriptor slice.
      for (Operation *op : setup)
        if (op != first && first->isBeforeInBlock(op)) op->moveBefore(first);
      issue->moveBefore(first);
    }
    // Keep completion as late as possible, immediately before the first access
    // that may observe/write the destination or mutate the source. A barrier
    // remains a hard boundary even if it does not name the copied buffer.
    // 第 2 步：找到同块内 token 相同的 wait，并逐条下沉。
    auto wait = dyn_cast_or_null<WaitBufferReadyOp>(issue->getNextNode());
    // Hoisting normally leaves wait at its original location; find its exact
    // same-block token match, stopping at a second issue into that tile.
    if (!wait || wait.getToken() != issue.getToken()) {
      wait = {};
      for (Operation *next = issue->getNextNode(); next; next = next->getNextNode()) {
        if (auto ready = dyn_cast<WaitBufferReadyOp>(next);
            ready && ready.getToken() == issue.getToken()) { wait = ready; break; }
        if (auto other = dyn_cast<AsyncCopyGmToShmOp>(next);
            other && other.getDst() == issue.getDst()) break;
      }
    }
    if (wait && !hasExplicitSchedule(wait))
      while (Operation *next = wait->getNextNode()) {
        if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(next)) {
          if (logical || !directWaitMotion.canCross(wait, issue, copy))
            break;
        } else if (next->hasTrait<OpTrait::IsTerminator>() ||
                   !independent(next, issue)) {
          break;
        }
        wait->moveAfter(next);
      }
  }
}

class ScheduleAsyncCopies : public PassWrapper<ScheduleAsyncCopies, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleAsyncCopies)
  StringRef getArgument() const final { return "frisk-schedule-async-copies"; }
  StringRef getDescription() const final { return "Schedule tile DMA against logical or physical storage dependencies before reuse"; }
  void runOnOperation() final { scheduleAsyncCopies(getOperation()); }
};

class AnalyzeShmLiveness : public PassWrapper<AnalyzeShmLiveness, OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeShmLiveness)
  StringRef getArgument() const final { return "frisk-analyze-shm-liveness"; }
  StringRef getDescription() const final { return "Report logical or physical SHM lifetimes without allocating storage"; }
  void runOnOperation() final {
    llvm::errs() << "SHM liveness @" << getOperation().getSymName() << '\n';
    dumpShmLiveness(analyzeShmLiveness(getOperation()), llvm::errs());
    markAllAnalysesPreserved();
  }
};

// 确定 LDS 物理布局。以下顺序不能随意调整：
//   packSharedMemory                       打包 LDS，确定每块 buffer 的形状。
//   synchronizeSharedMemoryLoopBackedges   补回边 barrier，保护旧读者。
//   scheduleAsyncCopies                    按物理信息再调度一次。
//   synchronizeSharedMemoryLoopBackedges   调度改变了访问位置，再检查并补一次。
//   reuseSharedMemory                      按最终生命周期分配物理地址，所以必须在调度之后。
//   loopUnrollFull                         最后展开带 frisk.loopUnrollFull 的循环。
class LowerFriskFragments : public impl::LowerFriskFragmentsBase<LowerFriskFragments> {
  void runOnOperation() override {
    if (failed(packSharedMemory(getOperation()))) return signalPassFailure();
    // Establish old-reader protection before considering moving readiness
    // barriers across direct launches. Recheck after the final schedule too.
    synchronizeSharedMemoryLoopBackedges(getOperation());
    scheduleAsyncCopies(getOperation());
    // The prefetch schedule is now fixed, and all physical reads are visible.
    // Allocate shared storage only against these final operation lifetimes.
    synchronizeSharedMemoryLoopBackedges(getOperation());
    reuseSharedMemory(getOperation());
    SmallVector<Loop> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](Loop loop) {
      auto full = loop->getAttrOfType<BoolAttr>("frisk.loopUnrollFull");
      if (full && full.getValue())
        loops.push_back(loop);
    });
    // 展开预算：累计最多新增 65536 个 op，且只展开 trip count <= 64 的循环。
    uint64_t remaining = 65536;
    for (auto loop : loops) {
      auto count = affine::getConstantTripCount(loop);
      if (!count || *count == 0 || *count > 64)
        continue;
      uint64_t size = 0;
      loop.getBody()->walk([&](Operation *) { ++size; });
      uint64_t cost = size * (*count - 1);
      if (cost > remaining)
        continue;
      if (failed(affine::loopUnrollFull(loop))) {
        loop.emitError("failed late fragment unrolling");
        signalPassFailure();
        return;
      }
      remaining -= cost;
    }
  }
};

class FriskFragIRReorder : public impl::FriskFragIRReorderBase<FriskFragIRReorder> {
  void runOnOperation() override {
    if (failed(normalizeTypedFragmentIR(getOperation()))) return signalPassFailure();
    // Restart after each mutation: nested loop handles and alias caches may
    // otherwise refer to erased operations. Each successful step removes a loop.
    // 融合后 IR 变了，所以每次成功都中断遍历，从头重建 AliasAnalysis 再来。
    // 每次成功都减少一个循环，因此必定终止。
    bool changed;
    do {
      if (failed(normalizeTypedFragmentIR(getOperation()))) return signalPassFailure();
      changed = false;
      AliasAnalysis aliases(getOperation());
      // a 往后找第一个循环 b，中间只能隔纯计算。
      getOperation().walk<WalkOrder::PreOrder>([&](Loop a) {
        for (Operation *next = a->getNextNode(); next; next = next->getNextNode()) {
          if (auto b = dyn_cast<Loop>(next)) {
            changed = tryFuse(a, b, aliases);
            break;
          }
          if (!isMemoryEffectFree(next) || !isSpeculatable(next) || next->getNumRegions())
            break;
        }
        return changed ? WalkResult::interrupt() : WalkResult::advance();
      });
    } while (changed);
    // 融合结束后再做一级 lookahead。
    SmallVector<Loop> loops;
    getOperation().walk<WalkOrder::PostOrder>([&](Loop loop) { loops.push_back(loop); });
    for (auto loop : loops)
      pipelineFragments(loop);
    if (failed(normalizeTypedFragmentIR(getOperation()))) return signalPassFailure();
  }
};
} // namespace
std::unique_ptr<Pass> createFriskFragIRReorderPass() {
  return std::make_unique<FriskFragIRReorder>();
}
std::unique_ptr<Pass> createLowerFriskFragmentsPass() {
  return std::make_unique<LowerFriskFragments>();
}
std::unique_ptr<Pass> createScheduleAsyncCopiesPass() {
  return std::make_unique<ScheduleAsyncCopies>();
}
std::unique_ptr<Pass> createAnalyzeShmLivenessPass() {
  return std::make_unique<AnalyzeShmLiveness>();
}
} // namespace mlir::frisk
