// ===========================================================================
// LDS（shared memory）生命周期分析与物理复用。优化汇总.py 第 12、15 项的实现主体。
//
// 提供 4 个入口：
//   analyzeShmLiveness                   分析每块 LDS alloc 的所有别名、访问和活跃区间（第 12 项）。
//   planShmReuse                         根据冲突关系决定每块 buffer 的字节偏移，只生成计划。
//   reuseSharedMemory                    执行计划：建一个大 pool，alloc 改成 view，补 barrier（第 15 项）。
//   synchronizeSharedMemoryLoopBackedges 给循环回边补 barrier，防止下一轮写覆盖本轮还没读完的旧数据。
//
// 核心概念：
//   活跃区间   一块 buffer 从第一次访问到最后一次访问，用函数入口块的 op 位置表示。
//   迭代局部   buffer 的内容不跨越某循环的回边（iterationLocalLoops）。两块 buffer 只有都在同一个
//              循环内迭代局部，才能在循环内部继续判断先后（findSeparation）。
//   集合循环   边界对所有线程一致的循环（isCollectiveLoop），只有这种循环才能安全地插 barrier。
//   交接       两块 buffer 共用同一段字节时，后者写之前前者的所有读者都要读完，需要 barrier。
// ===========================================================================
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <optional>
#include <utility>

namespace mlir::frisk {
namespace {

// poolAttr：标记复用后的大 pool alloc。分析遇到它会跳过，所以 reuseSharedMemory 重复调用是安全的。
constexpr StringLiteral poolAttr = "frisk.shm_pool";
constexpr uint64_t maxBytes = std::numeric_limits<int64_t>::max();

bool isShared(MemRefType type) {
  auto space = dyn_cast_or_null<IntegerAttr>(type.getMemorySpace());
  return space && space.getInt() == 3;
}

// 这三种 op lowering 后都带跨 wave 同步，都算 barrier。
bool isBarrier(Operation *op) {
  return isa<gpu::BarrierOp, SyncThreadsInBlockOp, WaitBufferReadyOp>(op);
}

// Iteration-local full-overwrite proofs still require completion in this block;
// the more general task-flow analysis below must not imply iteration locality.
bool hasLocalCompletion(AsyncCopyGmToShmOp issue) {
  for (Operation *next = issue->getNextNode(); next; next = next->getNextNode()) {
    if (auto wait = dyn_cast<WaitBufferReadyOp>(next);
        wait && wait.getToken() == issue.getToken()) return true;
    if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(next);
        copy && copy.getDst() == issue.getDst()) return false;
  }
  return false;
}

// Follow task identity forward, independently of destination descriptor aliases.
// Provenance alone is a may-analysis: finding one reachable wait does not prove
// completion on every execution. A loop carrier must be drained both on its
// next iteration and on exit (including the zero-trip path for an init).
// Solve these AND/OR obligations to a least fixed point: an unconsumed cycle
// cannot prove itself complete. FIFO pipelines with waits on the head and all
// exit slots are supported without treating different iterations as sequential.
// 沿 token 的数据流向前追，得到这条 DMA 可能存活到哪些 op，并证明它在所有路径上都被 wait 过。
//   accesses          token 的所有使用者，都算作目的 buffer 的访问。
//   acrossIterations  token 经过循环 carrier，说明 DMA 跨迭代。
// 返回 false 表示 token 逃逸或不能保证完成。
bool collectTaskLifetime(AsyncCopyGmToShmOp issue,
                         SmallVectorImpl<Operation *> &accesses,
                         bool &acrossIterations) {
  // work：待处理的 token SSA 值。
  // seen：已处理的值。
  // complete：已证明“完成”的值。
  // obligations[v]：v 完成的条件，是 OR-of-AND：任一组 targets 全部完成，v 就完成。
  // 例如 token 传给循环 init 时，要求 iter_arg 和循环结果两者都完成（下一轮和退出路径都要 wait）。
  SmallVector<Value> work{issue.getToken()};
  DenseSet<Value> seen, complete;
  DenseMap<Value, SmallVector<SmallVector<Value, 2>>> obligations;
  for (unsigned i = 0; i < work.size(); ++i) {
    Value value = work[i];
    if (!seen.insert(value).second)
      continue;
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      accesses.push_back(user);
      auto forward = [&](std::initializer_list<Value> targets) {
        // A use in a nested conditional/zero-trip loop may never execute.
        // Track its possible lifetime, but only unconditional forwarding can
        // discharge completion. Branch-local producers still forward through
        // their own yield, and loop inits explicitly cover the zero-trip path.
        if (user->getBlock() == value.getParentBlock())
          obligations[value].emplace_back(targets);
        work.append(targets.begin(), targets.end());
      };
      if (auto wait = dyn_cast<WaitBufferReadyOp>(user)) {
        // A conditional wait on an outer token is not a completion guarantee.
        if (wait->getBlock() == value.getParentBlock())
          complete.insert(value);
        continue;
      }
      if (auto loop = dyn_cast<scf::ForOp>(user)) {
        unsigned first = loop->getNumOperands() - loop.getInitArgs().size();
        if (use.getOperandNumber() < first)
          return false;
        unsigned n = use.getOperandNumber() - first;
        forward({loop.getRegionIterArgs()[n], loop.getResult(n)});
        acrossIterations = true;
        continue;
      }
      if (auto loop = dyn_cast<affine::AffineForOp>(user)) {
        unsigned first = loop->getNumOperands() - loop.getInits().size();
        if (use.getOperandNumber() < first)
          return false;
        unsigned n = use.getOperandNumber() - first;
        forward({loop.getRegionIterArgs()[n], loop.getResult(n)});
        acrossIterations = true;
        continue;
      }
      if (isa<scf::YieldOp, affine::AffineYieldOp>(user)) {
        Operation *parent = user->getParentOp();
        unsigned n = use.getOperandNumber();
        if (auto loop = dyn_cast<scf::ForOp>(parent)) {
          forward({loop.getRegionIterArgs()[n], loop.getResult(n)});
          acrossIterations = true;
        } else if (auto loop = dyn_cast<affine::AffineForOp>(parent)) {
          forward({loop.getRegionIterArgs()[n], loop.getResult(n)});
          acrossIterations = true;
        } else if (isa<scf::IfOp, affine::AffineIfOp>(parent)) {
          // Whenever this branch's issue executes, this yield forwards it.
          forward({parent->getResult(n)});
        } else {
          return false;
        }
        continue;
      }
      if (auto select = dyn_cast<arith::SelectOp>(user)) {
        // Track either arm for lifetime, but a select can discard a pending
        // task. It proves completion only if both arms are this same task.
        work.push_back(select.getResult());
        if (select.getTrueValue() == value && select.getFalseValue() == value)
          forward({select.getResult()});
        continue;
      }
      return false; // Task escapes or has an unknown consumer.
    }
  }
  // 最小不动点求解。没有 wait 消费的环不能自证完成。
  bool changed;
  do {
    changed = false;
    for (auto &entry : obligations) {
      if (complete.contains(entry.first))
        continue;
      if (llvm::any_of(entry.second, [&](const auto &targets) {
            return llvm::all_of(targets, [&](Value v) { return complete.contains(v); });
          })) {
        complete.insert(entry.first);
        changed = true;
      }
    }
  } while (changed);
  return complete.contains(issue.getToken());
}

// These regions finish synchronously. async.execute and parallel regions must
// not be treated as sequential operations when computing storage lifetimes.
// 返回 op 在 entry 块顶层的祖先。路径上只允许同步控制流；遇到其它 region 返回 nullptr。
Operation *getTopLevelUser(Operation *op, Block *entry) {
  while (op->getBlock() != entry) {
    op = op->getParentOp();
    if (!op || !isa<scf::ForOp, scf::IfOp, affine::AffineForOp,
                    affine::AffineIfOp>(op))
      return nullptr;
  }
  return op;
}

bool isSynchronousAccess(Operation *op) {
  // MemoryEffectOpInterface alone is insufficient: async copies also report
  // reads/writes, but their last operand use is not their completion point.
  // A lowered matrix read completes its shared-memory access at this op and
  // returns registers. It does not make the source escape or stay live while
  // those registers are consumed. Keep block-form reads conservative until
  // thread tiling has made their accesses explicit.
  if (auto read = dyn_cast<ReadMatrixFromShmOp>(op))
    return !read.isBlockForm();
  // Fragment loads finish the SHM read here; uses of the register result do
  // not keep its source storage alive. Fragment stores write their destination.
  return isa<LoadFragOp, StoreFragOp,
             memref::LoadOp, memref::StoreOp, memref::CopyOp,
             memref::DimOp, memref::AssumeAlignmentOp,
             affine::AffineLoadOp, affine::AffineStoreOp,
             affine::AffineVectorLoadOp, affine::AffineVectorStoreOp,
             vector::LoadOp, vector::StoreOp, vector::TransferReadOp,
             vector::TransferWriteOp, vector::MaskedLoadOp,
             vector::MaskedStoreOp>(op);
}

// 遍历一块 alloc 的所有使用，填写 info。
// worklist 沿视图、cast、select、循环 carrier、yield 扩展别名集合；
// 真实访问通过 recordUse 更新 firstUse / lastUse。
// 遇到无法理解的使用就写 skipReason 并返回，此时 reusable 保持 false，这块 buffer 不参与复用。
void analyzeUses(ShmLiveInterval &info, Block *entry,
                 const DenseMap<Operation *, unsigned> &positions) {
  SmallVector<Value> worklist{info.allocation.getResult()};
  DenseSet<Value> visited;
  bool hasUse = false;
  info.firstUse = std::numeric_limits<unsigned>::max();
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    info.aliases.push_back(value);
    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      Operation *top = getTopLevelUser(user, entry);
      if (!top) {
        info.skipReason = "use inside unsupported/async control flow";
        return;
      }
      auto recordUse = [&] {
        info.accesses.push_back(user);
        unsigned position = positions.lookup(top);
        info.firstUse = std::min(info.firstUse, position);
        info.lastUse = std::max(info.lastUse, position);
        hasUse = true;
      };
      if (auto view = dyn_cast<BufferViewOp>(user)) {
        if (view.getSource() == value) {
          worklist.push_back(view.getResult());
          continue;
        }
      }
      if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
        if (view.getViewSource() == value && user->getNumRegions() == 0 &&
            isMemoryEffectFree(user)) {
          for (Value result : user->getResults())
            if (isa<BaseMemRefType>(result.getType()))
              worklist.push_back(result);
          continue; // Constructing a descriptor does not touch storage.
        }
      }
      if (isa<memref::CastOp, arith::SelectOp>(user)) {
        worklist.push_back(user->getResult(0));
        continue;
      }
      // Loop inits also reach the result along the zero-trip path.
      if (auto loop = dyn_cast<scf::ForOp>(user)) {
        for (auto init : llvm::enumerate(loop.getInitArgs()))
          if (init.value() == value) {
            worklist.push_back(loop.getRegionIterArgs()[init.index()]);
            worklist.push_back(loop.getResult(init.index()));
          }
        recordUse();
        continue;
      }
      if (auto loop = dyn_cast<affine::AffineForOp>(user)) {
        for (auto init : llvm::enumerate(loop.getInits()))
          if (init.value() == value) {
            worklist.push_back(loop.getRegionIterArgs()[init.index()]);
            worklist.push_back(loop.getResult(init.index()));
          }
        recordUse();
        continue;
      }
      if (isa<scf::YieldOp, affine::AffineYieldOp>(user)) {
        Operation *parent = user->getParentOp();
        unsigned index = use.getOperandNumber();
        if (isa<scf::IfOp, affine::AffineIfOp, scf::ForOp,
                affine::AffineForOp>(parent)) {
          worklist.push_back(parent->getResult(index));
          if (auto loop = dyn_cast<scf::ForOp>(parent))
            worklist.push_back(loop.getRegionIterArgs()[index]);
          if (auto loop = dyn_cast<affine::AffineForOp>(parent))
            worklist.push_back(loop.getRegionIterArgs()[index]);
          recordUse(); // Covers loop backedges and branch-selected aliases.
          continue;
        }
      }
      if (!isSynchronousAccess(user)) {
        // DMA 的“使用”不止发起点：buffer 要一直活到 wait 才算结束，所以把 token 的使用者也计入。
        if (auto issue = dyn_cast<AsyncCopyGmToShmOp>(user);
            issue && use.getOperandNumber() == 1) {
          recordUse();
          SmallVector<Operation *> taskAccesses;
          if (!collectTaskLifetime(issue, taskAccesses, info.asyncAcrossIterations)) {
            info.skipReason = "async task has no guaranteed completion or escapes";
            return;
          }
          for (Operation *access : taskAccesses) {
            Operation *completion = getTopLevelUser(access, entry);
            if (!completion) { info.skipReason = "unsupported task completion scope"; return; }
            info.accesses.push_back(access);
            info.firstUse = std::min(info.firstUse, positions.lookup(completion));
            info.lastUse = std::max(info.lastUse, positions.lookup(completion));
          }
          continue;
        }
        info.skipReason = "unsupported or escaping use: " +
                          user->getName().getStringRef().str();
        return;
      }
      if (auto assume = dyn_cast<memref::AssumeAlignmentOp>(user))
        info.alignment = std::max<uint64_t>(info.alignment, assume.getAlignment());
      recordUse();
    }
  }
  if (!hasUse) {
    info.skipReason = "no storage uses";
    return;
  }
  info.reusable = true;
}

bool overlaps(uint64_t a, uint64_t sizeA, uint64_t b, uint64_t sizeB) {
  return a < b + sizeB && b < a + sizeA;
}

Block *getLoopBody(Operation *op) {
  if (auto loop = dyn_cast<scf::ForOp>(op))
    return loop.getBody();
  if (auto loop = dyn_cast<affine::AffineForOp>(op))
    return loop.getBody();
  return nullptr;
}

// isCollectiveLoop 和 isBlockUniform 互相递归，这里先声明。
bool isCollectiveLoop(Operation *op, Block *entry);

// Only scalar kernel arguments, block-wide GPU queries and pure arithmetic
// derived from them are uniform. Loads, thread IDs and loop-carried values are
// deliberately not inferred uniform, even if a loop already contains a barrier.
// 判断 value 是否对同一 block 内所有线程都相同。
bool isBlockUniform(Value value, Block *entry) {
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() == entry)
      return isa<IndexType, IntegerType, FloatType>(value.getType());
    auto *parent = arg.getOwner()->getParentOp();
    return getLoopBody(parent) == arg.getOwner() && arg.getArgNumber() == 0 &&
           isCollectiveLoop(parent, entry);
  }
  Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  if (op->hasTrait<OpTrait::ConstantLike>() ||
      isa<gpu::BlockIdOp, gpu::BlockDimOp, gpu::GridDimOp>(op))
    return true;
  if (op->getNumRegions() || !isMemoryEffectFree(op) ||
      !(isa<affine::AffineApplyOp>(op) ||
        op->getName().getDialectNamespace() == "arith"))
    return false;
  return llvm::all_of(op->getOperands(),
                      [&](Value operand) { return isBlockUniform(operand, entry); });
}

// 循环边界 uniform，且所有外层也是集合循环。只有这样所有线程才执行相同次数的 barrier。
bool isCollectiveLoop(Operation *op, Block *entry) {
  if (!getLoopBody(op))
    return false;
  if (op->getBlock() != entry &&
      !isCollectiveLoop(op->getParentOp(), entry))
    return false; // Never enter a conditional or unsupported region.
  if (auto loop = dyn_cast<scf::ForOp>(op))
    return isBlockUniform(loop.getLowerBound(), entry) &&
           isBlockUniform(loop.getUpperBound(), entry) &&
           isBlockUniform(loop.getStep(), entry);
  auto loop = cast<affine::AffineForOp>(op);
  return llvm::all_of(loop.getLowerBoundOperands(),
                      [&](Value v) { return isBlockUniform(v, entry); }) &&
         llvm::all_of(loop.getUpperBoundOperands(),
                      [&](Value v) { return isBlockUniform(v, entry); });
}

// Evaluate the common affine thread-to-storage mapping without changing IR.
// Unknown indices conservatively disable the full-overwrite proof.
// 把索引表达式在给定线程号下求值，用于 fullyOverwrites。
std::optional<int64_t> evaluateIndex(Value value, int64_t thread) {
  APInt constant;
  if (matchPattern(value, m_ConstantInt(&constant)))
    return constant.getSExtValue();
  if (auto id = value.getDefiningOp<gpu::ThreadIdOp>()) {
    if (id.getDimension() == gpu::Dimension::x)
      return thread;
    return std::nullopt;
  }
  if (auto apply = value.getDefiningOp<affine::AffineApplyOp>()) {
    SmallVector<Attribute> operands;
    for (Value operand : apply.getMapOperands()) {
      auto index = evaluateIndex(operand, thread);
      if (!index)
        return std::nullopt;
      operands.push_back(IntegerAttr::get(IndexType::get(value.getContext()), *index));
    }
    SmallVector<Attribute> results;
    if (succeeded(apply.getAffineMap().constantFold(operands, results)))
      return cast<IntegerAttr>(results.front()).getInt();
  }
  return std::nullopt;
}

// An allocation outside the loop is not necessarily iteration-local: partial
// stores, a read before the write, or a rotated pipeline buffer can carry data
// across the backedge. Require the first storage access to overwrite the whole
// allocation on every iteration. For tiled vector stores, check the actual
// address mapping across all threads, rather than trusting the 'tiled' marker.
// 证明 op 是这块 buffer 的“每轮第一次访问”且写满全部元素。
// 满足时，上一轮的内容不会流到下一轮，alloc 在循环外也算迭代局部。
bool fullyOverwrites(Operation *op, const ShmLiveInterval &info,
                     func::FuncOp kernel) {
  auto allocation = info.allocation;
  Value buffer = allocation.getResult();
  if (auto copy = dyn_cast<AsyncCopyGmToShmOp>(op))
    return copy.getDst() == buffer && hasLocalCompletion(copy) &&
        cast<MemRefType>(copy.getSrc().getType()).getNumElements() ==
            cast<MemRefType>(buffer.getType()).getNumElements();
  if (auto copy = dyn_cast<memref::CopyOp>(op))
    return copy.getTarget() == buffer && copy.getSource() != buffer;
  ValueRange indices;
  int64_t width = 1;
  if (auto store = dyn_cast<vector::StoreOp>(op)) {
    auto vectorType = store.getVectorType();
    if (store.getBase() != buffer || vectorType.getRank() != 1 ||
        vectorType.isScalable())
      return false;
    indices = store.getIndices();
    width = vectorType.getDimSize(0);
  } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
    if (store.getMemref() != buffer)
      return false;
    indices = store.getIndices();
  } else {
    return false;
  }
  auto type = allocation.getType();
  int64_t elements = type.getNumElements();
  auto threads = kernel->getAttrOfType<IntegerAttr>("thread_num");
  // Bound compile-time work; larger or unknown footprints stay conservative.
  if (!threads || threads.getInt() <= 0 || threads.getInt() > 1024 ||
      elements > (1 << 20) || type.getRank() == 0)
    return false;
  // written：枚举所有线程的写入位置，必须覆盖全部元素。
  llvm::SmallBitVector written(elements);
  for (int64_t thread = 0; thread < threads.getInt(); ++thread) {
    int64_t offset = 0;
    for (auto [dim, index] : llvm::enumerate(indices)) {
      auto coordinate = evaluateIndex(index, thread);
      if (!coordinate || *coordinate < 0 || *coordinate >= type.getDimSize(dim))
        return false;
      if (dim + 1 == indices.size() && width > type.getDimSize(dim) - *coordinate)
        return false;
      offset = offset * type.getDimSize(dim) + *coordinate;
    }
    written.set(offset, offset + width);
  }
  return written.all();
}

// 从内到外找满足条件的集合循环：所有访问都在循环内，且 alloc 在循环内或第一次访问写满 buffer。
// 有多个别名或 DMA 跨迭代的 buffer 不处理，退回到整循环生命周期。
void findIterationLocalLoops(ShmLiveInterval &info, func::FuncOp kernel) {
  if (!info.reusable || info.aliases.size() != 1 || info.asyncAcrossIterations)
    return; // Alias/iter_arg cases retain the existing whole-loop lifetime.
  Block *entry = &kernel.getBody().front();
  for (Operation *loop = info.accesses.front()->getParentOp();
       loop && loop != kernel.getOperation(); loop = loop->getParentOp()) {
    Block *body = getLoopBody(loop);
    if (!body || !isCollectiveLoop(loop, entry))
      continue;
    Operation *first = nullptr;
    bool allInside = llvm::all_of(info.accesses, [&](Operation *access) {
      if (!loop->isProperAncestor(access))
        return false;
      auto *top = getTopLevelUser(access, body);
      if (!top)
        return false;
      if (!first || top->isBeforeInBlock(first))
        first = top;
      return true;
    });
    if (allInside &&
        (loop->isProperAncestor(info.allocation) ||
         fullyOverwrites(first, info, kernel)))
      info.iterationLocalLoops.push_back(loop);
  }
}

// 两块 buffer 在某个块内的先后关系，证明它们不冲突。
//   block        判断发生在哪个块（入口块或某循环体）。
//   order        该块内的 op 顺序。
//   earlierFirst/earlierLast  先用的 buffer 在 order 中的首尾位置。
//   laterFirst/laterLast      后用的 buffer 的首尾位置。
struct Separation {
  Block *block;
  SmallVector<Operation *> order;
  unsigned earlierFirst, earlierLast, laterFirst, laterLast;
};

// Descend only when BOTH buffers are confined to one collective loop and their
// contents do not survive its backedge. Other regions remain atomic; in
// particular, barriers will never be inserted inside a divergent conditional.
// 在 block 内比较两块 buffer 的区间。不相交就返回；如果都被同一个循环包住且都是迭代局部，就进入循环体
// 递归比较。返回空表示冲突，不能共用字节。
std::optional<Separation> findSeparation(const ShmLiveInterval &a,
                                       const ShmLiveInterval &b, Block *block) {
  Separation result;
  result.block = block;
  DenseMap<Operation *, unsigned> positions;
  for (Operation &op : *block) {
    positions[&op] = result.order.size();
    result.order.push_back(&op);
  }
  auto range = [&](const ShmLiveInterval &info) {
    unsigned first = std::numeric_limits<unsigned>::max(), last = 0;
    for (Operation *access : info.accesses) {
      auto *top = getTopLevelUser(access, block);
      assert(top && "access must be inside the common scope");
      unsigned pos = positions.lookup(top);
      first = std::min(first, pos);
      last = std::max(last, pos);
    }
    return std::make_pair(first, last);
  };
  auto [af, al] = range(a);
  auto [bf, bl] = range(b);
  if (bl < af) {
    std::swap(af, bf);
    std::swap(al, bl);
  }
  if (al < bf) {
    result.earlierFirst = af;
    result.earlierLast = al;
    result.laterFirst = bf;
    result.laterLast = bl;
    return result;
  }
  if (af == al && af == bf && bf == bl) {
    Operation *loop = result.order[af];
    if (llvm::is_contained(a.iterationLocalLoops, loop) &&
        llvm::is_contained(b.iterationLocalLoops, loop))
      return findSeparation(a, b, getLoopBody(loop));
  }
  return std::nullopt;
}

} // namespace

bool ShmLivenessResult::interferes(unsigned lhs, unsigned rhs) const {
  const auto &a = buffers[lhs];
  const auto &b = buffers[rhs];
  return lhs == rhs || !a.reusable || !b.reusable ||
         !findSeparation(a, b, programOrder.front()->getBlock());
}

// 只处理单块函数中的静态形状、identity 布局、整数/浮点元素的 LDS alloc。
ShmLivenessResult analyzeShmLiveness(func::FuncOp kernel) {
  ShmLivenessResult result;
  if (kernel.isExternal())
    return result;
  Block *entry = &kernel.getBody().front();
  DenseMap<Operation *, unsigned> positions;
  for (Operation &op : *entry) {
    positions[&op] = result.programOrder.size();
    result.programOrder.push_back(&op);
  }
  kernel.walk([&](memref::AllocOp alloc) {
    auto type = alloc.getType();
    if (!isShared(type))
      return;
    result.buffers.emplace_back();
    auto &info = result.buffers.back();
    info.allocation = alloc;
    auto skip = [&](StringRef reason) { info.skipReason = reason.str(); };
    if (!llvm::hasSingleElement(kernel.getBody()))
      return skip("multi-block function");
    if (alloc->hasAttr(poolAttr))
      return skip("existing shared-memory pool");
    if (!getTopLevelUser(alloc, entry))
      return skip("allocation inside unsupported control flow");
    if (!type.hasStaticShape() || !type.getLayout().isIdentity())
      return skip("dynamic shape or non-identity layout");
    Type element = type.getElementType();
    if (!isa<IntegerType, FloatType>(element))
      return skip("unsupported element storage size");
    unsigned bits = element.getIntOrFloatBitWidth();
    if (bits < 8 || bits % 8 || !llvm::isPowerOf2_64(bits / 8))
      return skip("unsupported element storage size");
    uint64_t size = bits / 8;
    for (int64_t dim : type.getShape()) {
      if (dim <= 0 || uint64_t(dim) > maxBytes / size)
        return skip("empty or overflowing allocation");
      size *= dim;
    }
    info.sizeBytes = size;
    info.alignment = std::max<uint64_t>(bits / 8, alloc.getAlignment().value_or(1));
    analyzeUses(info, entry, positions);
    findIterationLocalLoops(info, kernel);
  });
  return result;
}

// 贪心分配：按大小降序，每块放到与所有冲突 buffer 不重叠的最低对齐地址（first fit）。
// 确定性算法，大小相同时保持稳定顺序。
ShmReusePlan planShmReuse(const ShmLivenessResult &liveness) {
  ShmReusePlan plan;
  SmallVector<unsigned> order;
  for (auto indexed : llvm::enumerate(liveness.buffers)) {
    const auto &info = indexed.value();
    if (!info.reusable)
      continue;
    if (info.sizeBytes > maxBytes - plan.originalBytes)
      return {};
    plan.originalBytes += info.sizeBytes;
    plan.alignment = std::max(plan.alignment, info.alignment);
    order.push_back(indexed.index());
  }
  llvm::stable_sort(order, [&](unsigned a, unsigned b) {
    return liveness.buffers[a].sizeBytes > liveness.buffers[b].sizeBytes;
  });
  for (unsigned index : order) {
    const auto &info = liveness.buffers[index];
    uint64_t offset = 0;
    // Move past conflicts until the lowest aligned free byte range is found.
    bool conflict;
    do {
      conflict = false;
      for (const auto &placed : plan.placements) {
        const auto &other = liveness.buffers[placed.bufferIndex];
        if (!liveness.interferes(index, placed.bufferIndex) ||
            !overlaps(offset, info.sizeBytes, placed.offsetBytes, other.sizeBytes))
          continue;
        uint64_t end = placed.offsetBytes + other.sizeBytes;
        if (info.alignment > maxBytes - end)
          return {};
        offset = llvm::alignTo(end, info.alignment);
        if (info.sizeBytes > maxBytes - offset)
          return {};
        conflict = true;
        break;
      }
    } while (conflict);
    plan.placements.push_back({index, offset});
    plan.pooledBytes = std::max(plan.pooledBytes, offset + info.sizeBytes);
  }
  return plan;
}

void dumpShmLiveness(const ShmLivenessResult &liveness, llvm::raw_ostream &os) {
  for (auto indexed : llvm::enumerate(liveness.buffers)) {
    const auto &info = indexed.value();
    os << "shm[" << indexed.index() << "] " << info.sizeBytes
       << " bytes, align " << info.alignment;
    if (info.reusable)
      os << ", live [" << info.firstUse << ", " << info.lastUse << "]";
    else
      os << ", skipped: " << info.skipReason;
    if (info.asyncAcrossIterations)
      os << ", async across iterations";
    os << '\n';
  }
}

bool isInBlockUniformControlFlow(Operation *op) {
  auto kernel = op->getParentOfType<func::FuncOp>();
  if (!kernel || !llvm::hasSingleElement(kernel.getBody()))
    return false;
  Block *entry = &kernel.getBody().front();
  return op->getBlock() == entry || isCollectiveLoop(op->getParentOp(), entry);
}

// 循环回边保护：防止快的 wave 进入下一轮后，覆盖慢的 wave 还没读完的数据。
// 后序遍历，内层循环自己补 barrier。
// 规则：
//   循环含 DMA：循环尾必须有 barrier（wait 只发布新数据，不等旧读者）。
//   无 DMA：如果 buffer 在循环内既写又读，prefix 或 tail 要有 barrier。
// 第 33 项 elideBackedgeBarrier 删除的就是这里插入的循环尾 barrier。
unsigned synchronizeSharedMemoryLoopBackedges(func::FuncOp kernel) {
  if (kernel.isExternal() || !kernel->hasAttr("thread_num") ||
      !llvm::hasSingleElement(kernel.getBody()))
    return 0;
  auto liveness = analyzeShmLiveness(kernel);
  Block *entry = &kernel.getBody().front();
  unsigned inserted = 0;
  // Inner loops must synchronize their own backedges: an enclosing loop's
  // barrier cannot stop a faster wave overwriting Q during the dk reduction.
  kernel.walk<WalkOrder::PostOrder>([&](Operation *loop) {
    Block *body = getLoopBody(loop);
    if (!body || !isCollectiveLoop(loop, entry))
      return;
    // Also protect async buffers that could not prove a local completion and
    // therefore do not participate in reuse (e.g. rotated pipeline slots).
    // Readiness waits publish new data; they do not release old readers.
    bool asyncCopy = false;
    loop->walk([&](frisk::AsyncCopyGmToShmOp) { asyncCopy = true; });
    if (asyncCopy) {
      Operation *tail = body->getTerminator()->getPrevNode();
      if (!tail || !isBarrier(tail)) {
        OpBuilder builder(body->getTerminator());
        builder.create<gpu::BarrierOp>(loop->getLoc());
        ++inserted;
      }
      return;
    }
    for (const auto &info : liveness.buffers) {
      if (!info.reusable)
        continue;
      // firstWrite / lastRead：这块 buffer 在循环体内最早的写和最晚的读。
      // barrier 只有在 firstWrite 之前或 lastRead 之后，才能保护跨回边的交接。
      Operation *firstWrite = nullptr, *lastRead = nullptr;
      for (Operation *access : info.accesses) {
        if (!loop->isProperAncestor(access))
          continue;
        auto effects = dyn_cast<MemoryEffectOpInterface>(access);
        Operation *top = getTopLevelUser(access, body);
        if (!effects || !top)
          continue;
        SmallVector<MemoryEffects::EffectInstance> instances;
        effects.getEffects(instances);
        for (const auto &effect : instances) {
          if (!llvm::is_contained(info.aliases, effect.getValue()))
            continue;
          if (isa<MemoryEffects::Write>(effect.getEffect()) &&
              (!firstWrite || top->isBeforeInBlock(firstWrite)))
            firstWrite = top;
          if (isa<MemoryEffects::Read>(effect.getEffect()) &&
              (!lastRead || lastRead->isBeforeInBlock(top)))
            lastRead = top;
        }
      }
      if (!firstWrite || !lastRead)
        continue;
      // Only a barrier in the tail or prefix protects this wrapped handoff.
      // The publication barrier AFTER the write does not protect old readers.
      bool protectedBackedge = false;
      for (Operation &op : *body)
        if (isBarrier(&op) &&
            (op.isBeforeInBlock(firstWrite) || lastRead->isBeforeInBlock(&op)))
          protectedBackedge = true;
      if (!protectedBackedge) {
        OpBuilder builder(body->getTerminator());
        builder.create<gpu::BarrierOp>(loop->getLoc());
        ++inserted;
        break; // The tail barrier protects every buffer in this loop.
      }
    }
  });
  return inserted;
}

// 真正落实 LDS 复用。分三步：算需要的 barrier；建 pool；alloc 替换成 pool 上的 view。
// 还有 fragment 或 to/from_thread_tile 时直接返回，物理访问尚未确定。
// 只有总字节数确实减少时才改 IR。
bool reuseSharedMemory(func::FuncOp kernel, ShmReuseStats *stats) {
  if (stats)
    *stats = {};
  if (!kernel->hasAttr("thread_num"))
    return false;
  bool hasDeferredStorage = false;
  kernel.walk([&](Operation *op) {
    hasDeferredStorage |= isa<ToThreadTileOp, FromThreadTileOp>(op) ||
        llvm::any_of(op->getOperandTypes(), [](Type t) { return isa<FragmentType>(t); }) ||
        llvm::any_of(op->getResultTypes(), [](Type t) { return isa<FragmentType>(t); });
  });
  if (hasDeferredStorage)
    return false;
  auto liveness = analyzeShmLiveness(kernel);
  auto plan = planShmReuse(liveness);
  if (stats) {
    stats->originalBytes = plan.originalBytes;
    stats->pooledBytes = plan.originalBytes;
  }
  if (plan.placements.size() < 2 || plan.pooledBytes >= plan.originalBytes)
    return false;

  // Operand liveness does not guarantee every warp has finished reading.
  // 每个作用域块的交接需求。
  //   order  块内 op 顺序。
  //   gaps   (deadline, lastUse)：barrier 必须放在 (lastUse, deadline] 之间，也就是 order[deadline] 之前。
  // scopes / scopeIndices：按作用域块归类。
  struct ScopeHandoffs {
    SmallVector<Operation *> order;
    SmallVector<std::pair<unsigned, unsigned>> gaps;
  };
  SmallVector<ScopeHandoffs> scopes;
  DenseMap<Block *, unsigned> scopeIndices;
  // 遍历所有字节重叠的 buffer 对，已有 barrier 的跳过，否则记录 gap。
  for (auto [i, a] : llvm::enumerate(plan.placements)) {
    const auto &lhs = liveness.buffers[a.bufferIndex];
    for (auto b : ArrayRef(plan.placements).drop_front(i + 1)) {
      const auto &rhs = liveness.buffers[b.bufferIndex];
      if (!overlaps(a.offsetBytes, lhs.sizeBytes, b.offsetBytes, rhs.sizeBytes))
        continue;
      auto separation = findSeparation(lhs, rhs, &kernel.getBody().front());
      assert(separation && "overlapping placements must have disjoint lifetimes");
      auto [it, inserted] = scopeIndices.try_emplace(separation->block, scopes.size());
      if (inserted)
        scopes.push_back({separation->order, {}});
      auto &scope = scopes[it->second];
      auto hasBarrier = [&](unsigned begin, unsigned end) {
        return llvm::any_of(ArrayRef(scope.order).slice(begin, end - begin),
                            isBarrier);
      };
      if (!hasBarrier(separation->earlierLast + 1, separation->laterFirst))
        scope.gaps.emplace_back(separation->laterFirst, separation->earlierLast);
      // The last user of this iteration must finish before the first writer of
      // the next one. A prefix barrier also covers the wrapped handoff; otherwise
      // require a barrier at the loop tail. Even a zero-trip path stays safe.
      // 循环内的两块 buffer 还要考虑跨回边的交接：后者的最后使用 -> 下一轮前者的第一次写。
      if (separation->block != &kernel.getBody().front() &&
          !hasBarrier(0, separation->earlierFirst) &&
          !hasBarrier(separation->laterLast + 1, scope.order.size()))
        scope.gaps.emplace_back(scope.order.size() - 1, separation->laterLast);
    }
  }
  // One barrier can satisfy several handoffs. Process the earliest deadline
  // first, and reuse a previously selected barrier whenever it covers the gap.
  // 经典区间选点：按 deadline 排序后贪心，一个 barrier 能满足多个交接。
  // barrierPositions：barrier 要插在这些 op 之前。
  SmallVector<Operation *> barrierPositions;
  for (auto &scope : scopes) {
    llvm::sort(scope.gaps);
    std::optional<unsigned> previous;
    for (auto [deadline, lastUse] : scope.gaps)
      if (!previous || *previous <= lastUse) {
        barrierPositions.push_back(scope.order[deadline]);
        previous = deadline;
      }
  }
  OpBuilder builder(kernel.getContext());
  for (Operation *before : barrierPositions) {
    builder.setInsertionPoint(before);
    builder.create<gpu::BarrierOp>(before->getLoc());
  }
  builder.setInsertionPointToStart(&kernel.getBody().front());
  // pool：memref<Nxi8, 3>，所有可复用 buffer 都变成它上面的 memref.view。
  auto poolType = MemRefType::get({int64_t(plan.pooledBytes)}, builder.getI8Type(),
                                MemRefLayoutAttrInterface{},
                                builder.getI64IntegerAttr(3));
  auto pool = builder.create<memref::AllocOp>(
      kernel.getLoc(), poolType, builder.getI64IntegerAttr(plan.alignment));
  pool->setAttr(poolAttr, builder.getUnitAttr());
  pool->setAttr("tiled", builder.getBoolAttr(true));
  for (auto placement : plan.placements) {
    auto alloc = liveness.buffers[placement.bufferIndex].allocation;
    builder.setInsertionPoint(alloc);
    Value offset = builder.create<arith::ConstantIndexOp>(
        alloc.getLoc(), placement.offsetBytes);
    auto view = builder.create<memref::ViewOp>(alloc.getLoc(), alloc.getType(),
                                              pool, offset, ValueRange{});
    alloc.getResult().replaceAllUsesWith(view.getResult());
    alloc.erase();
  }
  if (stats) {
    stats->pooledBytes = plan.pooledBytes;
    stats->reusedAllocations = plan.placements.size();
    stats->insertedBarriers = barrierPositions.size();
  }
  return true;
}

} // namespace mlir::frisk
