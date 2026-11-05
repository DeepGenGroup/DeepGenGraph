// ===========================================================================
// 本文件包含两个方向相反的 Pass：
//
// PromoteGlobalToSharedAsync（frisk-promote-gm-to-shm-async）   优化汇总.py 第 4 项，block 阶段
//   把 GM -> LDS 的同步 frisk.copy 改写成 async_copy_gm_to_shm + wait_buffer_ready 两步，
//   发起和完成拆开，后面的调度 Pass 才能分别移动它们。这是所有“提前发 DMA”优化的前提。
//   本 Pass 只拆分，wait 紧跟在 issue 后面，完成点不变。
//
// LowerFriskAsyncToSync（frisk-lower-async-to-sync）   copyMode=sync 时才运行，不是优化
//   反向操作：把异步拷贝退回同步 copy + block 同步，作为对照配置。
// ===========================================================================
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace {
// Preserve the original completion point. A pipeline scheme can subsequently
// put wait_buffer_ready in the consumer stage, independently of the issue.
struct PromoteGlobalToSharedAsyncPass
    : PassWrapper<PromoteGlobalToSharedAsyncPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteGlobalToSharedAsyncPass)
  PromoteGlobalToSharedAsyncPass() = default;
  PromoteGlobalToSharedAsyncPass(const PromoteGlobalToSharedAsyncPass &other)
      : PassWrapper(other) {}
  // assumeAlignment：调用方保证的对齐字节数，0 表示按元素类型推断。
  // 优先级（低到高）：元素大小 < 本选项 < copy 自身的 alignment 属性。
  Option<int64_t> assumeAlignment{*this, "assume-alignment",
      llvm::cl::desc("Caller guarantee for both tile origins in bytes; 0 infers element alignment"),
      llvm::cl::init(0)};
  StringRef getArgument() const final { return "frisk-promote-gm-to-shm-async"; }
  StringRef getDescription() const final {
    return "Represent global/shared copies with explicit buffer readiness";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<frisk::FriskDialect>();
  }
  void runOnOperation() final {
    if (assumeAlignment < 0 ||
        (assumeAlignment && !llvm::isPowerOf2_64(assumeAlignment))) {
      getOperation().emitError("assume-alignment must be zero or a positive power of two");
      return signalPassFailure();
    }
    // Schemes have one entry per top-level operation. Rebuild them when one
    // synchronous copy becomes issue+wait, retaining its original schedule.
    // schedule：op -> (pipeline stage, pipeline order)。
    // pipeline.stage / pipeline.order 属性是“循环体内每个顶层 op 一项”的数组。一条 copy 变成
    // issue + wait 后数组长度对不上，所以先拆成 map，改写完再重建。
    // loops：带有效流水调度标注的循环，末尾用来回写属性。
    DenseMap<Operation *, std::pair<int64_t, int64_t>> schedule;
    SmallVector<affine::AffineForOp> loops;
    getOperation().walk([&](affine::AffineForOp loop) {
      auto stages = loop->getAttrOfType<DenseI64ArrayAttr>("pipeline.stage");
      auto orders = loop->getAttrOfType<DenseI64ArrayAttr>("pipeline.order");
      if (!stages || !orders || stages.size() != orders.size() ||
          stages.size() != std::distance(loop.getBody()->begin(),
                                        std::prev(loop.getBody()->end())))
        return;
      loops.push_back(loop);
      unsigned i = 0;
      for (Operation &op : loop.getBody()->without_terminator()) {
        schedule[&op] = {stages[i], orders[i]};
        ++i;
      }
    });
    SmallVector<frisk::CopyOp> copies;
    getOperation().walk([&](frisk::CopyOp op) { copies.push_back(op); });
    // 只处理 GM(addrspace 1) -> LDS(addrspace 3)、静态形状、元素类型相同的拷贝。
    for (auto copy : copies) {
      auto src = dyn_cast<MemRefType>(copy.getSrc().getType());
      auto dst = dyn_cast<MemRefType>(copy.getDst().getType());
      if (!src || !dst || src.getMemorySpaceAsInt() != 1 ||
          dst.getMemorySpaceAsInt() != 3 || !src.hasStaticShape() ||
          !dst.hasStaticShape() || src.getRank() == 0 || dst.getRank() == 0 ||
          src.getElementType() != dst.getElementType())
        continue;
      OpBuilder b(copy);
      auto loc = copy.getLoc();
      Value source = copy.getSrc(), destination = copy.getDst();
      // same         源和目的形状相同，不需要切片。
      // sourceSlice  源比目的大，要在源上切片；否则如果形状不同，就是在目的上切片。
      // sentinel     offset map 是“单个常量 = rank”的特殊写法，表示没有偏移。
      // zero         offset map 全为 0，同样表示没有偏移。
      bool same = src.getShape() == dst.getShape();
      bool sourceSlice = !same && src.getNumElements() > dst.getNumElements();
      auto shape = sourceSlice ? dst.getShape() : src.getShape();
      AffineMap map = copy.getOffsetMap();
      bool sentinel = map.getNumInputs() == 0 && map.getNumResults() == 1 &&
          isa<AffineConstantExpr>(map.getResult(0)) &&
          cast<AffineConstantExpr>(map.getResult(0)).getValue() == dst.getRank();
      bool zero = llvm::all_of(map.getResults(), [](AffineExpr e) {
        auto c = dyn_cast<AffineConstantExpr>(e);
        return c && c.getValue() == 0;
      });
      // 有切片或偏移时，用 buffer_view 显式表达，异步 op 本身不再携带 offset map。
      if (!same || (!sentinel && !zero)) {
        auto mapped = sourceSlice ? src : dst;
        if (map.getNumResults() != mapped.getRank() ||
            map.getNumInputs() != copy.getMapOperands().size())
          continue;
        auto view = b.create<frisk::BufferViewOp>(loc,
            sourceSlice ? source : destination, copy.getMapOperands(), map, shape);
        if (schedule.count(copy)) schedule[view] = schedule.lookup(copy);
        if (sourceSlice) source = view.getResult();
        else destination = view.getResult();
      }
      // Do not invent a stronger alignment than the original element access.
      // A caller may explicitly guarantee more via the copy's alignment attr.
      int64_t alignment = std::max<int64_t>(1, src.getElementTypeBitWidth() / 8);
      if (!llvm::isPowerOf2_64(alignment)) alignment = 1;
      if (assumeAlignment) alignment = assumeAlignment;
      if (auto attr = copy->getAttrOfType<IntegerAttr>("alignment"))
        alignment = attr.getInt();
      auto issue = b.create<frisk::AsyncCopyGmToShmOp>(loc, source, destination,
                                                     alignment);
      auto wait = b.create<frisk::WaitBufferReadyOp>(loc, issue.getToken());
      if (schedule.count(copy)) {
        schedule[issue] = schedule.lookup(copy);
        schedule[wait] = schedule.lookup(copy);
      }
      if (copy.hasValueResult()) copy.getValueResult().replaceAllUsesWith(copy.getDst());
      copy.erase();
    }
    // 按改写后的循环体重建 pipeline.stage / pipeline.order。
    for (auto loop : loops) {
      SmallVector<int64_t> stages, orders;
      for (Operation &op : loop.getBody()->without_terminator()) {
        auto [stage, order] = schedule.lookup(&op);
        stages.push_back(stage);
        orders.push_back(order);
      }
      loop->setAttr("pipeline.stage", DenseI64ArrayAttr::get(&getContext(), stages));
      loop->setAttr("pipeline.order", DenseI64ArrayAttr::get(&getContext(), orders));
    }
  }
};
struct LowerFriskAsyncToSyncPass
    : PassWrapper<LowerFriskAsyncToSyncPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerFriskAsyncToSyncPass)
  StringRef getArgument() const final { return "frisk-lower-async-to-sync"; }
  StringRef getDescription() const final {
    return "Select ordinary tile copies and block synchronization before thread tiling";
  }
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<frisk::FriskDialect, gpu::GPUDialect, arith::ArithDialect>();
  }
  void runOnOperation() final {
    // sync 模式的回退 Pass。frisk.source_map 表示 packed LDS：目的形状已不等于源的逻辑形状，
    // 不能生成普通 copy，只打 frisk.force_sync，留给后面的 lowering 按置换处理。
    if (failed(frisk::validateAsyncTaskTokens(getOperation()))) return signalPassFailure();
    SmallVector<Operation *> ops;
    getOperation().walk([&](Operation *op) {
      if (isa<frisk::AsyncCopyGmToShmOp, frisk::WaitBufferReadyOp>(op))
        ops.push_back(op);
    });
    for (Operation *op : ops) {
      OpBuilder b(op);
      if (auto issue = dyn_cast<frisk::AsyncCopyGmToShmOp>(op)) {
        if (issue->hasAttr("frisk.source_map")) {
          // A packed destination no longer has the source's logical shape.
          // Preserve the permutation for late scalar lowering instead of
          // constructing an invalid ordinary copy between unequal shapes.
          issue->setAttr("frisk.force_sync", b.getUnitAttr());
          continue;
        }
        auto copy = b.create<frisk::CopyOp>(op->getLoc(), issue.getSrc(), issue.getDst());
        // Readiness is explicitly supplied by the separate wait op. Do not
        // reinsert a publication barrier immediately after this copy.
        copy->setAttr("frisk.explicit_wait", b.getUnitAttr());
      } else {
        auto wait = cast<frisk::WaitBufferReadyOp>(op);
        SmallVector<frisk::AsyncCopyGmToShmOp> issues;
        if (failed(frisk::collectAsyncTaskIssues(wait.getToken(), issues))) {
          wait.emitError("cannot resolve async task for synchronous lowering");
          return signalPassFailure();
        }
        bool packed = llvm::any_of(issues, [](frisk::AsyncCopyGmToShmOp issue) {
          return issue->hasAttr("frisk.source_map");
        });
        // Packed tasks still need late permutation-aware lowering and their wait.
        if (packed) continue;
        b.create<frisk::SyncThreadsInBlockOp>(op->getLoc());
      }
      if (auto issue = dyn_cast<frisk::AsyncCopyGmToShmOp>(op)) {
        Value completed = b.create<frisk::AsyncTaskCompletedOp>(op->getLoc(),
            frisk::friskAsyncTaskType::get(&getContext()));
        issue.getToken().replaceAllUsesWith(completed);
      }
      op->erase();
    }
    if (failed(frisk::lowerAsyncTaskTokens(getOperation()))) signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> mlir::frisk::createPromoteGlobalToSharedAsyncPass() {
  return std::make_unique<PromoteGlobalToSharedAsyncPass>();
}

std::unique_ptr<Pass> mlir::frisk::createLowerFriskAsyncToSyncPass() {
  return std::make_unique<LowerFriskAsyncToSyncPass>();
}
