#ifndef DEEPGENGRAPH_ANALYSIS_SHARED_MEMORY_PLAN_H
#define DEEPGENGRAPH_ANALYSIS_SHARED_MEMORY_PLAN_H

#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Analysis/LivelinessAnalyze.h"
#include <string>

namespace mlir::frisk {

enum class SharedAccessKind { Read, Write, AsyncCopy, SyncCopy, MatrixRead, Unknown };

struct SharedAccessDesc {
  Operation *op = nullptr;
  unsigned operand = 0;
  Value buffer;
  DictionaryAttr attributes;
  AffineMap computationCoordinates;
  SharedAccessKind kind = SharedAccessKind::Unknown;
  std::optional<LogicalAccess> logical;
  unsigned requiredPacketBytes = 0; // Implicit matrix footprint, not a preference.
  Value source;                    // Copy source view; preserves its origin SSA.
  std::optional<coordXY_t> sourceStrides;
  unsigned sourceAlignment = 0;    // Guarantee at this view's origin.
  unsigned requestedDMABytes = 0;
  unsigned copyThreads = 0;
  std::string reason;
};

struct SharedBufferGroup {
  unsigned id = 0;
  SmallVector<Operation *> allocations;
  SmallVector<Value> aliases;
  SmallVector<DictionaryAttr> allocationAttributes;
  SmallVector<SharedAccessDesc, 0> accesses;
  coordXY_t shape{0, 0};
  int64_t elementBytes = 0;
  unsigned threads = 0;
  unsigned alignment = 0;
  bool complete = true;
  std::string reason;
};

struct SharedAccessPlan {
  unsigned access = 0;
  unsigned vectorBytes = 0;
  uint64_t waveInstructions = 0;
  std::optional<unsigned> worstBankWays;
  std::optional<uint64_t> bankCost; // Sum of phase-count * worst ways per issue.
  // Matrix reads have a collective footprint. Rank their packet placement
  // using both 128-bit lane partitions, without claiming a measured bank cost.
  std::optional<uint64_t> packetCost;
};

struct SharedCopyPlan {
  unsigned access = 0;
  unsigned bytesPerLane = 0; // Zero denotes synchronous lowering.
  unsigned copyThreads = 0;
  uint64_t rounds = 0;
  uint64_t waveInstructions = 0;
  // (copy tid, round) -> allocation-relative physical element offset.
  // Active iff destination < totalElements. Complete-wave legality guarantees
  // that this only masks whole waves in the last round.
  uint64_t totalElements = 0;
  AffineMap destination;
  // Physical element -> source-view coordinates. Source SSA retains origin.
  AffineMap sourceCoordinates;
  AffineMap sourceOffsets; // Element offset relative to source view origin.
  bool exportable = false;
};

struct SharedLayoutCandidate {
  ShmDataIODesc layout;
  SmallVector<SharedAccessPlan> accesses;
  SmallVector<SharedCopyPlan> copies;
  bool legal = false;
  bool exportable = false;
  bool dominated = false;
  std::optional<uint64_t> bankCost;
  uint64_t dmaInstructions = 0;
  uint64_t ldsInstructions = 0;
  unsigned addressCost = 0;
  std::string reason;
};

struct SharedGroupPlan {
  SharedBufferGroup group;
  SmallVector<SharedLayoutCandidate, 0> candidates;
  std::optional<unsigned> selected;
  bool required = false;
};

struct SharedPlanningOptions {
  // Hardware capabilities and enabled emission paths are separate contracts.
  // Buffer asm supports 8B too; intrinsic-only callers must opt into it.
  SmallVector<unsigned> exportableDMABytes{4, 16};
  DenseMap<Operation *, SmallVector<std::pair<int, int>, 2>> requestedSwizzles;
  DenseMap<Operation *, unsigned> requestedDMABytes;
  bool searchLayouts = true;
  bool searchSwizzles = true;
  bool requireMaximumDMA = false;
  bool reduceCopyWaves = false;
  bool allowSynchronousFallback = false;
  uint64_t maxProofElements = 1 << 20;
};

// The execution phase runs after physical access maps have been materialized,
// but before allocations become views of a shared pool. It validates and
// materializes AutoPipeline's provisional stage/slot/issue-after schedule, and
// owns physical placement and cross-iteration scheduling. Lowering must not
// choose another slot or issue time.
struct SharedPipelineStage {
  AsyncCopyGmToShmOp issue;
  WaitBufferReadyOp ready;
  SmallVector<ReadMatrixFromShmOp> reads;
  unsigned bufferIndex = 0;
  unsigned slot = 0;
  Operation *issueAfter = nullptr;
  SmallVector<Operation *> addressDependencies;
};

struct SharedBufferingPlan {
  SmallVector<SharedPipelineStage> stages;
  uint64_t slotBytes = 0;
  std::array<uint64_t, 2> slotOffsets{};
};

// One-iteration lookahead. Task SSA connects the prologue/guarded issue to
// the original readiness in the next iteration. Allocation uses a circular
// lifetime, not an iteration-local lifetime or a whole-loop approximation.
struct SharedCrossIterationPlan {
  Operation *loop = nullptr;
  AsyncCopyGmToShmOp issue;
  WaitBufferReadyOp ready;
  WaitBufferReadyOp issueAfter;
  unsigned bufferIndex = 0;
  SmallVector<Operation *> addressDependencies;
};

struct SharedExecutionPlan {
  func::FuncOp kernel;
  ShmLivenessResult lifetimes; // Includes the planned earlier DMA issues.
  ShmReusePlan storage;       // Includes required storage-handoff barriers.
  SmallVector<SharedBufferingPlan, 0> pipelines;
  SmallVector<SharedCrossIterationPlan, 0> crossIterations;
  SmallVector<std::string> rejectedPipelines;
  SmallVector<std::string> rejectedCrossIterations;
  std::string inputSnapshot;
  bool materializePool = false;
};

// Owns a snapshot. Any operand/layout/alias change invalidates it. No analysis
// routine writes IR. publish() validates every group before writing any attrs.
struct SharedMemoryPlan {
  SmallVector<SharedGroupPlan, 0> groups;
  std::optional<SharedExecutionPlan> execution;
  void dump(llvm::raw_ostream &os) const;
  LogicalResult publish(LowerInfoMap &layouts) const;
  LogicalResult materializeExecution() const;
};

SmallVector<SharedBufferGroup, 0> collectSharedBufferAccesses(
    func::FuncOp kernel, LowerInfoMap &layouts, HWSpecification &hw);
SharedMemoryPlan planSharedMemory(func::FuncOp kernel, LowerInfoMap &layouts,
    HWSpecification &hw, const SharedPlanningOptions &options = {});

// Read-only scheduling and allocation against one immutable IR snapshot.
// Double buffering consumes frisk.shared_pipeline_plan from AutoPipeline and
// preserves the unbuffered plan's LDS budget. No unmarked stream is scheduled.
SharedMemoryPlan planSharedMemoryExecution(func::FuncOp kernel,
                                          bool doubleBuffer = false,
                                          bool crossIteration = false);
void dumpSharedMemoryExecution(const SharedExecutionPlan &plan, llvm::raw_ostream &os);

// Shared decoding/encoding for the planner and physical packer. Legacy single
// XOR attrs are supported; the generic ordered form is frisk.shared_swizzles.
std::optional<ShmDataIODesc> readSharedLayout(Operation *storage, MemRefType type);
void writeSharedLayout(Operation *storage, const ShmDataIODesc &layout);

} // namespace mlir::frisk
#endif
