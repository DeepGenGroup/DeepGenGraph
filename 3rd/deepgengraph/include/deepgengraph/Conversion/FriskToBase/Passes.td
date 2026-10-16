#ifndef FRISKTOBASE_CONVERSION_PASSES
#define FRISKTOBASE_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def OptimizeTensorLoops: Pass<"optimize-tensor-loops", "::mlir::ModuleOp"> {
  let summary = "Share invariant tensor work through checked unroll-and-jam, recurrence CSE and LICM";
  let constructor = "mlir::frisk::createOptimizeTensorLoopsPass()";
  let options = [Option<"maxJamFactor", "max-jam-factor", "unsigned", "8",
    "Maximum unroll-and-jam factor (divisors of the trip count are considered)">,
    Option<"maxCarrierWords", "max-carrier-words", "uint64_t", "24576",
    "Maximum nested live tensor carrier footprint per block, in 32-bit words; 0 disables the budget">];
  let dependentDialects = ["mlir::affine::AffineDialect", "mlir::arith::ArithDialect"];
}

def ConvertFriskToBase: Pass<"convert-frisk-to-base", "::mlir::ModuleOp"> {
  let summary = "convert frisk to base";
  let description = [{}];

  let constructor = "mlir::frisk::createConvertFriskToBasePass()";

  let dependentDialects = [
    "mlir::frisk::FriskDialect"
  ];
}

def NormalizeAttentionKLayout: Pass<"normalize-attention-k-layout", "::mlir::ModuleOp"> {
  let summary = "Use a contiguous BHSD K input for the Attn_p2 example";
  let description = [{
    Changes the external ABI from BHDS to BHSD, keeping a zero-copy transpose
    view for the internal QK tile. Run immediately after convert-frisk-to-base.
    This is deliberately specific to four-argument Attn_p2 self-attention:
    Q is argument 0, O is argument 3, and K is the unique argument among 1/2
    whose trailing shape is reversed relative to Q. For S == D, key-arg must
    explicitly identify the BHDS argument. Callers must supply BHSD storage.
  }];
  let constructor = "mlir::frisk::createNormalizeAttentionKLayoutPass()";
  let options = [Option<"keyArg", "key-arg", "int", "-1",
      "BHDS K argument index (1 or 2); -1 detects the example's reversed shape">];
  let dependentDialects = ["mlir::memref::MemRefDialect"];
}

def PlanWmmaLayoutConversions: Pass<"plan-wmma-layout-conversions", "::mlir::func::FuncOp"> {
  let summary = "Plan wave-local WMMA C layout conversions with XOR shuffles";
  let description = [{
    Request contiguous GEMM C fragments before layout inference, and classify
    convert_layout operations carrying frisk.source_layout, frisk.target_layout,
    frisk.layout_role = "wmmaC", and frisk.wmma_instruction. A proven square
    lane/register transpose receives a serializable frisk.shuffle_plan.
    Thread tiling invokes this planner automatically at both boundaries.
  }];
  let constructor = "mlir::frisk::createPlanWmmaLayoutConversionsPass()";
  let options = [Option<"contiguousOutputs", "contiguous-outputs", "bool", "true",
      "Request contiguous C fragments for supported GEMM instructions">];
  let dependentDialects = ["mlir::frisk::FriskDialect"];
}

def ConvertFriskBaseToThreadLevelIR: Pass<"convert-friskbase-to-thread", "::mlir::func::FuncOp"> {
  let summary = "frisk base 转换到thread级别的实现";
  let description = [{}];

  let constructor = "mlir::frisk::createConvertFriskBaseToThreadLevelIRPass()";
  let options = [Option<"packSharedOperands", "pack-shared-operands", "bool", "true",
      "Plan packed LDS layouts for owned GEMM operands">,
    Option<"matrixLoads", "matrix-loads", "bool", "true",
      "Prefer compatible warp matrix reads for shared GEMM B operands">,
    Option<"asyncDirectOnly", "async-direct-only", "bool", "false",
      "Use vectorized synchronous copies when packed halfword packets require LDS staging">,
    Option<"dmaPackMode", "dma-pack-mode", "std::string", "\"legacy\"",
      "legacy, useKPack (caller coarsens tiles), or noKPack (reduce copy waves); require maximum hardware DMA width">,
    Option<"uniformMasks", "uniform-masks", "bool", "false",
      "Skip elementwise upper-triangular masks for provably complete tiles">,
    Option<"fuseMaskAdd", "fuse-mask-add", "bool", "true",
      "Generate single-use masks with their adjacent addition inside each fragment">,
    Option<"wmmaShuffle", "wmma-shuffle", "bool", "true",
      "Legacy auto-mode switch for contiguous WMMA C and shuffle conversions">,
    Option<"wmmaLayout", "wmma-layout", "std::string", "\"auto\"",
      "GEMM A-input producer layout (attention P): shared (3.1), shuffle (3.2), b-permute (3.3; unsupported shapes use shuffle), auto (legacy switch). Terminal C outputs retain shuffle.">];

  let dependentDialects = [
    "mlir::frisk::FriskDialect"
  ];
}

def PackSharedMemory: Pass<"pack-shared-memory", "::mlir::func::FuncOp"> {
  let summary = "Pack owned shared buffers and coalesce proven contiguous accesses";
  let constructor = "mlir::frisk::createPackSharedMemoryPass()";
  let dependentDialects = ["mlir::affine::AffineDialect", "mlir::arith::ArithDialect", "mlir::memref::MemRefDialect", "mlir::vector::VectorDialect"];
}

def PromoteGlobalToSharedAsync: Pass<"frisk-promote-gm-to-shm-async", "::mlir::func::FuncOp"> {
  let summary = "Represent global/shared copies with explicit buffer readiness";
  let constructor = "mlir::frisk::createPromoteGlobalToSharedAsyncPass()";
}

def PromoteGemmMatrixLoads: Pass<"frisk-promote-gemm-matrix-loads", "::mlir::func::FuncOp"> {
  let summary = "Select collective shared-to-register matrix reads for GEMM B";
  let constructor = "mlir::frisk::createPromoteGemmMatrixLoadsPass()";
}

def LowerFriskAsyncToSync: Pass<"frisk-lower-async-to-sync", "::mlir::func::FuncOp"> {
  let summary = "Select ordinary copies and block synchronization before thread tiling";
  let constructor = "mlir::frisk::createLowerFriskAsyncToSyncPass()";
}

def SinkWaitBufferReady: Pass<"frisk-sink-wait-buffer-ready", "::mlir::func::FuncOp"> {
  let summary = "Sink buffer readiness to the first dependent access or synchronization boundary";
  let description = [{
    Move same-block waits past independent work, tracking logical Frisk views
    and physical memref aliases. Preserve source lifetime, destination accesses,
    explicit synchronization, DMA ordering and existing pipeline schedules.
    Thread tiling invokes this after layout planning and async/sync selection.
  }];
  let constructor = "mlir::frisk::createSinkWaitBufferReadyPass()";
}

def FinalizeThreadTiling: Pass<"finalize-thread-tiling", "::mlir::func::FuncOp"> {
  let summary = "基于thread级别的实现 将其转为 mlir built-in IR 表达的形式。完全去除 to_threadtile 和 from_threadtile";
  let description = [{}];

  let constructor = "mlir::frisk::createFinalizeThreadTilingPass()";

  let dependentDialects = [
    "mlir::frisk::FriskDialect"
  ];
}



def FriskFragIRReorder: Pass<"frisk-frag-ir-reorder", "::mlir::func::FuncOp"> {
  let summary = "Fuse fragment loops and pipeline read-only fragment loads";
  let constructor = "mlir::frisk::createFriskFragIRReorderPass()";
  let dependentDialects = ["mlir::affine::AffineDialect", "mlir::arith::ArithDialect", "mlir::frisk::FriskDialect"];
}

def LowerFriskFragments: Pass<"lower-frisk-fragments", "::mlir::func::FuncOp"> {
  let summary = "Lower scheduled fragment regions, plan storage and unroll marked loops";
  let constructor = "mlir::frisk::createLowerFriskFragmentsPass()";
  let dependentDialects = ["mlir::affine::AffineDialect", "mlir::arith::ArithDialect", "mlir::memref::MemRefDialect", "mlir::vector::VectorDialect", "mlir::gpu::GPUDialect", "mlir::frisk::FriskDialect"];
}

#endif // FRISKTOBASE_CONVERSION_PASSES
