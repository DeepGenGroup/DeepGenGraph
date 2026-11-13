#ifndef DEEPGENGRAPH_ANALYSIS_REGISTERPRESSURE_H
#define DEEPGENGRAPH_ANALYSIS_REGISTERPRESSURE_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <memory>

namespace mlir {
class AnalysisManager;
class Liveness;
class Pass;
namespace frisk {

/// Target-independent, per-thread SSA estimates in 32-bit words. dataWords
/// charges the surviving originating scalar slots (subword slots stay unpacked);
/// packedDataWords packs surviving bits within each originating SSA payload.
/// idealPackedDataWords() also allows packing across independent SSA origins.
/// addressWords counts explicit LLVM pointer SSA values. descriptorWords is a
/// separate diagnostic for full memref descriptors, excluded from words().
/// Memory contents never contribute to any register estimate.
/// These are not allocated VGPR/SGPR counts or occupancy predictions.
struct RegisterPressureUsage {
  uint64_t dataWords = 0;
  uint64_t packedDataWords = 0;
  uint64_t descriptorWords = 0;
  unsigned unknownValues = 0;
  // Add bits before rounding: two independent f16 origins have 32 data bits,
  // even though their individually rounded packedDataWords sum to two.
  uint64_t dataBits = 0;
  uint64_t addressWords = 0;
  uint64_t words() const { return dataWords + addressWords; }
  uint64_t diagnosticWords() const { return words() + descriptorWords; }
  uint64_t idealPackedDataWords() const { return (dataBits + 31) / 32; }
};

struct RegisterPressurePoint {
  Operation *operation = nullptr;
  unsigned block = 0;
  unsigned depth = 0;
  RegisterPressureUsage before, after, overlap;
  RegisterPressureUsage ssaBefore, ssaAfter, ssaOverlap;
  // Actual SSA values, before representation-alias deduplication. Includes
  // enclosing live-through values and externally captured loop invariants.
  SmallVector<Value, 0> liveBefore, liveAfter;
};

/// A range of bits in an originating value. Zero width denotes an indivisible
/// descriptor/unsupported aggregate, whose cost is recorded in costs.
struct RegisterPressureSlice {
  Value origin;
  uint64_t offset = 0, bitWidth = 0;
};

struct RegisterPressureStorage {
  SmallVector<RegisterPressureSlice, 1> slices;
  // Exact mappings concatenate slices in logical element order. An unresolved
  // insert instead retains a conservative union of destination and input.
  bool exact = true;
};

struct RegisterPressureResult {
  func::FuncOp function;
  SmallVector<RegisterPressurePoint, 0> points;
  DenseMap<Operation *, unsigned> pointIndices;
  DenseMap<Value, Value> storageRoots;
  DenseMap<Value, RegisterPressureUsage> costs;
  DenseMap<Value, RegisterPressureStorage> storage;
  DenseMap<Value, uint64_t> elementBits;
  // An unresolved extract can reuse its source when that source's entire
  // content is already represented in the current live-content union.
  DenseMap<Value, Value> subsetSources;
  unsigned blocks = 0;
  unsigned unknownRegionOps = 0;
  unsigned unresolvedExtracts = 0, conservativeInserts = 0;

  const RegisterPressurePoint *lookup(Operation *op) const;
  /// Whole-value aliases used by the legacy SSA comparator.
  Value storageRoot(Value value) const;
  /// Union of originating content, including slice/insert representation reuse.
  RegisterPressureUsage measure(ArrayRef<Value> values) const;
  RegisterPressureUsage measureSSA(ArrayRef<Value> values) const;
};

/// Reuses MLIR's CFG liveness and last-use queries; adds nested-region
/// live-through values and repetitive-region captures. No IR mutation.
/// All stored handles become stale after IR changes: recompute the analysis.
RegisterPressureResult analyzeRegisterPressure(func::FuncOp function,
                                              const Liveness &liveness);

/// Shared PassManager analysis. Reuses the manager's cached mlir::Liveness;
/// normal MLIR invalidation applies when a transformation changes the IR.
class RegisterPressureAnalysis {
public:
  RegisterPressureAnalysis(Operation *operation, AnalysisManager &manager);
  const RegisterPressureResult &getResult() const { return result; }
private:
  RegisterPressureResult result;
};

void dumpRegisterPressure(const RegisterPressureResult &result,
                          llvm::raw_ostream &os, StringRef stage = "",
                          unsigned top = 8, bool allPoints = false);
std::unique_ptr<Pass> createAnalyzeRegisterPressurePass(StringRef stage = "");

} // namespace frisk
} // namespace mlir
#endif
