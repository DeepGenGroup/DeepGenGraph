# Generic Thread Scheduling Implementation Plan

> **For agentic workers:** Execute this authorized plan inline with superpowers:executing-plans; validate each task before continuing.

**Goal:** Merge the four attention-named scheduling entries into two mechanism-based passes, while retaining legacy command-line aliases.

**Architecture:** MMA chain interleaving selects operand sources and bounded load motion through options. LDS pairing chooses a register-computation window rather than a separate tail algorithm. Both validate a complete local order before mutation and use common scheduling markers.

**Tech Stack:** C++17, MLIR/LLVM, Python integration checks.

**Spec:** The approved design in this conversation: generic MMA interleaving and LDS pairing, with preserved chain order, memory boundaries, budgets, and repeatability.

## Global Constraints

- Keep the existing supported MMAC instruction; do not infer purity from arbitrary inline assembly.
- Preserve each accumulator chain's order and floating-point attributes.
- Bounded read motion never crosses writes, synchronization, unknown effects, or existing protected read pairs.
- Existing MyTest flags retain their defaults and map to semantic options.
- Do not modify frontend numerical policies or unrelated shared-memory planning.
- Leave changes uncommitted for review.

## Task 1: Generic integration contract

**Files:** Create `3rd/deepgengraph/test/unittest/check_generic_thread_scheduling.py`.

- [x] Add a real ThreadTilingTest check for two register-only MMA chains. The literal expected order is `0:0, 1:0, 0:1, 1:1, 0:2, 1:2`.
- [x] Add existing matrix/ordinary LDS input cases with exact SSA graph preservation, bounded-read budget rejection, barrier rejection, and replay.
- [x] Add ordinary-GEMM LDS reads consumed through the A operand, with local add/mul between reads. Require pairing without changing the returned DAG.
- [x] Run `python3 test/unittest/check_generic_thread_scheduling.py --tool build/test/ThreadTilingTest`; expect failure because `frisk-thread-interleave-mma-chains` is not registered yet.

## Task 2: Generic pass implementations and compatibility

**Files:** Create `include/deepgengraph/Conversion/ConvertToLLVM/ThreadScheduling.h`, `lib/Conversion/ConvertToLLVM/ThreadSchedulePlan.h`, and `lib/Conversion/ConvertToLLVM/ThreadSchedulingCompatibility.cpp`. Rename `ThreadInterleavePV.cpp` to `ThreadInterleaveMMAChains.cpp`. Modify `ThreadPrefetchPairedLDS.cpp`, `Passes.h`, CMake, `test_src.cpp`, and `thread_tiling_test.cpp`.

**Interfaces:**

```cpp
struct MMAChainInterleaveOptions {
  unsigned width = 0;
  int guardGroup = 2;
  std::string operandSource = "any"; // any, register, lds, matrix
  std::string loadMotion = "none";   // none, bounded
  unsigned prefetchWords = 32;
};
struct PairedLDSPrefetchOptions {
  unsigned distance = 0;
  std::string window = "register"; // address, register, reduction
};
std::unique_ptr<Pass> createThreadInterleaveMMAChainsPass(
    const MMAChainInterleaveOptions &options = {});
std::unique_ptr<Pass> createThreadPrefetchPairedLDSPass(
    const PairedLDSPrefetchOptions &options = {});
```

- [x] Separate accumulator-chain discovery from source recognition. Register operands need no memory packet; bounded motion is available only to supported LDS reads.
- [x] Compute distinct operand packet words from type width, retaining the legacy B-footprint budget definition.
- [x] Use one LDS pairing implementation; register windows admit ordinary speculatable register computations, while compatibility reduction windows retain the original narrower selection.
- [x] Add `LocalSchedulePlan::validate/apply` for unique membership, SSA order, and external definition availability.
- [x] Use `frisk.mma_interleave` and `frisk.lds_pair`; record LDS policy/distance without skipping newly introduced opportunities. Allocate new pair identifiers after existing identifiers.
- [x] Register the two canonical passes and legacy option mappings; the ordinary LDS CLI keeps its name, and the other three names are pipeline aliases.
- [x] Update MyTest to construct the canonical options explicitly at the original pipeline positions.
- [x] Build with `cmake --build build --target ThreadTilingTest MyTest -j2` and run the generic contract.

## Task 3: Regression and documentation

**Files:** Modify `check_pv_interleave.py`, `check_thread_fragment_schedules.py`, `PVInterleave.md`, `PairedLDS.md`, and `优化汇总.py`.

- [x] Update graph fingerprinting to ignore only canonical scheduling markers, preserving checks of operands, FP flags, addresses, and chain order.
- [x] Give the fragment schedule checker a `--tool` argument with the correct repository-relative default.
- [x] Run generic checks, legacy PV/QK checks, paired/softmax checks, and LLVM/O3 export checks.
- [x] Document canonical flags, semantic windows, legacy mappings, and the unchanged distance/budget meanings.
- [x] Review the complete diff and confirm only scheduling-related changes remain.

## Verification

- ThreadTilingTest and MyTest builds succeeded.
- Generic scheduling: 16 contracts; legacy PV: 40; MMA/QK: 133; LDS/softmax: 63. All 252 scheduling checks passed, including LLVM/O3 verification in the existing suites.
- MyTest named/legacy CLI diagnostics passed.
- Complete attention compilation with both MMA source policies succeeded; the exported LLVM IR passed llvm-as. Artifacts: `/tmp/deepgengraph-generic-scheduling-4nqtlgob`.
- Read-only independent review found two generality gaps; both were reproduced before fixes. Follow-up review found no remaining important issues.
- GPU timing was not performed. Changes remain uncommitted.
