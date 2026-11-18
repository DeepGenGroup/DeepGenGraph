# Attention Block Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Shared tensor attention with independent, real block-level QK and V/O tiling.

**Architecture:** Generator hints preserve one shared attention recurrence. A block pass pushes slices to global loads and splits output carriers before layouts and physical memory planning.

**Tech Stack:** Python, C++, MLIR affine/Frisk, NumPy regression interpreter.

**Spec:** docs/superpowers/specs/2026-11-18-attention-block-tiling-design.md

## Global Constraints

- Preserve online/legacy softmax, causal/noncausal, ABI and global strides.
- No GPU speedup claims without measurements.
- Reject QK regrouping without allow-reassociation; independently tile VO.
- Preserve the existing optimization-summary edit; the user authorized committing and pushing the completed changes to GitHub on 2026-11-18.

### Task 1: Shared generator

**Files:** `generate_attention_ir.py`, `3rd/deepgengraph/test/check_attention_shared_generator.py`, historical generator regression call sites.
**Interface:** `generate_ir(...)`; kernel attribute `frisk.attention_tiles = array<i64: QK, VO>`.

- [x] Write a failing generator regression asserting one dot per QK/PV, no output/reduction D loops, full-D O shape, valid causal/noncausal online/legacy modes, and hint preservation.
- [x] Run `python3 3rd/deepgengraph/test/check_attention_shared_generator.py`; observe the old output-loop failure.
- [x] Generate explicit-origin full-D attention with Q/K/V loads in the key loop; delete `split_d_kernel` and the old driver retile helper.
- [x] Run generator tests and update historical integration tests to shared semantics; retain archived fixtures for generic tensor-loop tests only.

### Task 2: Block slicing and carriers

**Files:** new `lib/Conversion/FriskToBase/TileAttention.cpp`, Passes.h/td, CMakeLists.txt, thread_tiling_test.cpp, new `check_attention_block_tiles.py`.
**Interface:** `createTileAttentionPass()` / `--frisk-tile-attention=qk-tile=32 vo-tile=32 allow-reassociation=true` operating on `func.func`.

- [x] Test real block input with full QK/PV: QK64/VO32 and QK32/VO64, tile staging shapes, one probability, invalid tiles and strict rejection.
- [x] Observe absent-pass failure, then implement slice pushdown for global views, owned shared allocations/copies, constants, zero, cast, elementwise operations.
- [x] Build `MyTest` and `ThreadTilingTest`; run tests with `--verify-each` and CPU before/after execution.
- [x] Split full-D O vector carriers into independent column carriers, preserve shared L/M and ordered QK reduction, and test replay idempotence.

### Task 3: Driver integration and verification

**Files:** `test_src.cpp`, `SplitDAttentionTiles.h`, generator integration regressions, `优化汇总.py`.
**Interface:** argv[17]/argv[18] override hints; the old tensor retile helper is removed.

- [x] Compile shared generator with explicit overrides and low tensor carrier budget; expect shared tensor IR and successful block tiling.
- [x] Transfer generator hints after base conversion; run block tiling before FriskOpFuse and downstream planning. Remove legacy retile guard.
- [x] Verify online/legacy, causal/noncausal, at least two independent tile pairs, invalid/strict rejection and LLVM verification. Run existing relevant fusion/fragment/memory tests.
- [x] Update optimization summary with actual pass ordering and independent-tiling rationale; request independent code review and address important findings.
- [x] Migrate TuneAttention default input and QK pipeline/unroll controls to shared kernel metadata; verify actual CLI exports, LLVM CPU numerics and independent scoped review.

## Validation record

### Follow-up: Remove obsolete tensor loop optimization

User approved removing OptimizeTensorLoops, its registration, dedicated tests and carrier budget. The shared generator has no matching outer loop, so the pass already returns without transforming current attention.

- [x] Remove C++ pass, TableGen/CMake registration and MyTest invocation; delete carrier positional argument and shift later positions together.
- [x] Migrate TuneAttention, candidate generation, maintained tests and scripts to the shorter positional interface; remove carrier sweeps and reporting.
- [x] Preserve unrelated loop-layout recurrence regression in a separately named test; remove unroll-and-jam/budget-only tests.
- [x] Update current pass documentation and optimization summary to 39 active entries, preserving existing IDs 2..40.
- [x] Rebuild tools, run driver/candidate CLI checks, shared block and independent LLVM numerical regressions; request scoped review. User subsequently authorized the GitHub submission.

- Generator and shapes: 5 + 4 tests; all softmax/mask modes, unequal extents and CLI.
- Block tiling: five independent tile pairs, actual CPU before/after, idempotence, strict grouping, mutation/offset rejection, pipeline hints, bounded QK exposure.
- Driver: six generated tile variants and full-size tensor/block numerics; carrier budget 1.
- LLVM: independent tile variants execute equal dynamic MMAC/exp2 work; invalid CLI rejected.
- Fragment fusion/scheduling and shared-memory tests pass, including double buffering and cross-iteration ownership.
- AutoPipeline: 5 variants; cleanup: 4 variants, actual asynchronous ownership and last query tile.
- Full noncausal mainline: LLVM verifies; CPU head0/block0 and head1/block31 pass with retention, V double buffering, cross-key prefetch and lazy softmax (8192 MMAC each).
- Tune shared QK metadata: runner 9 tests; nohint/rolled and stages3/unroll real exports verify and execute CPU head1/block0 correctly (256 MMAC each).
- Tensor loop cleanup: MyTest/ThreadTilingTest rebuilt; runner 13 tests; shared block and independent LLVM numerical regressions pass. Six candidate exports verify; baseline LLVM bytes match the pre-removal artifact after normalizing paths. Full noncausal mainline matches after normalizing input/debug file locations, and both CPU query-tile cases retain the same errors. Both CP-SAT baselines regenerated with exact Base/LLVM identity replay; five solver-free bridge/DMA-mode regressions pass. Scoped review has no unresolved findings. Complete solver suite unavailable because ortools is absent.
- GPU timing and target code generation were not performed.
