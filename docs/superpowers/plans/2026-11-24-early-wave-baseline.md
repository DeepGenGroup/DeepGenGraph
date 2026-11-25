# Early-wave baseline Implementation Plan

> **For agentic workers:** Execute this single task inline with test-driven development.

**Goal:** Use the user's full measured configuration through the existing runners.
**Architecture:** Extend preset dictionaries, named CLI forwarding, and validation;
keep input generation and external HSACO optimization independent.
**Tech Stack:** Python standard library and unittest.
**Spec:** ../specs/2026-11-24-early-wave-baseline-design.md

## Global Constraints

- Preserve all historical named presets and default compiler parameters.
- User runs GPU benchmarks; no MyTest rebuild or automatic GPU scan.
- New baseline: internal llvm-opt=0; external HSACO opt-level remains 3.

## Review Focus

- Complete flag forwarding, including options absent from the old tuner.
- Finite lazy-base bounds and invalid boolean/enum values.
- Overrides and sweeps preserve independent internal/external optimization.
- Default runner and both causal modes retain launch metadata.
- Original measured input and shape performance are not assumed reproduced.

### Task 1: Add the full baseline

**Files:** TuneAttention.py, RunAttention.py, RunAttention.md,
调优参数说明.md, 3rd/deepgengraph/test/unittest/check_attention_runner.py.

- [x] Add failing CLI/pipeline tests for the preset, validation, and sweeps.
- [x] Observe failures caused by missing preset/options.
- [x] Add seven named options, validation, and `early-wave-baseline`.
- [x] Select it by default in RunAttention and document explicit case/sweep use.
- [x] Run runner/generator tests and validate generated arguments with MyTest's
      parser using a missing input (no lowering or device run).
- [x] Review the diff and report verification plus a user-run command.

Execution record: existing runner baseline passed 14 tests. Work is confined to
the requested checkout and entry scripts so the user can run the result there;
no branch, binary, existing artifact, or user modification will be overwritten.

Verification: all 17 runner tests pass, including complete preset forwarding,
override/sweep combinations, validation, default selection and independent HSACO
O3. The existing MyTest accepts the generated flags and reaches the expected
missing-input parse error. No lowering, rebuild, or GPU benchmark was run.

Broader discovery with Python 3.12 and the shared NumPy installation passed all
27 collected runner/shape/shared-generator tests, but two other modules fail to
import: check_attention_block_tiles looks for attention_full_d_block.mlir in
the unittest directory, and check_attention_diagnostics cannot load
libgalaxyhip.so.5. These are outside this preset change; the broad suite is not
reported as passing. Scoped whitespace verification excludes the user's
pre-existing rebuild.sh modification. GPU correctness and performance remain
for the user to measure with the intended input and toolchain.

2026-11-25 integration: user-run benchmarks now cover 8 shapes and both modes;
all 16 selected measurements passed the configured numerical tolerance. The
portable report and CSV are in docs/benchmarks/2026-11-25-early-wave-baseline.*.
After fast-forwarding the two upstream commits through 70b3a94f, all 28
runner/shape/shared-generator tests passed. The existing MyTest binary and the
user's rebuild.sh modification are preserved; benchmark observations refer to
the binary used during those runs, before the upstream compiler changes.
