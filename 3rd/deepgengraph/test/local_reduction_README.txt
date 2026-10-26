LocalReductionPass
==================

Public API: include/deepgengraph/Conversion/ConvertToLLVM/LocalReduction.h
Implementation: lib/Conversion/ConvertToLLVM/LocalReduction.cpp
This is an LLVM new-PM FunctionPass. It can run on ordinary CPU functions or
GPU kernels. It does not inspect attention names, SSA names, tile dimensions,
thread IDs, tensor layouts, shuffle intrinsics, or any target GPU feature.

Transformation
--------------
Find a long spine of identical associative operations in one basic block.
Each interior instruction must have exactly one use. A spine may appear on
either operand side; intermediate uses and branching trees bound the match.
Partition inputs round-robin among K independent partial accumulators, then
combine the K results with a balanced tree. Keep an existing constant identity
initialization as the first partial, e.g. max(x0, -inf). Replace all uses of the
final result, including when that result has multiple users.

Each new partial is inserted at the original accumulation position for its
new input, preserving dominance even when inputs are produced incrementally.
Other instructions, memory operations, calls, barriers, and CFG remain in place.
Loop PHIs are opaque leaves; the pass never walks across a block boundary or
changes a loop-carried recurrence. This handles thread-local reductions; it
does not rewrite subgroup/workgroup communication or vector.reduce intrinsics.

Supported operations, scalar and fixed-length vector:
  add: integer add, floating fadd
  mul: integer mul, floating fmul
  and/or/xor: integer bitwise operations
  min: llvm.smin, llvm.umin, llvm.minnum, llvm.minimum
  max: llvm.smax, llvm.umax, llvm.maxnum, llvm.maximum
Different opcodes/intrinsics are never mixed. Scalable vectors, constrained FP,
convergent/noduplicate calls and operand bundles are excluded. StrictFP and
OptNone functions are skipped. Matching is bounded to 2048 instructions.

Legality and modes
------------------
Default off: no change to existing compiler presets.
legal:
  Integer operations use modular arithmetic where applicable. Regrouped
  operations do not inherit nsw/nuw/disjoint flags: the new partials may overflow
  even when the original partials do not. Results remain equal for inputs on
  which the original computation is defined.
  fadd/fmul require reassoc AND nsz on every matched instruction.
  Floating min/max require nnan AND nsz on every matched instruction.
  These are deliberately conservative gates. Missing strictfp is not permission.
approx:
  Explicitly authorize floating reassociation and disregard signed-zero/NaN
  distinctions in the selected reductions. Rounding, cancellation, intermediate
  overflow and NaN behavior may change; this is not strict bitwise equivalence.
  Does not add fast-math flags or assume infinities are absent. StrictFP and
  constrained operations remain excluded. MyTest requires math=approx as well.

New instructions carry !frisk.local_reduction metadata to prevent repeated
rewrites. Changing tuning settings requires re-running on the original IR.
No metadata requests a backend scheduling order; downstream LLVM may vectorize
or otherwise optimize the resulting graph. More partials may increase register
pressure: this is a tunable optimization, not a promise of speedup on all targets.

Options (MyTest and standalone LegacyLLVMExportTest)
---------------------------------------------------
  --local-reduce=off|legal|approx        default off
  --local-reduce-chains=N               default 4; range 2..16
  --local-reduce-min-inputs=N           default 16; range 2*chains..2048
  --local-reduce-ops=all                default all, or comma-separated subset
                                       of add,mul,and,or,xor,min,max
The input threshold counts the retained identity seed as one input.
Non-power-of-two chain counts and input lengths are supported.

Generic standalone use, independent of MLIR and attention:
  build/test/LegacyLLVMExportTest --local-reduce=legal input.ll > output.ll
  build/test/LegacyLLVMExportTest --local-reduce=approx \
    --local-reduce-ops=add,max --local-reduce-chains=4 input.ll > output.ll
The standalone route only runs the requested local-reduction/mask-bypass passes
and LLVM verification; it does not run unrelated export canonicalization.

Embedding in another LLVM pipeline:
  llvm::FunctionPassManager pipeline;
  mlir::frisk::LocalReductionOptions options;
  pipeline.addPass(mlir::frisk::LocalReductionPass(options));
or call mlir::frisk::optimizeLocalReductions(module, options).

MyTest runs it after register math and mask bypass, before prefetch/lifetime
scheduling. TuneAttention.py forwards the same four options, supports sweeps,
filters incompatible math/chain/threshold combinations, and records matched
groups, inputs, chains and mode in result.json's local_reduction_reports.
For the validated attention experiment (commands from repository root):
  python3 TuneAttention.py \
    --input generated_ir/online_softmax/test_input_1x32x4096x128-128-128-32-online.mlir \
    --case online-baseline --set mask-add-bypass=approx \
    --set local-reduce=approx --set local-reduce-ops=add,max \
    --llvm-as /data2/xsl/install/bin/llvm-as \
    --output-dir generated_ir/local_reduction_tuning
Use --sweep 'local-reduce-ops=add;max;add,max' to isolate the two reductions.

Build and regression (repository root):
  cmake --build 3rd/deepgengraph/build --target LegacyLLVMExportTest MyTest -j 2
  OPENBLAS_NUM_THREADS=1 python3 3rd/deepgengraph/test/check_local_reduction.py \
    3rd/deepgengraph/build/test/LegacyLLVMExportTest
  python3 -m unittest script.test_tune_attention script.test_portable_tuning
Regression covers legality, unsupported forms, independent CPU functions,
multiple types/lengths/chain counts, load dominance, loop PHIs, escaping uses,
integer poison flags, option validation, LLVM O3 verification and idempotence.
The attention-specific validation supplements these general tests.
