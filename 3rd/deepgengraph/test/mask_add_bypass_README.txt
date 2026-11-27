FragmentIR mask-add bypass with ThreadIR fallback
================================================

Primary implementation: lib/Conversion/FriskToBase/FragmentMaskAddBypass.cpp
Pass: --frisk-bypass-fragment-mask-adds=mode=off|approx
Fallback implementation: lib/Conversion/ConvertToLLVM/ThreadMaskAddBypass.cpp
Pass: --frisk-bypass-mask-adds=mode=off|approx
MyTest/TuneAttention: --mask-add-bypass=off|approx (default off)

MyTest runs the FragmentIR pass after fragment retention/prefetch scheduling,
before thread lowering. This keeps the new control-flow region and its cleanup
from changing operand retention and read-ahead planning. For a fused mask_frag with uniform=true and a
recognized causal predicate whose clear value is +0, it uses the logical tile
starts and shape to generate a guarded scf.if returning the existing score on
the clear arm. The masked arm retains mask_frag with uniform=false, preserving
its original predicate, addition order and independent causal-mask policy.
Fusion also accepts the downstream reduction's shared-memory annotation on
the score-add result; shape, element type and layout must still agree, and
the replacement preserves the consumer type and layout. This does not require
finite scores and does not by itself remove the addition.
Unsigned range-wrap guards are shared with the existing mask lowering. No
score computation, loads or synchronization are moved. Reapplication is a no-op.

The fallback pass runs after thread loop unrolling, scalar replacement and
register SSA cleanup, before firstLowering and LLVM dialect conversion. The LLVM FunctionPass,
its wrapper and its production/legacy export entry points have been removed.
legal is rejected explicitly; no other pass's legal mode or FP policy changes.
MyTest still requires math=approx when mask-add-bypass=approx is requested.

ThreadIR fallback shapes:
- Frisk uniform-mask lowering with fused additions: remove proven +0 additions
  only in the clear arm. Preserve the masked arm and existing control flow.
- A single-result scf.if whose clear yield is +0: follow static vector views to
  one or more floating additions. Sink mask views/additions into the masked arm
  and extend the existing if with results that yield each score on the clear arm.
  Either then or else can be clear; original region effects/results are retained.
- Scalar/per-lane arith.select with a +0 arm: distribute supported static lane
  views and produce select(clear, score, score+masked). Vector predicates undergo
  exactly the same extract/shuffle/reshape/transpose/slice as the data.
  Select remains select: this does not claim hardware skips inactive-lane fadds,
  and it does not introduce per-lane divergent branches.

Views: static vector.extract, vector.shuffle, vector.shape_cast,
vector.broadcast, vector.transpose, vector.extract_strided_slice and
vector.from_elements register repacking.
GPU wave shuffles are NOT lane views. Every consumed clear lane must fold to
positive zero. Dynamic extraction, poison/undef sources, nonzero/negative-zero
clear values, escaping or unsupported consumers and scalable vectors fall back.

All consumers must lie after the mask in the same block. No loops/regions,
loads, calls, DMA or barriers may be crossed. Scores must dominate the mask,
or consist solely of duplicable static register views/constants of values
that do. Score arithmetic and memory reads are never moved. Add operand order
and FP attributes are retained. Analysis is bounded and reapplication idempotent.

approx explicitly permits signed-zero differences for matched mask additions:
  -0 + +0 -> +0, but bypass returns -0.
It does not imply finite scores, nnan, ninf or global fast-math. StrictFP,
OptNone and explicit denormal FP environments are skipped, even in approx mode.
NaN results remain NaNs, but bypassing arithmetic does not promise identical
NaN payloads, signaling/quiet bits or floating exception behavior. CPU tests
compare NaNs by classification and permit signed-zero differences; they are
not evidence of bitwise equivalence for those cases.
The stronger causal-mask=finite contract remains independent.

MaskAddBypassFragment and MaskAddBypassThread diagnostics report
function/groups/additions. Fragment counts are scalar elements per thread;
Thread counts are arith.addf operations (which may be vector operations).
Neither counter measures dynamic executions. TuneAttention's mask_add_bypass_reports includes stage="fragment"
or "thread". Standalone users run the FragmentIR pass before
convert-fragment-to-thread, and the fallback pass after thread SSA cleanup.
Both passes honor off/approx and share the same FP-environment protection.
Unfused, multi-use and nonuniform patterns remain handled by the fallback.

Build:
  cmake --build 3rd/deepgengraph/build --target ThreadTilingTest MyTest LegacyLLVMExportTest -j 4
Tests:
  python3 3rd/deepgengraph/test/unittest/check_mask_add_bypass.py 3rd/deepgengraph/build/test/ThreadTilingTest
  python3 3rd/deepgengraph/test/unittest/check_early_mask_add_bypass.py 3rd/deepgengraph/build/test/ThreadTilingTest
  python3 -m unittest script.test_tune_attention

The original mask_finite_pk_rtz preset keeps uniform-masks=false and already
has finite-score selects, so zero bypass matches are expected. The adapted
command in generated_ir/fragment_mask_add_bypass enables uniform-masks and
fuse-mask-add and explicitly exports with --llvm-opt=3. It retains causal-mask=finite
for mixed tiles. Earlier thread-only outputs remain in generated_ir/thread_mask_add_bypass.
