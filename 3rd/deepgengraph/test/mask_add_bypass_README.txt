ThreadIR mask-add bypass
=======================

Implementation: lib/Conversion/ConvertToLLVM/ThreadMaskAddBypass.cpp
Pass: --frisk-bypass-mask-adds=mode=off|approx
MyTest/TuneAttention: --mask-add-bypass=off|approx (default off)

The pass runs after thread loop unrolling, scalar replacement and register SSA
cleanup, before firstLowering and LLVM dialect conversion. The LLVM FunctionPass,
its wrapper and its production/legacy export entry points have been removed.
legal is rejected explicitly; no other pass's legal mode or FP policy changes.
MyTest still requires math=approx when mask-add-bypass=approx is requested.

Supported shapes:
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

MaskAddBypassThread diagnostics report function/groups/additions in thread IR.
TuneAttention's mask_add_bypass_reports includes stage="thread". Base/Fragment
lowering no longer owns bypass_mode or a separate bypass option: standalone
users run the ThreadIR pass explicitly after lowering the fragments.

Build:
  cmake --build 3rd/deepgengraph/build --target ThreadTilingTest MyTest LegacyLLVMExportTest -j 4
Tests:
  python3 3rd/deepgengraph/test/check_mask_add_bypass.py 3rd/deepgengraph/build/test/ThreadTilingTest
  python3 3rd/deepgengraph/test/check_early_mask_add_bypass.py 3rd/deepgengraph/build/test/ThreadTilingTest
  python3 -m unittest script.test_tune_attention

The pmc mask_finite_pk_rtz preset keeps uniform-masks=false and already has
finite-score selects, so zero bypass matches are expected. Test outputs live
in generated_ir/thread_mask_add_bypass with command, launch config and hashes.
