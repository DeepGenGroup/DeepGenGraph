Compiler-pass export of the measured cross-BN K0 + FP32 sum-anchor strategy

Main IR: cross_bn_k0.ll
Configuration: cross_bn_k0.launch.json
Control: control.ll (byte-identical to k_tail_prefetch4.ll)

Enable in MyTest or LegacyLLVMExportTest:
  --cross-bn-k0=1
The option defaults to 0. In TuneAttention.py use --set cross-bn-k0=1;
--sweep 'cross-bn-k0=0;1' is also supported. Keep the other tested settings,
including k-tail-prefetch=4, V buffering and softmax-fragments=pv.

Implementation: PrefetchNextKPass in PrefetchNextK.cpp/.h. It proves the LDS
handoffs and disjoint footprints, primes K0(0), issues K0(n+1) before final PV
under next<bound, and fixes the exact denominator additions at their fragments
with tied-register identity constraints. Unknown shapes/control/memory are
skipped conservatively. No source SHA, SSA name or tensor shape is hard-coded
in the C++ pass. The Python artifact generator fingerprints comparison inputs.

The generated cross_bn_k0.ll instruction graph matches the GPU-measured
../noncausal_cross_bn_k0_lifetimes/k0_sum_anchor.ll, including arithmetic,
operands, order, DMA and control flow, after normalizing SSA names, debug
metadata and attribute numbers. Its supplied measurement was 1.663345 ms;
this newly exported file has not itself been benchmarked on the GPU.

Regenerate and verify:
  python3 generated_ir/noncausal_cross_bn_k0_pass/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_cross_bn_k0_pass/verify.py

manifest.json records hashes, comparison status and pass diagnostics.
validation.json records actual-IR numerical/epoch checks and MyTest results.
mytest_smoke/ contains fresh MLIR exports, exact commands and logs.
The CPU model is not a simulation of independent wave scheduling or register
allocation. See the repository's PairedLDS.md for matching limits and checks.

GPU comparison:
  bash generated_ir/noncausal_cross_bn_k0_pass/run_gpu.sh

Default: control, cross_bn_k0, control. Runs have separate build, HSACO,
assembly, resource-check and performance/correctness results under runs/.
The script checks scratch/spill before benchmarking with --non-causal.
