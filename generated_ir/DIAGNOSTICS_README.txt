Attention diagnostics deployment bundle

Current status: the user reports PASS for the DMA-preserving address guard:
0/16777216 full-output mismatches, max error 0.00390625, mean error
3.0495710234390572e-05. All five prefix probes and the independent prefix
reference pass. HSACO SHA256:
7d023046e41d06f5a65350c07f970817382647d27b132d132ac4f49390710c02
Use this guarded kernel for the next benchmark; timings have not been supplied.
The remaining sections retain the reproduction steps and earlier comparisons.

Extract into a new directory on the GPU machine. Run commands from that directory.
Requires the GPU environment's existing Python, NumPy, PyTorch and HIP runtime.

1. Diagnose the existing failing HSACO without rebuilding:

python kernelLauncher.py --hsaco /workspace/deepgengraph/ktest.hsaco --kernel Attn_p2 --grid 32,32,1 --block 256,1,1 --k-layout bhsd --warmup 0 --repeat 1 --diagnose-attention

Adjust the HSACO path if necessary. Keep attentionDiagnostics.py next to
kernelLauncher.py, including if you rename the launcher to launchKernel.py.
The original numerical failure still returns exit status 1 after diagnostics.
The snapshot is written beside the HSACO as <hsaco>.prefix.npz.
Probe outputs, independent references and the first two heads of K/V are also
saved as <hsaco>.probes.npz. Failure logs report NaN/Inf counts, their locations,
affected rows and finite-only errors without changing the allclose verdict.
These single-launch timings are diagnostic, not performance measurements.

2. Build and compare the original IR and three controlled variants:

bash RunCorrectnessDiagnostics.sh generated_ir/test_input_spidD_best_k5_hybrid_syncfix.ll 3

Use the same optimization level as the original failing build (replace 3).
GenHsaco.sh accepts LLVM_DIR, ROCM_BC and GPU_ARCH environment overrides.
Default tools: /workspace/deepgengraph/dtk/aillvm/bin; bitcode: /opt/dtk/amdgcn/bitcode;
architecture: gfx936. Each variant gets its own directory under
generated_ir/correctness_diagnostics with build.log, run.log, HSACO and assembly.
summary.tsv lists the per-case status. The script finishes even if cases fail;
check summary.tsv and run.log for the actual correctness verdicts.

3. Optional CPU LLVM-model comparison against the captured GPU inputs:

python 3rd/deepgengraph/test/check_attention_llvm_numerics.py generated_ir/test_input_spidD_best_k5_hybrid_syncfix.ll --snapshot /workspace/deepgengraph/ktest.hsaco.prefix.npz

The model assumes repository MMAC semantics and immediate DMA completion;
it does not establish GPU correctness or simulate backend/hardware hazards.
Hardware results are specific to the tested input and artifact. The guarded
candidate and sync_copy now have user-reported correctness passes.

Earlier original-kernel diagnostics:
SDPA matches the independent reference; row 1 fits a V0/V1 mixture with the
wrong weight (0.21094081 vs 0.59050679). A positive score-scale change alone
cannot cross p=0.5. Zero Q has 782 mismatches among 32768 prefix elements;
the old NaN aggregate does not reveal how many mismatches are nonfinite.
These observations do not identify DMA or MMAC as the root cause. Compare
the full original-output correctness verdict for all four variants, not just
the zero-Q probe. Provide summary.tsv, per-case run.log, build.log and snapshots.

Earlier four-way hardware A/B report from the user:
reference=EXIT_1, sync_copy=PASS, mmac_wait=EXIT_1,
sync_copy_mmac_wait=PASS. This supports using sync_copy as the current tested
baseline and investigating the async transfer path. It does not distinguish
address semantics, completion/synchronization, or backend code generation.
PASS reflects the original random-input comparison; probe verdicts must still
be read from run.log. Remote build hashes and timings have not been supplied.

Use the already-built correctness_diagnostics/sync_copy/kernel.hsaco for the
next benchmark (warmup=10, repeat=50), with grid=32,32,1 and block=256,1,1.
The named fallback IR is test_input_spidD_best_k5_hybrid_sync_copy.ll.

To export only the same synchronous packet transformation from an existing IR:

python 3rd/deepgengraph/test/make_attention_diagnostic_variants.py generated_ir/test_input_spidD_best_k5_hybrid_syncfix.ll --sync-output generated_ir/test_input_spidD_best_k5_hybrid_sync_copy.ll

In a full repository checkout with the compiler build dependencies installed:

FRISK_GM_TO_SHM_MODE=hybrid FRISK_GM_TO_SHM_ALIGNMENT=4 FRISK_SYNC_DMA_PACKETS=1 bash RunFriskSHell.sh "$PWD/3rd/deepgengraph/test/test_input_spidD.mlir" generated_ir/test_input_spidD_best_k5_hybrid_sync_copy.ll 3.3 34816 approx 256 4,1 1 1 1 0

FRISK_SYNC_DMA_PACKETS is an opt-in post-export substitution in RunFriskSHell.sh.
The direct MyTest executable does not read this variable. This differs from
FRISK_GM_TO_SHM_MODE=sync, which changes the earlier compiler pipeline.
It retains asyncmark/wait/barriers and MMAC padding just like the tested variant;
only the DMA calls are replaced. Future IR/configurations still need GPU tests.

DMA-preserving address guard (assembly evidence and subsequent GPU PASS):
The user-supplied failing gfx936 assembly has six Q DMA immediates -48 through
-8, with M0 still equal to the IR's intended LDS wave base. The instruction
immediate affects both source and LDS, so these global-address folds also
shift LDS writes. A local LLVM 24 gfx900 minimal reproducer shows the same
pattern. An opaque tied VGPR identity on each final global pointer eliminates
these immediates in local codegen. The subsequent user gfx936 run passed the
full-output check and all prefix probes, as recorded above.

bash RunDMAAddressGuard.sh generated_ir/test_input_spidD_best_k5_hybrid_syncfix.ll 3

Use the original optimization level. This preserves all 16 DMA calls and the
original mark/wait/barriers/MMAC. It builds in generated_ir/dma_address_guard,
checks 16 zero-immediate DMA instructions in the assembly sidecar, then runs
the original-input comparison and probes. Outputs: build.log, address-check.log,
run.log, kernel.ll, kernel.hsaco, kernel.hsaco.amdgcn and both NPZ snapshots.
The named candidate IR is test_input_spidD_best_k5_hybrid_dma_guard.ll.
The already-passing sync_copy remains available as a fallback.

The full repository's MyTest and common LLVM export paths now automatically
protect final DMA source addresses. This requires rebuilding the compiler;
the old failing syncfix.ll is deliberately retained as a reproduction artifact.
The standalone script remains useful for previously exported, unguarded IR.
