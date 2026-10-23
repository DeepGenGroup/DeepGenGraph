Generic LDS read canonicalization A/B

baseline.ll is a byte-identical copy of the measured 1.779047 ms source:
  generated_ir/group8_reverse_k_gep/group8_reverse_k128_gep.ll
optimized.ll applies the generic CanonicalizeLDSGEPsPass and dead-code cleanup.
The historical time belongs to the source; neither new artifact has been GPU timed.

Both use BHSD=(1,32,4096,128), FP16 causal attention, contiguous BHSD K,
grid=(32,32,1), block=(256,1,1), dynamic shared=0, static LDS=16384 bytes.
Use baseline.launch.json / optimized.launch.json respectively.

Changes: 68 proven constant byte offsets; 16 ordinary LDS loads strengthened
from align 2 to align 8. All 144 ordinary LDS loads now have align 8.
DMA128 sites=24, block barriers=16, vmcnt(0) sites=12, MMAC sites=256 in both.
No buffer roles, SSA names, attention dimensions or Q/K/V identities are used
by the transformation. Ordinary nonvolatile/nonatomic address-space-3 loads
are eligible, with arbitrary scalar/vector types. Unproven cases are retained.
Alignment uses LLVM known bits/contracts plus bounded exact thread-layout proof.
Address rebasing requires a proven constant difference and a dominating base.
The pass neither forces a target instruction nor rewrites matrix-read intrinsics.

Reproduce from repository root:
  cmake --build 3rd/deepgengraph/build --target LegacyLLVMExportTest -j 2
  python script/optimize_lds_reads.py \
    generated_ir/group8_reverse_k_gep/group8_reverse_k128_gep.ll \
    --output-dir generated_ir/generic_lds_reads \
    --config generated_ir/bm_bn_sweep/bm128_bn128.launch.json
  python 3rd/deepgengraph/test/check_lds_read_canonicalization.py
  OPENBLAS_NUM_THREADS=1 python generated_ir/generic_lds_reads/validate.py

The generic export script accepts any LLVM input, an output directory and an
optional launch JSON. --export-tool and --llvm-bin override tool locations.
Its baseline remains byte-identical to the input. The independent
--canonicalize-lds-geps-only mode avoids unrelated MMAC/shared-memory export
rewrites. The existing LLVM-22 text helper prints exact FP constants.

GPU compile and benchmark in the target DTK/gfx936 environment:
  mkdir -p generated_ir/generic_lds_reads/build-baseline
  cd generated_ir/generic_lds_reads/build-baseline
  bash ../../../GenHsaco.sh ../baseline.ll 3 kernel.hsaco
  python ../../../launchKernel.py --hsaco kernel.hsaco \
    --config ../baseline.launch.json --warmup 50 --repeat 200 --json-output result.json

For optimized.ll use build-optimized, ../optimized.ll and ../optimized.launch.json.
Use the same GPU_ARCH, LLVM_DIR, ROCM_BC and compile options for both versions.
Alternate repeated measurements and rank by program median, not profiler time.

Validation: LLVM assembly/O3 verifier; role-independent positive/negative tests;
CPU delayed-DMA execution at q0/q7/q8/q31 over two heads and two seeds, bitwise
equal outputs, identical ordered memory-address traces, checked load alignment,
unchanged dynamic barriers/waits/MMAC and no pending DMA. See validation.json.
The CPU model does not simulate independent wave timing or gfx936 code generation.
No GPU speedup or final instruction selection is claimed before target testing.
