Stable online softmax: native PV accumulator optimization (2026-10-23)

Primary artifact: kernel.ll (BM=BN=128, BK=32, shape 1x32x4096x128)
Regression artifact: kernel_bn64.ll (BM=128, BN=64, BK=32)
Launch configuration for primary artifact: launch.json

Compiler changes:
- FriskBaseToTheadlevelIR_plan2.cpp: sinkLoopLayoutConversions now handles
  O_next = alpha * O + PV. Keep O in the native PV C layout through the loop;
  convert only at the loop exit. Row alpha lane/register mapping is proved
  unchanged; unsupported scale mappings retain the existing fallback.
- FuseMMACAccumulator.cpp: permit pure vector construction/extraction and
  unconstrained FMul scheduling to seed MMAC with alpha * O in approximate mode.
  Loads/calls are not moved; StrictFP functions and unsupported seeds retain
  the existing fallback. This reassociates floating-point accumulation.

Tuning: group=4, reverse=1, frag-prefetch=0, retain-fragment-words=32,
K swizzle=8:6, threads=256, waves=4,1, QK/VO tile=32, pipeline=0,
async alignment=16, DMA=asm128, simplify-loop-control=1, coalesce-waits=0.
Compared with best grid-000001, all tuning fields match except input query/head
axis names (input parallel_map order differs). The generated workgroup mapping
IR was checked identical. See parameter-comparison.json and config.json.

Verified final primary artifact:
- BN-loop ds.bpermute sites: 72 -> 8 (4 max + 4 sum).
- Loop-exit ds.bpermute sites: 0 -> 64 (O conversion once).
- All 16 PV MMAC chains use the scaled accumulator; no separate PV+O fadd.
- Static LDS allocation: 16384 bytes.
- Estimated shared operations: 9,224,192 -> 5,160,960 (-44.05%).
  This assumes unchanged QKV LDS codegen and the supplied causal launch.
  It is an instruction-accounting estimate, NOT a measured GPU counter.

Validation:
- MyTest / ThreadTilingTest / LegacyLLVMExportTest builds passed.
- check_wmma_layout_shuffle.py, check_plan2_thread_tiles.py,
  check_register_math.py passed (logs in this directory).
- llvm-as and opt default<O3>,verify passed for both final artifacts.
- Actual LLVM IR CPU emulation: six cases each for BN128 and BN64,
  including random/positive overflow/negative underflow/changing max;
  finite outputs, complete Q loaded once, delayed DMA checks all passed.
  Maximum absolute error among these cases: 0.001061586.
- GPU correctness, timing, VGPR/scratch and PMC of the new artifacts are pending.
  CPU MMAC emulation does not establish hardware rounding/performance.

Reproduce IR (from repository root):
  cmake --build 3rd/deepgengraph/build --target MyTest -j 2
  python3 TuneAttention.py \
    --input generated_ir/online_softmax/test_input_1x32x4096x128-128-128-32-online.mlir \
    --case retained32-baseline --set workgroup-group-size=4 --set frag-prefetch=0 \
    --llvm-as /data2/xsl/install/bin/llvm-as \
    --output-dir generated_ir/online_softmax_native_acc/reproduce

GPU test (on the existing DTK machine, with LLVM_DIR/ROCM_BC configured):
  ./GenHsaco.sh generated_ir/online_softmax_native_acc/kernel.ll 3 \
    generated_ir/online_softmax_native_acc/kernel.hsaco
  python3 launchKernel.py \
    --hsaco generated_ir/online_softmax_native_acc/kernel.hsaco \
    --config generated_ir/online_softmax_native_acc/launch.json \
    --warmup 10 --repeat 50 \
    --json-output generated_ir/online_softmax_native_acc/gpu-result.json

Final export run:
  final-export/run-20261023-182556-l6_z6lbl
SHA256 (primary): 2ebae1fa1f4366d3c4072d0aec4bd2fe2bcec623cc667bf154a409de0d358b1e
