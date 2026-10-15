STATUS UPDATE: TARGET COMPILATION FAILED; these are diagnostic artifacts.
REJECTED ON TARGET: user reported useKPack failure with gfx936 DTK
hcu global_load_*_lds not support on the target
The earlier host checks below do not establish target instruction support.
Do not treat the launch instructions below as a validated runnable release.
Next: run script/probe_gfx936_dma.py with the exact target DTK.

gfx936 HCU 128-bit DMA exports (replaces the earlier kpack_dma128 export)

The supplied IntrinsicsHCU.log defines HCUGlobalLoadLDS at lines 425-435:
  void llvm.hcu.global.load.dwordx4.lds(
    ptr addrspace(1) global, ptr addrspace(3) LDS,
    i32 offset, i32 aux/cpol, i32 bps)
The exact name is also confirmed by IntrinsicsHCU.h.log, line 182.
The correct arguments here are (global, LDS, 0, 0, 0).
Width is encoded by dwordx4; the old generic intrinsic's size=16 argument
must NOT be carried over as the HCU offset.

Upstream IntrinsicsAMDGPU.td documents generic 12/16-byte load-to-LDS sizes
for gfx950. Passing size=16 to that generic intrinsic was not sufficient
to select the gfx936 vendor instruction. This export uses the HCU intrinsic
at the final LLVM export boundary, after upstream optimization.

All Q/K/V transfers remain asynchronous 16-byte DMA with bps=0.
The complete source-address guards, wave-uniform LDS bases, asyncmark,
wait.asyncmark and block barriers remain in place. No synchronous fallback
or staging tile is introduced. The computation and address bodies are
identical to the previous exports after normalizing the intrinsic ABI.

             BK   copy threads   launch threads   static LDS   HCU call sites
useKPack     64       256              256           32768 B        16
noKPack      32       128              256           16384 B        24

Both retain best_k5: pipeline=0, reorder=1, packing=1, layout=3.3,
carrier=34816, math=approx, threads=256, waves=4,1, lifetimes=1,
uniform masks=1, prefetch=1, fuse mask/add=0, LDS reuse=0.
FRISK_GM_TO_SHM_MODE=async, FRISK_GM_TO_SHM_ALIGNMENT=16, kpack=2.
Input: unchanged 3rd/deepgengraph/test/test_input_spidD.mlir.

Regenerate from repository root:
  cmake --build 3rd/deepgengraph/build --target MyTest -j 2
  bash script/generate_attention_dma_pack.sh

Build and run on the target machine, from repository root:
  for mode in useKPack noKPack; do
    bash GenHsaco.sh "generated_ir/kpack_dma128_hcu/$mode.ll" 3 "$mode.hsaco"
    python kernelLauncher.py --hsaco "$mode.hsaco" --kernel Attn_p2 \
      --grid 32,32,1 --block 256,1,1 --shared-mem 0 --k-layout bhsd \
      --warmup 10 --repeat 50
  done
Set LLVM_DIR and ROCM_BC to the target DTK installation if needed.
GPU_ARCH defaults to gfx936. Both variants still launch 256 threads;
noKPack's 128 threads are an internal DMA-only participation limit.

Optional minimal instruction-selection probe on the target machine:
  "$LLVM_DIR/llc" -mtriple=amdgcn-amd-amdhsa -mcpu=gfx936 -O3 \
    generated_ir/kpack_dma128_hcu/hcu_dma_probe.ll -o hcu_dma_probe.s
The assembly must contain a dwordx4 global-to-LDS instruction, not a call
to an external llvm.hcu symbol. The probe excludes the attention/MMAC body.
An automated host/optional DTK regression is available as:
  python3 3rd/deepgengraph/test/check_hcu_dma_export.py \
    --target-llvm-bin "$LLVM_DIR"

Local validation:
  PASS: rebuilt MyTest and LegacyLLVMExportTest.
  PASS: HCU parameter remapping (including nonzero offset/aux test),
        narrow-copy preservation, guards, synchronization and idempotence.
  PASS: host LLVM O3 retains the HCU calls and address guards.
  PASS: llvm-as / LLVM verifier for both final exports.
  PASS: both LLVM bodies numerically simulated for (head,block) =
        (0,0), (0,1), (1,7), (1,31), at atol=rtol=0.01.
  PASS: old 32-bit DMA guard regression, including gfx900 code generation.
  FAILED remotely: gfx936/HCU instruction selection (user report above).
  NOT RUN locally: target instruction selection; no gfx936-capable DTK.
  NOT RUN: GPU execution.

validation.json records final hashes and the exact validation boundaries.

Feature investigation (targetfeature.log, line 239 after UTF-16 decoding):
  glb2lds-dword-x2x4 - support global_load_dwordx2/x4 to lds.
This is an available feature, not proof that gfx936 enables or supports it.
The default-feature remote probes select 4-byte generic async/legacy DMA;
8/16-byte generic async and every tested HCU width are rejected.
Compile-only followup with the exact target DTK:
  python3 script/probe_gfx936_dma.py \
    --llvm-bin /workspace/deepgengraph/dtk/aillvm/bin \
    --mattr=+glb2lds-dword-x2x4 --out dma_probe_features
This also checks the HCU dwordx4 path with asyncmark/wait.asyncmark.
The explicit feature probe ALSO FAILED remotely with the same HCU error.
GenHsaco.sh accepts the explicit
GPU_FEATURES=+glb2lds-dword-x2x4 environment override and passes it to opt,
object generation and assembly generation. It does not enable it by default.
Existing HCU IR does not need to be regenerated to test this compiler flag.

Compatibility export: ../kpack_dma32_compat uses four 4B generic async
instructions per logical 16B packet; it is not native 128-bit DMA.
