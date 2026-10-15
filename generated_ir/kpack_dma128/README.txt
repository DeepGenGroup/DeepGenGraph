STATUS UPDATE: TARGET COMPILATION FAILED; these are diagnostic artifacts.
REJECTED ON TARGET: user reported generic wide async DMA instruction-selection failure
Cannot select: intrinsic %llvm.amdgcn.global.load.async.lds
The earlier host checks below do not establish target instruction support.
Do not treat the launch instructions below as a validated runnable release.
Next: run script/probe_gfx936_dma.py with the exact target DTK.

test_input_spidD: best_k5 + 128-bit asynchronous DMA

Input: 3rd/deepgengraph/test/test_input_spidD.mlir (unchanged)
Target: repository HWSpecification BW1000 / gfx936, wave64.
Both exports use 16 bytes = 8 FP16 per lane for every Q/K/V DMA.
MMAC input is 8 bytes = 4 FP16 per lane, so kpack = 2.
No staging tile or synchronous GM->LDS fallback is present.
Complete source-address guards are retained on all DMA calls.

             BK   copy threads   compute threads   static LDS   DMA sites
useKPack     64       256              256           32768 B        16
noKPack      32       128              256           16384 B        24

useKPack merges adjacent D tiles before lowering: Q/V/O become 128x64,
K becomes 64x128, and both the QK reduction and output-D loops step by 64.
BM=BN=128 and the external tensor ABI remain unchanged. The additional data
is consumed by the enlarged computation. This MyTest coarsening helper is
deliberately restricted to the supplied split-D attention fixture.

noKPack keeps the original tile sizes. Threads 0..127 issue DMA packets;
all 256 threads execute completion/publication and participate in computation.
Copy lanes follow the inverse ShmDataIODesc map on the contiguous global axis;
the same physical mapping rewrites every LDS consumer address.

Regenerate from the repository root after building MyTest:
  cmake --build 3rd/deepgengraph/build --target MyTest -j 2
  bash script/generate_attention_dma_pack.sh

Exact MyTest invocation (run once for each MODE = useKPack / noKPack):
  FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
    3rd/deepgengraph/build/test/MyTest \
    3rd/deepgengraph/test/test_input_spidD.mlir 0 OUTPUT.ll \
    1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 MODE

Positional settings: pipeline=0, reorder=1, shared packing=1, layout=3.3,
carrier=34816 words/block, math=approx, threads=256, waves=4,1,
register lifetimes=1, uniform masks=1, packet prefetch=1, fuse mask/add=0,
LDS reuse budget=0, DMA pack mode=MODE.
These are the existing best_k5 settings; these new variants have not been
benchmarked on a GPU here. Input allocation and tile origins must be at
least 16-byte aligned (the supplied launcher uses contiguous torch tensors).

On the target machine, from the repository root:
  for mode in useKPack noKPack; do
    bash GenHsaco.sh "generated_ir/kpack_dma128/$mode.ll" 3 "$mode.hsaco"
    python kernelLauncher.py --hsaco "$mode.hsaco" --kernel Attn_p2 \
      --grid 32,32,1 --block 256,1,1 --shared-mem 0 --k-layout bhsd \
      --warmup 10 --repeat 50
  done

GenHsaco.sh accepts LLVM_DIR and ROCM_BC for the target DTK installation;
its default GPU_ARCH is gfx936. Use the target machine's gfx936-capable tools.
Both versions require block=256,1,1, including noKPack. Shared memory above
is static, already encoded in the kernel; --shared-mem is therefore 0.
The launcher supplies contiguous BHSD FP16 Q/K/V/O, B=1,H=32,S=4096,D=128.

Validation performed locally:
  - MyTest and ThreadTilingTest rebuilt successfully.
  - llvm-as and opt -passes=verify passed for both final .ll files.
  - Final IR inspected: all DMA widths 16, zero immediate offsets, source
    address guards, expected LDS sizes, no staging or synchronous fallback.
  - check_full_width_dma.py: actual lowered packet addresses executed on CPU,
    including reduced copy waves, partial tails, transposed source layouts,
    memory guards and rejection of invalid widths/alignment/wave counts.
  - check_frisk_async_copy.py and check_async_layout_integration.py passed.
  - Each final LLVM body executed by check_attention_llvm_numerics.py for
    (head,block)=(0,0),(0,1),(1,7),(1,31); all passed at atol=rtol=0.01.
    Largest absolute error against the dense reference: 0.0010877848.

CPU numerical execution models immediate DMA completion. GPU code generation,
asynchronous hazards, numerical correctness and performance require your
target-machine run. validation.json contains hashes and structural results.
