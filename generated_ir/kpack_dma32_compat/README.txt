test_input_spidD: best_k5 packed layouts, 4-byte async DMA compatibility export

Status: the user's gfx936 DTK selects generic 4-byte async DMA and asyncmark/
wait.asyncmark. Native generic 8/16-byte DMA and HCU global-to-LDS are rejected.
Explicit +glb2lds-dword-x2x4 also failed. These exports remove those rejected
intrinsics. Full target code generation and GPU execution still require testing.

This is NOT native 128-bit DMA. The logical layout keeps 16 bytes per lane and
kpack=2 so the approved tile merge and the two layout strategies remain intact.
Each logical packet is implemented by four 4-byte async instructions. Source
addresses are exchanged with ds_bpermute; payload data still goes directly
from global memory to LDS. No synchronous payload load/store or staging tile
is introduced. Extra instructions/address exchange can reduce performance.

                 BK   copy threads   launch threads   static LDS   DMA calls
useKPack         64       256              256           32768 B        64
noKPack          32       128              256           16384 B        96
DMA calls above are static call sites, not dynamic counts over kernel loops.
Both keep the existing best_k5 tuning; neither new variant is GPU-benchmarked.

Correctness of the split (lane is the physical wave64 lane, part=0..3):
  source lane = 16*part + floor(lane/4)
  source byte offset within the original packet = 4*(lane mod 4)
  instruction LDS base = original wave base + 256*part
  actual LDS write = instruction base + 4*lane
Thus every original base + 16*source_lane + word_offset byte is preserved.
Only full waves may participate. Packed-layout planning rejects partial-wave
packet tails. Both modes still execute block-collective waits and barriers.

Generate from repository root:
  cmake --build 3rd/deepgengraph/build --target MyTest -j 2
  bash script/generate_attention_dma_pack.sh

Exact invocation, MODE is useKPack or noKPack:
  FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
    3rd/deepgengraph/build/test/MyTest \
    3rd/deepgengraph/test/test_input_spidD.mlir 0 OUTPUT.ll \
    1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 MODE split32
New last argument is dma-export: split32 (default) or experimental hcu128.
The hcu128 mode is retained for other backends; it failed on the supplied DTK.

Build and run on the target machine, from repository root:
  for mode in useKPack noKPack; do
    GPU_ARCH=gfx936 GPU_FEATURES= \
      bash GenHsaco.sh "generated_ir/kpack_dma32_compat/$mode.ll" 3 "$mode.hsaco"
    python kernelLauncher.py --hsaco "$mode.hsaco" --kernel Attn_p2 \
      --grid 32,32,1 --block 256,1,1 --shared-mem 0 --k-layout bhsd \
      --warmup 10 --repeat 50
  done
LLVM_DIR defaults to /workspace/deepgengraph/dtk/aillvm/bin; set ROCM_BC to
this DTK's device bitcode directory when necessary.
Launch block=256,1,1 for BOTH modes. The noKPack copy participation limit
of 128 is internal. shared-mem=0 means no dynamic LDS; static LDS is in IR.
Q/K/V/O: contiguous BHSD FP16, B=1 H=32 S=4096 D=128, input fixture unchanged.

Optional isolated target compile before compiling full attention:
  /workspace/deepgengraph/dtk/aillvm/bin/llc \
    -mtriple=amdgcn-amd-amdhsa -mcpu=gfx936 -O3 -filetype=asm \
    -amdhsa-code-object-version=4 \
    generated_ir/kpack_dma32_compat/split_dma_probe.ll -o split_dma_probe.s
The probe compiles the real splitting sequence with readback, no MMAC.
Do not execute it with kernelLauncher: it has a separate test-only tensor ABI.

Local validation results and SHA256 hashes are recorded in validation.json.
CPU models check values and addresses, not GPU timing/hazards/performance.
