test_input_spidD: best_k5 + native 128-bit buffer-to-LDS inline assembly

Source clue: example.cpp, lines 58-63, emits:
  s_mov_b32 m0, <uniform LDS base>
  buffer_load_dwordx4 <VGPR byte offset>, <SGPRx4 resource>, 0 offen lds
This is the BUFFER path. The rejected llvm.hcu.global.load.*.lds intrinsics
use a different GLOBAL path. The user successfully compiled the buffer asm
opcode to an object with the target gfx936 DTK and default target features.

The initial probe used an M0 clobber and triggered a reserved-register warning.
These exports use an explicit i32 output constraint ={m0}, with memory clobber,
sideeffect and convergent; local object encoding confirms no such warning.
The revised constraint and full attention kernels still require target testing.
No extra target feature flag is required by the successful original probe.

These IR files issue ONE native 16-byte buffer_load_dwordx4 ... lds per logical
packet. No four-dword split, payload VGPR staging, or synchronous copy fallback.

             BK   copy threads   launch threads   static LDS   DMA call sites
useKPack     64       256              256           32768 B           16
noKPack      32       128              256           16384 B           24
kpack=2. Both retain the existing best_k5 settings; performance is not measured.
useKPack merges adjacent tiles and consumes BK=64; noKPack keeps BK=32.

Address handling:
  resource base = the original uniform Q/K/V kernel argument
  resource range = actual tensor size, 33554432 bytes for this input
  voffset = each lane's existing physical global pointer - resource base
  LDS base = the existing readfirstlane-selected wave base
  destination byte address = LDS base + physical lane * 16
Do not copy example.cpp's pointer - threadIdx.x*N shortcut into this layout:
our lane source pointers are not a simple contiguous sequence.
Descriptor word 3 is 0x00020000, the local CK gfx9 raw byte-addressed format;
stride and swizzle are zero. The included tl_templates/hcu descriptor header
was not provided with example.cpp. GPU testing must confirm this target format.
Input tensors must fit below 4GiB and all generated source offsets must be
within the descriptor range; the numerical model checks these conditions.

Completion:
  Inline asm hides the DMA issue from asyncmark bookkeeping. This export
  removes the old commit calls, replaces wait.asyncmark(0) with explicit
  sideeffect asm "s_waitcnt vmcnt(0)", and retains the publication barriers.
  Nonzero async group waits are rejected. Issuing DMA remains asynchronous;
  waits occur at the original readiness points, not after every DMA issue.

Regenerate from repository root:
  cmake --build 3rd/deepgengraph/build --target MyTest -j 2
  DMA_EXPORT=asm128 bash script/generate_attention_dma_pack.sh

Exact MyTest invocation for MODE = useKPack / noKPack:
  FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=16 \
    3rd/deepgengraph/build/test/MyTest \
    3rd/deepgengraph/test/test_input_spidD.mlir 0 OUTPUT.ll \
    1 1 3.3 34816 approx 256 4,1 1 1 1 0 0 MODE asm128
The script's default remains split32; select asm128 explicitly as above.

Build and run on the target machine, from repository root:
  for mode in useKPack noKPack; do
    GPU_ARCH=gfx936 GPU_FEATURES= \
      bash GenHsaco.sh "generated_ir/kpack_dma128_asm/$mode.ll" 3 "$mode.hsaco"
    python kernelLauncher.py --hsaco "$mode.hsaco" --kernel Attn_p2 \
      --grid 32,32,1 --block 256,1,1 --shared-mem 0 --k-layout bhsd \
      --warmup 10 --repeat 50
  done
LLVM_DIR defaults to /workspace/deepgengraph/dtk/aillvm/bin.
Set ROCM_BC to the matching DTK device bitcode directory if needed.
Both variants launch 256 threads. shared-mem=0 means no dynamic LDS.
Q/K/V/O ABI: contiguous BHSD FP16, B=1 H=32 S=4096 D=128.

Optional standalone test of the exact export's descriptor and M0 constraints:
  /workspace/deepgengraph/dtk/aillvm/bin/llc \
    -mtriple=amdgcn-amd-amdhsa -mcpu=gfx936 -O3 -filetype=obj \
    -amdhsa-code-object-version=4 \
    generated_ir/kpack_dma128_asm/buffer_asm_copy_probe.ll -o buffer_asm_probe.o
  /workspace/deepgengraph/dtk/aillvm/bin/llvm-objdump \
    -d --mcpu=gfx936 buffer_asm_probe.o
This probe is compile-only and has a different tensor ABI; do not run it with
the attention launcher. Object generation matters: printing .s alone may not
validate inline-asm instruction encoding.

validation.json and *.log record the validation boundaries and hashes.
CPU models execute addresses/math; they do not model GPU DMA timing or hazards.

Additional endpoint validation: both kernels passed head=31, block=31 with
full 32MiB tensors and descriptor bounds checks (last_head_bounds.log).
