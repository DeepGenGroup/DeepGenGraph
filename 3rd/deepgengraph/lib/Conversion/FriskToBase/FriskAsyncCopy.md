# Frisk GM → SHM copy and buffer readiness

`frisk.async_copy_gm_to_shm` describes a **block tile** copy. It does not
produce a register tile or imply completion. `frisk.wait_buffer_ready` names
the shared tile that must be complete and visible to the whole block:

```mlir
frisk.async_copy_gm_to_shm %gm to %shm {alignment = 4 : i64}
    : memref<64x32xf16, 1>, memref<64x32xf16, 3>
// Independent work can go here.
frisk.wait_buffer_ready %shm : memref<64x32xf16, 3>
// All block threads may now consume the tile.
```

Both operations are block collective: their descriptors and control flow must
be uniform across the block. The source and destination have identical static
shapes and element types. Slices use `frisk.buffer_view` or memref views; no
implicit element conversion or out-of-bounds padding is performed. The source
must remain unchanged and both buffers must remain live until completion.
Before overwriting a previously consumed shared tile, its readers must also
finish: readiness is not a release of those readers.

The `alignment` attribute guarantees alignment of **both tile origins**, not
merely the allocation roots. Its default on explicitly authored async copies
is 4 bytes. Do not use that default for an unaligned slice; specify its actual
alignment (1 or 2 are accepted and select the ordinary-copy fallback).

## Passes and modes

1. `frisk-promote-gm-to-shm-async` runs on block IR after
   `convert-frisk-to-base` (and after attention K layout normalization).
   It rewrites eligible global→shared `frisk.copy` operations to issue+wait,
   preserves destination aliases and slice origins, and expands existing
   per-operation pipeline annotations. It deliberately keeps the original
   completion point. Moving a wait into the consumer stage is a separate
   scheduling choice, not an automatic assertion of independence.
2. `frisk-pipeline-schedule` recognizes async destinations as writes and waits
   as uses of those destinations. It rotates the same slot into copy, wait,
   and consumer. It rejects schemes that schedule a consumer without a matching
   wait between issue and consumption.
3. To disable asynchronous semantics, run **`frisk-lower-async-to-sync` after
   scheduling and before thread tiling**. It selects ordinary `frisk.copy`
   and `frisk.sync_threads_in_block`. Copies carry `frisk.explicit_wait` so
   their thread lowering does not add an earlier publication barrier. Existing
   barrier analysis may remove synchronization it proves unnecessary.
4. Thread tiling leaves async tile operations intact and recognizes their
   destinations as physical storage, including when adapting register-only
   GEMM producer/consumer layouts. Finalization materializes logical views as
   strided memref descriptors. Shared packing rewrites async destinations,
   waits and readers together. It records `frisk.source_map`, an inverse affine
   permutation from flat physical destination elements to logical source
   coordinates. The op verifier checks the permutation is bounded and bijective.
   After layout insertion and hybrid async/sync selection, it invokes
   `frisk-sink-wait-buffer-ready`. This standalone function pass delays waits
   past independent arithmetic and descriptor construction to the first
   dependent access. It understands logical Frisk views/allocations and
   memref views/casts/selects, protects source writes/frees and destination
   reads/writes/frees, and leaves explicit pipeline schedules alone. Motion
   stays within the original block and stops at regions, unknown effects,
   synchronization or another DMA issue/wait. In the split-D hybrid example,
   K readiness moves after Q scaling and immediately before GEMM. Q readiness
   remains before the K DMA because separate DMA completion is still the
   backend policy; this pass does not enable grouped DMA or remove waits.

   To run it independently on logical or physical tile IR:

   ```bash
   build/test/ThreadTilingTest input.mlir --frisk-sink-wait-buffer-ready \
     --verify-each -o scheduled.mlir
   ```
5. After fragment inlining, `frisk-schedule-async-copies` moves issues earlier
   and waits toward consumers within the same block. Pure descriptor setup and
   static owned LDS allocations move with the issue. Alias checks protect source writes and destination
   reads/writes; barriers, other async operations and unknown effects bound
   motion. This happens **before** storage reuse, so pending transfers extend
   the allocation lifetime. A matching same-block, exact-descriptor wait is
   required for async storage reuse; incomplete/conditional completion stays
   conservative. Collective loop backedges and storage handoffs retain barriers
   to protect previous readers. This is intra-iteration scheduling, not
   automatic double buffering or cross-iteration pipelining.
6. `frisk-lower-async-copy` runs at the start of `secondLowering`, after
   fragment/storage optimization and before memref/SCF/LLVM conversion. It
   distributes the tile and lowers to ROCDL. Its `use-async-copy=false` option
   also supports late selection of ordinary scalar copies plus block barriers;
   prefer the earlier `frisk-lower-async-to-sync` for the existing optimized
   synchronous-copy pipeline.

The normal backend now expands strided metadata before final memref-to-LLVM
conversion, including dynamic origins introduced by async tile views.

## gfx936 implementation

`HWSpecification::dataCopyInfo.dmaTransfers` describes the target-confirmed
4-byte and 16-byte per-lane DMA capabilities, including lane stride and
alignment. This is separate from the widths currently emitted below.
`getBankConflictPhases(bytes, Read/Write)` models ordinary LDS b32/b64/b128
instructions as explicit wave-local lane groups; b128 read and write groups
differ. Unknown instruction widths/targets return no model. For read2/write2,
query each address component's width, not the combined instruction width.

`LowerInfo::shmIO` holds an optional `ShmDataIODesc` after allocation-wide
packing negotiation. Its consumer layout snapshot is owned by value, and a
coordinate transpose invalidates the plan. Packing attributes preserve the
parameters across lowering; `PackSharedMemory` reconstructs the descriptor
and uses it for both consumer addresses and the inverse `frisk.source_map`.
Logical/physical maps use element offsets; bank analysis uses byte addresses.

`getShmAccessMap()` composes the full consumer layout with physical packing.
`getGlobalSourceMap()` combines inverse packing with actual source strides and
the view origin. `selectDMATransferBytes()` checks every physical packet for
source contiguity and alignment before choosing a hardware-supported width.
For row-major Q, four-f16 packets usually permit only 4-byte direct DMA;
eight-f16 packets can permit 16 bytes. This query does not change packing or
enable wider emission automatically. `getBankConflictWays()` evaluates actual
per-lane addresses with the target phase groups, including inactive lanes and
read broadcasts; it does not infer conflict freedom from packet size alone.

The asynchronous path requires a one-dimensional block of 64, 128, …, 1024
threads (`thread_num`) and wave64 execution. It issues 4-byte DMA packets:

```text
amdgpu.gather_to_lds async       (temporary memref representation)
  → rocdl.global.load.async.lds  (size = 4)
  → rocdl.asyncmark             (once after each logical tile issue)

frisk.wait_buffer_ready
  → rocdl.wait.asyncmark 0
  → gpu.barrier → rocdl.barrier
```

The conversion to ROCDL uses the standard LLVM memref descriptor helpers but
does not query upstream's chip database for `gfx936`. The LDS operand is the
**wave base**, made uniform with `rocdl.readfirstlane`. Hardware adds
`lane_id * 4`; passing a per-thread LDS pointer would add that offset twice.
Inactive tail lanes issue no load. Source row strides and descriptor offsets
are preserved.

Full tiles with at most sixteen packets per thread emit constant DMA rounds.
They retain one mark per tile, while removing the per-lane 64-bit copy loop and
its execution-mask bookkeeping. Partial waves and larger tiles keep the
bounded loop; no tail load is speculated.
The register-lifetime pass handles DMA source pointers and the address input
to `readfirstlane`, as well as ordinary loads/stores. It rematerializes pure
address expressions at the existing calls without moving or cloning the
convergent operations. Otherwise constant rounds can leave many wave-base
addresses live across the complete attention loop.

The initial wait implementation conservatively drains all previously committed
groups. The named buffer determines IR dependencies and slot selection; it does
not yet select a minimal hardware wait count. Group counts are not packet or
byte counts. This implementation makes no claim of optimal overlap.

Readiness waits do not move across another DMA issue, even when alias analysis
proves the tiles disjoint. Lowering preserves each named wait and its block
barrier. Both the grouped and original/separate-wait hybrid exports fail the
reported gfx936 numerical check. The earlier report that the original passed
was corrected by the user; grouping has NOT been established as the cause.
Grouping remains disabled while the common numerical failure is investigated.
The CPU DMA model verified addresses and assumed completion semantics, but did
not validate target code generation or hardware synchronization. The precise
cause remains unverified.

Scheduling still runs **before** LDS storage reuse. Within a single completion
interval it can overlap independent computation; barriers, other issues/waits,
aliasing accesses and unknown effects remain boundaries. This policy applies
to every tile and thread count, without recognizing attention tensor names.

DMA is selected for 8/16/32-bit elements when each physical 32-bit destination
packet maps to consecutive, aligned source elements, the physical destination
is dense, and both tile origins are at least 4-byte aligned. A bounded exact
address proof uses source strides and the inverse packing map. In particular,
FP16 logical transpose views can use DMA when row-axis LDS packing makes their
physically adjacent elements a dword. Logical row-major copying of such a view
still falls back if its physical source dwords are not contiguous.

An external row-major `[S,D]` K tensor viewed as `[D,S]` correctly has strides
`[1,D]`; this is a zero-copy transpose, not column-major external storage.
Neither `LowerInfo` lane ownership nor the logical GEMM axes should be changed
to pretend that the last logical axis is physically contiguous.

If a packed destination dword contains nonconsecutive source halfwords, direct
DMA cannot preserve that layout. With a proven source-order DMA mapping and an
exact same-block wait, lowering instead allocates a static LDS staging tile.
It issues DMA early into staging; at the named wait it drains DMA, synchronizes,
repackages into the original physical destination, and synchronizes again.
Staging allocations are reused for equal-sized, nonoverlapping same-block
issue/wait intervals. Overlapping or incomparable intervals get distinct
storage. Staging never silently replaces the GEMM consumer layout.

For the split-D k5 fixture, Q/K use direct DMA, while four V slices reuse one
8 KiB staging tile. The original consumer pool remains 16 KiB, giving 24 KiB
total static LDS. The first V issue overlaps softmax, and subsequent V issues
overlap the preceding PV computation. This preserves the old packed reads at
the cost of staging space, LDS repacking and synchronization; performance must
be measured rather than inferred from the presence of DMA.

Insufficient alignment, unsupported/dynamic strides or unprovable staging
use bit-preserving ordinary loads/stores **with the same inverse map**. Waits
retain the same block-wide readiness contract. Early sync selection on an
already packed copy preserves its map with `frisk.force_sync` for late scalar
lowering; it does not construct an unequal-shaped ordinary copy.

## Entry points

`MyTest` retains its existing behavior by default. Select either implementation
without changing the input or the pipeline scheme:

```bash
FRISK_GM_TO_SHM_MODE=async build/test/MyTest test/test_input.mlir 0 /tmp/async.ll
FRISK_GM_TO_SHM_MODE=sync  build/test/MyTest test/test_input.mlir 0 /tmp/sync.ll
```

`FRISK_GM_TO_SHM_MODE=hybrid` promotes copies, then selects the existing
vectorized synchronous path for sub-dword packed operands whose packing axis
differs from the physical contiguous source axis. Selection runs after layout
planning and before thread tiling (`async-direct-only=true`). It preserves
the consumer layout, global packet coalescing and packed LDS stores. Only an
adjacent matching issue/wait pair is replaced; nonlocal pipeline waits retain
their original contract. Other unsupported cases keep the existing fallback.

For the split-D k5 input this keeps Q/K direct DMA and makes V use the original
vectorized copy, eliminating the 8 KiB staging allocation and its scalar LDS
repack. This targets the measured regression in `pmc_test.log`: 5.105 ms,
96-byte scratch, 256 VGPRs and 4.223% bank conflict versus the original k5's
2.383 ms, zero scratch, 248 VGPRs and zero measured bank conflict. The mixed
path itself still needs gfx936 timing and register-allocation validation.

From the repository root, regenerate the mixed candidate with:

```bash
FRISK_GM_TO_SHM_MODE=hybrid FRISK_GM_TO_SHM_ALIGNMENT=4 \
  3rd/deepgengraph/build/test/MyTest \
  "$PWD/3rd/deepgengraph/test/test_input_spidD.mlir" 0 \
  "$PWD/generated_ir/test_input_spidD_best_k5_hybrid.ll" \
  1 1 3.3 34816 approx 256 4,1 1 1 1 0 \
  > generated_ir/test_input_spidD_best_k5_hybrid.log 2>&1
```

Promotion infers only element alignment by default, or uses an existing
`alignment` attribute on `frisk.copy`. Thus an unannotated FP16 copy remains a
synchronous fallback. If the caller can guarantee alignment of **every promoted
tile origin**, specify it explicitly:

```bash
FRISK_GM_TO_SHM_MODE=async FRISK_GM_TO_SHM_ALIGNMENT=4 \
  build/test/MyTest test/test_input.mlir 1 /tmp/async.ll
```

The standalone promotion pass exposes the same guarantee as
`--frisk-promote-gm-to-shm-async='assume-alignment=4'`. These are assumptions, not
runtime alignment checks. `FRISK_GM_TO_SHM_MODE=legacy` bypasses promotion.

## Validation

```bash
cmake --build build --target ThreadTilingTest MyTest -j4
python3 test/check_frisk_async_copy.py
python3 test/check_async_layout_integration.py
python3 test/check_sink_wait_buffer_ready.py
python3 test/check_async_attention_export.py output.ll output.log
# For the mixed candidate:
python3 test/check_async_attention_export.py output.ll output.log /path/to/llvm/bin hybrid
python3 test/check_async_address_lifetimes.py build/test/LegacyLLVMExportTest
python3 test/check_hybrid_copy_equivalence.py original_k5.ll hybrid.ll
```

The regression checks positive/negative op verification, two-wave and partial
wave copies, both synchronous selection points, dynamic and strided views,
buffer-slot scheduling, and complete LLVM export. It executes the exported
address calculations on x86 with a delayed DMA shim that implements the
hardware's implicit lane offset. Packed/staged cases cover alternate shapes,
transposed and padded source strides, alignment fallback, and malformed maps.
Staged repacking additionally runs concurrent CPU lanes with real block
barriers, checking source data and destination/staging guards bit-for-bit.
Scheduling tests cover descriptor dependencies, aliasing writes, nested writes,
barriers and idempotence; reuse tests reject incomplete/overlapping transfers.
The attention export test checks register provenance of P, LDS sizes, all four
V overlap windows and preservation through LLVM O3. These tests do not measure
gfx936 performance or emulate GPU wave scheduling; device validation remains
necessary for those properties.
