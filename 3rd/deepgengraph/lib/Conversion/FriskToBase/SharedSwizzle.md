# Shared swizzle and grouped workgroups

These optimizations are opt-in. Existing callers retain the previous layout and
launch order. The implementation does not depend on `Attn_p2`, fixed SSA names,
32 heads, a 128-dimensional head, or a fixed number of waves.

The LLVM `CanonicalizeLDSGEPsPass` can also run independently of swizzling via
`LegacyLLVMExportTest --canonicalize-lds-geps-only input.ll`. This entry point
does not run the MMAC/shared-memory export preparation. The pass considers
ordinary nonvolatile, nonatomic loads from any address-space-3 buffer, including
pointer arguments and arbitrary scalar/vector element types. It uses LLVM's
known alignment (including argument contracts and dynamic offset known bits),
with bounded exact thread-layout evaluation as an additional proof. Constant
offset rebasing requires dominance; unproven cases remain unchanged. Memory
operations are not moved or widened, and matrix-read intrinsics are not rewritten.
`script/optimize_lds_reads.py` exports an unchanged baseline and a candidate with
this pass, plus verifier results and optional matching launch configurations.

## Pipeline

1. `convert-frisk-to-base` materializes logical `gpu.block_id` operations.
2. `frisk-reorder-workgroups` optionally permutes these IDs, before async
   scheduling and thread tiling. Launch dimensions remain unchanged.
3. `convert-friskbase-to-thread` runs LowerInfo inference and shared packing,
   then selects DMA packing. **After the final DMA packing decision**, it
   negotiates the requested swizzle and publishes `LowerInfo::shmIO` plus
   allocation attributes.
4. `pack-shared-memory` (also called by `lower-frisk-fragments`) reconstructs
   that `ShmDataIODesc`, rewrites every supported alias/consumer, and uses the
   inverse layout for the DMA source map. Failure of an explicit swizzle
   request is a compilation error, not a partial layout change.
5. LLVM `CanonicalizeLDSGEPsPass`, after register lifetime/fragment scheduling,
   exposes proven constant address differences as byte GEPs. `MyTest` enables
   this automatically when a Q/K/V swizzle is enabled.

The ownership/MMAC layout in `LowerInfo::base_layout` is unchanged. Physical LDS
swizzling belongs to `ShmDataIODesc`, not to the instruction's register layout.
Doing it in initial LowerInfo inference alone would be incorrect: subsequent DMA
planning can replace the original packing.

## User options

`MyTest` accepts named options without shifting its existing positional options:

```text
--swizzle-q=off|on|SOURCE_BIT:TARGET_BIT
--swizzle-k=off|on|SOURCE_BIT:TARGET_BIT
--swizzle-v=off|on|SOURCE_BIT:TARGET_BIT
--swizzle-q-arg=N
--swizzle-k-arg=N
--swizzle-v-arg=N
--workgroup-group-size=N
--workgroup-reverse=true|false
--workgroup-query-axis=x|y|z
--workgroup-head-axis=x|y|z
```

The corresponding MLIR thread-tiling pass options omit `--`; workgroup options
belong to `frisk-reorder-workgroups` and omit the `workgroup-` prefix.

All three swizzles default to `off`. `on` / `1` is shorthand for `8:6`, and
`off` / `0` disables the swizzle. Bits are **byte-address bits relative to the
allocation**: target bit XORs source bit. Thus `8:6` implements
`physical_byte ^= ((physical_byte >> 2) & 64)` on the already packed offset.
Different roles may select different bit pairs. This is an explicit layout
choice, not an automatic prediction that every enabled operand will get faster.

Argument indices refer to the lowered function ABI. They default to `-1`, which
uses string argument attributes `frisk.shared_role = "q"`, `"k"`, or `"v"`.
The existing attention K-layout normalization helper adds these attributes after
validating its example ABI. Other frontends can attach them or pass explicit
indices. The generic planner traces views/transposes/casts/subviews and never
assumes that operand B is K, or that identical tensor shapes distinguish K/V.
Duplicate bindings and untraceable enabled producers are rejected in a kernel
being tiled. Functions with no block operations are left alone.

Example for an already lowered Q/V/K/O ABI:

```text
convert-friskbase-to-thread="dma-pack-mode=useKPack swizzle-k=8:6 swizzle-q-arg=0 swizzle-v-arg=1 swizzle-k-arg=2"
```

### Legality and limits

- Static, owned, supported rank-two shared tiles and complete alias components.
- Distinct source/target bits in `[4,30]`; low four byte bits remain unchanged.
- Tile byte size must contain complete swizzle periods. No padding is added.
- All async producers of an alias component must agree on the layout and pass
  the existing all-packet 16-byte DMA proof with their actual source strides and
  alignment. The proof is repeated after view finalization.
- Scalar/vector coalescing preserves packet offsets explicitly. Collective
  matrix readers must have a proven aligned, contiguous 16-byte footprint.
- Unsupported escaping users/alias contracts are rejected for explicit requests.
  The current role planner supports async DMA producers; it does not invent a
  synchronous staging path or split DMA into smaller packets.

In particular, the old Q `7:3` permutation is deliberately rejected: it moves
bytes inside a 16-byte packet and caused the measured DMA/instruction regression.
Q/V `8:6` are legal experimental layouts, not the same as that old Q scheme.
Defaults remain off; the measured recommendation is K-only.

### Workgroup ordering

Requires independent workgroups and consistent positive static `range`
attributes on both selected `gpu.block_id` axes. Axes must differ; the third
axis is untouched. No function-name or tensor-shape recognition is used.

Group size defaults to zero (disabled); the default query/head axes are `y/x`.
For physical IDs, flatten the selected pair with the lower-numbered hardware
axis varying fastest. With query extent `Q`, head extent `H`, and
`G = min(group_size, Q)`:

```text
start  = (linear / (G*H)) * G
width  = min(G, Q-start)
within = linear % (G*H)
head   = within / width
query  = start + within % width
if reverse: query = Q-1-query
```

This remains bijective for a short last group, `group_size > Q`, unequal extents,
and either axis order. A repeated identical pass is a no-op; conflicting second
configurations are rejected. Helpers with no relevant block IDs are skipped.

### LLVM constant GEP pass

The pass first uses ScalarEvolution to prove constant pointer differences.
For nonlinear XOR layouts it also evaluates the **entire proven legal
workitem-x domain**, bounded to at most 1024 IDs. It accepts only supported pure
integer address expressions and GEPs rooted in the same shared global. Unknown
data, loop-dependent expressions and unknown calls fall back to the SCEV proof.
The register lifetime pass's marked, empty, tied-register identity asm is
recognized explicitly. A wider domain containing counterexamples prevents
rewriting. Proven allocation-relative offsets also establish load alignment.

Loads are not moved, widened, or speculated. Volatile/atomic loads are excluded;
DMA, waits, barriers and stores are untouched. GEPs do not acquire an unproved
`inbounds` promise. Backend scheduling still decides whether to emit `ds_read2`.

## Reproduction and verification

From the repository root:

```bash
cmake --build 3rd/deepgengraph/build \
  --target MyTest ThreadTilingTest ShmDataIOTest LegacyLLVMExportTest -j 6
3rd/deepgengraph/build/test/ShmDataIOTest
python 3rd/deepgengraph/test/check_swizzle_passes.py
python 3rd/deepgengraph/test/check_shared_packing.py
python 3rd/deepgengraph/test/check_shared_buffer_aliases.py
bash generated_ir/swizzle_pass/generate.sh k128
OPENBLAS_NUM_THREADS=1 python generated_ir/swizzle_pass/validate.py
OPENBLAS_NUM_THREADS=1 python generated_ir/swizzle_pass/probe.py
```

`generate.sh qkv128` exercises all three independent switches; `generate.sh off`
keeps only grouped/reversed workgroups. Output is isolated under
`generated_ir/swizzle_pass/`; no measured baseline is overwritten.

The generated K-only and Q/K/V variants retain 24 `buffer_load_dwordx4 ... lds`
sites, 144 b64 LDS load sites, and 16384 bytes of shared storage. CPU delayed-DMA
execution for `(head,query)=(0,0),(1,7),(1,31)` is bitwise equal to the measured
hand-patched K/GEP winner, with equal wait/barrier/MMAC counts. The K-only probe
checks 16 unique K addresses across 256 threads, preserves the two-way bank
model, and emits eight `ds_read2_b64` in local gfx900 code generation.

These are semantic and local code-generation checks. Neither the full gfx936
assembly nor GPU runtime has been measured for the new compiler-generated IR;
address arithmetic and scheduling can still differ from the hand-patched winner.
