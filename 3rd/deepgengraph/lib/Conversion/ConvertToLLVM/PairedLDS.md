# Paired K/B LDS prefetch

`--k-lds-prefetch=0..8` controls a final LLVM function pass. Zero is the
default and leaves the export unchanged. A nonzero value is the maximum
number of independent MMAC instructions to move a read pair ahead of its
first consumer. `TuneAttention.py` accepts the same key with `--set` or
`--sweep`. `online-vdb-k64-lds2` enables distance 2 on the existing DMA64
configuration; the older `online-vdb-k64` preset remains unchanged.

The pass runs after PV interleaving, V buffering and K DMA/layout conversion.
Enabled runs perform EarlyCSE, InstCombine and EarlyCSE before scheduling,
to eliminate repeated reads and expose common pointer bases. It recognizes
ordinary aligned `<4 x half>` LDS loads used exclusively as the B operand
of supported register MMACs. In attention these are K loads; in another
GEMM they may be B. Matrix V readers and the A operand are not selected.

Within a single basic block, DMA, stores, fences, hardware waits/barriers,
wave operations, existing scheduling hints and unknown operations split
the movable regions. Loads never move across these boundaries. Pairing
requires the same pointer base and an encodable constant byte difference:
normal read2 offsets are preferred, followed by read2st64. Ties use the
first MMAC consumer. Neither SSA names, old assembly register numbers,
fixed tile sizes, nor global argument positions select the pairs.

All selected pairs in a region are first closed by hoisting the later
read and its pure address dependencies. Complete pairs are then hoisted
before an independent MMAC, without delaying any load. Before committing,
the pass checks SSA order and that no scheduling hint splits any selected
pair. It brackets each anchor with `llvm.amdgcn.sched.barrier(0)`; these
are compiler scheduling hints, not workgroup synchronization. It preserves
MMAC order, floating-point flags, DMA and hardware synchronization order,
and shared-memory capacity. Unsupported movements are skipped and reported.

The deterministic address-based pairs can differ from the historical
assembly-derived pairing table. The optimization implements paired prefetch
and protection, not a promise of bit-identical target assembly. Target
code generation still decides read2 selection, waits, registers and speed.

The function attribute `frisk.k_lds_prefetch` records the applied distance.
Repeated application with the same distance is a no-op. Conflicting explicit
reapplication fails instead of silently stacking scheduling hints.

Tests: `python 3rd/deepgengraph/test/check_paired_lds.py` and
`python -m unittest script.test_tune_attention`. Attention IR and numerical
validation are saved under `generated_ir/noncausal_k_lds_prefetch_pass/`.
