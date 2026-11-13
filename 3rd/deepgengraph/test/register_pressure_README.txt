Shared register-pressure analysis
=================================

API: include/deepgengraph/Analysis/RegisterPressure.h
Implementation: lib/Analysis/RegisterPressure.cpp

This analysis does not change IR, select reduction chains, schedule instructions,
or impose a register budget. FragmentIR and ThreadIR use the same analysis.
It reports whole-function shared-content demand in 32-bit words per thread,
and keeps the original whole-SSA estimate as a comparator.
It is NOT physical VGPR/SGPR allocation and does NOT predict occupancy.

MLIR infrastructure used
------------------------
* mlir::Liveness: CFG fixed point, live-in/live-out sets and last ancestor uses.
  Last-use events are built once per block; the expensive per-op
  currentlyLiveValues/resolveLiveness query is not repeated at every instruction.
* RegionBranchOpInterface and LoopLikeOpInterface: repetitive regions, loop
  bounds/steps. getUsedValuesDefinedAbove identifies externally captured values.
* DataLayout::closest: scalar/index and LLVM pointer sizes, including DLTI.
* AnalysisManager: cached Liveness and RegisterPressureAnalysis, with normal
  invalidation after IR mutation. The logging pass preserves all analyses.
* AsmState: contributor SSA names agree with the printed function.
* StaticValueUtils: resolve constant vector/fragment coordinates. MLIR's
  Liveness provides SSA lifetimes, not fragment-element storage identities;
  the content-provenance layer below is implemented by this analysis.

Region handling
---------------
In each nested region, include values live after its parent operation even if
the body never uses them. Exclude the parent operation's results, which do not
exist until it returns. Inside repetitive regions, retain external captures and
dynamic loop bounds/steps across the backedge, including after their last lexical
use. Loop initial operands are not automatically counted as extra copies of the
body carriers. Sibling regions are analyzed separately, with CFG live-in unions
provided by MLIR. Unknown region operations conservatively retain outer values
and are counted in unknown-region-ops. No uniform/divergent hardware scheduling
or register-coalescing claim is made by this SSA-level control-flow model.

Cost model and limitations
--------------------------
data: union of surviving originating scalar slots, rounded up to 32 bits.
      A newly computed vector<4xf16> is 4 words and vector<4xf64> is 8 words.
      A bitcast of vector<2xi32> to vector<4xf16> reuses its two originating
      words. Subranges of wider elements only charge the intersecting words.
packed-data: alternative size of the distinct surviving bits, packed within
      each originating payload (a whole vector<4xf16> is 2 words). Not added
      to data. This does not predict target packing or the required copies.
ideal-packed-data: ceil(sum of distinct live data bits / 32). Unlike packed-data,
      this permits packing across independent SSA origins: two f16 scalar SSA
      values have data=2, packed-data=2, ideal-packed-data=1; three f16 values
      have ideal-packed-data=2. A vector<2xf16> has packed-data=1 too. Compute
      the bit union first and round only after summing; aliases introduce no
      extra bits. Target operand/register-class/packing constraints and needed
      copies are not modeled. This is an optimistic payload size, not an
      allocated VGPR count or a hardware lower bound. Pointers are separate.
addresses: explicit LLVM pointer SSA values, sized using DataLayout. These
      represent address values, not the global/shared storage they point to.
      Explicit integer offsets/index calculations count as data. An early-IR
      memref does not implicitly allocate a full descriptor's worth of registers.
descriptor-diagnostic: full ranked memref descriptor (two pointers, offset,
      rank sizes and rank strides), including its lowered LLVM struct. Excluded
      from words(), peak-live, overlap and hotspot ranking. Static fields may
      fold, and surviving address/offset fields are counted once they become
      explicit SSA values. Global/shared memory CONTENTS are never registers.
      General LLVM arrays/structs recursively sum their data/pointer fields;
      only recognized memref descriptor shapes are entirely diagnostic.
unknown: unsupported types (including scalable vectors) explicitly counted;
      their missing costs are never a reason to accept an optimization budget.
      unmodeled-type lines identify the types and representative SSA values.

Constants are assumed rematerializable and compile-time !frisk.async_task
dependencies cost zero. Each scalar/vector/fragment has a logical mapping to
bit ranges of originating SSA values. pack_frag/unpack_frag, vector.shape_cast
and vector.bitcast preserve that mapping. Static extract_frag/vector.extract
select ranges; insert_frag/vector.insert replace only the selected ranges.
Fragment coordinates are multiplied by fragment dimensions before projection.
vector.from_elements assembles input mappings; scalar vector.broadcast repeats
one mapping. Shared ranges are counted once. Arithmetic/load results introduce
new origins; no assumption of in-place arithmetic is made. If old and updated
vectors coexist, overwritten OLD content stays live through the old vector.

For unresolved extract coordinates (or a source with an unresolved mapping),
the selected content is covered when the full source content is represented in
the current union. Otherwise charge the result slice independently; different
dynamic selectors may select different elements. An unresolved insert retains
the destination+input union, including potentially overwritten old elements.
This may exceed the standalone output size, but avoids inventing another full
destination allocation. unresolved-extracts/conservative-inserts expose these
fallbacks. The union is a content estimate, not a claim that hardware supports
runtime indexing or arbitrary register aliases without extra instructions.

Whole-value SSA lifetimes still retain all components of a live vector, even
if only some will be used later. Loop carriers and CFG phi values are still
separate origins. In particular, FragmentIR can keep an entire captured source
and entire partially updated carriers live together; tracking per-iteration
element liveness is future work. Instruction operand constraints, register
copies/packing, constants' materialization temporaries, calls, spills,
private-memory promotion and opaque operations' internal lowering temporaries
are not modeled. This is a diagnostic estimate, not a proven bound on hardware
resources. SGPR/VGPR classes and allocation granularity need a later target
model. Do not use peak-live or peak-data as a hard hardware limit such as 255.

Log fields
----------
peak-live: maximum data+explicit-address words across before/after boundaries.
      Excludes memrefs, descriptor diagnostics and all global/shared contents.
peak-overlap: additionally assumes dying operands and new results coexist for
      ordinary operations; useful when evaluating a rewrite before coalescing.
      Region-container inputs and outputs are NOT forced to coexist.
peak-data/peak-packed-data/peak-ideal-packed-data/peak-addresses: independent
      category maxima, which need not occur together and must not be summed.
peak-ideal-packed-data-overlap: ideal data bit packing at overlap boundaries.
peak-descriptor-diagnostic: full descriptor maximum, reported separately.
peak-with-descriptors: historical data+pointer+full-descriptor combined estimate
      for comparison only, NOT the register pressure metric.
ssa-peak-live/ssa-peak-data/ssa-peak-overlap: original whole-SSA model for
      comparison (only pack/unpack/shape_cast are deduplicated in that model),
      with descriptors excluded just as in peak-live. ssa-peak-with-descriptors
      preserves the older combined SSA+descriptor diagnostic.
ssa-before/ssa-after/ssa-overlap: that comparator at each printed point.
block-peak: one peak per basic block, with nesting depth and source location.
hotspot/content-origin: largest distinct-content contributors at hotspots.
      An origin can be an old SSA definition whose bits are now used through
      another representation. Its printed words describe only surviving ranges.
      Contributions plus other-content sum to the selected before/after total
      for additive fields (words/data/packed-data/addresses/diagnostics). The
      optimistic ideal-packed-data field is NOT additive across contributors:
      its total rounds only after summing their bits.
reduction: every reduce_frag, even if not among the largest hotspots.
point: optional full before/after curve (all-points=true).
Top contributor lists are truncated to top=N; the peak computation is not.

Usage (repository root)
-----------------------
  cmake --build 3rd/deepgengraph/build --target ThreadTilingTest MyTest -j 2
  python3 3rd/deepgengraph/test/check_register_pressure.py

Analyze serialized FragmentIR or ThreadIR, output unchanged IR separately:
  3rd/deepgengraph/build/test/ThreadTilingTest input.mlir \
    '--frisk-analyze-register-pressure=stage=fragment top=8' \
    -o /dev/null 2> pressure.log

For all points, add all-points=true to the pass options. For stable function
report order on a multi-function module, also use --mlir-disable-threading.

MyTest: append --analyze-register-pressure=1 to an existing command. Default is
0. The reports are written to stderr at the end of typed fragment optimization
and at the last ThreadIR boundary before firstLowering. Other LLVM-level passes
(including local reduction/prefetch) run later and are not reflected in these
reports. Extract reports from the ordinary lowering log between
"RegisterPressure:" and "EndRegisterPressure".

In a transformation pass, getAnalysis<frisk::RegisterPressureAnalysis>() exposes
getResult().lookup(op), with liveBefore/liveAfter and their measured costs.
getResult().measure(values) measures any existing set with shared-content
deduplication; measureSSA(values) supplies the old comparator. For standalone
callers, analyzeRegisterPressure(func, liveness)
accepts an existing MLIR Liveness instance. Load the LLVM dialect when using
the descriptor cost model; the reporting pass declares this dependency.
Recompute after scheduling, retention, unrolling or other IR mutations. Future
transforms must account for their proposed values and hidden temporaries too;
these diagnostics alone do not establish an occupancy/spill safety gate.
