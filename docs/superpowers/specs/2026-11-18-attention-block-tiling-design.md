# Attention D tiling at the block boundary

The user approved keeping shared QK/softmax in tensor IR and moving independent QK reduction and V/O output decisions to the block-to-fragment path.

Generate one key loop carrying full-D O, L and M. BK is a lowering hint, stored as `frisk.attention_tiles = array<i64: QK, VO>` on the tensor kernel; it does not expand tensor computations. The user subsequently authorized deleting the historical split-D generation mode and driver compatibility branch. Online/legacy softmax and causal/noncausal remain independent options.

A dedicated `frisk-tile-attention` function pass runs after ConvertFriskToBase and before layout inference, asynchronous promotion or scheduling. It pushes Q/K/V slices through buffer copies and elementwise producers, allocates only tile-sized staging, and emits an ordered QK reduction plus independent output carriers. The shared score and probability are computed once per key tile. Buffer views preserve original global strides; sliced copies adjust their origins, never the input ABI.

QK/VO CLI overrides independently replace hints. Hints and overrides must be positive 16 multiples dividing their respective extents. `useKPack` doubles an unspecified hint according to the existing hardware DMA/MMAC ratio. Historical split-D inputs are no longer a driver mode. The user also approved deleting the now-unused tensor unroll-and-jam pass and its carrier budget. MyTest's positional interface now places math immediately after layout, followed by threads/waves; all later arguments shift by one.

Strict mode must retain its original QK grouping: requested smaller QK is rejected when reassociation is unavailable. V/O columns may be tiled because their sums are independent. No fastmath flags are introduced by the new pass.

Validation covers generator structure, tile independence, global origins, block numerical execution, causal/online semantics, idempotence, invalid options, full LLVM lowering, and staging dimensions. CPU checks are not GPU performance measurements.

New QK loops carry `frisk.simplify_loop` so budgeted fragment loop simplification exposes constant staging phases before Q retention and cross-key prefetch. Kernel `frisk.qk_simplify_loop = false` disables both this marker and the late full-unroll request; absence defaults to true. An optional kernel `frisk.qk_pipeline_hint` transfers to the new loop as `frisk.pipeline_hint`; authored schedules must run after tiling. TuneAttention automatically targets these kernel hints for shared tensor inputs and edits only metadata, preserving the device body, tile hints and parallel mapping.

Slice pushdown currently accepts global views sourced directly from function arguments. Computed global alias sources (including an already-normalized K transpose) reject rather than bypass possible writes to an underlying argument. Run the pass before NormalizeAttentionKLayout.
