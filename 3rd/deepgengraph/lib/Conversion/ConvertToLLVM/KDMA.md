# Configurable K DMA width

`MyTest`, `LegacyLLVMExportTest`, `RunFriskSHell.sh` and `TuneAttention.py`
accept `--k-dma=128|64|64-swizzle` (TuneAttention uses `--set k-dma=...`).
The default, `128`, does not transform the exported IR.

* `64`: split each selected 16-byte packet into two `buffer_load_dwordx2 ...
  offen lds` instructions, preserving its LDS layout.
* `64-swizzle`: the same 8-byte DMA, plus allocation-relative byte bit3 XOR
  bit9 on the selected K lifetime. Existing `swizzle-k=8:6` remains in place;
  this is an additional permutation, not a replacement for that option.

`--k-dma-arg=N` selects the lowered global-pointer ABI argument (default 1 for
Q,K,V,O). It does not use an operand name or assume that all GEMM B operands
are K. Use an explicit index for other ABIs. The argument must be the sole
kernel argument underlying the selected DMA resource descriptor.

The final export pass runs after asm128 conversion, PV interleave and V double
buffering. Therefore `dma-export=asm128` still describes the initial export;
Q/V transfers remain 128-bit. `auto` is accepted with async/hybrid copies.
The user-confirmed gfx936 `buffer_load_dwordx2 ... offen lds` form is required;
the local upstream gfx900 assembler does not implement this form.

The physical destination of half h is `m0 + 512*h + 8*lane`. The original
16-byte packet owner is `wave_base + 32*h + lane/2`, and its intra-packet
offset is `8*(lane%2)` (or `8*((lane%2) XOR h)` for 64-swizzle). The pass
recomputes the actual source expression for that owner; it does not introduce
ds_bpermute or hard-code source strides. Uniform loop/workgroup expressions
are retained. Publication waits, barriers and matrix arithmetic are unchanged.

The pass requires fixed one-dimensional wave64 workgroups (64..1024 threads),
one static LDS allocation, aligned and nonoverlapping 1024-byte destination
packets per wave, and provable integer source/address expressions. All K copies
and ordinary consumers must belong to one compute block, optionally with a pure
loop header. It tracks byte ownership through the compute block and rejects K
lifetimes escaping the block, unknown shared side effects, unsupported shared
stores, mixed ownership and non-b64 swizzle consumers. Matrix reader checks are
conservative over aligned 8KiB regions. Unsupported explicit requests fail
compilation; they are not silently treated as successful tuning candidates.

For swizzled consumers, exact evaluation over every workitem proves constant
differences and emits dominating common bases plus GEP offsets. This preserves
the opportunity to select ds_read2_b64; final instruction counts and VGPR usage
must still be measured with the target compiler. No LDS capacity is added.

Tests: `python3 3rd/deepgengraph/test/check_k_dma.py` and
`python3 -m unittest script.test_tune_attention`. Full attention numerical,
delayed-DMA and epoch-ownership validation is recorded with generated IR under
`generated_ir/noncausal_k_dma_pass/`.
