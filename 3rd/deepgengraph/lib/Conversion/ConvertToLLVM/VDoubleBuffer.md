# V double buffering in existing LDS

`DoubleBufferVPass` is an opt-in LLVM function pass, run at final export after
DMA lowering and optional `InterleavePVPass`. `MyTest` and
`LegacyLLVMExportTest` accept `--v-double-buffer=0|1`; the default is `0`.
`TuneAttention.py` exposes `v-double-buffer` through `--set` and `--sweep`.
`pv-interleave=0|1|2|4` is also exposed so both optimizations can be tuned together.
No floating-point operation, accumulator edge or MMAC order is changed.

The pass detects a sequence of V matrix reads separated by DMA-ready
`vmcnt(0) / release / s_barrier / acquire` groups. Each next-slice DMA
originally follows a reader barrier, overwriting the same LDS slice. The pass
finds an adjacent unused slice in the existing allocation, alternates slots,
and advances the next DMA to just after the current slice's ready barrier.
The redundant overwrite barriers are removed; all ready waits/barriers and
the final reader barrier remain. LDS capacity is never increased.

For the user-confirmed best `pv_interleave4` baseline, the inferred slices are
8192 bytes, using `[0,8192)` and `[8192,16384)`. V1/V2/V3 DMA moves from PV
MMAC ordinals 4/36/68 to 0/32/64. V0's initial issue is unchanged. The next
slice can overlap the current slice's LDS reads and all 32 PV MMACs.
Neither the 8192-byte size, four stages, 256 threads nor four PV chains is
hard-coded in the matcher.

## Legality and scope

The supported export uses wave64, a fixed one-dimensional workgroup of
64–1024 threads, one static LDS allocation, full-wave 16-byte buffer-to-LDS
inline-assembly DMA (`asm128`), and
`llvm.hcu.ds.read.m32x16.b16.v4i32.p3` readers. The sequence has 2–32 unrolled
slices in one compute block. A separate pure loop header is accepted, as in
fresh MLIR export; nested or more complex loops are rejected.

Before mutation the pass:

- Evaluates the restricted integer address DAG for every lane, including
  GEPs, vector packing, lane broadcasts and swizzles. It infers the contiguous
  DMA footprint, checks unique writes, bounds, and matrix-read containment.
  Unknown addresses fail closed. Tile sizes and packet counts are inferred.
- Matches the resource descriptor by SSA identity or identical pure DAGs,
  supporting fresh exports before CSE and archived optimized LLVM alike.
- Proves workgroup-uniform branch control and exact reader/ready fence groups.
- Rejects unrelated shared-memory users or unknown effects in the V region.
  This is the proof that retained Q and completed K permit LDS reuse, rather
  than assuming that a retention tuning flag implies safe storage ownership.
- Checks that any earlier same-loop read of the alternate slot is preceded
  by an overwrite in that iteration. Readers after the V region or in
  unrelated blocks are rejected, protecting loop-carried LDS contents.
- Preflights dependencies of relocated DMA. Only pure address computations
  and proven lane broadcasts can be cloned at the new issue point. No load,
  other DMA, branch, MMAC, or unrecognized effect is pulled forward.

The current ready barrier releases the previous owner of the alternate slot:
K's last slice before V1, V0 before V2, and V1 before V3. The final barrier
protects the next outer iteration's K writes. `frisk.v_double_buffer`
metadata makes the transformation idempotent. Rejected candidates are left
unchanged, and the lowering log reports the reason.

`result.json` includes `v_double_buffer_reports` (slice count/bytes,
displacement, barriers removed, LDS bytes) and `v_double_buffer_skips`.
`pv_interleave_reports` records the interleave match separately. An enabled
option alone is not evidence that a candidate was transformed.

## K buffering

The best case already alternates K's DK slices between the same two 8 KiB
slots: K0/K2 in the low slot and K1/K3 in the high slot. The new pass preserves
all K DMA, reads, waits and MMACs. It does not allocate another pair of K
buffers or reinterpret K's ordinary vector reads as V matrix reads.

K buffering along DK and prefetching the next outer BN tile are different
optimizations. The latter must account for live V slots and the loop edge;
this pass does not introduce it. The existing MLIR `AutoPipeline` /
`PipelineSchedule` infrastructure supports general loop buffer rotation.
`TuneAttention.py --sweep 'pipeline-stages=none;2'` controls a pipeline hint on
the selected loop (normally `dk`), subject to that pass's legality checks. It
can add allocations and is not equivalent to this late V storage-reuse pass.
Their effects should be measured separately using actual match reports and
LDS usage, rather than unconditionally doubling an already buffered K stream.

## Build and use

```sh
cmake --build 3rd/deepgengraph/build --target MyTest LegacyLLVMExportTest -j 3
python3 3rd/deepgengraph/test/check_v_double_buffer.py
python3 3rd/deepgengraph/test/check_pv_interleave.py
python3 -m unittest script.test_tune_attention
```

Append `--v-double-buffer=1 --pv-interleave=4` to a normal `MyTest` command.
`RunFriskSHell.sh` forwards these options too. The other tuning parameters
still determine whether the supported export is produced.

For a fresh non-causal input and the corresponding parameter family:

```sh
python3 TuneAttention.py \
  --input 3rd/deepgengraph/test/test_input_spidD_noncasual.mlir \
  --sweep-base online-baseline \
  --set workgroup-group-size=32 --set workgroup-reverse=0 \
  --set mask-add-bypass=approx --set local-reduce=approx \
  --set defer-lane-sum=approx --set pv-interleave=4 \
  --sweep 'v-double-buffer=0;1'
```

The archived best case also contains earlier K grouped-prefetch and V
handoff experiments. A fresh `MyTest` command does not recreate those
independent experiments. To preserve the exact best-case baseline, use:

```sh
python3 generated_ir/noncausal_v_double_buffer_pass_group32/generate.py
OPENBLAS_NUM_THREADS=1 python3 generated_ir/noncausal_v_double_buffer_pass_group32/verify.py
bash generated_ir/noncausal_v_double_buffer_pass_group32/run_gpu.sh
```

The experiment exports through the compiled pass, retains the original
baseline, validates LLVM/O3, and executes the CPU delayed-DMA model against
the baseline and the previously validated Python experiment. GPU numerical
validation, register allocation, scratch, and performance still require the
target gfx936 toolchain and hardware.
