TuneAttention: fixed retained32 baseline and retention tuning

New preset: --case retained32-baseline
  Same compiler settings as measured q_bk_retention_sweep/retained_32words.ll:
  group8 reverse, K swizzle 8:6, retain-fragment-words=32, existing nohint
  unrolling / approximate math / fragment prefetch configuration.
  Presets configure compilation; input MLIR still determines BM/BN/BK/shape.
  Use the exact measured input below to reproduce the current best baseline.
  Historical baseline/nohint/group8/legacy/k5 presets keep retention disabled.
  MyTest's own default remains 0; no retention Pass implementation is changed.

Export the measured baseline (development environment):
  cd /data2/xsl/DeepGenGraph
  python3 TuneAttention.py \
    --input generated_ir/pipeline_cleanup/nohint_unrolled.mlir \
    --case retained32-baseline \
    --output-dir generated_ir/tune_retained32 \
    --llvm-as /data2/xsl/install/bin/llvm-as

Compile HSACO, check GPU correctness and benchmark in the existing DTK environment:
  python3 TuneAttention.py \
    --input generated_ir/pipeline_cleanup/nohint_unrolled.mlir \
    --case retained32-baseline --benchmark --warmup 50 --repeat 200 \
    --output-dir generated_ir/tune_retained32

Only change the retention budget; preserve all other new-baseline settings:
  python3 TuneAttention.py \
    --input generated_ir/pipeline_cleanup/nohint_unrolled.mlir \
    --sweep-base retained32-baseline \
    --sweep 'retain-fragment-words=0;8;16;24;32' \
    --benchmark --warmup 50 --repeat 200 \
    --output-dir generated_ir/tune_retention_budget

For a single overridden value:
  python3 TuneAttention.py \
    --input generated_ir/pipeline_cleanup/nohint_unrolled.mlir \
    --case retained32-baseline --set retain-fragment-words=24 --dry-run

For future group-size experiments at fixed 32 words:
  python3 TuneAttention.py \
    --input generated_ir/pipeline_cleanup/nohint_unrolled.mlir \
    --sweep-base retained32-baseline --sweep 'workgroup-group-size=4;8;16' \
    --dry-run

--list displays parameter sets without compilation. --dry-run emits replayable
commands. No --benchmark means export only. --benchmark inherits GenHsaco.sh's
existing LLVM_DIR / ROCM_BC / GPU_ARCH / GPU_FEATURES settings.
Each real invocation creates a new run-* directory, preserving previous outputs.

Sweep semantics:
  --sweep-base requires --sweep and accepts one exact explicit-parameter preset.
  Without it, sweeps still start from the historical baseline.
  Priority: --sweep > --set > selected sweep-base/preset.
  --case in sweep mode continues to filter grid-* names; do not use
  --case retained32-baseline to select the sweep base.
  --retain-fragment-words is a MyTest option; TuneAttention accepts it through
  --set / --sweep. Values are unsigned 32-bit integers; 0 disables retention.
  The budget describes retained 32-bit payload words/thread, not allocated VGPRs.

Results:
  config.json / result.json / summaries preserve parameters.retain-fragment-words.
  command.sh includes the explicit compiler flag; run.json records sweep_base.
  result.json.retention_reports records the actual compiler report, for example:
    [{"operand_tiles": 4, "budget_words_per_thread": 32}]
  result.json.retained_loop_words_observed contains distinct word counts observed
  in lowered MLIR snapshots. Repeated snapshots are not summed. Empty reports
  indicate no reported Pass run, and zero tiles do not imply successful retention.
  These fields are compiler evidence, not VGPR/occupancy or GPU timing measurements.

Validation performed for this integration:
  18 TuneAttention tests; 9 portable-entrypoint compatibility tests.
  Real retained32-baseline export through TuneAttention, with llvm-as verification.
  Its result.json reports 4 retained tiles, observed 32 words, static LDS 16384.
  After normalizing only the source filename/directory metadata, its LLVM output
  is byte-identical to the previously measured retained_32words.ll.
  See baseline_equivalence.json and the retained32-baseline result in run-*.
  No new GPU performance measurement was performed for this frontend change.
