Address-paired LDS scheduling experiment

Inputs (each has a matching .launch.json):
  baseline.ll               Byte-identical retained_8words.ll (1.646087 ms).
  fragment_pair.ll          Byte-identical previous candidate (1.622721 ms).
  ds_pair_prefetch.ll       New primary candidate: address-paired LDS loads,
                            with scheduling hints around the independent MMAC.
  ds_pair_prefetch_soft.ll  Same new load order, without any NEW scheduling hints.
                            Retains the baseline's eight existing hints.

All timings above are previous user measurements, not new measurements.
BM=BN=128, BK=32, BHSD=(1,32,4096,128), grid=(32,32,1), block=(256,1,1).
Group8 reverse, K128 swizzle, retained first Q fragment, LDS=16384 bytes.
No wait_tail2 transformation is included.

What changes:
  Match surviving b64 reads with the same address base and a 4096-byte offset
  difference. In a small window containing two MMACs and pure address operations,
  the first MMAC must be independent of this pair. Move the high-half read and
  its necessary address operations before that independent MMAC:

    Before: load A; MMAC(previous pair); MMAC(A); load A+4096; ...
    After:  load A; load A+4096; [hint]; MMAC(previous pair); [hint]; MMAC(A); ...

  Brackets denote NEW hints in ds_pair_prefetch only. No hint separates A and
  A+4096. This gives the address pair an opportunity to combine and leaves an
  independent MMAC between the pair and its first consumer.
  The rule does not depend on Q/K/V names. This particular input exposes 28
  eligible pairs (7 in each BK stage), compared with 27 adjacent-load groups in
  the old experiment. These are different pair boundaries, not extra reads.
  No read crosses another memory operation, synchronization, opaque call or
  control flow. Memory instruction order and MMAC order are unchanged.
  The first pair in each BK stage has no independent intervening MMAC and is
  left alone. There is no broader prefetch-distance sweep in this experiment.

Important measurement:
  Previous shared memory operation: baseline 5271552; fragment_pair 7096320.
  Check whether the new candidate reduces this extra work while preserving or
  improving median time. No expected count or speedup is guaranteed.
  Compare ds_read_b64 / ds_read2* / ds_read_b128, lgkmcnt waits, and actual
  load-to-consumer distance in TARGET assembly. Offset matching and local O3
  verification alone do not prove gfx936 DS combining.
  Observe final VGPR, scratch and timing; additional operand lifetime may change
  register allocation even though all inputs have the same logical traffic.

Reproduce/validate:
  python3 generated_ir/retained_ds_pair_schedule/generate.py
  OPENBLAS_NUM_THREADS=1 python3 generated_ir/retained_ds_pair_schedule/verify.py

Target testing, from repository root in the existing DTK environment:
  bash generated_ir/retained_ds_pair_schedule/run.sh baseline
  bash generated_ir/retained_ds_pair_schedule/run.sh fragment_pair
  bash generated_ir/retained_ds_pair_schedule/run.sh ds_pair_prefetch
  bash generated_ir/retained_ds_pair_schedule/run.sh ds_pair_prefetch_soft
  bash generated_ir/retained_ds_pair_schedule/run.sh baseline

Each build-<variant>/ retains kernel.hsaco.amdgcn, assembly_counts.json and
result.json. Copy result.json before repeating a variant if preserving all
rounds. GenHsaco.sh uses the same target/compiler/options as previous runs.
*.opt.ll files are local O3 inspection artifacts, not benchmark inputs.

Validation checks:
  LLVM assembly and default<O3>,verify for all four inputs.
  All have 80 surviving ordinary LDS loads after local O3, 12 explicit static
  VM waits and 18 static workgroup barriers. All 28 new pairs survive before
  their independent MMAC, without an internal scheduling hint.
  Static sched hints: baseline 8, previous fragment_pair 62, new hard 64, soft 8.
  CPU execution checks five query/head/seed cases including q=0 and q=31,
  bitwise output equality, reference tolerance, identical global/LDS traces,
  pending DMA and cross-wave LDS hazards. See validation.json after running.

This is a fingerprint-restricted IR experiment, not an integrated compiler
Pass. The target DTK/gfx936 assembly and performance are not verified locally.
