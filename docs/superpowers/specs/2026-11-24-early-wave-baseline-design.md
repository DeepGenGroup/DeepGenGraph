# Early-wave tuning baseline

The user approved using the complete measured non-causal configuration as the
baseline for subsequent shape testing. Add `early-wave-baseline`, preserving the
existing named presets as controls. Make it the default of `RunAttention.py`;
`TuneAttention.py` retains its explicit case/coverage and sweep-base behavior.

Extend the existing preset/CLI machinery rather than adding a parallel runner.
Expose `softmax-fragments`, `lazy-softmax-base`, `softmax-cvt-pack`,
`hoist-lds-bases`, `hoist-lane-shuffles`, `dump-llvm-passes`, and `llvm-opt` to
`--set` and `--sweep`. Existing presets use the current MyTest defaults for these
options. The new preset adds K LDS prefetch 8, softmax fragments pv, K tail
prefetch 4, cross-BN K0 prefetch 1, lazy softmax base 8, RTZ conversion, both
hoists, LLVM pass dumps, and internal LLVM optimization 0 to `online-vdb-k64`.
HSACO optimization remains controlled independently by `--opt-level` (default 3).

Validate thresholds as finite numbers in (0,15], or off; retain MyTest's enum
and boolean ranges. Do not change input shape, causal semantics, or the compiler
binary. The original 20261120 input is unavailable locally, so matching options
does not promise identical IR or performance. GPU runs remain with the user.

Alternative: changing historical `baseline` would invalidate existing control
comparisons. A separate preset keeps those comparisons reproducible.
