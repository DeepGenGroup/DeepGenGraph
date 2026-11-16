#!/usr/bin/env python3
"""Execute lazy-base IR numerically, with delayed-DMA and trace audits against control.

Each wave decides the slow path on its own, which the base interpreter cannot
run because it executes all waves in lockstep. The slow path is checked to be pure ALU, so
this model evaluates it for every lane and selects the join PHIs per wave. That
is equivalent for such a block. It is NOT a model of gfx936 timing, register
allocation or MMAC internal rounding.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cross_bn_checks', HERE.parent / 'noncausal_cross_bn_k0/verify.py')
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
from check_gemm_pv_addresses import split_args  # noqa: E402  (path set by the import chain)

# Upper bound on W (and therefore P) per variant. lazy_always matches the running max.
W_BOUND = {'control': 1.0, 'lazy_t8': 2.0 ** 8, 'lazy_t4': 2.0 ** 4, 'lazy_always': 1.0}
IMPURE = ('load ', 'store ', 'fence ', 's.barrier', 'buffer_load', 's_waitcnt', 'ds.', 'v_mmac')


class Model(previous.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slow_counts = np.zeros(self.t // 64, dtype=np.int64)
        self.max_w = -np.inf
        self.lazy_condition = None
        self.branches = 0
        if 'lazy.slow' in self.blocks:
            self.if_convert()

    def if_convert(self):
        slow = self.blocks['lazy.slow']
        assert slow[-1] == 'br label %lazy.join', slow[-1]
        for line in slow:
            assert not any(op in line for op in IMPURE), ('impure slow path', line)
        owner = [b for b, lines in self.blocks.items() if lines and 'label %lazy.slow' in lines[-1]]
        assert len(owner) == 1, owner
        args = split_args(self.blocks[owner[0]][-1][3:])
        assert len(args) == 3 and args[2] == 'label %lazy.join', args
        self.blocks[owner[0]][-1:] = ['lazy.record ' + args[0], 'br label %lazy.slow']
        join = self.blocks['lazy.join']
        for i, line in enumerate(join):
            m = re.match(r'(%[\w.-]+) = phi (.*)', line)
            if not m:
                continue
            ty, rest = self.parse_type(m[2])
            incoming = {split_args(x[1:-1])[1]: split_args(x[1:-1])[0] for x in split_args(rest)}
            assert len(incoming) == 2 and '%lazy.slow' in incoming, line
            taken = incoming.pop('%lazy.slow')
            join[i] = f'{m[1]} = lazy.phi {ty}|{taken}|{next(iter(incoming.values()))}'

    def operation(self, expr):
        if expr.startswith('lazy.record '):
            condition = self.typed(expr[len('lazy.record '):]).astype(bool)
            waves = condition.reshape(-1, 64)
            assert np.all(waves == waves[:, :1]), 'slow-path decision is not uniform within a wave'
            self.lazy_condition = condition
            self.slow_counts += waves[:, 0]
            self.branches += 1
            return None
        if expr.startswith('lazy.phi '):
            ty, taken, fast = expr[len('lazy.phi '):].split('|')
            a, b = self.value(taken, ty), self.value(fast, ty)
            shape = (self.t,) + (1,) * (a.ndim - 1)
            return np.where(self.lazy_condition.reshape(shape), a, b)
        if expr.startswith('fptrunc <4 x float> %math.result'):
            value = self.typed(expr.split(' ', 1)[1].split(' to ')[0])
            self.max_w = max(self.max_w, float(np.nanmax(value)))
        if expr.startswith('fcmp '):
            # O3 canonicalizes llvm.amdgcn.fcmp into fcmp + llvm.amdgcn.ballot.
            body = re.sub(r'^(?:(?:nnan|ninf|nsz|arcp|contract|afn|reassoc|fast) )*', '', expr[5:])
            predicate, body = body.split(' ', 1)
            a, b = split_args(body)
            ty, _ = self.parse_type(a)
            a, b = self.typed(a), self.value(b, ty)
            ordered = ~(np.isnan(a) | np.isnan(b))
            compare = {'gt': np.greater, 'ge': np.greater_equal, 'lt': np.less,
                       'le': np.less_equal, 'eq': np.equal, 'ne': np.not_equal}
            assert predicate[0] in 'ou' and predicate[1:] in compare, predicate
            result = compare[predicate[1:]](a, b)
            return result & ordered if predicate[0] == 'o' else result | ~ordered
        return super().operation(expr)

    def call(self, body):
        # The real results are lane masks; only their nonzero test is used.
        if 'llvm.amdgcn.fcmp.i64.f32' in body:
            a, b, predicate = map(self.typed, split_args(re.search(r'\((.*)\)', body)[1]))
            assert np.all(predicate == 2), 'only FCMP_OGT is modeled'
            return self.wave_any(a > b)
        if 'llvm.amdgcn.ballot.i64' in body:
            (condition,) = map(self.typed, split_args(re.search(r'\((.*)\)', body)[1]))
            return self.wave_any(condition.astype(bool))
        return super().call(body)

    def wave_any(self, condition):
        live = condition & self.active
        return np.repeat(live.reshape(-1, 64).any(axis=1), 64).astype(np.int64)


def dataset(kind, seed):
    rng = np.random.default_rng(seed)
    arrays = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
    q, k, v = arrays
    if kind == 'max_changes':
        q.fill(1); k.fill(-16); k[:, :128] = .25; k[:, 128:256] = .5
    elif kind == 'future_key_dominates':
        q.fill(1); k.fill(-1); k[:, -128:] = 1; v.fill(0); v[:, -128:] = 1
    elif kind in ('positive_overflow', 'negative_underflow'):
        q.fill(16); k.fill(16 if kind == 'positive_overflow' else -16)
    elif kind == 'gradual_growth':
        # Score grows by about 3 per key tile, so T=8 rescales repeatedly.
        q.fill(1)
        for tile in range(32):
            k[:, tile * 128:(tile + 1) * 128] = .18 * tile
    elif kind == 'one_wave_triggers':
        # Only query row 0 jumps at tile 5, so only the wave holding it rescales again.
        q.fill(.01); q[:, 128] = 1; k.fill(0); k[:, 5 * 128:6 * 128] = 1
    elif kind == 'large_jump':
        # Without a rescale, S' reaches about 2^261 and exp2 overflows FP32.
        q.fill(1); k.fill(0); k[:, 128:256] = 16
    return arrays


CASES = [('random', 0, 0), ('random', 1, 31), ('max_changes', 1, 2), ('future_key_dominates', 1, 0),
         ('positive_overflow', 1, 1), ('negative_underflow', 1, 1), ('gradual_growth', 1, 0),
         ('one_wave_triggers', 1, 1), ('large_jump', 1, 1)]


def run_case(text, arrays, head, query):
    model = Model(text, *arrays, threads=256, physical=(query, head))
    full = model.run().reshape(arrays[0].shape)
    return model, full, full[head, query * 128:(query + 1) * 128]


def rejected(text, kind, reason):
    arrays = dataset(kind, 42)
    expected = previous.helper.kdma.vdb.reference(*arrays, 1, 1, causal=False)
    model, _, actual = run_case(text, arrays, 1, 1)
    bad = not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=.01, rtol=.01)
    assert bad, ('negative control unexpectedly passed', reason)
    print('PASS negative', reason, flush=True)
    return dict(case=reason, dataset=kind, detected=True)


def main():
    manifest = json.loads((HERE / 'manifest.json').read_text())
    texts = {}
    for name, info in manifest['variants'].items():
        for suffix, key in [('', 'sha256'), ('.o3', 'o3_sha256')]:
            raw = (HERE / (name + suffix + '.ll')).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == info[key], name + suffix
            texts[name + suffix] = previous.helper.model_text(raw.decode())
    records = []
    for kind, head, query in CASES:
        arrays = dataset(kind, 42 + query)
        expected = previous.helper.kdma.vdb.reference(*arrays, head, query, causal=False)
        names = list(manifest['variants'])
        if (kind, head, query) == CASES[0]:
            names += [n + '.o3' for n in manifest['variants'] if n != 'control']
        base = base_full = None
        for name in names:
            variant = name.removesuffix('.o3')
            model, full, actual = run_case(texts[name], arrays, head, query)
            assert np.isfinite(actual).all(), (kind, name)
            np.testing.assert_allclose(actual, expected, atol=.01, rtol=.01)
            assert model.k0_issues == [(t, t * 256 - (32 if t else 0)) for t in range(32) for _ in range(4)]
            assert model.k0_overlap == 32 * 31
            assert (model.vm_waits, model.barriers, model.mmac_count, model.dma64_count) == (260, 293, 8192, 512)
            assert not model.pending and not model.k0_pending
            assert model.v_reads_with_pending_dma == 1016
            assert len(model.q_elements) == 16384 and set(model.q_elements.values()) == {1}
            assert np.isnan(full).sum() == full.size - 16384
            # O3 may rename or scalarize the packing, so only the delivered IR must hit the hook.
            assert name != variant or model.max_w > -np.inf, 'W bound was not checked'
            assert model.max_w <= W_BOUND[variant] * (1 + 1e-6), (kind, name, model.max_w)
            if base is None:
                base, base_full = model, full.copy()
            assert model.read_records == base.read_records, (kind, name, 'LDS provenance/epoch')
            assert model.lds_values.digest() == base.lds_values.digest()
            assert model.global_trace.digest() == base.global_trace.digest()
            assert (model.v_reads, model.v_dma) == (base.v_reads, base.v_dma)
            if variant != 'control':
                assert model.branches == 32
                assert np.all(model.slow_counts >= 1), 'first iteration must rescale'
            if variant == 'lazy_always':
                assert np.all(model.slow_counts == 32)
            if kind == 'one_wave_triggers' and variant == 'lazy_t8':
                assert sorted(model.slow_counts.tolist()) == [1, 1, 1, 2], model.slow_counts
            live = ~np.isnan(base_full)
            records.append(dict(kind=kind, head=head, query=query, variant=name,
                max_abs_error=float(np.max(np.abs(actual - expected))),
                max_abs_diff_from_control=float(np.max(np.abs(full[live] - base_full[live]))),
                slow_path_per_wave=model.slow_counts.tolist(), max_W=model.max_w,
                global_trace_LDS_values_and_V_schedule_equal=True))
            print('PASS', kind, head, query, name, 'slow', model.slow_counts.tolist(),
                  'maxW', round(model.max_w, 3), 'err', records[-1]['max_abs_error'], flush=True)
    good = texts['lazy_t8']
    no_first = good.replace('%lazy.take = or i1 %lazy.first, %lazy.any', '%lazy.take = or i1 false, %lazy.any')
    no_slow = good.replace('%lazy.any = icmp ne i64 %lazy.mask, 0', '%lazy.any = icmp ne i64 0, 0')
    assert no_first != good and no_slow != good
    negatives = [rejected(no_first, 'negative_underflow', 'no forced first-iteration rescale'),
                 rejected(no_slow, 'large_jump', 'threshold never triggers')]
    (HERE / 'validation.json').write_text(json.dumps(dict(cases=records, negative_controls=negatives,
        variants=manifest['variants'], reference_atol=.01, reference_rtol=.01, gpu_tested=False,
        target_codegen_tested=False, independent_wave_timing_simulated=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
