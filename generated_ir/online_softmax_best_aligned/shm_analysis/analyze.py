#!/usr/bin/env python3
"""Reproduce the LDS accounting for the supplied 2026-10-23 GPU capture.

Counts Attn_p2 only. LLVM CFG cycles distinguish per-BN shuffles from exit
shuffles. Static counts are converted to dynamic counts only for this fixed
BHSD=(1,32,4096,128), BM=BN=128, 256-thread causal launch.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def decode(name, output):
    raw = (ROOT / name).read_bytes()
    text = raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8')
    (HERE / output).write_text(text)
    return text, hashlib.sha256(raw).hexdigest()


def llvm_shuffles(path):
    text = path.read_text().split('define amdgpu_kernel void @Attn_p2', 1)[1].split('\n}', 1)[0]
    blocks, current = {'entry': []}, 'entry'
    for line in text.splitlines()[1:]:
        match = re.match(r'([\w.]+):', line)
        if match:
            current = match[1]
            blocks[current] = []
        else:
            blocks[current].append(line)
    edges = {key: re.findall(r'label %([\w.]+)', '\n'.join(lines))
             for key, lines in blocks.items()}

    def cyclic(key):
        pending, seen = list(edges[key]), set()
        while pending:
            other = pending.pop()
            if other == key:
                return True
            if other not in seen:
                seen.add(other)
                pending.extend(edges.get(other, []))
        return False

    counts = Counter()
    for key, lines in blocks.items():
        counts['inside_loop' if cyclic(key) else 'outside_loop'] += sum(
            'call ' in line and '@llvm.amdgcn.ds.bpermute' in line for line in lines)
    return dict(counts)


pmc, pmc_hash = decode('pmc_test.log', 'pmc.txt')
assembly, asm_hash = decode('k_test.gcn.log', 'assembly.txt')
kernel = assembly.split('Attn_p2:', 1)[1].split('.Lfunc_end0:', 1)[0]
counts, locations = Counter(), defaultdict(Counter)
location = 'unknown'
for line in kernel.splitlines():
    match = re.search(r'\.loc\s+1\s+(\d+)\s+(\d+)', line)
    if match:
        location = ':'.join(match.groups())
    match = re.match(r'\s+(ds_\w+|s_barrier\w*|v_mmac\w*|buffer_load\w*)\b', line)
    if match:
        counts[match[1]] += 1
        locations[location][match[1]] += 1

old = ROOT / 'generated_ir/retained32_group_prefetch/run-20261023-163255-p16epokv/grid-000001/kernel.ll'
new = ROOT / 'generated_ir/online_softmax_best_aligned/kernel.ll'
old_shuffles, new_shuffles = llvm_shuffles(old), llvm_shuffles(new)
assert old_shuffles == {'outside_loop': 64, 'inside_loop': 4}
assert new_shuffles == {'outside_loop': 0, 'inside_loop': 72}
assert counts['ds_bpermute_b32'] == 72
assert locations['89:20']['ds_bpermute_b32'] == 4  # reduce MAX
assert locations['95:20']['ds_bpermute_b32'] == 4  # reduce ADD
assert locations['102:23']['ds_bpermute_b32'] == 64  # PV output layout
assert sum(count for op, count in counts.items() if op.startswith('ds_write')) == 0
assert counts['ds_read2_b64'] + counts['ds_read2st64_b64'] == 40
assert counts['ds_read_m32x16_b16'] == 32

waves = 32 * 32 * 4
iterations = 32 * sum(range(1, 33)) * 4
components = {
    'Q_prologue_reads': [8 * waves, 8 * waves],
    'K_V_loop_reads': [64 * iterations, 64 * iterations],
    'sum_shuffles': [4 * iterations, 4 * iterations],
    'max_shuffles': [0, 4 * iterations],
    'PV_layout_shuffles': [64 * waves, 64 * iterations],
}
totals = [sum(pair[i] for pair in components.values()) for i in (0, 1)]
observed_new = int(re.search(r'shared memory operation\s+(\d+)', pmc)[1])
old_section = pmc.split('[retained_32words.ll]', 1)[1]
observed_old = int(re.search(r'shared memory operation\s+(\d+)', old_section)[1])
assert totals == [observed_old, observed_new] == [4890624, 9224192]
report = dict(
    inputs_sha256={'pmc_test.log': pmc_hash, 'k_test.gcn.log': asm_hash},
    counter_comparator='retained_32words original; group4 best has timing only in supplied log',
    assembly_kernel_static_counts=dict(counts), assembly_counts_by_source_line=dict(locations),
    llvm_shuffle_sites={'unsafe_best': old_shuffles, 'stable_aligned': new_shuffles},
    waves=waves, BN_wave_iterations=iterations,
    dynamic_components_old_new=components, reconstructed_totals_old_new=totals,
    observed_totals_old_new=[observed_old, observed_new],
    increase_from_PV_layout=64 * (iterations - waves), increase_from_max=4 * iterations,
    ideal_after_sinking_PV_layout=totals[0] + 4 * iterations,
    ideal_note='Accounting target only: excludes any new alpha layout adaptation; not measured performance',
)
(HERE / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'components': components, 'totals': totals}, indent=2))
print('PASS: dynamic accounting exactly matches both supplied PMC totals')
