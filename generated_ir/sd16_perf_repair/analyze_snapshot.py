#!/usr/bin/env python3
"""Attribute the supplied sd16 PMC snapshot; not a generic GPU cost model.

Loop factors below come from sd16_extract_fix/sd16.frag-prefetch.ll:
4 output groups, 8 QK iterations, 4 iterations in each staging-copy loop.
Debug locations identify source operations; they are not inferred from opcode
names alone. ds_bpermute is included in this GPU's SHM operation counter.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def decode(path):
    raw = path.read_bytes()
    return raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8')


def main():
    asm = decode(ROOT / 'k_test.gcn.log').split('.Lfunc_end0:')[0]
    pmc = decode(ROOT / 'pmc_test.log')
    by_source = defaultdict(Counter)
    source_line = None
    for line in asm.splitlines():
        location = re.search(r'\.loc\s+1\s+(\d+)\s+', line)
        if location:
            source_line = int(location[1])
        opcode = re.match(r'\s*(ds_\w+)\s', line)
        if opcode:
            by_source[source_line][opcode[1]] += 1
    expected = {
        56: {'ds_bpermute_b32': 16},
        59: {'ds_bpermute_b32': 64},
        67: {'ds_read_u16': 2, 'ds_write_b32': 1},
        68: {'ds_read2st64_b64': 1},
        69: {'ds_read_m32x16_b16': 4},
        86: {'ds_bpermute_b32': 4},
        92: {'ds_read_u16': 4, 'ds_write_b32': 2},
        93: {'ds_read_b128': 8},
    }
    assert dict(by_source) == expected, dict(by_source)
    base_visits = 32 * sum(range(1, 33)) * 4  # heads, causal tiles, waves
    output_visits = base_visits * 4
    qk_visits = output_visits * 8
    dynamic = {
        'Q_operand_reads': qk_visits,
        'K_staging_copy': 3 * 4 * qk_visits,
        'K_matrix_reads': 4 * qk_visits,
        'V_staging_copy': 2 * 3 * 4 * output_visits,
        'V_operand_reads': 8 * output_visits,
        'P_layout_bpermutes': 64 * output_visits,
        'row_reduction_bpermutes': 4 * output_visits,
        'output_layout_bpermutes': 16 * 4 * (32 * 32 * 4),
    }
    measured = int(re.search(r'shared memory operation\s+(\d+)', pmc)[1])
    assert sum(dynamic.values()) == measured == 64061440
    slow_ir = (ROOT / 'generated_ir/sd16_extract_fix/sd16.frag-prefetch.ll').read_text()
    assert not re.search(r'call void @llvm.amdgcn.sched.barrier', slow_ir)
    report = {
        'scope': 'Supplied sd16 0.084 snapshot, not all shapes or targets',
        'input_sha256': {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT / 'k_test.gcn.log', ROOT / 'pmc_test.log',
                      ROOT / 'generated_ir/sd16_extract_fix/sd16.frag-prefetch.ll']
        },
        'static_ds_opcodes_by_mlir_source_line': dict(by_source),
        'dynamic_ds_operations': dynamic,
        'reconstructed_total': sum(dynamic.values()),
        'pmc_total': measured,
        'actual_prefetched_packets_in_slow_kernel': 0,
        'scratch_bytes': 0,
        'old_compile_profile': {
            'layout': '3.2', 'carrier_words': 24576, 'math': 'strict',
            'dma_pack': 'legacy', 'dma_export': 'split32',
            'qk_tile': 16, 'output_groups': 4,
        },
        'replacement_candidate_profile': {
            'layout': '3.3', 'carrier_words': 34816, 'math': 'approx',
            'dma_pack': 'useKPack', 'dma_export': 'asm128',
            'qk_tile': 32, 'vo_tile': 32, 'output_groups': 1,
            'target_gpu_measured': False,
            'qualification': 'See qk_probe_counts.json: local gfx90a probe still splits read2 with prefetch',
        },
        'known_target_measured_fast_artifact': 'generated_ir/frag_prefetch_v2/temp.frag-prefetch.ll',
    }
    (OUT / 'diagnosis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dynamic, indent=2))
    print(f'PASS exact PMC reconstruction: {measured:,}')


if __name__ == '__main__':
    main()
