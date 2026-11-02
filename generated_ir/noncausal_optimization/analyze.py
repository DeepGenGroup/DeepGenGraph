#!/usr/bin/env python3
"""Attribute measured DS counts to the exact Attn_p2 loop, excluding helpers."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def read_log(path):
    raw = path.read_bytes()
    return raw.decode('utf-16' if raw[:2] in (b'\xff\xfe', b'\xfe\xff') else 'utf-8')


def main():
    asm = read_log(ROOT/'k_test.gcn.log')
    pmc = read_log(ROOT/'pmc_test.log')
    lines = asm.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('Attn_p2:'))
    stop = next(i for i in range(start, len(lines)) if '-- End function' in lines[i])
    loop = next(i for i in range(start, stop) if 'This Inner Loop Header' in lines[i])
    label = lines[loop].split(':')[0]
    backedge = next(i for i in range(loop, stop) if re.search(r'\bs_cbranch_\w+\s+'+re.escape(label)+r'\b', lines[i]))
    sections = dict(entry=lines[start:loop], loop=lines[loop:backedge+1], exit=lines[backedge+1:stop])
    counts = {}
    for name, section in sections.items():
        counts[name] = dict(Counter(m[1] for l in section
            if (m := re.match(r'\s*((?:s_|v_|ds_|buffer_|global_)[\w.]+)\s', l))))
    ds = {n: sum(v for k, v in c.items() if k.startswith('ds_')) for n, c in counts.items()}
    assert ds == dict(entry=8, loop=68, exit=68), ds
    assert counts['loop']['v_mmac_f32_16x16x16_f16'] == 256
    noncausal_record = '\n'.join(pmc.splitlines()[:28])
    causal_record = pmc.split('/data2/xsl/DeepGenGraph/generated_ir/online_softmax_deferred_sum/kernel.ll', 1)[1]

    def measured(record):
        return dict(kernel_ms=float(re.search(r'time_our_mid=([\d.]+)', record)[1]),
                    sdpa_ms=float(re.search(r'time_base_mid=([\d.]+)', record)[1]),
                    shm_ops=int(re.search(r'shared memory operation\s+(\d+)', record)[1]))

    causal, noncausal = measured(causal_record), measured(noncausal_record)
    waves = 32*32*4
    expected = dict(causal=int(waves*(ds['entry']+ds['exit']+16.5*ds['loop'])),
                    noncausal=waves*(ds['entry']+ds['exit']+32*ds['loop']))
    assert expected == dict(causal=causal['shm_ops'], noncausal=noncausal['shm_ops'])
    result = dict(source_sha256={n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest()
                                  for n in ('k_test.gcn.log', 'pmc_test.log')},
        kernel_only_line_bounds=[start+1, stop+1], loop_line_bounds=[loop+1, backedge+1],
        instruction_counts=counts, ds_per_wave=ds, grid_waves=waves,
        measured=dict(causal=causal, noncausal=noncausal), predicted_shm_ops=expected,
        ratios=dict(tile_work=32/16.5, shm_ops=noncausal['shm_ops']/causal['shm_ops'],
            kernel_time=noncausal['kernel_ms']/causal['kernel_ms'],
            sdpa_time=noncausal['sdpa_ms']/causal['sdpa_ms'],
            kernel_tiles_per_second_improvement=(32/16.5)/(noncausal['kernel_ms']/causal['kernel_ms']),
            noncausal_sdpa_over_kernel=noncausal['sdpa_ms']/noncausal['kernel_ms']),
        needed_time_reduction_to_match_sdpa=1-noncausal['sdpa_ms']/noncausal['kernel_ms'],
        caveat='Exact count fit for these two measurements; DS operation count is not LDS bytes or occupancy.')
    (HERE/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'instruction_counts'}, indent=2))


if __name__ == '__main__':
    main()
