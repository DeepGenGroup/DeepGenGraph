#!/usr/bin/env python3
"""Record actual opcodes/resources; reject scratch/spills before benchmarking."""
import json
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
descriptor = re.search(r'\.amdhsa_kernel Attn_p2\b(.*?)\.end_amdhsa_kernel', text, re.S)
assert descriptor, 'Attn_p2 descriptor missing'
def field(name):
    found = re.search(r'\.' + name + r'\s+(\d+)', descriptor[1])
    assert found, name
    return int(found[1])

body = text.split('.Lfunc_end0:', 1)[0]
opcodes = re.findall(r'^\s*(v_\w+)\b', body, re.M)
counts = {stem: sum(op.startswith(stem) for op in opcodes) for stem in
          ('v_cvt_pk_f16_f32', 'v_cvt_pkrtz_f16_f32', 'v_cvt_f16_f32',
           'v_cvt_f16_f32_sdwa', 'v_pack_b32_f16', 'v_lshl_or_b32', 'v_mmac_')}
mode = sys.argv[2]
expected = {'pk_rne': 'v_cvt_pk_f16_f32', 'pk_rtz': 'v_cvt_pkrtz_f16_f32',
            'sdwa_rne': 'v_cvt_f16_f32_sdwa'}
instructions_ok = mode == 'baseline' or counts[expected[mode]] == 32
result = dict(vgpr_next_free=field('amdhsa_next_free_vgpr'),
              scratch_bytes=field('amdhsa_private_segment_fixed_size'),
              spill_stores=body.count('Folded Spill'), spill_reloads=body.count('Folded Reload'),
              whole_kernel_static_opcode_counts=counts, expected_pack_sites_present=instructions_ok,
              mode=mode)
result['passes'] = instructions_ok and not any(result[k] for k in ('scratch_bytes', 'spill_stores', 'spill_reloads'))
print(json.dumps(result, indent=2))
sys.exit(0 if result['passes'] else 2)
