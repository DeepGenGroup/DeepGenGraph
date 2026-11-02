#!/usr/bin/env python3
"""Report resources of the first (Attn_p2) kernel and reject scratch variants."""
import json
from pathlib import Path
import re
import sys

text = Path(sys.argv[1]).read_text()
kernel = re.search(r'\.amdhsa_kernel Attn_p2\b(.*?)\.end_amdhsa_kernel', text, re.S)
assert kernel, 'Attn_p2 kernel descriptor missing'
def descriptor(name):
    m = re.search(r'\.' + name + r'\s+(\d+)', kernel[1])
    assert m, name
    return int(m[1])
info = dict(VGPR_next_free=descriptor('amdhsa_next_free_vgpr'),
            scratch_bytes=descriptor('amdhsa_private_segment_fixed_size'))
body = text.split('.Lfunc_end0:', 1)[0]
info.update(static_spill_stores=body.count('Folded Spill'), static_spill_reloads=body.count('Folded Reload'))
info['passes_no_scratch_check'] = not (info['scratch_bytes'] or info['static_spill_stores'] or info['static_spill_reloads'])
print(json.dumps(info, indent=2))
sys.exit(0 if info['passes_no_scratch_check'] else 2)
