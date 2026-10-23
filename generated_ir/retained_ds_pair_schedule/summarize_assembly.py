#!/usr/bin/env python3
"""Static opcode counts, not dynamic PMC counts or read byte counts."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

p = Path(sys.argv[1])
raw = p.read_bytes()
text = raw.decode()
counts = Counter()
waits = Counter()
metadata = {}
for line in text.splitlines():
    m = re.match(r'^\s+(ds_\w+|s_waitcnt\w*|s_barrier\w*|v_mmac\w*|buffer_load\w*)\b(.*)',line)
    if m:
        counts[m[1]] += 1
        if m[1].startswith('s_waitcnt'):
            waits[(m[1]+' '+m[2].split('//')[0].split(';')[0].strip()).strip()] += 1
    m = re.match(r'^\s*\.(amdhsa_(?:next_free_vgpr|next_free_sgpr|group_segment_fixed_size|private_segment_fixed_size))\s+(\d+)',line)
    if m: metadata[m[1]] = int(m[2])
assert counts, 'no AMDGPU instruction text found'
print(json.dumps(dict(assembly=str(p),sha256=hashlib.sha256(raw).hexdigest(),
                      static_opcodes=dict(sorted(counts.items())),
                      static_wait_forms=dict(sorted(waits.items())),metadata=metadata,
                      note='Whole-file static counts; inspect BN loop assembly for combining and actual load/use distance.'),indent=2))
