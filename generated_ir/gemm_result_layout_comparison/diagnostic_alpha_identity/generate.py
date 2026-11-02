#!/usr/bin/env python3
"""Exact-source diagnostic: bypass the ten mislabelled-alpha LDS reloads.

This isolates Exp2OpTiling's producer layout error; it is not a general fix.
The original 3.1 export and compiler source remain unchanged.
"""
import hashlib
from pathlib import Path
import re

here = Path(__file__).resolve().parent
source = here.parent / 'layout_3_1/kernel.ll'
assert hashlib.sha256(source.read_bytes()).hexdigest() == 'eebb622e7db989f092e57e293311db9ba534967f8a2273ba273dab34eb95d2ff'
text = source.read_text()
loads = re.findall(r'  (%\w+) = load float, ptr addrspace\(3\) (%[\w.]+),.*!dbg !(?:15|36)\n', text)
assert len(loads) == 10
for index, (value, _) in enumerate(loads):
    text = re.sub(r'^  ' + re.escape(value) + r' = load float[^\n]*\n', '', text, flags=re.M)
    text = re.sub(re.escape(value) + r'(?![\w.])', '%1228' if index % 2 == 0 else '%1231', text)
(here / 'kernel.ll').write_text(text)
