#!/usr/bin/env python3
"""Exhaustive single-XOR address sweep for this layout-3.4 export.

32 banks, 4 bytes/bank, 16 consecutive lanes per b64 phase; identical
words broadcast. This is not a measurement of gfx936 performance.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'swizzle_pass'))
from probe import Addresses
from check_packed_lds_banks import multiplicity


def reads(path):
    text = path.read_text()
    model = Addresses(text)
    # The paired and tail-prefetch passes tag K loads after ownership analysis.
    pointers = re.findall(r'load <4 x half>, ptr addrspace\(3\) (%[\w.]+),[^\n]*!frisk.k_(?:lds|tail)_pair', text)
    assert len(pointers) == 64, len(pointers)
    result = [model.value(p) for p in pointers]
    assert all(np.all((a >= 0) & (a + 8 <= 16384) & (a % 8 == 0)) for a in result)
    return result


def xor(addresses, source, target):
    return [a ^ (((a >> source) & 1) << target) for a in addresses]


def histogram(addresses):
    return dict(Counter(max(multiplicity(a[i:i+16].tolist(), 8)
                            for i in range(0, 256, 16)) for a in addresses))


def main():
    old, actual = reads(HERE / 'baseline.ll'), reads(HERE / 'kernel.ll')
    # Undo the old composition in reverse order, then substitute a new layout.
    raw = xor(xor(old, 9, 3), 8, 6)
    sweep = []
    settings = [None] + [(s, t) for s in range(4, 13) for t in range(4, 13) if s != t]
    for setting in settings:
        packed = raw if setting is None else xor(raw, *setting)
        for mode in ('128', '64', '64-swizzle'):
            addresses = xor(packed, 9, 3) if mode == '64-swizzle' else packed
            h = histogram(addresses)
            sweep.append(dict(swizzle_k='off' if setting is None else '%d:%d' % setting,
                              k_dma=mode, max_multiplicity=max(h), site_histogram=h))
    winners = [row for row in sweep if row['max_multiplicity'] == 1]
    assert [(r['swizzle_k'], r['k_dma']) for r in winners] == [('7:9', '64-swizzle')]
    expected = xor(xor(raw, 7, 9), 9, 3)
    assert Counter(a.tobytes() for a in expected) == Counter(a.tobytes() for a in actual)
    assert histogram(old) == {2: 64} and histogram(actual) == {1: 64}
    record = dict(scope='BHSD=1,32,4096,128; BM=BN=128; QK/VO=32; layout=3.4',
                  model='32 banks, 4B words, 16 consecutive lanes/b64 phase, broadcast deduplicated',
                  searched_configs=len(sweep), source_target_bits=list(range(4, 13)),
                  note='Address sweep; only baseline and winner compiled. Bits >=13 cannot improve intra-8KiB bank distribution.',
                  baseline=histogram(old), winner=histogram(actual), winners=winners,
                  first_16_baseline=old[0][:16].tolist(), first_16_winner=actual[0][:16].tolist(),
                  emitted_addresses_match_prediction=True, gpu_tested=False,
                  sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (HERE/'baseline.ll', HERE/'kernel.ll')}, sweep=sweep)
    (HERE/'swizzle_analysis.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'sweep'}, indent=2))


if __name__ == '__main__':
    main()
