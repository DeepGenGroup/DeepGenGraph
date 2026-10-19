#!/usr/bin/env python3
"""Compare full exported kernels before/after fragment prefetch.

Usage: check_frag_prefetch_export.py baseline.ll prefetched.ll
Checks every address/FP/MMAC DAG and loop recurrence; loads may change order.
"""
import collections
import re
import sys
from pathlib import Path

from check_register_lifetimes_export import describe


def compare(before, after):
    old, new = describe(before), describe(after)
    for index, name in [(0, 'stores'), (2, 'FP/MMAC DAGs'), (3, 'loop recurrences')]:
        assert old[index] == new[index], name
    assert collections.Counter(old[1]) == collections.Counter(new[1]), 'load address multiset'
    for pattern in [r's_waitcnt[^"\\\n]*', r'call void @llvm\.amdgcn\.s\.barrier\(\)',
                    r'\bfence [^\n!]*', r'call <4 x i32> @llvm\.hcu\.ds\.read']:
        assert collections.Counter(re.findall(pattern, before)) == collections.Counter(re.findall(pattern, after)), pattern
    print('PASS all stores, load addresses, FP/MMAC operand DAGs, loop recurrences and synchronization')


if __name__ == '__main__':
    compare(*(Path(path).read_text() for path in sys.argv[1:3]))
