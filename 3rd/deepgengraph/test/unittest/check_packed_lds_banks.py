#!/usr/bin/env python3
"""Check emitted addresses against the AMD 32-bank LDS phase model.

This is an address-model regression, not a gfx936 hardware measurement.
Read phases: https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
"""
import collections
import re
import sys
from pathlib import Path

from check_packed_attention_copy import ByteAddresses
from check_gemm_pv_addresses import split_args


def multiplicity(addresses, byte_width):
    banks = collections.defaultdict(set)
    for address in addresses:
        for word in range(address // 4, (address + byte_width - 1) // 4 + 1):
            banks[word % 32].add(word)
    return max(map(len, banks.values()))


def check(text):
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    def builtin(name):
        return next(n for n,e in definitions.items() if '@llvm.amdgcn.' + name + '()' in e)
    tid, head, tile = map(builtin, ['workitem.id.x', 'workgroup.id.x', 'workgroup.id.y'])
    args = re.findall(r'ptr addrspace\(1\) (%\w+)',
                      re.search(r'define amdgpu_kernel void @Attn_p2\((.*)\) #', text)[1])
    common = {a: (a, 0) for a in args}
    common.update({head: 0, tile: 0})
    common.update({n: 0 for n,e in definitions.items() if e.startswith('phi i64 ')})
    traces = [ByteAddresses(definitions, {**common, tid: lane}) for lane in range(128)]
    # Computed contiguous C fragments also use b64 stores. Restrict this
    # K/V check to the copy prologue, before the first register MMA.
    prologue = text[:text.index('v_mmac_')]
    stores = re.findall(r'^  store <(\d+) x half> (%\w+), ptr addrspace\(3\) (%\w+)', prologue, re.M)
    store_max, store_count = 0, 0
    for width, value, dest in stores:
        if int(width) != 4:
            continue  # Q and computed-producer stores have other widths.
        data = traces[0].value(value)
        if data[0][0] not in args[1:3]:
            continue
        assert int(width) == 4, width
        store_count += 1
        addresses = [trace.value(dest)[1] for trace in traces]
        # Each 64-bit store half is checked separately; read2/write2 combines
        # two addresses, not a larger contiguous per-lane memory footprint.
        for start in range(0, 128, 16):
            store_max = max(store_max, multiplicity(addresses[start:start+16], 8))
    read_max, read_count = 0, 0
    phases = [(0,20), (4,16), (8,28), (12,24)]
    for expr in definitions.values():
        m = re.match(r'load <8 x half>, ptr addrspace\(3\) (%\w+)', expr)
        if not m:
            continue
        read_count += 1
        addresses = [trace.value(m[1])[1] for trace in traces]
        for wave in range(0, 128, 64):
            for upper in (0, 32):
                for a,b in phases:
                    lanes = [wave+upper+x for x in [*range(a,a+4), *range(b,b+4)]]
                    read_max = max(read_max, multiplicity([addresses[l] for l in lanes], 16))
    assert store_count == 16, store_count
    assert read_count == 32, read_count
    assert store_max <= 2, ('store bank concentration', store_max)
    assert read_max == 1, ('read bank concentration', read_max)
    print(f'PASS 32-bank address model: {store_count} K/V stores, max {store_max}-way; '
          f'{read_count} b128 read packets, conflict-free phases')


def check_row_packets(text, expected_read=2):
    """Model the actual emitted P packets; this is not a hardware measurement.

    DWORD reads use 32-lane phases, b128 writes contiguous eight-lane phases.
    The operand-coordinate regression separately proves every packed value.
    """
    from check_direct_shared_copy import Trace
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    tid = next(n for n,e in definitions.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    mma = [n for n,e in definitions.items() if 'asm sideeffect ' in e and 'v_mmac_' in e]
    region = text[text.index(f'  {mma[31]} ='):text.index(f'  {mma[-1]} =')]
    stores = list(re.finditer(r'^  store <(4|8) x half> %\w+, ptr addrspace\(3\) (%\w+),', region, re.M))
    assert [int(s[1]) for s in stores] in ([8, 8], [4, 4, 4, 4])
    label = re.findall(r'^(\d+):', region[:stores[0].start()], re.M)[-1]
    guard = re.findall(r'br i1 (%\w+), label %' + label + r', label %(\w+)', region)
    assert len(guard) == 1
    traces = [Trace(definitions, {tid: lane}) for lane in range(128)]
    store_max = 0
    for store in stores:
        byte_width = 2 * int(store[1])
        phase = 128 // byte_width
        for start in range(0,128,phase):
            addresses = [t.value(store[2])[1] for t in traces[start:start+phase] if t.value(guard[0][0])]
            if addresses:
                store_max = max(store_max, multiplicity(addresses, byte_width))
    reads = re.findall(r'load <(2|4) x half>, ptr addrspace\(3\) (%\w+)', region[stores[-1].end():])
    assert reads, 'Expected explicit packed reads for the computed operand'
    read_max = 0
    for width, ptr in reads:
        addresses = [t.value(ptr)[1] for t in traces]
        for start in range(0,128,32):
            read_max = max(read_max, multiplicity(addresses[start:start+32], 2 * int(width)))
    assert read_max == expected_read, ('P read concentration', read_max)
    assert store_max == 1, ('P write concentration', store_max)
    print(f'PASS P/A 32-bank model: packed reads max {read_max}-way, stores conflict-free')
    return {'read_max': read_max, 'store_max': store_max}


if __name__ == '__main__':
    text = Path(sys.argv[1]).read_text()
    check(text)
    check_row_packets(text)
