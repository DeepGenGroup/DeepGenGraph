#!/usr/bin/env python3
"""Static LLVM address model, not a gfx936 bank-counter prediction."""
import collections
import json
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '3rd/deepgengraph/test'))
from check_direct_shared_copy import Trace
from check_packed_lds_banks import multiplicity

def analyze(path):
    text = Path(path).read_text()
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    tid = next(n for n,e in definitions.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    threads = int(re.search(r'"amdgpu-flat-work-group-size"="\d+,(\d+)"', text)[1])
    context = {n: 0 for n,e in definitions.items() if e.startswith('phi i64 ') or '@llvm.amdgcn.workgroup.id.' in e}
    traces = [Trace(definitions, {**context, tid: lane}) for lane in range(threads)]
    stats = collections.defaultdict(collections.Counter)
    locations = dict((i, int(line)) for i,line in re.findall(r'!(\d+) = !DILocation\(line: (\d+)', text))
    for line in text.splitlines():
        match = re.search(r'= load (<\d+ x half>|half|float), ptr addrspace\(3\) (%[\w.]+)', line)
        matrix = re.search(r'= call <4 x i32> @llvm.hcu.ds.read.*?\(ptr addrspace\(3\) (%[\w.]+), i32 (\d+)\)', line)
        if matrix:
            kind, width, pointer, extra = 'matrix-packet-proxy', 16, matrix[1], int(matrix[2])
        elif match:
            kind, pointer, extra = 'load', match[2], 0
            width = 4 if match[1] == 'float' else 2 if match[1] == 'half' else int(match[1].split()[0][1:]) * 2
        else:
            continue
        addresses = [trace.value(pointer)[1] + extra for trace in traces]
        if width == 16:
            phases = [[wave+upper+x for x in [*range(a,a+4), *range(b,b+4)]]
                      for wave in range(0,threads,64) for upper in (0,32)
                      for a,b in ((0,20),(4,16),(8,28),(12,24))]
        else:
            lanes = min(32,128//width)
            phases = [range(start,start+lanes) for start in range(0,threads,lanes)]
        ways = max(multiplicity([addresses[l] for l in phase],width) for phase in phases)
        location = locations[re.search(r'!dbg !(\d+)', line)[1]]
        stats[f'mlir:{location} {kind} {width}B'][str(ways)] += 1
    return dict(stats)

if __name__ == '__main__':
    print(json.dumps({p:analyze(p) for p in sys.argv[1:]},indent=2))
