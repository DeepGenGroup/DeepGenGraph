#!/usr/bin/env python3
"""Trace every packed K/V store back to its global source, across all lanes."""
import re
import sys
from pathlib import Path

from check_gemm_pv_addresses import Addresses, split_args


class ByteAddresses(Addresses):
    def value(self, name):
        if name not in self.cache and name in self.definitions:
            opcode, body = self.definitions[name].split(' ', 1)
            if opcode == 'getelementptr':
                ty, pointer, index = split_args(body)
                base, offset = self.typed(pointer)
                size = {'i8': 1, 'half': 2, 'float': 4}[ty.split()[-1]]
                self.cache[name] = (base, offset + size*self.typed(index))
            elif opcode == 'load' and body.startswith('<'):
                ty, pointer, *_ = split_args(body)
                count = int(ty.split()[0][1:])
                assert 'half>' in ty
                base, offset = self.typed(pointer)
                self.cache[name] = [(base, offset+2*r) for r in range(count)]
        return super().value(name)


def check(text, k_layout="bhds"):
    assert k_layout in ("bhds", "bhsd")
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', text, re.M))
    def builtin(name):
        return next(n for n,e in definitions.items() if '@llvm.amdgcn.' + name + '()' in e)
    tid, head, tile = [builtin(x) for x in ('workitem.id.x', 'workgroup.id.x', 'workgroup.id.y')]
    args = re.findall(r'ptr addrspace\(1\) (%\w+)',
                      re.search(r'define amdgpu_kernel void @Attn_p2\((.*)\) #', text)[1])
    assert len(args) == 4
    # The only scalar i64 phi is the normalized sequence-tile induction var.
    ivs = [n for n,e in definitions.items() if e.startswith('phi i64 ')]
    assert len(ivs) == 1, ivs
    mma = [e for e in definitions.values() if 'asm sideeffect ' in e and 'v_mmac_' in e]
    assert len(mma) == 64
    b_operands = [split_args(re.search(r'"\((.*)\)', mma[i])[1])[1] for i in (0,32)]
    # Contiguous WMMA C now also emits four-half P packets. K/V copies in
    # this non-pipelined fixture precede QK; P is checked by the value tracer.
    prologue = text[:text.index('v_mmac_')]
    stores = re.findall(r'^  store <(\d+) x half> (%\w+), ptr addrspace\(3\) (%\w+)', prologue, re.M)
    assert stores
    for h, step in [(0,0), (7,3), (31,127)]:
        common = {a:(a,0) for a in args}
        common.update({head:h, tile:63, ivs[0]:step})
        trace = ByteAddresses(definitions, {**common, tid:0})
        bases = [trace.typed(v)[0] for v in b_operands]
        coverage = [set(),set()]
        for lane in range(128):
            trace = ByteAddresses(definitions, {**common, tid:lane})
            for width, value, dest in stores:
                if int(width) != 4:
                    continue  # Q and computed-producer packets are checked separately.
                data = trace.value(value)
                # Ignore the Q prologue; identify K/V by their GLOBAL sources,
                # rather than assuming pooled LDS buffers have unique bases.
                source = data[0][0]
                if source not in args[1:3]:
                    continue
                which = args[1:3].index(source)
                base, offset = trace.value(dest)
                physical_base, origin = bases[which]
                assert base == physical_base
                n = 32 if which == 0 else 128
                for r in range(int(width)):
                    flat = (offset-origin)//2+r
                    group = flat//(n*8)
                    k = group//4*32+group%4*4+flat%8//4*16+flat%4
                    col = flat//8%n
                    column_group = col//8
                    col = (column_group*8 + (col//4-column_group//4-group//2)%2*4
                           + (col-column_group)%4)
                    logical = (k,col)
                    assert 0 <= k < (128 if which == 0 else 32)
                    assert logical not in coverage[which], (which,logical)
                    coverage[which].add(logical)
                    global_index = (h*128*4096 + k*4096 + step*32 + col if which==0 else
                                    h*4096*128 + (step*32+k)*128 + col)
                    if which == 0 and k_layout == "bhsd":
                        global_index = h*4096*128 + (step*32+col)*128 + k
                    assert data[r] == (source,2*global_index), (lane,which,logical,data[r],global_index)
        assert all(len(c)==4096 for c in coverage), list(map(len,coverage))
    print('PASS packed K/V copies: 24576 distinct global/LDS coordinates, no gaps or duplicate writers')


if __name__ == '__main__':
    check(Path(sys.argv[1]).read_text())
