#!/usr/bin/env python3
"""Audit supplied gfx936 assembly and LLVM addresses; no GPU timing prediction.

Run from any directory. Writes analysis.json beside this script.
The ordinary b64 model uses 32 four-byte banks and 16 contiguous lanes/phase.
The V matrix result is ONLY a b128 packet proxy, not a model of gfx936's
ds_read_m32x16_b16 internal phases. read2 operands are evaluated separately.
"""
import collections
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test"))
from check_direct_shared_copy import Trace
from check_packed_lds_banks import multiplicity


def read(path):
    data = path.read_bytes()
    return data.decode("utf-16" if data[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8")


def main():
    paths = [ROOT / "k_test.gcn.log", ROOT / "pmc_test.log",
             ROOT / "generated_ir/block_order/group8_reverse.ll"]
    asm, pmc, ir = map(read, paths)
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*?)(?:, !dbg !\d+)?$', ir, re.M))
    tid = next(n for n, e in definitions.items() if '@llvm.amdgcn.workitem.id.x()' in e)
    context = {n: 0 for n, e in definitions.items()
               if e.startswith('phi i64 ') or '@llvm.amdgcn.workgroup.id.' in e}
    traces = [Trace(definitions, {**context, tid: t}) for t in range(256)]
    locations = {i: int(line) for i, line in re.findall(r'!(\d+) = !DILocation\(line: (\d+)', ir)}
    transforms = {
        "original": lambda a: a,
        "K_16B_packet_xor_bit8_into_bit6": lambda a: a ^ ((a >> 2) & 64),
        "Q_8B_packet_xor_bit7_into_bit3": lambda a: a ^ ((a >> 4) & 8),
        "K_8B_packet_xor_bits8_9_into_bits3_6": lambda a: a ^ ((a >> 5) & 8) ^ ((a >> 3) & 64),
    }
    address_stats = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    examples = {}
    for line in ir.splitlines():
        m = re.search(r'= load <4 x half>, ptr addrspace\(3\) (%[\w.]+).*?!dbg !(\d+)$', line)
        if not m:
            continue
        kind = {68: "Q", 69: "K"}[locations[m[2]]]
        addresses = [t.value(m[1])[1] for t in traces]
        examples.setdefault(kind, {"pointer": m[1], "lane_0_to_15_bytes": addresses[:16],
                                   "lane_0_to_15_first_bank": [a // 4 % 32 for a in addresses[:16]]})
        for name, transform in transforms.items():
            ways = max(multiplicity([transform(a) for a in addresses[start:start+16]], 8)
                       for start in range(0, 256, 16))
            address_stats[kind][name][ways] += 1
    assert dict(address_stats["Q"]["original"]) == {2: 16}
    assert dict(address_stats["K"]["original"]) == {4: 128}
    assert dict(address_stats["K"]["K_16B_packet_xor_bit8_into_bit6"]) == {2: 128}
    assert dict(address_stats["Q"]["Q_8B_packet_xor_bit7_into_bit3"]) == {1: 16}
    assert dict(address_stats["K"]["K_8B_packet_xor_bits8_9_into_bits3_6"]) == {1: 128}

    # A 16B packet permutation fits inside each DMA wave's 1024B destination.
    # Physical lane t fetches the original source for t'; the consumer reads F(a).
    # This validates byte provenance of the proposed permutation, not emitted IR.
    F = transforms["K_16B_packet_xor_bit8_into_bit6"]
    for base in range(8192, 16384, 1024):
        written = {}
        for lane in range(64):
            source_lane = lane ^ ((lane >> 2) & 4)
            for byte in range(16):
                written[base + 16 * lane + byte] = base + 16 * source_lane + byte
        assert all(written[F(a)] == a for a in range(base, base + 1024))

    phase_pairs = ((0, 20), (4, 16), (8, 28), (12, 24))
    matrix_proxy = collections.Counter()
    for pointer in re.findall(r'= call <4 x i32> @llvm.hcu.ds.read.m32x16.b16.v4i32.p3\(ptr addrspace\(3\) (%[\w.]+), i32 0\)', ir):
        addresses = [t.value(pointer)[1] for t in traces]
        ways = max(multiplicity([addresses[w+u+x] for x in [*range(a,a+4), *range(b,b+4)]], 16)
                   for w in range(0, 256, 64) for u in (0, 32) for a, b in phase_pairs)
        matrix_proxy[ways] += 1

    # Use assembly .loc records to separate Q and K even when read2 is merged.
    loop_start, loop_end = asm.index('.LBB0_2:'), asm.index('.LBB0_6:')
    loop, epilogue = asm[loop_start:loop_end], asm[loop_end:]
    static = collections.Counter()
    location = None
    for line in loop.splitlines():
        loc = re.search(r'\.loc\s+1\s+(\d+)', line)
        if loc:
            location = int(loc[1])
        op = re.match(r'\s+(ds_\w+)\s', line)
        if op:
            kind = "shuffle" if op[1] == "ds_bpermute_b32" else {68: "Q", 69: "K", 93: "V"}[location]
            static[kind] += 1
    assert dict(static) == {"shuffle": 4, "V": 32, "Q": 8, "K": 32}
    epilogue_count = len(re.findall(r'^\s+ds_bpermute_b32 ', epilogue, re.M))
    assert epilogue_count == 64
    wave_iterations = 32 * sum(range(1, 33)) * 4
    dynamic = {name: n * wave_iterations for name, n in static.items()}
    dynamic["epilogue_shuffle"] = epilogue_count * 1024 * 4
    group8_record = pmc.split("group8-reverse", 1)[1].split("kernel-name:", 1)[1]
    measured = int(re.search(r'shared memory operation\s+(\d+)', group8_record)[1])
    assert sum(dynamic.values()) == measured == 5398528
    denominator_check = []
    # First seven records are the current block-order variants. Later logs
    # include older baselines and different kernels, so do not mix them here.
    names = ["query_fast_reverse", "group8_reverse", "head_fast", "group16", "group8", "group4", "group2"]
    for name, record in zip(names, pmc.split('kernel-name:')[1:8]):
        seconds = float(re.search(r'kernel time\s+([\d.]+)', record)[1])
        percent = float(re.search(r'bank conflict\s+([\d.]+)', record)[1])
        denominator_check.append({"variant": name, "profiler_seconds": seconds,
                                  "conflict_percent": percent,
                                  "product_ms_not_recoverable_runtime": seconds * percent * 10})
    result = {
        "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "gpu_tested": False,
        "model": "32 banks x 4 bytes; ordinary b64 operands: 16 contiguous lanes per phase, distinct words per bank",
        "limitations": "gfx936 instruction phases and hipprof raw metric formula are not independently verified. V uses a b128 packet proxy only. Percent*time is a normalization diagnostic, not a speedup bound. Proposed permutations have no generated target code or GPU timing.",
        "ordinary_load_address_stats": address_stats,
        "examples": examples,
        "V_matrix_b128_packet_proxy_only": matrix_proxy,
        "K_16B_DMA_permutation_byte_provenance_verified": True,
        "static_ds_per_loop": static,
        "static_epilogue_ds": epilogue_count,
        "wave_iterations": wave_iterations,
        "dynamic_ds": dynamic,
        "dynamic_ds_total": sum(dynamic.values()),
        "measured_shared_memory_operations": measured,
        "normalization_check": denominator_check,
    }
    output = Path(__file__).with_name("analysis.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
