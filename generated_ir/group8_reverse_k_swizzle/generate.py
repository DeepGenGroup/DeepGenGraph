#!/usr/bin/env python3
"""Generate the K-only 16-byte DMA packet swizzle from the measured winner."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / "generated_ir/block_order/group8_reverse.ll"
SOURCE_SHA256 = "b82cdd9f10f130753d896a15fc0acb46841bd955e9cacae2bb4f44f453816eb2"
BASELINE = HERE / "group8_reverse_control.ll"
CANDIDATE = HERE / "group8_reverse_k_swizzle128.ll"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(text):
    output, undo, pointers, dma_offsets = [], {}, {}, []
    for line in text.splitlines():
        if "buffer_load_dwordx4" in line and line.endswith("!dbg !16"):
            old = re.search(r'\(i32 %[^,]+, i32 (%[\w.]+), <4 x i32>', line)[1]
            new = f"%ksw.dma.offset{len(dma_offsets)}"
            output.append(f"  {new} = xor i32 {old}, %ksw.dma.mask, !dbg !16")
            changed = line.replace(f"i32 {old},", f"i32 {new},")
            undo[changed] = line
            line = changed
            dma_offsets.append({"original": old, "swizzled": new})
        m = re.search(r'= load <4 x half>, ptr addrspace\(3\) (%[\w.]+), align 2, !dbg !17$', line)
        if m:
            old = m[1]
            if old not in pointers:
                prefix = f"%ksw.read{len(pointers)}"
                output += [
                    f"  {prefix}.address = ptrtoint ptr addrspace(3) {old} to i32, !dbg !17",
                    f"  {prefix}.xor = xor i32 {prefix}.address, %ksw.read.mask, !dbg !17",
                    f"  {prefix}.pointer = inttoptr i32 {prefix}.xor to ptr addrspace(3), !dbg !17",
                ]
                pointers[old] = prefix + ".pointer"
            changed = line.replace(f"ptr addrspace(3) {old},", f"ptr addrspace(3) {pointers[old]},")
            undo[changed] = line
            line = changed
        output.append(line)
        if line.startswith("  %24 = call i32 @llvm.amdgcn.workitem.id.x()"):
            output += [
                "  %ksw.dma.bit = and i32 %24, 16",
                "  %ksw.dma.mask = shl i32 %ksw.dma.bit, 6",
                "  %ksw.read.bit = and i32 %24, 1",
                "  %ksw.read.mask = shl i32 %ksw.read.bit, 6",
            ]
    assert len(dma_offsets) == 8 and len(pointers) == 16 and len(undo) == 136
    restored = "\n".join(undo.get(line, line) for line in output
                         if not line.startswith("  %ksw.")) + "\n"
    assert restored == text, "Unexpected modification outside K addresses"
    return "\n".join(output) + "\n", dma_offsets, pointers


def main():
    assert sha(SOURCE) == SOURCE_SHA256, "Measured baseline has changed"
    source = SOURCE.read_text()
    candidate, dma_offsets, pointers = transform(source)
    BASELINE.write_text(source)
    CANDIDATE.write_text(candidate)
    manifest = {
        "baseline": str(SOURCE.relative_to(ROOT)), "baseline_sha256": SOURCE_SHA256,
        "baseline_program_median_ms": 1.8489584922790527,
        "candidate": CANDIDATE.name, "candidate_sha256": sha(CANDIDATE),
        "control": BASELINE.name, "control_sha256": sha(BASELINE),
        "gpu_tested": False, "static_lds_bytes": 16384,
        "lds_transform": "F(A) = A ^ ((A >> 2) & 64)",
        "producer_lane_permutation": "t' = t ^ ((t >> 2) & 4)",
        "producer_implementation": "K buffer offset ^= (thread_id & 16) << 6; same m0/resource and 16B DMA",
        "consumer_implementation": "K LDS byte address ^= (thread_id & 1) << 6",
        "dma_offsets": dma_offsets, "read_pointers": pointers,
        "unchanged": ["group8_reverse entry", "Q", "V", "MMAC", "waits/barriers", "output", "DMA widths/count"],
        "launch": {"kernel": "Attn_p2", "shape": [1, 32, 4096, 128],
                   "grid": [32, 32, 1], "block": [256, 1, 1], "shared_mem": 0, "k_layout": "bhsd"},
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {CANDIDATE}: 8 K DMA offsets, 16 pointers / 128 K loads; control byte-identical")


if __name__ == "__main__":
    main()
