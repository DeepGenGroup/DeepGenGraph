#!/usr/bin/env python3
"""Add an 8B Q layout swizzle using DWORD DMA to the measured K128 winner."""
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / "generated_ir/group8_reverse_k_swizzle/group8_reverse_k_swizzle128.ll"
SOURCE_SHA256 = "73234d33f58e848a06f363a42f1090a2952e71791ec6e2e409e762f944a880c1"
BASELINE = HERE / "group8_reverse_k128_control.ll"
CANDIDATE = HERE / "group8_reverse_q32_k128_swizzle.ll"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def transform(source):
    result, restored, pointers = [], [], {}
    dma_count = read_count = 0
    for line in source.splitlines():
        original = line
        if "buffer_load_dwordx4" in line and line.endswith("!dbg !11"):
            dst, off, rsrc = re.search(r'\(i32 (%[\w.]+), i32 (%[\w.]+), <4 x i32> (%[\w.]+)\)', line).groups()
            old_result = re.match(r'  (%\w+) =', line)[1]
            assert len(re.findall(re.escape(old_result) + r'(?![\w.])', source)) == 1
            prefix = f"%qsw.dma{dma_count}"
            result.append(f"  {prefix}.base = add i32 {off}, %qsw.source.delta, !dbg !11")
            for quarter in range(4):
                # Lane l writes DWORD at M + 256*j + 4*l. Its original
                # 16B packet owner is j*16 + l/4; F toggles packet bit3.
                result += [
                    f"  {prefix}.off{quarter} = add i32 {prefix}.base, {4096 * quarter}, !dbg !11",
                    f"  {prefix}.lds{quarter} = add i32 {dst}, {256 * quarter}, !dbg !11",
                ]
                name = old_result if quarter == 3 else f"{prefix}.call{quarter}"
                result.append(
                    f'  {name} = call i32 asm sideeffect "s_mov_b32 $0, $1\\0A\\09buffer_load_dword $2, $3, 0 offen lds", '
                    f'"={{m0}},s,v,s,~{{memory}}"(i32 {prefix}.lds{quarter}, i32 {prefix}.off{quarter}, <4 x i32> {rsrc}) #10, !dbg !11')
            dma_count += 1
            restored.append(original)
            continue
        m = re.search(r'= load <4 x half>, ptr addrspace\(3\) (%[\w.]+), align 2, !dbg !12$', line)
        if m:
            pointer = m[1]
            if pointer not in pointers:
                prefix = f"%qsw.read{len(pointers)}"
                result += [
                    f"  {prefix}.address = ptrtoint ptr addrspace(3) {pointer} to i32, !dbg !12",
                    f"  {prefix}.xor = xor i32 {prefix}.address, %qsw.read.mask, !dbg !12",
                    f"  {prefix}.pointer = inttoptr i32 {prefix}.xor to ptr addrspace(3), !dbg !12",
                ]
                pointers[pointer] = prefix + ".pointer"
            line = line.replace(f"ptr addrspace(3) {pointer},", f"ptr addrspace(3) {pointers[pointer]},")
            read_count += 1
        result.append(line)
        restored.append(original)
        if line.startswith("  %24 = call i32 @llvm.amdgcn.workitem.id.x()"):
            result += [
                "  %qsw.lane = and i32 %24, 63",
                "  %qsw.owner = lshr i32 %qsw.lane, 2",
                "  %qsw.row.delta = sub i32 %qsw.owner, %qsw.lane",
                "  %qsw.byte.delta = shl i32 %qsw.row.delta, 8",
                "  %qsw.word = and i32 %qsw.lane, 3",
                "  %qsw.word.byte = shl i32 %qsw.word, 2",
                "  %qsw.half = lshr i32 %qsw.lane, 2",
                "  %qsw.half.mask = and i32 %qsw.half, 8",
                "  %qsw.intra = xor i32 %qsw.word.byte, %qsw.half.mask",
                "  %qsw.source.delta = add i32 %qsw.byte.delta, %qsw.intra",
                "  %qsw.read.mask = and i32 %24, 8",
            ]
    assert dma_count == 8 and read_count == 16 and len(pointers) == 4
    assert "\n".join(restored) + "\n" == source
    candidate = "\n".join(result) + "\n"
    # Every original instruction outside Q DMA/read sites is retained verbatim.
    for line in source.splitlines():
        if ("buffer_load_dwordx4" in line and line.endswith("!dbg !11")) or (
                "= load <4 x half>, ptr addrspace(3)" in line and line.endswith("!dbg !12")):
            continue
        assert line in candidate.splitlines(), line
    return candidate


def main():
    assert sha(SOURCE) == SOURCE_SHA256
    source = SOURCE.read_text()
    BASELINE.write_text(source)
    CANDIDATE.write_text(transform(source))
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": sha(SOURCE),
        "source_program_median_ms": 1.8204780220985413,
        "candidate": CANDIDATE.name, "candidate_sha256": sha(CANDIDATE),
        "control": BASELINE.name, "control_sha256": sha(BASELINE),
        "Q_transform": "F(A) = A ^ ((A >> 4) & 8)",
        "Q_DMA": "8 dwordx4 sites replaced by 32 dword sites; no address bpermute, no staging",
        "Q_read": "16 b64 load sites; byte address ^= thread_id & 8",
        "K_V_DMA": "Unchanged buffer_load_dwordx4 lds (128 bits)",
        "static_lds_bytes": 16384, "waits_barriers_entry_arithmetic_unchanged": True,
        "gpu_tested": False, "target_gfx936_codegen_tested": False,
        "launch": {"kernel": "Attn_p2", "shape": [1, 32, 4096, 128],
                   "grid": [32, 32, 1], "block": [256, 1, 1], "shared_mem": 0, "k_layout": "bhsd"},
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Generated {CANDIDATE.name}; Q32/K128/V128, 16 KiB LDS")


if __name__ == "__main__":
    main()
