#!/usr/bin/env python3
"""Generate entry-only permutations of the pinned bx=query/by=head baseline."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "baseline_swap_bx_by.ll"
SOURCE_SHA256 = "24bdcbd39aa0761fee22119b1ff6c80bda2d637eaa8e691b51deda885ce2214c"
VARIANTS = {
    "baseline": (32, False),
    "head_fast": (1, False),
    "group2": (2, False),
    "group4": (4, False),
    "group8": (8, False),
    "group16": (16, False),
    "query_fast": (32, False),
    "group8_reverse": (8, True),
    "query_fast_reverse": (32, True),
}
# query_fast is retained as a compatibility alias, not a second default benchmark.
DEFAULT_VARIANTS = [name for name in VARIANTS if name != "query_fast"]
QUERY_CALL = "  %7 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5"
HEAD_CALL = "  %9 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def entry(group, reverse):
    if group == 32 and not reverse:
        return QUERY_CALL, HEAD_CALL
    if group == 1 and not reverse:
        return QUERY_CALL.replace("id.x()", "id.y()"), HEAD_CALL.replace("id.y()", "id.x()")
    lines = [
        "  %bo.x = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5",
        "  %bo.y = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5",
    ]
    if group == 32:
        query, head = "%bo.x", "%bo.y"
    else:
        # t=x+32*y; q=(t//(32*G))*G+t%G; h=(t//G)%32.
        # All operations are uniform and their divisors are powers of two.
        shift = group.bit_length() - 1
        lines += [
            "  %bo.row = shl i32 %bo.y, 5",
            "  %bo.linear = add i32 %bo.row, %bo.x",
            f"  %bo.within = and i32 %bo.linear, {group - 1}",
            f"  %bo.chunk = lshr i32 %bo.linear, {5 + shift}",
            f"  %bo.start = shl i32 %bo.chunk, {shift}",
            "  %bo.query = add i32 %bo.start, %bo.within",
            f"  %bo.head_chunk = lshr i32 %bo.linear, {shift}",
            "  %bo.head = and i32 %bo.head_chunk, 31",
        ]
        query, head = "%bo.query", "%bo.head"
    if reverse:
        lines.append(f"  %bo.reverse = sub i32 31, {query}")
        query = "%bo.reverse"
    # Retain numeric SSA slots so every downstream instruction stays identical.
    lines.append(f"  %7 = add i32 {query}, 0, !dbg !5")
    return "\n".join(lines), f"  %9 = add i32 {head}, 0, !dbg !5"


def main():
    if digest(SOURCE) != SOURCE_SHA256:
        raise SystemExit("Source changed: inspect its ABI/SSA mapping before updating the pinned hash.")
    source = SOURCE.read_text()
    assert source.count(QUERY_CALL) == source.count(HEAD_CALL) == 1
    manifest = {
        "schema_version": 2,
        "source": str(SOURCE.relative_to(HERE.parent.parent)),
        "source_sha256": SOURCE_SHA256,
        "baseline_mapping": {"head": "workgroup.id.y", "query_tile": "workgroup.id.x"},
        "reference_variant": "baseline",
        "default_variants": DEFAULT_VARIANTS,
        "shape_bhsd": [1, 32, 4096, 128],
        "tile_bm_bn_bk": [128, 128, 32],
        "grid": [32, 32, 1], "block": [256, 1, 1],
        "kernel": "Attn_p2", "k_layout": "bhsd",
        "static_lds_bytes": 16384, "gpu_tested": False,
        "variants": {},
    }
    for name, (group, reverse) in VARIANTS.items():
        query, head = entry(group, reverse)
        text = source.replace(QUERY_CALL, query).replace(HEAD_CALL, head)
        assert text.replace(query, QUERY_CALL).replace(head, HEAD_CALL) == source
        path = HERE / f"{name}.ll"
        path.write_text(text)
        manifest["variants"][name] = {
            "file": path.name, "sha256": digest(path),
            "query_group_size": group, "reverse_query": reverse,
            "entry_only_change": True,
            "alias_of": "baseline" if name == "query_fast" else None,
        }
        print(name, digest(path))
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
