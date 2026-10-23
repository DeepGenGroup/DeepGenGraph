#!/usr/bin/env python3
"""Wrap the measured best kernel in a two-query-tile loop (fixed BHSD)."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / "generated_ir/group8_reverse_k_gep/group8_reverse_k128_gep.ll"
NAME = "group8_reverse_k128_gep_two_bm"


def generate(source, reverse=True):
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)

    replace("  %bo.row = shl i32 %bo.y, 5\n"
            "  %bo.linear = add i32 %bo.row, %bo.x",
            "  %bo.row = shl i32 %bo.y, 4\n"
            "  %bm.physical = add i32 %bo.row, %bo.x\n"
            "  %bo.linear = shl i32 %bm.physical, 1")
    replace("  %7 = add i32 %bo.reverse, 0, !dbg !5",
            "  br label %bm.tile\n\n"
            "bm.tile:\n"
            "  %bm.iter = phi i32 [ 0, %4 ], [ %bm.next, %bm.sync ]\n"
            "  %7 = sub i32 %bo.reverse, %bm.iter, !dbg !5")
    # The original K/V loop now starts from bm.tile on EACH query tile, so
    # its iteration index, denominator and all four O accumulators reset.
    assert source.count(", %4 ]") == 7  # six original PHIs plus bm.iter
    source = source.replace(", %4 ]", ", %bm.tile ]")
    replace("%bm.iter = phi i32 [ 0, %bm.tile ]",
            "%bm.iter = phi i32 [ 0, %4 ]")
    replace("; preds = %826, %4", "; preds = %826, %bm.tile")
    replace("  ret void, !dbg !5",
            "  %bm.next = add nuw nsw i32 %bm.iter, 1\n"
            "  %bm.more = icmp ult i32 %bm.next, 2\n"
            "  br i1 %bm.more, label %bm.sync, label %bm.exit\n\n"
            "bm.sync:\n"
            "  call void @llvm.amdgcn.s.barrier()\n"
            "  br label %bm.tile, !llvm.loop !34\n\n"
            "bm.exit:\n"
            "  ret void, !dbg !5")
    assert "!34 =" not in source and "!35 =" not in source
    source += '\n!34 = distinct !{!34, !35}\n!35 = !{!"llvm.loop.unroll.disable"}\n'
    if not reverse:
        replace("  %bo.reverse = sub i32 31, %bo.query\n", "")
        replace("  %7 = sub i32 %bo.reverse, %bm.iter, !dbg !5",
                "  %7 = add i32 %bo.query, %bm.iter, !dbg !5")
    order = "reverse" if reverse else "forward"
    pairs = "q31/q30, q29/q28" if reverse else "q0/q1, q2/q3"
    return (f"; Two consecutive BM=128 tiles per workgroup; serial, {order} order.\n"
            "; Launch ONLY grid=(16,32,1), block=(256,1,1), BHSD=(1,32,4096,128).\n"
            f"; Original group8 order is paired: {pairs}, ... .\n"
            "; The two tiles independently reset accumulators and reuse 16 KiB LDS.\n"
            + source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward", action="store_true", help="Generate non-reverse group8 order")
    args = parser.parse_args()
    name = NAME.replace("group8_reverse", "group8_forward") if args.forward else NAME
    output = HERE / (name + ".ll")
    output.write_text(generate(SOURCE.read_text(), reverse=not args.forward))
    config = dict(schema_version=1, kernel="Attn_p2", shape=[1, 32, 4096, 128],
                  grid=[16, 32, 1], block=[256, 1, 1], shared_mem=0, k_layout="bhsd")
    (HERE / (name + ".launch.json")).write_text(json.dumps(config, indent=2) + "\n")
    manifest = dict(source=str(SOURCE.relative_to(ROOT)),
                    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    source_measured_ms=1.7790470123291016,
                    timing_evidence="pmc_test.log (UTF-16), group8_reverse_k128_gep.ll",
                    ir=output.name, sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    bm=128, bn=128, bk=32, tiles_per_block=2, static_lds_bytes=16384,
                    mapping="p=by*16+bx; l=2*p; h=(l>>3)&31; q0=31-((l>>8)*8+(l&7)); q1=q0-1",
                    launch=config, gpu_tested=False)
    if args.forward:
        manifest["mapping"] = "p=by*16+bx; l=2*p; h=(l>>3)&31; q0=(l>>8)*8+(l&7); q1=q0+1"
    manifest_name = "manifest.forward.json" if args.forward else "manifest.json"
    (HERE / manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
