#!/usr/bin/env python3
"""Rebuild the pinned group8_reverse with only wait coalescing changed."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "generated_ir/block_order"))
from generate import entry


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "generated_ir/pipeline_cleanup/nohint_unrolled.mlir"
    original = ROOT / "generated_ir/block_order/group8_reverse.ll"
    assert sha(original) == "b82cdd9f10f130753d896a15fc0acb46841bd955e9cacae2bb4f44f453816eb2"
    tool = ROOT / "3rd/deepgengraph/build/test/MyTest"
    env = {**os.environ, "FRISK_GM_TO_SHM_MODE": "async", "FRISK_GM_TO_SHM_ALIGNMENT": "16"}
    records = {}
    for waits in (0, 1):
        name = "group8_reverse" + ("_coalesce_waits" if waits else "_control")
        raw = HERE / (name + ".head_fast.ll")
        command = [str(tool), str(source), "0", str(raw),
                   "1", "1", "3.3", "34816", "approx", "256", "4,1",
                   "1", "1", "1", "0", "0", "useKPack", "asm128", "32", "32",
                   "--frag-prefetch=1", "--frag-prefetch-words=4",
                   "--simplify-loop-control=1", f"--coalesce-waits={waits}"]
        (HERE / (name + ".command.sh")).write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n" +
            "cd -- " + shlex.quote(str(HERE)) + "\n" +
            shlex.join(["env", "FRISK_GM_TO_SHM_MODE=async", "FRISK_GM_TO_SHM_ALIGNMENT=16", *command]) + "\n")
        with (HERE / (name + ".log")).open("w") as log:
            subprocess.run(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=240)
        text = raw.read_text()
        query_call = "  %7 = call i32 @llvm.amdgcn.workgroup.id.y(), !dbg !5"
        head_call = "  %9 = call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !5"
        assert text.count(query_call) == text.count(head_call) == 1
        query, head = entry(8, True)
        mapped = text.replace(query_call, query).replace(head_call, head)
        output = HERE / (name + ".ll")
        output.write_text(mapped)
        if not waits:
            assert mapped == original.read_text(), "Rebuilt control differs from pinned group8_reverse"
        llvm = Path("/data2/xsl/install/bin")
        subprocess.run([llvm / "llvm-as", output, "-o", HERE / (name + ".bc")], check=True)
        subprocess.run([llvm / "opt", "-passes=default<O3>,verify", output, "-disable-output"], check=True)
        assert "@shm_0 = addrspace(3) global [16384 x i8] undef, align 16" in mapped
        records[name] = {
            "file": output.name, "sha256": sha(output), "coalesce_waits": bool(waits),
            "byte_identical_to_original": mapped == original.read_text(),
            "command": command, "llvm_verify": True, "llvm_o3_verify": True,
            "static_waits": mapped.count("s_waitcnt vmcnt(0)"),
            "static_barriers": len(re.findall(r"call void @llvm.amdgcn.s.barrier\(", mapped)),
            "static_lds_bytes": 16384,
        }
        print("PASS", name, records[name], flush=True)
    (HERE / "manifest.json").write_text(json.dumps({
        "source": str(source), "source_sha256": sha(source), "compiler_sha256": sha(tool),
        "baseline": str(original), "baseline_sha256": sha(original),
        "control_byte_identical": True, "only_compiler_flag_change": "--coalesce-waits=0 -> 1",
        "coalescing_changed_ir": records["group8_reverse_control"]["sha256"] != records["group8_reverse_coalesce_waits"]["sha256"],
        "mapping": "group8_reverse", "gpu_tested": False, "variants": records,
    }, indent=2) + "\n")
    (HERE / "launch.json").write_text(json.dumps({
        "kernel": "Attn_p2", "shape": [1, 32, 4096, 128], "grid": [32, 32, 1],
        "block": [256, 1, 1], "shared_mem": 0, "k_layout": "bhsd",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
