#!/usr/bin/env python3
"""用当前 MyTest 导出 group4/forward 的 control、prefetch4、prefetch8。

仅改变基线的 reverse 和 fragment prefetch；BM=BN=128、BK=32、
threads=256、waves=4,1、retain=32、K swizzle=8:6 均保持不变。
control 也重新导出，以避免与旧 MyTest 产物混比。
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "generated_ir/spidD_noncasual/run-20261027-162059-vbwzllhc/online-baseline/config.json"
INPUT = ROOT / "3rd/deepgengraph/test/test_input_spidD_noncasual.mlir"
MYTEST = ROOT / "3rd/deepgengraph/build/test/MyTest"
LLVM = Path("/data2/xsl/install/bin")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    baseline = json.loads(BASE.read_text())
    if sha(INPUT) != baseline["input_sha256"]:
        raise SystemExit("输入与归档基线不一致，请先核对算法和 tile 参数")
    env_updates = {"FRISK_GM_TO_SHM_MODE": "async", "FRISK_GM_TO_SHM_ALIGNMENT": "16"}
    launch = {
        "schema_version": 1, "kernel": "Attn_p2", "shape": [1, 32, 4096, 128],
        "grid": [32, 32, 1], "block": [256, 1, 1], "shared_mem": 0,
        "k_layout": "bhsd", "causal": False,
    }
    manifest = {"baseline_config": str(BASE), "input_sha256": sha(INPUT),
                "mytest_sha256": sha(MYTEST), "gpu_tested": False, "variants": {}}
    for name, enabled, words in [("control", 0, 4), ("prefetch4", 1, 4), ("prefetch8", 1, 8)]:
        folder = HERE / name
        folder.mkdir(exist_ok=True)
        source = folder / "input.mlir"
        source.write_bytes(INPUT.read_bytes())
        output = folder / "kernel.ll"
        command = baseline["command"][3:].copy()  # 去掉 env 和两个环境赋值。
        command[0], command[1], command[3] = str(MYTEST), str(source), str(output)
        changes = {"workgroup-reverse": "0", "frag-prefetch": str(enabled),
                   "frag-prefetch-words": str(words)}
        for i, arg in enumerate(command):
            if arg.startswith("--") and "=" in arg:
                key = arg[2:].split("=", 1)[0]
                if key in changes:
                    command[i] = f"--{key}={changes[key]}"
        parameters = {**baseline["parameters"], **changes}
        (folder / "config.json").write_text(json.dumps(
            {"parameters": parameters, "command": command, "env": env_updates}, indent=2) + "\n")
        (folder / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")
        shell = shlex.join(["env", *(f"{k}={v}" for k, v in env_updates.items()), *command])
        (folder / "command.sh").write_text(
            f"#!/usr/bin/env bash\nset -euo pipefail\ncd -- {shlex.quote(str(folder))}\n{shell}\n")
        print(f"Generating {name}: frag-prefetch={enabled}, words={words}, group=4, reverse=0", flush=True)
        with (folder / "lowering.log").open("w") as log:
            subprocess.run(command, cwd=folder, env={**os.environ, **env_updates},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run([str(LLVM / "llvm-as"), str(output), "-o", "/dev/null"], check=True)
        subprocess.run([str(LLVM / "opt"), "-passes=default<O3>,verify", "-disable-output",
                        str(output)], check=True)
        reports = [line for line in (folder / "lowering.log").read_text().splitlines()
                   if line.startswith(("FragPrefetch:", "RetainInvariantFragments:"))]
        manifest["variants"][name] = {"parameters": parameters, "ir_sha256": sha(output),
                                      "llvm_verified": True, "reports": reports}
        print(f"Verified {output}\n" + "\n".join(reports), flush=True)
    def normalized(name):
        text = (HERE / name / "kernel.ll").read_text().replace(str(HERE / name), "VARIANT")
        return re.sub(r'"frisk.frag_prefetch"="[48]"', '"frisk.frag_prefetch"="BUDGET"', text)
    manifest["prefetch4_vs_8_equal_except_source_path_and_budget_attribute"] = (
        normalized("prefetch4") == normalized("prefetch8"))
    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
