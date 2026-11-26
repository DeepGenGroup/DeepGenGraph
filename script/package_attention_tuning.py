#!/usr/bin/env python3
"""Build a self-contained script/template/MyTest bundle (no GPU toolchain)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
FILES = [
    "TuneAttention.py", "generate_attention_ir.py", "GenHsaco.sh", "launchKernel.py", "LICENSE",
    "调优参数说明.md",
    "3rd/deepgengraph/test/test_input.mlir",
    "3rd/deepgengraph/test/test_input_spidD.mlir",
    "script/test_tune_attention.py", "script/test_launch_kernel.py", "script/test_portable_tuning.py",
]
ASSETS = ["run_tuning.sh", "run_tuning.py", "tuning.json", "tuning.combinations.json",
          "tuning.combinations.simple.json", "tuning.swizzle.json", "tuning.online.json",
          "env.sh.example", "README.txt"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_of(command):
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mytest", type=Path, default=ROOT / "3rd/deepgengraph/build/test/MyTest")
    parser.add_argument("--llvm-license", type=Path, required=True, help="LLVM LICENSE.TXT for the linked LLVM/MLIR")
    args = parser.parse_args()
    created = datetime.now(timezone.utc)
    archive = (args.output or ROOT / "dist" / ("attention-tuning-" + created.strftime("%Y%m%d-%H%M%S") + ".tar.gz")).resolve()
    if archive.exists():
        parser.error(f"不覆盖已有文件：{archive}")
    sources = [ROOT / path for path in FILES]
    sources += [ROOT / "portable/attention-tuning" / path for path in ASSETS]
    sources += [args.mytest, args.llvm_license]
    for source in sources:
        if not source.is_file():
            parser.error(f"缺少文件：{source}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="attention-package-") as temp:
        bundle = Path(temp) / "attention-tuning"
        bundle.mkdir()
        for name in FILES:
            dest = bundle / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, dest)
        for name in ASSETS:
            shutil.copy2(ROOT / "portable/attention-tuning" / name, bundle / name)
        shutil.copy2(args.mytest, bundle / "MyTest")
        (bundle / "licenses").mkdir()
        shutil.copy2(args.llvm_license, bundle / "licenses/LLVM-LICENSE.txt")
        for name in ("MyTest", "run_tuning.sh", "run_tuning.py", "GenHsaco.sh"):
            (bundle / name).chmod(0o755)
        versions = output_of(["readelf", "--version-info", str(bundle / "MyTest")])
        requirements = sorted(set(re.findall(r"Name: ((?:GLIBCXX|GLIBC|CXXABI)_[0-9.]+)", versions)))
        record = {
            "created_utc": created.isoformat(),
            "source_commit": output_of(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip(),
            "source_state": "working tree snapshot; SHA256SUMS identifies the packaged files",
            "mytest_sha256": sha256(bundle / "MyTest"),
            "mytest_bytes": (bundle / "MyTest").stat().st_size,
            "mytest_file": output_of(["file", "-b", str(bundle / "MyTest")]).strip(),
            "mytest_version_requirements": requirements,
            "mytest_linked_libraries_on_build_host": output_of(["ldd", str(bundle / "MyTest")]),
            "default_gpu_arch": "gfx936",
            "bundled_gpu_toolchain": False,
            "bundled_python_environment": False,
            "hardware_benchmark_validated_on_build_host": False,
        }
        (bundle / "BUILD_INFO.json").write_text(json.dumps(record, indent=2) + "\n")
        checksums = [f"{sha256(path)}  {path.relative_to(bundle).as_posix()}\n"
                     for path in sorted(bundle.rglob("*")) if path.is_file()]
        (bundle / "SHA256SUMS").write_text("".join(checksums))
        with tarfile.open(archive, "x:gz", compresslevel=6) as tar:
            tar.add(bundle, arcname=bundle.name)
    checksum = sha256(archive)
    archive.with_name(archive.name + ".sha256").write_text(f"{checksum}  {archive.name}\n")
    print(archive)
    print(f"{archive.stat().st_size / 1024**2:.1f} MiB; SHA256={checksum}")


if __name__ == "__main__":
    main()
