#!/usr/bin/env python3
"""解压后运行 ./run_tuning.sh；所有相对配置路径以包根目录为基准。"""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parent
BACKEND_TOOLS = ("llvm-link", "opt", "llc", "ld.lld")


def executable(value):
    """Preserve venv interpreter symlinks."""
    if os.sep not in value:
        result = shutil.which(value)
        if result:
            return result
    else:
        path = Path(value).expanduser()
        path = path if path.is_absolute() else ROOT / path
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.absolute())
    raise ValueError(f"找不到可执行文件：{value}")


def settings(path, smoke):
    config = json.loads(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("调优配置必须是 JSON 对象")
    config.setdefault("softmax", "online")
    config.setdefault("sweep_base", None)
    if config["softmax"] not in ("online", "legacy"):
        raise ValueError("softmax 必须为 online 或 legacy")
    if config["sweep_base"] is not None and (
            not isinstance(config["sweep_base"], str) or not config["sweep_base"]):
        raise ValueError("sweep_base 必须为非空预设名或 null")
    for key in ("bm", "bn", "bk"):
        values = config[key]
        minimum = 0 if key == "bk" else 1
        if not isinstance(values, list) or not values or any(type(x) is not int or x < minimum for x in values):
            constraint = "非负整数列表（0 表示不传 --bk）" if key == "bk" else "正整数列表"
            raise ValueError(f"{key} 必须为非空{constraint}")
        config[key] = list(dict.fromkeys(values))
    for key in ("cases", "set", "sweep"):
        if not isinstance(config[key], list) or any(not isinstance(x, str) for x in config[key]):
            raise ValueError(f"{key} 必须为字符串列表")
    if not config["cases"] and not config["sweep"]:
        raise ValueError("cases 或 sweep 至少指定一项，避免意外运行全部 coverage")
    for key, minimum in (("device", 0), ("warmup", 0), ("repeat", 1), ("opt_level", 0)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f"{key} 数值不合法")
    if config["opt_level"] > 3:
        raise ValueError("opt_level 只能为 0..3")
    if (type(config["timeout"]) not in (int, float) or
            not math.isfinite(config["timeout"]) or config["timeout"] <= 0):
        raise ValueError("timeout 必须为有限正数")
    if type(config["seed"]) is not int:
        raise ValueError("seed 必须为整数")
    if not isinstance(config["output_dir"], str) or not config["output_dir"]:
        raise ValueError("output_dir 必须为非空路径")
    if config["sweep"] and config["cases"]:
        raise ValueError("使用 sweep 时请将 cases 设为 []；扫描候选名是 grid-*")
    if config["sweep_base"] and not config["sweep"]:
        raise ValueError("sweep_base 必须与 sweep 一起使用")
    if smoke:
        config.update(bm=[128], bn=[128], bk=[32], softmax="online",
                      cases=["online-baseline"], sweep_base=None,
                      set=[], sweep=[], warmup=1, repeat=3)
    return config


def backend_environment():
    env = os.environ.copy()
    if env.get("LLVM_DIR"):
        candidates = [Path(env["LLVM_DIR"]).expanduser()]
    else:
        candidates = [Path(p) for p in (
            "/opt/dtk/llvm/bin", "/opt/dtk/aillvm/bin", "/opt/rocm/llvm/bin",
            "/workspace/deepgengraph/dtk/aillvm/bin")]
        if shutil.which("llc"):
            candidates.append(Path(shutil.which("llc")).parent)
    llvm = next((p for p in candidates if all(
        (p / tool).is_file() and os.access(p / tool, os.X_OK) for tool in BACKEND_TOOLS)), None)
    if llvm is None:
        raise ValueError("找不到完整 DTK/ROCm LLVM 工具链（llvm-link/opt/llc/ld.lld）。"
                         "请在 env.sh 设置 LLVM_DIR。")
    env["LLVM_DIR"] = str(llvm.resolve())
    arch = env.get("GPU_ARCH", "gfx936")
    env["GPU_ARCH"] = arch
    libraries = ["ocml", "ockl", "oclc_isa_version_" + arch.removeprefix("gfx"),
                 "oclc_abi_version_400", "oclc_wavefrontsize64_on",
                 "oclc_correctly_rounded_sqrt_off", "oclc_finite_only_off",
                 "oclc_unsafe_math_off", "oclc_daz_opt_off"]
    candidates = ([Path(env["ROCM_BC"]).expanduser()] if env.get("ROCM_BC") else
                  [Path("/opt/dtk/amdgcn/bitcode"), Path("/opt/rocm/amdgcn/bitcode"),
                   llvm.parent.parent / "amdgcn/bitcode",
                   llvm.parent.parent.parent / "amdgcn/bitcode"])
    bitcode = next((p for p in candidates if all((p / (name + ".bc")).is_file()
                                                for name in libraries)), None)
    if bitcode is None:
        raise ValueError(f"找不到 {arch} 所需的完整 device bitcode。请在 env.sh 设置 ROCM_BC。")
    env["ROCM_BC"] = str(bitcode.resolve())
    return env


def check_environment(mode, config):
    tool = executable(os.environ.get("MYTEST", str(ROOT / "MyTest")))
    # The known MyTest returns 1 and prints usage with no input; never invokes GPU.
    probe = subprocess.run([tool], capture_output=True, text=True, timeout=20)
    usage = probe.stdout + probe.stderr
    if probe.returncode not in (0, 1) or "usage:" not in usage.lower():
        raise ValueError("MyTest 无法运行。随包版本需要 x86-64 Linux、glibc>=2.34、"
                         "GLIBCXX_3.4.30；可用 MYTEST 指向同版源码在目标机编译的程序。\n" + usage[-2000:])
    print("MyTest:", tool, flush=True)
    env = os.environ.copy() if mode == "ir" else backend_environment()
    gpu_python = executable(env.get("GPU_PYTHON", sys.executable)) if mode == "benchmark" else sys.executable
    record = {"mode": mode, "mytest": tool, "python": sys.executable, "gpu_python": gpu_python}
    if mode != "ir":
        print("LLVM_DIR:", env["LLVM_DIR"], "\nROCM_BC:", env["ROCM_BC"],
              "\nGPU_ARCH:", env["GPU_ARCH"], flush=True)
        record.update({key: env.get(key) for key in
                       ("LLVM_DIR", "ROCM_BC", "GPU_ARCH", "GPU_FEATURES", "DWARF_VERSION")})
    if mode == "benchmark":
        probe = subprocess.run([gpu_python, "-c", """
import ctypes, json, sys
import numpy
import torch
ctypes.CDLL("libamdhip64.so")
if not torch.version.hip or not torch.cuda.is_available():
    raise RuntimeError("需要能访问 AMD/DCU GPU 的 HIP 版 PyTorch")
device = int(sys.argv[1])
torch.cuda.set_device(device)
p = torch.cuda.get_device_properties(device)
print("GPU_CHECK_JSON=" + json.dumps({
    "device": device, "name": p.name, "arch": getattr(p, "gcnArchName", None),
    "pytorch": torch.__version__, "hip": torch.version.hip, "numpy": numpy.__version__}))
""", str(config["device"])], env=env, capture_output=True, text=True, timeout=60)
        if probe.returncode:
            raise ValueError("GPU 环境检查失败；请设置 GPU_PYTHON、HIP 动态库路径和设备权限。\n"
                             + probe.stderr[-3000:])
        data = next((line.split("=", 1)[1] for line in probe.stdout.splitlines()
                     if line.startswith("GPU_CHECK_JSON=")), None)
        if data is None:
            raise ValueError("GPU 环境检查未返回设备信息")
        record["gpu"] = json.loads(data)
        actual_arch = record["gpu"]["arch"]
        if actual_arch and actual_arch.split(":")[0] != env["GPU_ARCH"]:
            raise ValueError(f"GPU 架构 {actual_arch} 与 GPU_ARCH={env['GPU_ARCH']} 不匹配。"
                             "需使用匹配该 GPU 的 MyTest 和代码生成配置。")
        print("GPU:", record["gpu"]["name"], flush=True)
    return tool, gpu_python, env, record


def commands(config, mode, session, tool, gpu_python):
    generate = [sys.executable, str(ROOT / "generate_attention_ir.py"),
                "--output-dir", str(session / "inputs"),
                "--softmax", config.get("softmax", "online")]
    for key in ("bm", "bn"):
        generate += ["--" + key, *map(str, config[key])]
    # Separate invocations let BK=0 use the generator's existing no-BK mode.
    # Keep every invocation's paths in one manifest, including mixed scans.
    generations = [generate + (["--bk", str(bk)] if bk else []) for bk in config["bk"]]
    tune = [sys.executable, str(ROOT / "TuneAttention.py"),
            "--input-list", str(session / "inputs.txt"), "--tool", tool,
            "--output-dir", str(session / "tuning"), "--summary-dir", str(session),
            "--launch-python", gpu_python,
            "--gen-hsaco", str(ROOT / "GenHsaco.sh"), "--launcher", str(ROOT / "launchKernel.py")]
    for key in ("cases", "set", "sweep"):
        for value in config[key]:
            tune += ["--case" if key == "cases" else "--" + key, value]
    if config.get("sweep_base"):
        tune += ["--sweep-base", config["sweep_base"]]
    for key in ("opt_level", "device", "warmup", "repeat", "seed", "timeout"):
        tune += ["--" + key.replace("_", "-"), str(config[key])]
    if mode != "ir":
        tune += ["--benchmark" if mode == "benchmark" else "--hsaco"]
    if os.environ.get("LLVM_AS"):
        tune += ["--llvm-as", executable(os.environ["LLVM_AS"])]
    return generations, tune


def run_tuning(command, env):
    """Forward cancellation to the tuner so it can stop GPU children and checkpoint."""
    with subprocess.Popen(command, env=env, start_new_session=True) as process:
        try:
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def interrupted(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "tuning.json")
    parser.add_argument("--mode", choices=("benchmark", "hsaco", "ir"), default="benchmark")
    parser.add_argument("--smoke", action="store_true", help="只跑 online 128x128x32 online-baseline，预热1次/测量3次")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="只检查当前模式的运行环境")
    modes.add_argument("--dry-run", action="store_true", help="只打印完整命令，不创建文件或检查 GPU")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        config = settings(args.config, args.smoke)
        output = (args.output_dir or ROOT / config["output_dir"]).resolve()
        if args.dry_run:
            session = output / "PREVIEW"
            generations, tune = commands(config, args.mode, session,
                                      os.environ.get("MYTEST", str(ROOT / "MyTest")),
                                      os.environ.get("GPU_PYTHON", sys.executable))
            print("mkdir -p -- " + shlex.quote(str(session)))
            for index, generate in enumerate(generations):
                redirect = " > " if index == 0 else " >> "
                print(shlex.join(generate) + redirect + shlex.quote(str(session / "inputs.txt")))
            print(shlex.join(tune))
            return 0
        tool, gpu_python, env, record = check_environment(args.mode, config)
        if args.check:
            print("环境检查通过。")
            return 0
        output.mkdir(parents=True, exist_ok=True)
        session = Path(tempfile.mkdtemp(prefix=time.strftime("session-%Y%m%d-%H%M%S-"), dir=output))
        (session / "settings.json").write_text(json.dumps(config, indent=2) + "\n")
        (session / "environment.json").write_text(json.dumps(record, indent=2) + "\n")
        generations, tune = commands(config, args.mode, session, tool, gpu_python)
        print("本次目录:", session, flush=True)
        with (session / "inputs.txt").open("w") as stream:
            for generate in generations:
                print("生成输入:", shlex.join(generate), flush=True)
                subprocess.run(generate, stdout=stream, env=env, check=True)
        print("开始调优:", shlex.join(tune), flush=True)
        print("实时进度:", session / "summary.json",
              "\n实时最大加速比及配置:", session / "best_by_shape.json", flush=True)
        result = run_tuning(tune, env)
        if result:
            print("存在失败候选；已完成的结果仍保存在上述目录。", file=sys.stderr)
        return result
    except KeyboardInterrupt:
        print("\n调优已中断；已完成结果保存在本次 session 的 JSON 文件中。", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print("错误:", exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
