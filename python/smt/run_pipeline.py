#!/usr/bin/env python3
"""Export a MyTest problem, solve it with CP-SAT, then import and lower to LLVM.

Pass MyTest arguments after the output filename after a standalone --, e.g.:
  python python/smt/run_pipeline.py input.mlir -o kernel.ll -- --k-tail-prefetch=4
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

try:
    from .lds_budget import DEFAULT_LDS_BUDGET_BYTES, inspect_llvm_lds, validate_lds_budget
    from .check_assembly_resources import inspect_assembly
except ImportError:
    from lds_budget import DEFAULT_LDS_BUDGET_BYTES, inspect_llvm_lds, validate_lds_budget
    from check_assembly_resources import inspect_assembly

ROOT = Path(__file__).resolve().parents[2]
SOLVER = Path(__file__).with_name("cpsat_scheduler.py")


def run_stage(name, command, work, env, artifacts):
    log = artifacts / f"{name}.log"
    print(f"[{name}] {shlex.join(command)}\n  log: {log}", flush=True)
    with log.open("w") as stream:
        stream.write(f"$ {shlex.join(command)}\n")
        stream.flush()
        result = subprocess.run(command, cwd=work, env=env, stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} failed (exit {result.returncode}); see {log}")


def require_file(path):
    if not path.is_file() or not path.stat().st_size:
        raise RuntimeError(f"expected nonempty output was not produced: {path}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    mytest_args = []
    if "--" in argv:
        split = argv.index("--")
        argv, mytest_args = argv[:split], argv[split + 1:]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path,
                        help="also copy successful LLVM output here; default: WORK_DIR/kernel.ll")
    parser.add_argument("--mytest", type=Path,
                        default=ROOT / "3rd/deepgengraph/build/test/MyTest")
    parser.add_argument("--work-dir", type=Path,
                        help="new directory for artifacts/logs; default: generated_ir/cpsat_runs/run-*")
    parser.add_argument("--provenance", type=Path,
                        help="load MyTest arguments and environment from a baseline provenance.json; "
                             "arguments after -- are appended")
    parser.add_argument("--loop", help="explicit MyTest loop selector, e.g. Attn_p2:0")
    parser.add_argument("--unroll-max-trip-count", type=int, default=0,
                        help="legacy Base compatibility; MyTest Fragment mode always fully unrolls static inner loops")
    parser.add_argument("--unroll-max-expanded-ops", type=int, default=2048,
                        help="legacy Base compatibility; ignored by MyTest Fragment full unrolling")
    # None leaves the solver's defaults as the single source of truth.
    parser.add_argument("--mode", choices=("intra", "pipeline"))
    parser.add_argument('--objective', choices=('makespan', 'ii'))
    parser.add_argument('--trip-count-upper-bound', type=int)
    parser.add_argument("--require-cross-iteration", action="store_true",
                        help="require actual cross-iteration overlap; needs --mode=pipeline")
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--costs", type=Path)
    parser.add_argument("--scale-work", action="store_true")
    parser.add_argument('--vgpr-budget', type=int, default=250)
    parser.add_argument('--register-reserve', type=int, default=32)
    parser.add_argument('--lds-read-ahead', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--llc', type=Path,
                        help='optional target-compatible llc; compile and enforce physical resources before publishing IR')
    parser.add_argument('--mcpu', help='target GPU for --llc, e.g. gfx936')
    parser.add_argument("--lds-budget", type=int, default=DEFAULT_LDS_BUDGET_BYTES,
                        help="hard LDS limit in bytes for solving and final LLVM (default: 16384 / 16 KiB)")
    for option in ("max-stages", "max-slots", "dma-capacity", "workers", "seed"):
        parser.add_argument(f"--{option}", type=int)
    parser.add_argument("--shm-check", choices=("auto", "constraint", "postcheck", "sum"))
    args = parser.parse_args(argv)
    try:
        validate_lds_budget(args.lds_budget)
        if args.vgpr_budget <= 0 or args.register_reserve < 0:
            raise ValueError('VGPR budget must be positive and register reserve nonnegative')
        if args.trip_count_upper_bound is not None and args.trip_count_upper_bound < 0:
            raise ValueError('trip count upper bound must be nonnegative')
        if bool(args.llc) != bool(args.mcpu):
            raise ValueError('--llc and --mcpu must be supplied together')
        source, mytest = args.input.resolve(), args.mytest.resolve()
        if args.unroll_max_trip_count < 0 or args.unroll_max_expanded_ops < 0:
            raise ValueError("unroll limits must be nonnegative")
        if args.require_cross_iteration and args.mode != 'pipeline':
            raise ValueError("--require-cross-iteration requires --mode=pipeline")
        require_file(source)
        if not mytest.is_file() or not os.access(mytest, os.X_OK):
            raise ValueError(f"MyTest is not executable: {mytest}; build MyTest first")
        env = dict(os.environ)
        if args.provenance:
            provenance = json.loads(args.provenance.read_text())
            mytest_args = provenance["mytest_arguments_after_output"] + mytest_args
            env.update(provenance["environment"])
        if any(arg.startswith("--cpsat-") for arg in mytest_args):
            raise ValueError("--cpsat-* flags are managed by this script; use --loop for loop selection")
        env.setdefault("FRISK_GM_TO_SHM_MODE", "async")
        solver_args = []
        for option in ("mode", "time-limit", "max-stages", "max-slots", "lds-budget",
                       "dma-capacity", "workers", "seed", "shm-check", "costs",
                       "vgpr-budget", "register-reserve", "objective", "trip-count-upper-bound"):
            value = getattr(args, option.replace("-", "_"))
            if value is not None:
                if isinstance(value, Path):
                    value = value.resolve()
                solver_args.append(f"--{option}={value}")
        if args.scale_work:
            solver_args.append("--scale-work")
        if args.lds_read_ahead is not None:
            solver_args.append('--lds-read-ahead' if args.lds_read_ahead else '--no-lds-read-ahead')
        if args.require_cross_iteration:
            solver_args.append("--require-cross-iteration")
        if args.work_dir:
            work = args.work_dir.resolve()
            work.mkdir(parents=True, exist_ok=False)
        else:
            parent = ROOT / "generated_ir/cpsat_runs"
            parent.mkdir(parents=True, exist_ok=True)
            work = Path(tempfile.mkdtemp(prefix="run-", dir=parent))
        print(f"Artifacts: {work}", flush=True)
        for stage in ("export", "lower"):
            (work / stage).mkdir()
        problem, schedule, llvm = (work / name for name in
                                   ("problem.json", "schedule.json", "kernel.ll"))
        prefix = [str(mytest), str(source), "0"]
        shared = mytest_args + ([f"--cpsat-loop={args.loop}"] if args.loop else [])
        if args.unroll_max_trip_count:
            shared += [f"--cpsat-unroll-max-trip-count={args.unroll_max_trip_count}",
                       f"--cpsat-unroll-max-expanded-ops={args.unroll_max_expanded_ops}"]
        run_stage("export", prefix + [str(work / "export/unused.ll"), *shared,
                  f"--cpsat-export={problem}", f"--cpsat-baseline={work / 'baseline.json'}",
                  f"--cpsat-dump-ir={work / 'base.mlir'}", "--cpsat-stop-after=1"],
                  work / "export", env, work)
        require_file(problem)
        run_stage("solve", [sys.executable, str(SOLVER), str(problem), "-o", str(schedule),
                            *solver_args], work, env, work)
        require_file(schedule)
        # MyTest re-analyzes the IR and validates the imported schedule before lowering.
        candidate = work / "lower/kernel.ll"
        run_stage("lower", prefix + [str(candidate), *shared, f"--cpsat-import={schedule}",
                  f"--cpsat-dump-ir={work / 'scheduled.mlir'}"], work / "lower", env, work)
        require_file(candidate)
        report_path = work / "lds_check.json"
        try:
            report = inspect_llvm_lds(candidate, args.lds_budget)
        except ValueError as error:
            report_path.write_text(json.dumps(dict(budget_bytes=args.lds_budget,
                within_budget=False, error=str(error)), indent=2) + "\n")
            raise
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        if not report['within_budget']:
            raise RuntimeError(f"final LDS {report['physical_lds_bytes']} bytes exceeds budget "
                               f"{args.lds_budget} bytes; see {report_path}")
        resources = dict(vgpr_budget=args.vgpr_budget, physical_registers_verified=False,
                         reason='target codegen not requested; IR estimates do not prove no spill')
        if args.llc:
            assembly = work / 'kernel.amdgcn'
            run_stage('codegen', [str(args.llc.resolve()), '-march=amdgcn',
                      f'-mcpu={args.mcpu}', '-filetype=asm', str(candidate), '-o', str(assembly)],
                      work, env, work)
            require_file(assembly)
            resources = inspect_assembly(assembly, args.vgpr_budget, args.lds_budget)
            resources['physical_registers_verified'] = True
        (work / 'assembly_check.json').write_text(json.dumps(resources, indent=2) + '\n')
        if args.llc and not resources['accepted']:
            raise RuntimeError(f'final register/spill check failed; see {work / "assembly_check.json"}')
        candidate.replace(llvm)
        output = args.output.resolve() if args.output else llvm
        if output != llvm:
            output.parent.mkdir(parents=True, exist_ok=True)
            # Preserve an existing output if any compiler/solver stage fails.
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.",
                                             delete=False) as stream:
                staging = Path(stream.name)
            try:
                shutil.copyfile(llvm, staging)
                staging.replace(output)
            finally:
                staging.unlink(missing_ok=True)
        print(f"Success: {output}\nProblem/schedule/IR/logs: {work}")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"run_pipeline: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("run_pipeline: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
