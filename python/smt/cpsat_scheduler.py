#!/usr/bin/env python3
"""Solve a Frisk Base scheduling problem exported by MyTest.

Examples:
  python python/smt/cpsat_scheduler.py problem.json -o schedule.json
  python python/smt/cpsat_scheduler.py problem.json -o pipeline.json --mode pipeline
  python python/smt/cpsat_scheduler.py problem.json -o baseline.json --baseline

The schedule contains op identities, stages, order and verified buffer counts.
Model times are advisory only: the compiler emits real waits and barriers.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from math import ceil
from pathlib import Path
import sys
import time

# ==================== 全局资源限额（默认配置） ====================
# 这些是调度预算，不是自动探测到的 GPU 硬件规格；CLI / solve() 参数可覆盖
# 已建模的限额。None 表示不限制，保留原来的默认行为。

# SHM / LDS：所选 loop 的逻辑活跃峰值（字节），含跨迭代重叠。
# 不包含布局/对齐/后续 scratch 和 loop 外无关存储，仍需后端物理容量检查。
SHM_LIMIT_BYTES: int | None = None
SHM_CHECK_MODE = "auto"          # auto / constraint / postcheck / sum（旧总量模型）
SHM_MAX_CUMULATIVE_INTERVALS = 512  # auto 超过此规模时改用 post-check
SHM_MAX_POSTCHECK_REJECTIONS = 32   # 超限候选加 no-good 后重求；耗尽则不输出非法解

# 寄存器：以 32-bit 寄存器为单位；VGPR 按 thread，SGPR 按 wave。
# 当前 JSON 没有寄存器活跃区间/分配信息，以下两项仅预留，尚未建模。
# 设成非 None 时求解器明确报错，避免看似设置了上限却没有实际约束。
VGPR_LIMIT_PER_THREAD: int | None = None
SGPR_LIMIT_PER_WAVE: int | None = None

DMA_MAX_IN_FLIGHT = 4       # 同时在途的 block 级 DMA op 数，不是底层 packet 数
MAX_PIPELINE_STAGES = 2    # 流水级数；还必须满足 MLIR problem.limits.max_stages
MAX_BUFFER_SLOTS = 2       # 每个 buffer 的最大版本数；还受 MLIR max_slots 限制
# 同步 op 的 issue 流当前固定为串行（容量 1），不支持配置成多发射。

# ==================== 求解器预算与模型规模保护 ====================
SOLVER_TIME_LIMIT_SECONDS = 15.0
SOLVER_WORKERS = 1
SOLVER_RANDOM_SEED = 0
MAX_OP_LATENCY = 100_000           # 单个 op 的相对耗时上限，不是 GPU cycles
MAX_TOTAL_LATENCY = 1_000_000      # 所有 op 的相对耗时之和上限
MAX_DMA_PERIOD_RADIUS = 128       # 周期 DMA 模型向前/后展开的最大周期数

from ortools.sat.python import cp_model

try:
    from .cost_model import CostModel
    from .schedule_schema import (baseline_schedule, make_schedule, read_problem,
                                  validate_problem, validate_schedule, write_json)
    from .shm_liveness import add_shm_constraint, analyze_shm, dma_transfers, interval_count, lifetime_specs
except ImportError:
    from cost_model import CostModel
    from schedule_schema import (baseline_schedule, make_schedule, read_problem,
                                 validate_problem, validate_schedule, write_json)
    from shm_liveness import add_shm_constraint, analyze_shm, dma_transfers, interval_count, lifetime_specs


def solve(problem: dict, *, mode: str = "intra", costs: CostModel | None = None,
          time_limit: float = SOLVER_TIME_LIMIT_SECONDS,
          max_stages: int = MAX_PIPELINE_STAGES, max_slots: int = MAX_BUFFER_SLOTS,
          lds_budget: int | None = SHM_LIMIT_BYTES, dma_capacity: int = DMA_MAX_IN_FLIGHT,
          shm_check: str = SHM_CHECK_MODE,
          workers: int = SOLVER_WORKERS, seed: int = SOLVER_RANDOM_SEED) -> dict:
    validate_problem(problem)
    if VGPR_LIMIT_PER_THREAD is not None or SGPR_LIMIT_PER_WAVE is not None:
        raise ValueError("VGPR/SGPR limits are not modeled yet; register allocation/liveness "
                         "information is missing from the problem JSON")
    if mode not in ("intra", "pipeline"):
        raise ValueError("mode must be intra or pipeline")
    if not 1 <= max_stages <= problem["limits"]["max_stages"]:
        raise ValueError("max_stages exceeds the materializer limit")
    if not 1 <= max_slots <= problem["limits"]["max_slots"]:
        raise ValueError("max_slots exceeds the materializer limit")
    if time_limit <= 0 or workers < 1 or dma_capacity < 1:
        raise ValueError("time limit, workers and DMA capacity must be positive")
    if lds_budget is not None and lds_budget < 0:
        raise ValueError("LDS budget must be nonnegative")
    if shm_check not in ("auto", "constraint", "postcheck", "sum"):
        raise ValueError("shm_check must be auto, constraint, postcheck or sum")
    if mode == "pipeline" and not problem["loop"]["pipeline_eligible"]:
        raise ValueError("pipeline is not supported: " + problem["loop"]["pipeline_reason"])
    costs = costs or CostModel()
    ops = problem["operations"]
    n = len(ops)
    stage_limit = max_stages if mode == "pipeline" else 1
    transfers = dma_transfers(problem)
    shm_specs = lifetime_specs(problem)
    if lds_budget is not None and any(b['bytes'] < 0 for b in shm_specs):
        raise ValueError("cannot enforce SHM budget with unknown buffer size")
    shm_intervals = interval_count(shm_specs, stage_limit)
    if shm_check == 'auto':
        shm_check = ('constraint' if lds_budget is not None and
                     shm_intervals <= SHM_MAX_CUMULATIVE_INTERVALS else 'postcheck')
    durations = [costs.operation(op) for op in ops]
    occupancy, latency = zip(*durations)
    if max(latency) > MAX_OP_LATENCY or sum(latency) > MAX_TOTAL_LATENCY:
        raise ValueError("cost horizon too large; increase reference work units")
    # The issue stream executes block ops sequentially; asynchronous DMA runs
    # independently after its issue. A period cannot end halfway through an op.
    minimum_ii = sum(occupancy)
    horizon = sum(latency) + n
    model = cp_model.CpModel()
    ii = model.new_int_var(minimum_ii, horizon, "period")
    phases = [model.new_int_var(0, horizon, f"phase_{i}") for i in range(n)]
    stages = [model.new_int_var(0, stage_limit - 1, f"stage_{i}") for i in range(n)]
    maximum = model.new_int_var(0, stage_limit - 1, "max_stage")
    minimum = model.new_int_var(0, stage_limit - 1, "min_stage")
    model.add_max_equality(maximum, stages)
    model.add_min_equality(minimum, stages)
    model.add(minimum == 0)
    pipelined = model.new_bool_var("pipelined")
    model.add(maximum > 0).only_enforce_if(pipelined)
    model.add(maximum == 0).only_enforce_if(pipelined.Not())

    starts, stream = [], []
    for i, op in enumerate(ops):
        base = model.new_int_var(0, horizon * stage_limit, f"base_{i}")
        model.add_multiplication_equality(base, [stages[i], ii])
        start = model.new_int_var(0, horizon * (stage_limit + 1), f"start_{i}")
        model.add(start == base + phases[i])
        starts.append(start)
        model.add(phases[i] + occupancy[i] <= ii)
        stream.append(model.new_fixed_size_interval_var(phases[i], occupancy[i], f"issue_{i}"))
        if op["pipeline_anchor"]:
            model.add(stages[i] == maximum)
    model.add_no_overlap(stream)

    # Strict phase order projects into a permutation, without ever emitting
    # sleeps or relying on model completion times for correctness.
    before = {}
    inversion_terms = []
    for a in range(n):
        for b in range(a + 1, n):
            lit = model.new_bool_var(f"before_{a}_{b}")
            model.add(phases[a] < phases[b]).only_enforce_if(lit)
            model.add(phases[a] > phases[b]).only_enforce_if(lit.Not())
            before[a, b] = lit
            inversion_terms.append(lit.Not())

    for edge in problem["dependencies"]:
        a, b = edge["src"], edge["dst"]
        constraints = []
        # Completion edges wait for the DMA, ordinary source dependencies wait
        # for the synchronous op. No estimated completion replaces a real wait.
        delay = latency[a] if edge["kind"] == "completion" else occupancy[a]
        constraints.append(model.add(starts[b] >= starts[a] + delay))
        if edge["same_stage"]:
            constraints.append(model.add(stages[a] == stages[b]))
        if edge["kind"] == "intra_sync":
            for constraint in constraints:
                constraint.only_enforce_if(pipelined.Not())

    # wait-all observes any previously issued DMA, including a future tile's
    # copy in the same steady step. If issue is later, it observes the previous
    # step's copy. This is why placing wait before future issue can be better.
    dmas = [i for i, op in enumerate(ops) if op["kind"] == "dma"]
    waits = [i for i, op in enumerate(ops) if op["kind"] == "wait"]
    if problem["wait_semantics"] != "wait_all":
        raise ValueError("v1 currently models the wait-all backend contract")
    for wait in waits:
        for dma in dmas:
            precedes = before[dma, wait] if dma < wait else before[wait, dma].Not()
            model.add(phases[wait] >= phases[dma] + latency[dma]).only_enforce_if(precedes)
            model.add(phases[wait] + ii >= phases[dma] + latency[dma]).only_enforce_if(precedes.Not())

    # Periodic DMA resource occupancy: enough translated intervals to cover
    # every overlap across a wrap boundary, including copies from prior steps.
    if dmas:
        reach = ceil(max(latency[i] for i in dmas) / minimum_ii) + 1
        if reach > MAX_DMA_PERIOD_RADIUS:
            raise ValueError("DMA cost/issue ratio too large for the bounded periodic resource model")
        dma_intervals = []
        for iteration in range(-reach, reach + 1):
            for i in dmas:
                shifted = model.new_int_var(-reach * horizon, (reach + 1) * horizon,
                                            f"dma_start_{i}_{iteration}")
                model.add(shifted == phases[i] + iteration * ii)
                dma_intervals.append(model.new_fixed_size_interval_var(
                    shifted, latency[i], f"dma_{i}_{iteration}"))
        model.add_cumulative(dma_intervals, [1] * len(dma_intervals), dma_capacity)

    buffer_slots, extra_bytes, allocated_bytes = [], [], []
    for b in problem["buffers"]:
        slot = model.new_int_var(1, max_slots, f"slots_{b['id']}")
        if b["rotatable"]:
            writer = b["writers"][0]
            model.add_max_equality(slot, [1] + [1 + stages[u] - stages[writer] for u in b["users"]])
            extra_bytes.append((slot - 1) * b["bytes"])
        else:
            model.add(slot == 1)
        buffer_slots.append(slot)
        if b["memory_space"] == 3:
            if b["bytes"] < 0 and lds_budget is not None:
                raise ValueError("cannot enforce LDS budget with unknown buffer size")
            allocated_bytes.append(slot * max(0, b["bytes"]))
    if lds_budget is not None and shm_check == 'sum':
        model.add(sum(allocated_bytes) <= lds_budget)
    # Rank time depends only on the emitted order, not on estimated idle gaps.
    # The same rank/stage pair always has the same SHM post-check result.
    ranks = [model.new_int_var(0, n - 1, f"rank_{i}") for i in range(n)]
    for i in range(n):
        model.add(ranks[i] == sum(before[j, i] for j in range(i)) +
                  sum(before[i, j].Not() for j in range(i + 1, n)))
    if lds_budget is not None and shm_check == 'constraint':
        add_shm_constraint(model, shm_specs, stages, ranks, buffer_slots,
                           n=n, stage_limit=stage_limit, budget=lds_budget)

    # Lexicographic solve: modeled period first, then extra LDS, then minimize
    # changes to source order. Later objectives never trade away a proven period.
    objectives = [ii, sum(extra_bytes), sum(inversion_terms)]
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    deadline = time.monotonic() + time_limit
    best = None
    proof = []
    rejected = 0
    failure_reason = "time budget exhausted without a feasible schedule"
    for objective in objectives:
        model.minimize(objective)
        candidate = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            solver.parameters.max_time_in_seconds = remaining
            status = solver.solve(model)
            name = solver.status_name(status)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                failure_reason = f"CP-SAT returned {name}; no SHM-feasible schedule was found"
                break
            stage_values = [solver.value(s) for s in stages]
            phase_values = [solver.value(t) for t in phases]
            orders = [solver.value(rank) for rank in ranks]
            candidate = make_schedule(problem, stage_values, orders, mode=mode, status=name)
            memory = analyze_shm(problem, candidate, budget=lds_budget, specs=shm_specs)
            if lds_budget is not None and not memory['within_budget']:
                if shm_check != 'postcheck':
                    raise RuntimeError('SHM cumulative constraint and independent liveness check disagree')
                rejected += 1
                failure_reason = (f"SHM post-check rejected {rejected} candidates; last logical peak "
                                  f"{memory['peak_bytes']} > budget {lds_budget}. "
                                  "Increase time/rejection budget or use --shm-check=constraint")
                candidate = None
                if rejected >= SHM_MAX_POSTCHECK_REJECTIONS:
                    break
                # Exclude this emitted schedule, not just its timing assignment.
                # Keep optimizing the current objective; never freeze an invalid II.
                model.add_forbidden_assignments(stages + ranks, [stage_values + orders])
                continue
            candidate = make_schedule(problem, stage_values, orders, mode=mode, status=name,
                             model={"period": solver.value(ii), "costs": asdict(costs),
                                    "dma_capacity": dma_capacity,
                                    "dma_transfers": transfers,
                                    "dma_summary": problem.get('dma_summary', {
                                        'direct_issues': len(transfers),
                                        'direct_waits': sum(o['kind'] == 'wait' for o in ops)}),
                                    "resource_limits": {
                                        "shm_bytes": lds_budget,
                                        "vgpr_per_thread": VGPR_LIMIT_PER_THREAD,
                                        "sgpr_per_wave": SGPR_LIMIT_PER_WAVE,
                                        "register_pressure_modeled": False,
                                        "dma_max_in_flight": dma_capacity,
                                        "max_stages": stage_limit, "max_slots": max_slots,
                                        "issue_stream_capacity": 1,
                                    },
                                    "shared_bytes_upper_bound": sum(solver.value(x) for x in allocated_bytes),
                                    "shm_liveness": memory,
                                    "shm_check": shm_check,
                                    "shm_cumulative_intervals": shm_intervals,
                                    "phase": phase_values,
                                    "latency": list(latency), "issue_occupancy": list(occupancy),
                                    "note": "relative cost optimum, not measured GPU time"})
            break
        if candidate is None:
            break
        best = candidate
        proof.append({"status": name, "objective": solver.objective_value,
                      "bound": solver.best_objective_bound})
        if status != cp_model.OPTIMAL:
            break
        model.add(objective == solver.value(objective))
    if best is None:
        raise RuntimeError(failure_reason + "; no schedule was written")
    best['model']['shm_postcheck_rejections'] = rejected
    best["model"]["lexicographic_solves"] = proof
    validate_schedule(problem, best)
    return best


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("problem", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("intra", "pipeline"), default="intra")
    parser.add_argument("--baseline", action="store_true", help="capture source order without solving")
    parser.add_argument("--costs", type=Path, help="JSON overrides for CostModel fields")
    parser.add_argument("--scale-work", action="store_true")
    parser.add_argument("--max-stages", type=int, default=MAX_PIPELINE_STAGES)
    parser.add_argument("--max-slots", type=int, default=MAX_BUFFER_SLOTS,
                        help="maximum versions per buffer, within the MLIR materializer limit")
    parser.add_argument("--lds-budget", type=int, default=SHM_LIMIT_BYTES,
                        help="logical SHM live-peak budget in bytes; sum mode uses unpooled total")
    parser.add_argument("--shm-check", choices=('auto', 'constraint', 'postcheck', 'sum'),
                        default=SHM_CHECK_MODE, help="SHM capacity enforcement strategy")
    parser.add_argument("--dma-capacity", type=int, default=DMA_MAX_IN_FLIGHT)
    parser.add_argument("--time-limit", type=float, default=SOLVER_TIME_LIMIT_SECONDS)
    parser.add_argument("--workers", type=int, default=SOLVER_WORKERS)
    parser.add_argument("--seed", type=int, default=SOLVER_RANDOM_SEED)
    args = parser.parse_args(argv)
    try:
        problem = read_problem(args.problem)
        settings = json.loads(args.costs.read_text()) if args.costs else {}
        if args.scale_work:
            settings["scale_work"] = True
        result = baseline_schedule(problem) if args.baseline else solve(
            problem, mode=args.mode, costs=CostModel(**settings), max_stages=args.max_stages,
            max_slots=args.max_slots,
            time_limit=args.time_limit, workers=args.workers, seed=args.seed,
            lds_budget=args.lds_budget, dma_capacity=args.dma_capacity, shm_check=args.shm_check)
        validate_schedule(problem, result)
        write_json(args.output, result)
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as error:
        print(f"cpsat_scheduler: {error}", file=sys.stderr)
        return 2
    print(f"{result['status']}: {args.output}; loop={result['loop_id']}; mode={result['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
