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
try:
    from .lds_budget import DEFAULT_LDS_BUDGET_BYTES, validate_lds_budget
except ImportError:
    from lds_budget import DEFAULT_LDS_BUDGET_BYTES, validate_lds_budget

# SHM / LDS：所选 loop 的逻辑活跃峰值（字节），含跨迭代重叠。
# 不包含布局/对齐/后续 scratch 和 loop 外无关存储，仍需后端物理容量检查。
SHM_LIMIT_BYTES: int = DEFAULT_LDS_BUDGET_BYTES
SHM_CHECK_MODE = "auto"          # auto / constraint / postcheck / sum（旧总量模型）
SHM_MAX_CUMULATIVE_INTERVALS = 512  # auto 超过此规模时改用 post-check
SHM_MAX_POSTCHECK_REJECTIONS = 32   # 超限候选加 no-good 后重求；耗尽则不输出非法解

# 寄存器：以 32-bit 寄存器为单位；VGPR 按 thread，SGPR 按 wave。
# 以下旧常量仍不代表物理寄存器分配约束。Fragment 数据活跃期使用
# solve(vgpr_budget=250, register_reserve=32)，SGPR 尚无独立模型。
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
    from .makespan import add_makespan, resolve_trip_count
    from .register_liveness import add_register_constraint, analyze_registers
    from .register_hint import register_hint
    from .lds_overlap import lds_overlap_rules, check_lds_overlap
    from .schedule_schema import (baseline_schedule, make_schedule, read_problem,
                                  validate_problem, validate_schedule, write_json)
    from .shm_liveness import add_shm_constraint, add_periodic_storage, analyze_shm, dma_transfers, interval_count, lifetime_specs
except ImportError:
    from cost_model import CostModel
    from makespan import add_makespan, resolve_trip_count
    from register_liveness import add_register_constraint, analyze_registers
    from register_hint import register_hint
    from lds_overlap import lds_overlap_rules, check_lds_overlap
    from schedule_schema import (baseline_schedule, make_schedule, read_problem,
                                 validate_problem, validate_schedule, write_json)
    from shm_liveness import add_shm_constraint, add_periodic_storage, analyze_shm, dma_transfers, interval_count, lifetime_specs


def pipeline_hint(problem, occupancy, latency, specs, alignment, budget,
                  vgpr_budget=None, register_reserve=32, order_edges=()):
    """Seed the joint search with one legal lookahead, never constrain it to it.

    Keep compute order, prefetch each DMA after the previous tile is ready,
    and place next-iteration lookahead near the final shared reader. Issuing
    the next iteration at its earliest legal point can monopolize half of a
    two-tile LDS pool for the entire iteration. Validate both dependencies
    and periodic packing; fall back to simpler seeds if lookahead does not fit.
    """
    ops, n = problem['operations'], len(problem['operations'])
    for b in sorted(specs, key=lambda b: -len(b['users'])):
        writer = b['writer']
        early = {writer}
        while True:
            previous = len(early)
            for edge in problem['dependencies']:
                if edge['kind'] == 'intra_sync':
                    continue
                if edge['dst'] in early or (edge['same_stage'] and edge['src'] in early):
                    early.update((edge['src'], edge['dst']))
            if len(early) == previous:
                break
        if any(ops[i]['pipeline_anchor'] for i in early):
            continue
        stages = [0 if i in early else 1 for i in range(n)]
        def at_cut(cut):
            return ([i for i in range(cut + 1) if i not in early] + sorted(early) +
                    [i for i in range(cut + 1, n) if i not in early])
        def validate_order(sequence, check_overlap=False):
            ranks = [0] * n
            for rank, i in enumerate(sequence):
                ranks[i] = rank
            validate_schedule(problem, make_schedule(problem, stages, ranks,
                              mode='pipeline', status='HINT'))
            if check_overlap and any(stages[a] != stages[b] or ranks[a] >= ranks[b] for a, b in order_edges):
                raise ValueError('seed violates LDS read-ahead order')
            return ranks
        last_reader = max((i for spec in specs for i in spec['users']
                           if ops[i]['kind'] == 'shm_reg'), default=max(b['users']))
        late = at_cut(max(max(b['users']), last_reader))
        prefetch = list(late)
        transfers = dma_transfers(problem)
        for previous, current in zip(transfers, transfers[1:]):
            issue, ready = current['issue'], previous['wait']
            if issue in early or ready in early:
                continue
            positions = {i: rank for rank, i in enumerate(prefetch)}
            movable = {issue}
            while True:
                old = set(movable)
                for edge in problem['dependencies']:
                    if edge['kind'] != 'intra_sync' and edge['dst'] in movable:
                        src = edge['src']
                        if positions[src] > positions[ready]:
                            movable.add(src)
                if movable == old:
                    break
            if any(i in early or ops[i]['kind'] not in ('dma', 'address', 'alloc') for i in movable):
                continue
            moved = [i for i in prefetch if i in movable]
            candidate = [i for i in prefetch if i not in movable]
            insertion = candidate.index(ready) + 1
            candidate[insertion:insertion] = moved
            try:
                validate_order(candidate)
            except ValueError:
                continue
            prefetch = candidate
        selected = None
        # A source-biased retirement policy avoids starting several independent
        # output rows at once; depth-first is useful for other DAG shapes.
        compact = [register_hint(problem, stages, prefetch, vgpr_budget, register_reserve,
                                 policy=policy, order_edges=order_edges) for policy in ('source', 'depth')]
        for sequence in (*compact, prefetch, late, at_cut(max(b['users']))):
            if sequence is None:
                continue
            try:
                ranks = validate_order(sequence, check_overlap=True)
                if vgpr_budget is not None and not analyze_registers(problem,
                        make_schedule(problem, stages, ranks, mode='pipeline', status='HINT'),
                        budget=vgpr_budget, reserve=register_reserve)['within_budget']:
                    continue
            except ValueError:
                continue
            packing = cp_model.CpModel()
            offsets = add_periodic_storage(packing, specs, stages, ranks, n=n,
                                           budget=budget, alignment=alignment)
            pack_solver = cp_model.CpSolver()
            pack_solver.parameters.max_time_in_seconds = 1
            pack_solver.parameters.num_search_workers = 1
            if pack_solver.solve(packing) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                selected = sequence
                break
        if selected is None:
            continue
        sequence = selected
        phases, cursor, pending = [0] * n, 0, 0
        for i in sequence:
            for edge in problem['dependencies']:
                a = edge['src']
                if edge['dst'] == i and edge['kind'] != 'intra_sync' and stages[a] == stages[i]:
                    delay = latency[a] if edge['kind'] in ('completion', 'ssa') else occupancy[a]
                    cursor = max(cursor, phases[a] + delay)
            if ops[i]['kind'] == 'wait':
                cursor = max(cursor, pending)
            phases[i] = cursor
            if ops[i]['kind'] == 'dma':
                pending = max(pending, cursor + latency[i])
            cursor += occupancy[i]
        waits = [phases[i] for i in range(n) if ops[i]['kind'] == 'wait']
        period = max(cursor, pending - min(waits, default=0))
        return stages, phases, period, {id: [pack_solver.value(v) for v in pair]
                                        for id, pair in offsets.items()}
    return None


def solve(problem: dict, *, mode: str = "intra", costs: CostModel | None = None,
          time_limit: float = SOLVER_TIME_LIMIT_SECONDS,
          max_stages: int = MAX_PIPELINE_STAGES, max_slots: int = MAX_BUFFER_SLOTS,
          lds_budget: int = SHM_LIMIT_BYTES, dma_capacity: int = DMA_MAX_IN_FLIGHT,
          shm_check: str = SHM_CHECK_MODE,
          workers: int = SOLVER_WORKERS, seed: int = SOLVER_RANDOM_SEED,
          require_cross_iteration: bool = False,
          vgpr_budget: int | None = None, register_reserve: int = 32,
          lds_read_ahead: bool = False, objective: str = 'makespan',
          trip_count_upper_bound: int | None = None) -> dict:
    validate_problem(problem)
    if objective not in ('makespan', 'ii'):
        raise ValueError('objective must be makespan or ii')
    trip_count, trip_count_source = resolve_trip_count(
        problem, trip_count_upper_bound, required=objective == 'makespan')
    if VGPR_LIMIT_PER_THREAD is not None or SGPR_LIMIT_PER_WAVE is not None:
        raise ValueError("legacy VGPR/SGPR constants do not constrain physical allocation; "
                         "use vgpr_budget for the Fragment data-liveness model")
    if vgpr_budget is None and 'register_liveness' in problem:
        vgpr_budget = 250
    if (vgpr_budget is not None and (type(vgpr_budget) is not int or vgpr_budget <= 0)) or \
            type(register_reserve) is not int or register_reserve < 0:
        raise ValueError('VGPR budget must be positive and register reserve nonnegative')
    if mode not in ("intra", "pipeline"):
        raise ValueError("mode must be intra or pipeline")
    if require_cross_iteration and mode != 'pipeline':
        raise ValueError("--require-cross-iteration requires --mode=pipeline")
    if not 1 <= max_stages <= problem["limits"]["max_stages"]:
        raise ValueError("max_stages exceeds the materializer limit")
    if not 1 <= max_slots <= problem["limits"]["max_slots"]:
        raise ValueError("max_slots exceeds the materializer limit")
    if time_limit <= 0 or workers < 1 or dma_capacity < 1:
        raise ValueError("time limit, workers and DMA capacity must be positive")
    validate_lds_budget(lds_budget)
    if shm_check not in ("auto", "constraint", "postcheck", "sum"):
        raise ValueError("shm_check must be auto, constraint, postcheck or sum")
    if mode == "pipeline" and not problem["loop"]["pipeline_eligible"]:
        raise ValueError("pipeline is not supported: " + problem["loop"]["pipeline_reason"])
    costs = costs or CostModel()
    if type(lds_read_ahead) is not bool:
        raise ValueError('lds_read_ahead must be boolean')
    overlap_rules = lds_overlap_rules(problem, enabled=lds_read_ahead)
    ops = problem["operations"]
    n = len(ops)
    stage_limit = max_stages if mode == "pipeline" else 1
    transfers = dma_transfers(problem)
    shm_specs = lifetime_specs(problem)
    physical_storage = (problem.get('storage_contract') == 'periodic-offsets-v1' and
                        bool(shm_specs) and mode == 'pipeline' and
                        all(b['local'] and problem['buffers'][b['id']]['rotatable'] for b in shm_specs))
    storage_alignment = max([16] + [b.get('alignment', 16) for b in problem['buffers']
                                   if b['memory_space'] == 3])
    if lds_budget is not None and any(b['bytes'] < 0 for b in shm_specs):
        raise ValueError("cannot enforce SHM budget with unknown buffer size")
    shm_intervals = interval_count(shm_specs, stage_limit)
    if shm_check == 'auto':
        shm_check = ('constraint' if lds_budget is not None and
                     shm_intervals <= SHM_MAX_CUMULATIVE_INTERVALS else 'postcheck')
    durations = [costs.operation(op) for op in ops]
    # Physical pool materialization inserts one old-reader handoff barrier at
    # each writer. Include its issue cost instead of silently treating it free.
    handoffs = {b['writer'] for b in shm_specs} if physical_storage else set()
    durations = [(issue + costs.wait, latency + costs.wait) if i in handoffs else (issue, latency)
                 for i, (issue, latency) in enumerate(durations)]
    occupancy, latency = zip(*durations)
    if max(latency) > MAX_OP_LATENCY or sum(latency) > MAX_TOTAL_LATENCY:
        raise ValueError("cost horizon too large; increase reference work units")
    # Preserve a strict emitted order for zero-issue views without inventing
    # hundreds of ALU cycles in which DMA could supposedly complete. One
    # epsilon tick per node sums to less than one modeled hardware time unit.
    time_scale = n + 1 if any(issue == 0 for issue in occupancy) else 1
    issue_costs, result_latencies = occupancy, latency
    if time_scale > 1:
        occupancy = tuple(value * time_scale + 1 for value in occupancy)
        latency = tuple(value * time_scale + 1 for value in latency)
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
    # These constraints survive projection to emitted order: virtual idle
    # time alone cannot satisfy the bounded read-ahead policy.
    for a, b in overlap_rules['edges']:
        model.add(stages[a] == stages[b])
        model.add(phases[b] >= phases[a] + occupancy[a])

    # Strict phase order projects into a permutation, without ever emitting
    # sleeps or relying on model completion times for correctness.
    before = {}
    inversion_terms = []
    compact_order = n > 256
    def precedes(a, b):
        if a > b:
            return precedes(b, a).Not()
        if (a, b) not in before:
            lit = model.new_bool_var(f"before_{a}_{b}")
            model.add(phases[a] < phases[b]).only_enforce_if(lit)
            model.add(phases[a] > phases[b]).only_enforce_if(lit.Not())
            before[a, b] = lit
        return before[a, b]
    if not compact_order:
        for a in range(n):
            for b in range(a + 1, n):
                inversion_terms.append(precedes(a, b).Not())

    for edge in problem["dependencies"]:
        a, b = edge["src"], edge["dst"]
        constraints = []
        # Completion edges wait for the DMA, ordinary source dependencies wait
        # for the synchronous op. No estimated completion replaces a real wait.
        delay = latency[a] if edge['kind'] in ('completion', 'ssa') else occupancy[a]
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
            earlier = precedes(dma, wait)
            model.add(phases[wait] >= phases[dma] + latency[dma]).only_enforce_if(earlier)
            model.add(phases[wait] + ii >= phases[dma] + latency[dma]).only_enforce_if(earlier.Not())

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
    # Advancing only an allocation is not a useful software pipeline. Require
    # actual producer/consumer overlap whenever the materializer will peel it.
    model.add(sum(slot - 1 for slot in buffer_slots) >= 1).only_enforce_if(pipelined)
    if require_cross_iteration:
        model.add(pipelined == 1)
    # Rank time depends only on the emitted order, not on estimated idle gaps.
    # The same rank/stage pair always has the same SHM post-check result.
    if compact_order:
        # NoOverlap gives a strict total order without O(N^2) Boolean pairs.
        # A common monotonic embedding of ranks preserves all lifetime overlaps,
        # including periodic copies: use the same horizon stride for every stage.
        ranks, memory_stride = phases, horizon + 1
        cursor = 0
        for i in range(n):
            distance = model.new_int_var(0, horizon, f"source_distance_{i}")
            model.add_abs_equality(distance, phases[i] - cursor)
            inversion_terms.append(distance)
            cursor += occupancy[i]
    else:
        memory_stride = n
        ranks = [model.new_int_var(0, n - 1, f"rank_{i}") for i in range(n)]
        for i in range(n):
            model.add(ranks[i] == sum(before[j, i] for j in range(i)) +
                      sum(before[i, j].Not() for j in range(i + 1, n)))
    if vgpr_budget is not None:
        add_register_constraint(model, problem, stages, ranks, stride=memory_stride,
                                stage_limit=stage_limit, budget=vgpr_budget,
                                reserve=register_reserve)
    if lds_budget is not None and shm_check == 'constraint':
        add_shm_constraint(model, shm_specs, stages, ranks, buffer_slots,
                           n=memory_stride, stage_limit=stage_limit, budget=lds_budget,
                           periodic=physical_storage)
    storage_offsets = (add_periodic_storage(model, shm_specs, stages, ranks,
                       n=memory_stride, budget=lds_budget, alignment=storage_alignment)
                       if physical_storage else {})
    for offsets in storage_offsets.values():
        model.add(offsets[0] == offsets[1]).only_enforce_if(pipelined.Not())
    physical_pool = model.new_int_var(0, lds_budget // storage_alignment, 'physical_pool_units')
    if physical_storage:
        model.add_max_equality(physical_pool, [offset + (b['bytes'] + storage_alignment - 1) // storage_alignment
            for b in shm_specs for offset in storage_offsets[b['id']]])
    else:
        model.add(physical_pool == 0)

    def storage_plan(solver):
        if not physical_storage:
            return {}
        buffers = [dict(id=b['id'], offsets=[solver.value(v) * storage_alignment
                   for v in storage_offsets[b['id']]]) for b in shm_specs]
        pool = max(max(entry['offsets']) + ((b['bytes'] + storage_alignment - 1) //
                   storage_alignment) * storage_alignment for entry, b in zip(buffers, shm_specs))
        return {'storage': dict(kind='periodic-offsets-v1', period=2, alignment=storage_alignment,
                                pool_bytes=pool, budget_bytes=lds_budget, buffers=buffers)}

    finite = (add_makespan(model, starts, ii, latency,
              [i for i, cost in enumerate(issue_costs) if cost > 0],
              trip_count=trip_count, horizon=horizon, stage_limit=stage_limit)
              if trip_count is not None else {})
    # Finite loop completion first; II remains a diagnostic and optional
    # legacy objective. Later objectives never trade away a proven makespan.
    primary = finite['makespan'] if objective == 'makespan' else ii
    objectives = [(objective, primary),
                  ('lds', physical_pool if physical_storage else sum(extra_bytes)),
                  ('source_order', sum(inversion_terms))]
    hint = (pipeline_hint(problem, occupancy, latency, shm_specs, storage_alignment, lds_budget,
                          vgpr_budget, register_reserve, overlap_rules['edges'])
            if compact_order and physical_storage and stage_limit > 1 else None)
    hint_policy = ('two_batch' if lds_read_ahead else 'unrestricted') if hint else None
    if compact_order and physical_storage and stage_limit > 1 and not lds_read_ahead:
        # Removing a search constraint must not discard a useful starting
        # schedule. Try the old read-ahead order as a HINT ONLY; none of its
        # policy edges are added to the optimization model above.
        seed_rules = lds_overlap_rules(problem, enabled=True)
        if seed_rules['edges']:
            overlap_seed = pipeline_hint(problem, occupancy, latency, shm_specs, storage_alignment,
                                         lds_budget, vgpr_budget, register_reserve, seed_rules['edges'])
            def hint_score(candidate):
                hs, hp, period, _ = candidate
                active = [i for i, cost in enumerate(issue_costs) if cost]
                if objective == 'ii':
                    return period
                if not active or not trip_count:
                    return 0
                begin = min(hs[i] * period + hp[i] for i in active)
                end = max(hs[i] * period + hp[i] + latency[i] for i in active)
                return (trip_count - 1) * period + end - begin
            if overlap_seed and (hint is None or hint_score(overlap_seed) < hint_score(hint)):
                hint, hint_policy = overlap_seed, 'two_batch_seed_only'
    if hint:
        hint_stages, hint_phases, hint_period, hint_offsets = hint
        for var, value in zip(stages + phases, hint_stages + hint_phases):
            model.add_hint(var, value)
        model.add_hint(ii, hint_period)
        model.add_hint(maximum, 1)
        model.add_hint(minimum, 0)
        model.add_hint(pipelined, 1)
        for id, pair in storage_offsets.items():
            for var, value in zip(pair, hint_offsets[id]):
                model.add_hint(var, value)
    elif compact_order and not require_cross_iteration:
        # Source order is a deterministic starting point, not an accepted
        # solution: CP-SAT must still satisfy all dependencies and LDS limits.
        cursor, pending = 0, 0
        for i, op in enumerate(ops):
            if op['kind'] == 'wait':
                cursor = max(cursor, pending)
            model.add_hint(phases[i], cursor)
            model.add_hint(stages[i], 0)
            if op['kind'] == 'dma':
                pending = max(pending, cursor + latency[i])
            cursor += occupancy[i]
        model.add_hint(ii, cursor)
        model.add_hint(maximum, 0)
        model.add_hint(minimum, 0)
        model.add_hint(pipelined, 0)
        for slot in buffer_slots:
            model.add_hint(slot, 1)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    deadline = time.monotonic() + time_limit
    if hint:
        # Complete the partial seed (including task/resource/packing auxiliary
        # variables) with a separate CP solve. The optimization model remains
        # unconstrained: these values are hints, never fixed decisions.
        seed_solver = cp_model.CpSolver()
        seed_solver.parameters.num_search_workers = min(workers, 4)
        seed_solver.parameters.fix_variables_to_their_hinted_value = True
        seed_solver.parameters.max_time_in_seconds = min(5, time_limit / 4)
        if seed_solver.solve(model.clone()) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            model.clear_hints()
            for i in range(len(model.proto.variables)):
                variable = model.get_int_var_from_proto_index(i)
                model.add_hint(variable, seed_solver.value(variable))
    best = None
    proof = []
    rejected = 0
    failure_reason = "time budget exhausted without a feasible schedule"
    for objective_name, objective_expr in objectives:
        model.minimize(objective_expr)
        candidate = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            solver.parameters.max_time_in_seconds = remaining
            status = solver.solve(model)
            name = solver.status_name(status)
            if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                failure_reason = (f"CP-SAT returned {name}; no SHM-feasible schedule was found "
                                  f"with LDS budget {lds_budget} bytes")
                break
            stage_values = [solver.value(s) for s in stages]
            phase_values = [solver.value(t) for t in phases]
            orders = [solver.value(rank) for rank in ranks]
            if compact_order:
                sequence = sorted(range(n), key=lambda i: phase_values[i])
                orders = [0] * n
                for rank, i in enumerate(sequence):
                    orders[i] = rank
            candidate = make_schedule(problem, stage_values, orders, mode=mode, status=name,
                                      **storage_plan(solver))
            memory = analyze_shm(problem, candidate, budget=lds_budget, specs=shm_specs)
            registers = (analyze_registers(problem, candidate, budget=vgpr_budget,
                                          reserve=register_reserve)
                         if vgpr_budget is not None else None)
            overlap_check = check_lds_overlap(overlap_rules, candidate)
            if not overlap_check['within_rules']:
                raise RuntimeError('LDS read-ahead model and emitted-order check disagree')
            if registers is not None and not registers['within_budget']:
                raise RuntimeError('register constraint and independent liveness check disagree')
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
                # Keep optimizing the current objective; never freeze an invalid candidate.
                model.add_forbidden_assignments(stages + ranks,
                    [stage_values + (phase_values if compact_order else orders)])
                continue
            candidate = make_schedule(problem, stage_values, orders, mode=mode, status=name,
                             **storage_plan(solver),
                             model={"objective": objective,
                                    "initial_hint_policy": hint_policy,
                                    "trip_count": trip_count,
                                    "trip_count_source": trip_count_source,
                                    "finite_horizon": {
                                        "contract": "finite-periodic-v1",
                                        "scope": "scheduled_loop",
                                        "formula": "(N-1)*II + max(real_start+latency)-min(real_start)",
                                        "excludes": "unexported prologue/epilogue work, kernel setup/finish, GPU occupancy",
                                        **{key + '_ticks': solver.value(var) for key, var in finite.items()}},
                                    **{key: solver.value(var) / time_scale for key, var in finite.items()},
                                    "period": solver.value(ii) / time_scale,
                                    "period_ticks": solver.value(ii), "time_scale": time_scale,
                                    "zero_issue_operations": sum(o.get('zero_issue', False) for o in ops),
                                    "storage_handoff_barriers": len(handoffs),
                                    "issue_costs": list(issue_costs),
                                    "result_latencies": list(result_latencies),
                                    "costs": asdict(costs),
                                    "dma_capacity": dma_capacity,
                                    "dma_transfers": transfers,
                                    "dma_summary": problem.get('dma_summary', {
                                        'direct_issues': len(transfers),
                                        'direct_waits': sum(o['kind'] == 'wait' for o in ops)}),
                                    "resource_limits": {
                                        "shm_bytes": lds_budget,
                                        "vgpr_per_thread": vgpr_budget,
                                        "sgpr_per_wave": SGPR_LIMIT_PER_WAVE,
                                        "register_pressure_modeled": registers is not None,
                                        "physical_registers_modeled": False,
                                        "dma_max_in_flight": dma_capacity,
                                        "max_stages": stage_limit, "max_slots": max_slots,
                                        "issue_stream_capacity": 1,
                                    },
                                    "shared_bytes_upper_bound": sum(solver.value(x) for x in allocated_bytes),
                                    "shm_liveness": memory,
                                    "register_liveness": registers,
                                    "lds_read_ahead": dict(rules=overlap_rules, check=overlap_check),
                                    "shm_check": shm_check,
                                    "order_encoding": "phase" if compact_order else "pairwise-rank",
                                    "shm_cumulative_intervals": shm_intervals,
                                    "phase": phase_values,
                                    "latency": list(latency), "issue_occupancy": list(occupancy),
                                    "note": "relative cost estimate, not measured GPU time"})
            break
        if candidate is None:
            break
        best = candidate
        proof.append({"name": objective_name, "status": name, "objective": solver.objective_value,
                      "bound": solver.best_objective_bound})
        if status != cp_model.OPTIMAL:
            break
        model.add(objective_expr == solver.value(objective_expr))
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
    parser.add_argument('--objective', choices=('makespan', 'ii'), default='makespan',
                        help='primary objective: finite loop makespan (default) or legacy steady II')
    parser.add_argument('--trip-count-upper-bound', type=int,
                        help='safe maximum iteration count over all legal bid/tid when bounds are dynamic')
    parser.add_argument("--require-cross-iteration", action="store_true",
                        help="require a real cross-iteration DMA/buffer overlap in pipeline mode")
    parser.add_argument("--baseline", action="store_true", help="capture source order without solving")
    parser.add_argument("--costs", type=Path, help="JSON overrides for CostModel fields")
    parser.add_argument("--scale-work", action="store_true")
    parser.add_argument("--max-stages", type=int, default=MAX_PIPELINE_STAGES)
    parser.add_argument("--max-slots", type=int, default=MAX_BUFFER_SLOTS,
                        help="maximum versions per buffer, within the MLIR materializer limit")
    parser.add_argument("--lds-budget", type=int, default=SHM_LIMIT_BYTES,
                        help="hard logical SHM live-peak budget in bytes (default: 16384 / 16 KiB); "
                             "sum mode uses unpooled total")
    parser.add_argument("--shm-check", choices=('auto', 'constraint', 'postcheck', 'sum'),
                        default=SHM_CHECK_MODE, help="SHM capacity enforcement strategy")
    parser.add_argument("--dma-capacity", type=int, default=DMA_MAX_IN_FLIGHT)
    parser.add_argument("--time-limit", type=float, default=SOLVER_TIME_LIMIT_SECONDS)
    parser.add_argument("--workers", type=int, default=SOLVER_WORKERS)
    parser.add_argument("--seed", type=int, default=SOLVER_RANDOM_SEED)
    parser.add_argument('--vgpr-budget', type=int,
                        help='Fragment data words plus reserve; defaults to 250 for new exports')
    parser.add_argument('--register-reserve', type=int, default=32,
                        help='words reserved for addresses/backend temporaries (default: 32)')
    parser.add_argument('--lds-read-ahead', action=argparse.BooleanOptionalAction, default=False,
                        help='opt in to the old two-batch LDS/MMAC order constraints (default: disabled)')
    args = parser.parse_args(argv)
    try:
        validate_lds_budget(args.lds_budget)
        if args.baseline and args.require_cross_iteration:
            raise ValueError("--baseline cannot require cross-iteration overlap")
        problem = read_problem(args.problem)
        settings = json.loads(args.costs.read_text()) if args.costs else {}
        if args.scale_work:
            settings["scale_work"] = True
        result = baseline_schedule(problem) if args.baseline else solve(
            problem, mode=args.mode, costs=CostModel(**settings), max_stages=args.max_stages,
            max_slots=args.max_slots,
            time_limit=args.time_limit, workers=args.workers, seed=args.seed,
            lds_budget=args.lds_budget, dma_capacity=args.dma_capacity, shm_check=args.shm_check,
            require_cross_iteration=args.require_cross_iteration,
            vgpr_budget=args.vgpr_budget, register_reserve=args.register_reserve,
            lds_read_ahead=args.lds_read_ahead, objective=args.objective,
            trip_count_upper_bound=args.trip_count_upper_bound)
        validate_schedule(problem, result)
        if args.baseline:
            memory = analyze_shm(problem, result, budget=args.lds_budget)
            if not memory['within_budget']:
                raise ValueError(f"baseline SHM peak {memory['peak_bytes']} exceeds or cannot "
                                 f"be checked against LDS budget {args.lds_budget}")
            result['model'] = {'resource_limits': {'shm_bytes': args.lds_budget},
                               'shm_liveness': memory}
            if args.vgpr_budget is not None or 'register_liveness' in problem:
                registers = analyze_registers(problem, result, budget=args.vgpr_budget or 250,
                                              reserve=args.register_reserve)
                if not registers['within_budget']:
                    raise ValueError('baseline exceeds Fragment register budget')
                result['model']['register_liveness'] = registers
        write_json(args.output, result)
    except (ValueError, RuntimeError, KeyError, TypeError, OSError) as error:
        print(f"cpsat_scheduler: {error}", file=sys.stderr)
        return 2
    print(f"{result['status']}: {args.output}; loop={result['loop_id']}; mode={result['mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
