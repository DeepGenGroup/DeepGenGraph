"""Versioned exchange contract; semantic dependencies originate in MLIR."""
from __future__ import annotations

import json
from pathlib import Path

PROBLEM_SCHEMA = "deepgengraph.cpsat.problem"
SCHEDULE_SCHEMA = "deepgengraph.cpsat.schedule"
VERSION = 1


def read_problem(path: str | Path) -> dict:
    problem = json.loads(Path(path).read_text())
    validate_problem(problem)
    return problem


def validate_problem(p: dict) -> None:
    if p.get("schema") != PROBLEM_SCHEMA or p.get("version") != VERSION:
        raise ValueError("unsupported problem schema/version")
    if not isinstance(p.get("fingerprint"), str) or len(p["fingerprint"]) != 64:
        raise ValueError("missing function SHA-256 fingerprint")
    if not isinstance(p.get("loop", {}).get("id"), str):
        raise ValueError("missing loop identity")
    ops, buffers = p["operations"], p["buffers"]
    if not 1 <= len(ops) <= 256:
        raise ValueError("v1 supports 1..256 atomic operations per loop")
    for records, label in ((ops, "operation"), (buffers, "buffer")):
        if [r["id"] for r in records] != list(range(len(records))):
            raise ValueError(f"{label} ids must be contiguous in source order")
    for op in ops:
        for field in ("read_bytes", "write_bytes", "compute_ops"):
            if type(op[field]) is not int or op[field] < -1:
                raise ValueError(f"invalid {field}; -1 denotes unknown")
        for buffer in op["reads"] + op["writes"]:
            if type(buffer) is not int or not 0 <= buffer < len(buffers):
                raise ValueError("invalid buffer reference")
    for edge in p["dependencies"]:
        if not 0 <= edge["src"] < edge["dst"] < len(ops):
            raise ValueError("v1 expects forward dependencies from original IR")
        if type(edge["same_stage"]) is not bool:
            raise ValueError("same_stage must be boolean")
    for b in buffers:
        if 'liveness' in b:
            info = b['liveness']
            if (not isinstance(info, dict) or type(info.get('iteration_local')) is not bool or
                    not isinstance(info.get('reason'), str)):
                raise ValueError('invalid buffer liveness ownership evidence')
        for op in b["users"] + b["writers"] + b["readers"]:
            if type(op) is not int or not 0 <= op < len(ops):
                raise ValueError("invalid buffer user")
        if b["rotatable"] and (len(b["writers"]) != 1 or b["bytes"] <= 0):
            raise ValueError("rotatable buffer needs one writer and known size")


def make_schedule(p: dict, stages: list[int], orders: list[int], *,
                  mode: str, status: str, **extra) -> dict:
    buffers = []
    for b in p["buffers"]:
        slots = 1
        if b["rotatable"]:
            slots += max([0] + [stages[u] - stages[b["writers"][0]] for u in b["users"]])
        buffers.append({"id": b["id"], "slots": slots})
    return dict(schema=SCHEDULE_SCHEMA, version=VERSION,
                fingerprint=p["fingerprint"], loop_id=p["loop"]["id"],
                mode=mode, status=status,
                operations=[dict(id=i, stage=s, order=o)
                            for i, (s, o) in enumerate(zip(stages, orders))],
                buffers=buffers, **extra)


def baseline_schedule(p: dict) -> dict:
    n = len(p["operations"])
    return make_schedule(p, [0] * n, list(range(n)), mode="intra", status="BASELINE")


def validate_schedule(p: dict, s: dict) -> None:
    """Independent structural check, repeated again by the C++ importer."""
    if (s.get("schema"), s.get("version"), s.get("fingerprint"), s.get("loop_id")) != (
        SCHEDULE_SCHEMA, VERSION, p["fingerprint"], p["loop"]["id"]
    ):
        raise ValueError("schedule identity mismatch")
    n = len(p["operations"])
    records = s["operations"]
    if len(records) != n or sorted(o["id"] for o in records) != list(range(n)):
        raise ValueError("missing/duplicate operation")
    records = sorted(records, key=lambda o: o["id"])
    stages, orders = [o["stage"] for o in records], [o["order"] for o in records]
    if any(type(v) is not int for v in stages + orders):
        raise ValueError("stages/orders must be integers")
    if sorted(orders) != list(range(n)) or min(stages) != 0:
        raise ValueError("orders must be a permutation and stages must start at zero")
    maximum = max(stages)
    if s["mode"] not in ("intra", "pipeline") or maximum >= p["limits"]["max_stages"]:
        raise ValueError("unsupported mode/stage count")
    if maximum and (s["mode"] == "intra" or not p["loop"]["pipeline_eligible"]):
        raise ValueError("cross-iteration scheduling is not supported for this region")
    for op in p["operations"]:
        if maximum and op["pipeline_anchor"] and stages[op["id"]] != maximum:
            raise ValueError("loop state must stay in final stage")
    for edge in p["dependencies"]:
        if maximum and edge["kind"] == "intra_sync":
            continue
        a, b = edge["src"], edge["dst"]
        if stages[a] > stages[b] or (edge["same_stage"] and stages[a] != stages[b]):
            raise ValueError("stage dependency violation")
        if stages[a] == stages[b] and orders[a] >= orders[b]:
            raise ValueError("order dependency violation")
    expected = make_schedule(p, stages, orders, mode=s["mode"], status="CHECK")["buffers"]
    if sorted(s["buffers"], key=lambda b: b["id"]) != expected:
        raise ValueError("buffer slot counts disagree with lifetimes")
    if any(b["slots"] > p["limits"]["max_slots"] for b in expected):
        raise ValueError("buffer slot limit exceeded")


def write_json(path: str | Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)
