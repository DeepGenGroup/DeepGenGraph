"""Relative integer cost model, not a GPU cycle simulator.

DMA issue occupancy and transfer latency are intentionally different. Unknown
work sizes fall back to one unit. Changing these estimates cannot relax any
semantic dependency. Optional scaling uses block-level work exported by MLIR.
"""
from dataclasses import asdict, dataclass
from math import ceil


@dataclass(frozen=True)
class CostModel:
    dma: int = 10
    shm_reg: int = 5
    shm_issue: int = 1
    alu: int = 1
    wmma: int = 1
    wmma_latency: int = 4
    issue: int = 1
    wait: int = 1
    reference_bytes: int = 8192
    reference_alu_ops: int = 4096
    reference_wmma_ops: int = 262144
    scale_work: bool = False

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key != "scale_work" and (type(value) is not int or not 1 <= value <= 1_000_000):
                raise ValueError(f"{key} must be a positive bounded integer")
        if type(self.scale_work) is not bool:
            raise ValueError("scale_work must be boolean")

    def operation(self, op: dict) -> tuple[int, int]:
        """Return (issue-stream occupancy, result/completion latency)."""
        kind = op["kind"]
        if op.get('zero_issue', False):
            return 0, 0
        scale = op.get('issue_units', 1)
        if self.scale_work and scale == 1:
            if kind in ("dma", "copy_gm_shm", "shm_reg", "copy"):
                work = max(op["read_bytes"], op["write_bytes"], 1)
                scale = max(1, ceil(work / self.reference_bytes))
            elif kind not in ("address", "alloc", "wait"):
                ref = self.reference_wmma_ops if kind == "wmma" else self.reference_alu_ops
                scale = max(1, ceil(max(1, op["compute_ops"]) / ref))
        if kind == "dma":
            return self.issue, max(self.issue, self.dma * scale)
        if kind == "copy_gm_shm":
            duration = self.dma * scale  # Synchronous copy: occupies issue stream.
        elif kind == "shm_reg":
            return self.shm_issue * scale, max(self.shm_issue, self.shm_reg) * scale
        elif kind == "wmma":
            return self.wmma * scale, max(self.wmma, self.wmma_latency) * scale
        elif kind == "wait":
            duration = self.wait  # Stall is derived from completion constraints.
        else:
            duration = self.alu * scale
        return duration, duration
