"""Shared LDS budget and conservative checks for MyTest's LLVM byte-array storage."""
from pathlib import Path
import re

DEFAULT_LDS_BUDGET_BYTES = 16 * 1024


def validate_lds_budget(budget):
    if type(budget) is not int or budget < 0:
        raise ValueError("LDS budget must be a nonnegative integer in bytes")


def inspect_llvm_lds(path: Path, budget: int) -> dict:
    """Count all module LDS allocations, including conservative alignment padding.

    MyTest emits one-line addrspace(3) byte-array globals. Unknown shared global
    forms (including dynamic/external storage) fail closed instead of counting as
    zero. Summing module globals is conservative for multiple kernels.
    """
    validate_lds_budget(budget)
    allocations = []
    total = 0
    # Gather global statements, also catching wrapped declarations.
    statements = re.split(r"(?m)(?=^\S)", path.read_text())
    for statement in statements:
        if not statement.startswith("@"):
            continue
        statement = re.sub(r";[^\n]*", "", statement).strip()
        if not re.search(r"addrspace\s*\(\s*3\s*\)", statement):
            continue
        match = re.fullmatch(
            r'(?P<name>@(?:"[^"\n]+"|[-\w.$]+))\s*=\s*'
            r'addrspace\(3\)\s+global\s+\[(?P<size>\d+)\s+x\s+i8\]\s+'
            r'(?:undef|zeroinitializer|poison),\s*align\s+(?P<align>\d+)', statement)
        if not match or int(match['size']) == 0:
            raise ValueError(f"cannot determine static LDS size: {statement[:180]}")
        size, alignment = int(match['size']), int(match['align'])
        if alignment < 1 or alignment & (alignment - 1):
            raise ValueError(f"invalid LDS alignment: {alignment}")
        reserved = (size + alignment - 1) // alignment * alignment
        total += reserved
        allocations.append(dict(name=match['name'], bytes=size, alignment=alignment,
                                reserved_bytes=reserved))
    return dict(budget_bytes=budget, physical_lds_bytes=total,
                within_budget=total <= budget, globals=allocations,
                accounting="module byte-array globals with conservative alignment padding")
