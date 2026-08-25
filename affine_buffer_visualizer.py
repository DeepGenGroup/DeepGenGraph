#!/usr/bin/env python3
"""
affine_buffer_visualizer.py

A lightweight MLIR Affine/Frisk access visualizer.

Designed for IR containing:
  * affine.for
  * affine.apply
  * arith.constant (integer/index)
  * arith.addi/subi/muli (integer/index)
  * affine.load / affine.store
  * frisk.copy ... at(...) [affine_map<...>]

It interprets the loop nest directly rather than generating Python source first.

No MLIR Python bindings are required.

Dependencies:
    pip install matplotlib numpy

Examples:
    # List discovered buffers/access sites
    python affine_buffer_visualizer.py frisk.mlir --list

    # Visualize every affine.load/store touching qkme
    python affine_buffer_visualizer.py frisk.mlir --buffer qkme --op affine --thread 0:128

    # Visualize frisk.copy accesses whose mapped (x,y) are on qkme
    python affine_buffer_visualizer.py frisk.mlir --buffer qkme --op copy --thread 0:128

    # Select one access site
    python affine_buffer_visualizer.py frisk.mlir --buffer qkme --site 3 --thread 0:128

    # Fix external SSA values used by loop bounds
    python affine_buffer_visualizer.py frisk.mlir --buffer qkme \
        --set block_id_y=0 --set block_id_x=0 --thread 0:128

    # Color points by thread / loop label / SSA induction variable
    python affine_buffer_visualizer.py frisk.mlir --buffer qkme \
        --thread 0:128 --color-by thread_id_x
    python affine_buffer_visualizer.py frisk.mlir --buffer qkme \
        --thread 0:128 --color-by br0

    # Plot only points produced under one dynamic SSA/loop value
    python affine_buffer_visualizer.py frisk.mlir --buffer qkme \
        --thread 0:1 --set block_id_x=0 --set block_id_y=1 --where arg4=128

Notes:
  1. For frisk.copy with at(...) [affine_map<...>], this script interprets the
     affine_map result as the (x,y,...) coordinates of the LARGE memref side.
     If exactly one side is rank-2+ and the other is a small local fragment,
     this normally matches the intended layout visualization.
  2. This is an analysis interpreter, not a complete MLIR executor.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# -----------------------------------------------------------------------------
# IR nodes
# -----------------------------------------------------------------------------

@dataclass
class Node:
    line_no: int

@dataclass
class Statement(Node):
    text: str

@dataclass
class AffineFor(Node):
    iv: str
    lower: str
    upper: str
    step: int
    body: List[Node] = field(default_factory=list)
    iter_label: Optional[str] = None

@dataclass
class AccessSite:
    site_id: int
    line_no: int
    kind: str
    buffer: str
    text: str
    rank: Optional[int] = None
    shape: Optional[Tuple[int, ...]] = None


# -----------------------------------------------------------------------------
# Basic text utilities
# -----------------------------------------------------------------------------

SSA_RE = re.compile(r"%[A-Za-z_.$][\w.$-]*|%\d+")

def strip_ssa(s: str) -> str:
    return s[1:] if s.startswith("%") else s

def remove_comment(line: str) -> str:
    # Current IR does not use // inside string literals relevant to us.
    return line.split("//", 1)[0].rstrip()

def split_top_level(text: str, delimiter: str = ",") -> List[str]:
    # 只按“最外层”的逗号切分，避免把 affine_map 结果或 memref 类型内部的逗号拆开。
    out = []
    start = 0
    depth = 0
    pairs = {"(": ")", "[": "]", "<": ">", "{": "}"}
    opens = set(pairs)
    closes = set(pairs.values())
    for i, ch in enumerate(text):
        if ch in opens:
            depth += 1
        elif ch in closes:
            depth -= 1
        elif ch == delimiter and depth == 0:
            out.append(text[start:i].strip())
            start = i + 1
    out.append(text[start:].strip())
    return [x for x in out if x]

def extract_balanced(text: str, start: int, open_ch: str, close_ch: str) -> Tuple[str, int]:
    assert text[start] == open_ch
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError(f"unbalanced {open_ch}{close_ch}: {text}")

def find_affine_map_span(text: str) -> Optional[Tuple[str, int]]:
    """Return (map_body, position_after_closing_>).

    A normal angle-bracket matcher is not enough because affine maps contain
    the token `->`; its `>` is not the closing angle bracket.
    """
    pos = text.find("affine_map<")
    if pos < 0:
        return None
    lt = text.find("<", pos)
    depth = 1
    i = lt + 1
    while i < len(text):
        ch = text[i]
        if ch == "<":
            depth += 1
        elif ch == ">":
            if i > 0 and text[i - 1] == "-":
                i += 1
                continue
            depth -= 1
            if depth == 0:
                return text[lt + 1:i].strip(), i + 1
        i += 1
    raise ValueError(f"unterminated affine_map: {text}")

def find_affine_map(text: str) -> Optional[str]:
    span = find_affine_map_span(text)
    return span[0] if span else None

def normalize_name(name: str) -> str:
    return name[1:] if name.startswith("%") else name


# -----------------------------------------------------------------------------
# Parser for nested affine.for blocks
# -----------------------------------------------------------------------------

FOR_RE = re.compile(
    r"^\s*affine\.for\s+(?P<iv>%[\w.$-]+|%\d+)\s*=\s*"
    r"(?P<lb>.*?)\s+to\s+(?P<ub>.*?)(?:\s+step\s+(?P<step>-?\d+))?\s*\{\s*$"
)

LABEL_RE = re.compile(r'iterLabel\s*=\s*"([^"]+)"')

def parse_ir(text: str) -> List[Node]:
    # 这里不做完整 MLIR 解析，只识别 affine.for 的嵌套结构；
    # 其他行保留为 Statement，交给后续解释阶段按需匹配。
    roots: List[Node] = []
    body_stack: List[List[Node]] = [roots]
    loop_stack: List[AffineFor] = []

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = remove_comment(raw).strip()
        if not line:
            continue

        m = FOR_RE.match(line)
        if m:
            # body_stack 永远指向当前应追加节点的语句列表；
            # 遇到新循环时把循环节点加入当前 body，再切入它自己的 body。
            loop = AffineFor(
                line_no=line_no,
                iv=m.group("iv"),
                lower=m.group("lb").strip(),
                upper=m.group("ub").strip(),
                step=int(m.group("step") or "1"),
            )
            body_stack[-1].append(loop)
            loop_stack.append(loop)
            body_stack.append(loop.body)
            continue

        if line.startswith("}"):
            # affine.for 的 iterLabel 出现在闭合大括号行上，因此在退出循环时回填标签。
            label = LABEL_RE.search(line)
            if loop_stack:
                loop = loop_stack.pop()
                if label:
                    loop.iter_label = label.group(1)
                if len(body_stack) > 1:
                    body_stack.pop()
            # Non-affine braces (func/module) are intentionally ignored.
            continue

        body_stack[-1].append(Statement(line_no=line_no, text=line))

    return roots


# -----------------------------------------------------------------------------
# Affine expression evaluator
# -----------------------------------------------------------------------------

TOKEN_RE = re.compile(
    r"""
    (?P<SPACE>\s+)
  | (?P<FLOORDIV>\bfloordiv\b)
  | (?P<CEILDIV>\bceildiv\b)
  | (?P<MOD>\bmod\b)
  | (?P<INT>-?\d+)
  | (?P<ID>[A-Za-z_][A-Za-z0-9_]*)
  | (?P<PLUS>\+)
  | (?P<MINUS>-)
  | (?P<MUL>\*)
  | (?P<LPAREN>\()
  | (?P<RPAREN>\))
    """,
    re.VERBOSE,
)

class ExprParser:
    def __init__(self, text: str, variables: Dict[str, int]):
        # affine 表达式只支持脚本需要的整数子集，变量名来自 dims/symbols 或运行时 env。
        self.variables = variables
        self.tokens = []
        pos = 0
        while pos < len(text):
            m = TOKEN_RE.match(text, pos)
            if not m:
                raise ValueError(f"unsupported affine syntax near: {text[pos:]} in {text!r}")
            pos = m.end()
            if m.lastgroup != "SPACE":
                self.tokens.append((m.lastgroup, m.group()))
        self.i = 0

    def peek(self, *kinds):
        return self.i < len(self.tokens) and self.tokens[self.i][0] in kinds

    def pop(self, kind=None):
        if self.i >= len(self.tokens):
            raise ValueError("unexpected end of expression")
        tok = self.tokens[self.i]
        if kind and tok[0] != kind:
            raise ValueError(f"expected {kind}, got {tok}")
        self.i += 1
        return tok

    def parse(self) -> int:
        v = self.parse_add()
        if self.i != len(self.tokens):
            raise ValueError(f"trailing tokens: {self.tokens[self.i:]}")
        return v

    def parse_add(self) -> int:
        v = self.parse_muldiv()
        while self.peek("PLUS", "MINUS"):
            op = self.pop()[0]
            rhs = self.parse_muldiv()
            v = v + rhs if op == "PLUS" else v - rhs
        return v

    def parse_muldiv(self) -> int:
        v = self.parse_unary()
        while self.peek("MUL", "FLOORDIV", "CEILDIV", "MOD"):
            op = self.pop()[0]
            rhs = self.parse_unary()
            if rhs == 0:
                raise ZeroDivisionError("division/modulo by zero")
            if op == "MUL":
                v *= rhs
            elif op == "FLOORDIV":
                v = math.floor(v / rhs)
            elif op == "CEILDIV":
                v = -math.floor(-v / rhs)
            else:
                v %= rhs
        return v

    def parse_unary(self) -> int:
        if self.peek("MINUS"):
            self.pop()
            return -self.parse_unary()
        if self.peek("LPAREN"):
            self.pop()
            v = self.parse_add()
            self.pop("RPAREN")
            return v
        kind, value = self.pop()
        if kind == "INT":
            return int(value)
        if kind == "ID":
            if value not in self.variables:
                raise KeyError(value)
            return int(self.variables[value])
        raise ValueError(f"unexpected token: {(kind, value)}")

def eval_affine_expr(expr: str, variables: Dict[str, int]) -> int:
    return ExprParser(expr.strip(), variables).parse()


@dataclass
class ParsedMap:
    dims: List[str]
    symbols: List[str]
    results: List[str]

@lru_cache(maxsize=None)
def parse_map(map_text: str) -> ParsedMap:
    # Examples:
    #   (d0, d1) -> (d0 * 8 + d1)
    #   ()[s0] -> (s0 * 128 + 128)
    # dims 对应圆括号里的 d0/d1，symbols 对应方括号里的 s0/s1；
    # 调用 eval_map 时会把实参按这两个列表绑定到表达式环境。
    m = re.match(
        r"^\s*\((?P<dims>.*?)\)\s*(?:\[(?P<syms>.*?)\])?\s*->\s*\((?P<results>.*)\)\s*$",
        map_text,
    )
    if not m:
        raise ValueError(f"cannot parse affine_map: {map_text}")
    dims = [x.strip() for x in split_top_level(m.group("dims"))] if m.group("dims").strip() else []
    syms_text = m.group("syms") or ""
    syms = [x.strip() for x in split_top_level(syms_text)] if syms_text.strip() else []
    results = split_top_level(m.group("results"))
    return ParsedMap(dims, syms, results)

def eval_map(map_text: str, dim_values: Sequence[int], symbol_values: Sequence[int] = ()) -> Tuple[int, ...]:
    amap = parse_map(map_text)
    if len(dim_values) != len(amap.dims):
        raise ValueError(f"map needs {len(amap.dims)} dims, got {len(dim_values)}")
    if len(symbol_values) != len(amap.symbols):
        raise ValueError(f"map needs {len(amap.symbols)} symbols, got {len(symbol_values)}")
    env = {}
    env.update(zip(amap.dims, map(int, dim_values)))
    env.update(zip(amap.symbols, map(int, symbol_values)))
    return tuple(eval_affine_expr(expr, env) for expr in amap.results)


# -----------------------------------------------------------------------------
# Runtime helpers
# -----------------------------------------------------------------------------

MEMREF_RE = re.compile(r"memref<([^>]+)>")
ALLOC_RE = re.compile(
    r"^(?P<res>%[\w.$-]+|%\d+)\s*=\s*(?:frisk\.alloc_buffer|memref\.alloca)"
    r".*?->?\s*memref<(?P<body>[^>]+)>"
)

def parse_memref_type(body: str) -> Optional[Tuple[int, ...]]:
    # e.g. "128x128xf16, 3" or "4x32xf32"
    dims = []
    for part in body.split("x"):
        part = part.strip()
        if re.fullmatch(r"\d+", part):
            dims.append(int(part))
        else:
            break
    return tuple(dims) if dims else None

def discover_shapes(text: str) -> Dict[str, Tuple[int, ...]]:
    shapes = {}
    # 通过 SSA 定义行上的 memref<...> 反推出 buffer shape，后面用于列表展示和
    # frisk.copy 的“大 buffer / 小 fragment”判断。
    # Generic SSA result whose line has "-> memref<...>"
    for line in text.splitlines():
        lhs = re.match(r"\s*(%[\w.$-]+|%\d+)\s*=", line)
        if not lhs:
            continue
        memrefs = MEMREF_RE.findall(line)
        if not memrefs:
            continue
        shape = parse_memref_type(memrefs[-1])
        if shape:
            shapes[normalize_name(lhs.group(1))] = shape
    return shapes

def get_value(token: str, env: Dict[str, int]) -> int:
    # token 可能是字面量、SSA 名称，或一个简单 affine 表达式。
    token = token.strip()
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    if token.startswith("%"):
        key = normalize_name(token)
        if key not in env:
            raise KeyError(key)
        return int(env[key])
    return eval_affine_expr(token, env)

@lru_cache(maxsize=None)
def parse_affine_apply(line: str):
    m = re.match(r"^(%[\w.$-]+|%\d+)\s*=\s*affine\.apply\s+", line)
    if not m:
        return None
    res = normalize_name(m.group(1))
    map_text = find_affine_map(line)
    if map_text is None:
        return None
    # operands 是紧跟 affine_map<...> 后面的 (...)，按顺序喂给 map 的 dims。
    span = find_affine_map_span(line)
    if span is None:
        return None
    _, after_map = span
    tail = line[after_map:].lstrip()
    if not tail.startswith("("):
        raise ValueError(f"cannot find affine.apply operands: {line}")
    operands_text, _ = extract_balanced(tail, 0, "(", ")")
    operands = split_top_level(operands_text)
    return res, map_text, operands

@lru_cache(maxsize=None)
def parse_index_constant(line: str):
    m = re.match(
        r"^(%[\w.$-]+|%\d+)\s*=\s*arith\.constant\s+(-?\d+)\s*:\s*(?:index|i\d+)\b",
        line,
    )
    if m:
        return normalize_name(m.group(1)), int(m.group(2))
    return None

@lru_cache(maxsize=None)
def parse_integer_arith(line: str):
    m = re.match(
        r"^(%[\w.$-]+|%\d+)\s*=\s*arith\.(addi|subi|muli)\s+"
        r"([^,]+),\s*([^:]+)\s*:",
        line,
    )
    if not m:
        return None
    return normalize_name(m.group(1)), m.group(2), m.group(3).strip(), m.group(4).strip()

@lru_cache(maxsize=None)
def parse_affine_mem_access(line: str):
    # affine.load %qkme[%28, %29] / affine.store %v, %qkme[%28, %29]
    # 返回统一的 read/write 语义，便于后面和 frisk.copy 的访问点合并处理。
    m = re.search(r"\baffine\.(load|store)\b", line)
    if not m:
        return None
    kind = "read" if m.group(1) == "load" else "write"

    if m.group(1) == "load":
        mm = re.search(r"affine\.load\s+(%[\w.$-]+|%\d+)\s*\[([^\]]*)\]", line)
    else:
        mm = re.search(r"affine\.store\s+[^,]+,\s*(%[\w.$-]+|%\d+)\s*\[([^\]]*)\]", line)
    if not mm:
        return None
    buffer = normalize_name(mm.group(1))
    indices = split_top_level(mm.group(2))
    return kind, buffer, indices

@lru_cache(maxsize=None)
def parse_frisk_copy(line: str):
    if "frisk.copy" not in line:
        return None
    m = re.search(
        r"frisk\.copy\s+(%[\w.$-]+|%\d+)\s+to\s+(%[\w.$-]+|%\d+)",
        line,
    )
    if not m:
        return None
    src = normalize_name(m.group(1))
    dst = normalize_name(m.group(2))

    at_values = []
    at_pos = line.find(" at(")
    if at_pos >= 0:
        # at(...) 是 copy 在大 buffer 上的起点/索引实参；后续再套 affine_map 得到坐标。
        p = line.find("(", at_pos)
        body, _ = extract_balanced(line, p, "(", ")")
        at_values = split_top_level(body)

    map_text = find_affine_map(line)
    return src, dst, at_values, map_text

@lru_cache(maxsize=None)
def parse_copy_memref_shapes(line: str):
    # Type list after ":" is src, dst.
    if ":" not in line:
        return None, None
    tail = line.rsplit(":", 1)[-1]
    memrefs = MEMREF_RE.findall(tail)
    if len(memrefs) < 2:
        return None, None
    return parse_memref_type(memrefs[0]), parse_memref_type(memrefs[1])

def choose_copy_mapped_buffer(src: str, dst: str, src_shape, dst_shape) -> str:
    # The at()+affine_map normally describes placement in the larger tiled buffer.
    # copy 常见形态是一侧为全局/大 tile buffer，另一侧为小片段；
    # 这里用元素总数做启发式判断 affine_map 描述的是哪一侧。
    def size(shape):
        if not shape:
            return -1
        p = 1
        for x in shape:
            p *= x
        return p
    return src if size(src_shape) >= size(dst_shape) else dst


# -----------------------------------------------------------------------------
# Bound evaluation
# -----------------------------------------------------------------------------

BOUND_MAP_RE = re.compile(r"affine_map<")

def eval_bound(text: str, env: Dict[str, int]) -> int:
    text = text.strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)

    # Form: affine_map<()[s0] -> (...)>()[%block_id_y]
    if text.startswith("affine_map<"):
        # 循环上下界有时本身就是 affine_map 调用，需要分别解析 dims 和 symbols。
        span = find_affine_map_span(text)
        if span is None:
            raise ValueError(f"cannot parse affine bound: {text}")
        map_text, end = span
        tail = text[end:].strip()
        dims = []
        syms = []
        if tail.startswith("("):
            dbody, pos = extract_balanced(tail, 0, "(", ")")
            dims = [get_value(x, env) for x in split_top_level(dbody)] if dbody.strip() else []
            tail2 = tail[pos:].strip()
            if tail2.startswith("["):
                sbody, _ = extract_balanced(tail2, 0, "[", "]")
                syms = [get_value(x, env) for x in split_top_level(sbody)] if sbody.strip() else []
        return eval_map(map_text, dims, syms)[0]

    # A bare SSA or simple expression.
    return get_value(text, env)


# -----------------------------------------------------------------------------
# Access collection
# -----------------------------------------------------------------------------

@dataclass
class Point:
    site_id: int
    line_no: int
    kind: str
    buffer: str
    coord: Tuple[int, ...]
    env: Dict[str, int]

EnvFilter = Tuple[str, int, Optional[int]]

class PointLimitReached(RuntimeError):
    pass

class Interpreter:
    def __init__(
        self,
        roots: List[Node],
        shapes: Dict[str, Tuple[int, ...]],
        selected_buffer: Optional[str],
        op_mode: str,
        selected_site: Optional[int],
        initial_env: Dict[str, int],
        env_filters: Sequence[EnvFilter],
        max_points: int,
        stop_after_line: Optional[int] = None,
        keep_env_keys: Sequence[str] = (),
        dedup_points: bool = False,
    ):
        self.roots = roots
        self.shapes = shapes
        self.selected_buffer = normalize_name(selected_buffer) if selected_buffer else None
        self.op_mode = op_mode
        self.selected_site = selected_site
        self.initial_env = dict(initial_env)
        self.env_filters = list(env_filters)
        self.max_points = max_points
        self.stop_after_line = stop_after_line
        self.keep_env_keys = tuple(keep_env_keys)
        self.dedup_points = dedup_points
        self._seen_points = set()
        self.points: List[Point] = []
        self.sites: List[AccessSite] = []
        self.site_lookup: Dict[int, AccessSite] = {}
        self._line_to_site: Dict[Tuple[int, str, str], int] = {}
        self.errors = Counter()

    def record_site(self, line_no, kind, buffer, text, shape=None):
        # access site 是静态访问位置；同一行同一种访问只登记一次，动态执行会产生多个 Point。
        key = (line_no, kind, buffer)
        if key in self._line_to_site:
            return self._line_to_site[key]
        site_id = len(self.sites)
        site = AccessSite(site_id, line_no, kind, buffer, text, len(shape) if shape else None, shape)
        self.sites.append(site)
        self.site_lookup[site_id] = site
        self._line_to_site[key] = site_id
        return site_id

    def may_record(self, site_id: int, buffer: str) -> bool:
        # --site 和 --buffer 都是采样过滤器；列表模式仍会先登记所有 site。
        if self.selected_site is not None and site_id != self.selected_site:
            return False
        if self.selected_buffer is not None and buffer != self.selected_buffer:
            return False
        return True

    def env_matches_filters(self, env: Dict[str, int]) -> bool:
        # --where 是动态过滤器：只有当前访问点的运行时 env 满足条件时才画出来。
        for name, lo, hi in self.env_filters:
            if name not in env:
                return False
            value = int(env[name])
            if hi is None:
                if value != lo:
                    return False
            elif not (lo <= value < hi):
                return False
        return True

    def append_point(self, site_id, line_no, kind, buffer, coord, env):
        if not self.env_matches_filters(env):
            return
        coord = tuple(coord)
        point_env = {k: int(env[k]) for k in self.keep_env_keys if k in env}
        if self.dedup_points:
            key = (site_id, line_no, kind, buffer, coord, tuple(sorted(point_env.items())))
            if key in self._seen_points:
                return
            self._seen_points.add(key)
        if len(self.points) >= self.max_points:
            raise PointLimitReached(
                f"point limit ({self.max_points}) reached. Narrow the selection with "
                f"--buffer/--site/--thread or raise --max-points."
            )
        self.points.append(Point(site_id, line_no, kind, buffer, coord, point_env))

    def exec_nodes(self, nodes: List[Node], env: Dict[str, int]):
        for node in nodes:
            if self.stop_after_line is not None and node.line_no > self.stop_after_line:
                break
            if isinstance(node, AffineFor):
                try:
                    lb = eval_bound(node.lower, env)
                    ub = eval_bound(node.upper, env)
                except Exception as e:
                    self.errors[f"loop-bound line {node.line_no}: {e}"] += 1
                    continue
                label = node.iter_label
                for v in range(lb, ub, node.step):
                    # 每次循环迭代复制一份环境，避免子循环/语句计算污染兄弟迭代。
                    child = dict(env)
                    child[normalize_name(node.iv)] = v
                    if label:
                        child[label] = v
                    self.exec_nodes(node.body, child)
                continue

            line = node.text

            c = parse_index_constant(line)
            if c:
                # arith.constant 产生运行时可引用的 SSA/index 值。
                env[c[0]] = c[1]
                continue

            a = parse_integer_arith(line)
            if a:
                name, op, lhs, rhs = a
                try:
                    lv, rv = get_value(lhs, env), get_value(rhs, env)
                    env[name] = {"addi": lv + rv, "subi": lv - rv, "muli": lv * rv}[op]
                except Exception as e:
                    self.errors[f"arith line {node.line_no}: {e}"] += 1
                continue

            ap = parse_affine_apply(line)
            if ap:
                name, map_text, operands = ap
                try:
                    # affine.apply 的结果继续写回 env，供之后的 load/store 下标使用。
                    vals = [get_value(x, env) for x in operands]
                    result = eval_map(map_text, vals)
                    if len(result) != 1:
                        raise ValueError("affine.apply is expected to have one result")
                    env[name] = result[0]
                except Exception as e:
                    self.errors[f"apply line {node.line_no}: {e}"] += 1
                continue

            if self.op_mode in ("all", "affine"):
                ma = parse_affine_mem_access(line)
                if ma:
                    kind, buffer, indices = ma
                    shape = self.shapes.get(buffer)
                    site_id = self.record_site(node.line_no, f"affine.{kind}", buffer, line, shape)
                    if self.may_record(site_id, buffer):
                        try:
                            # 真正的“动态访问点”：把当前环境里的 SSA 下标求值成整数坐标。
                            coord = tuple(get_value(x, env) for x in indices)
                            self.append_point(site_id, node.line_no, kind, buffer, coord, env)
                        except PointLimitReached:
                            raise
                        except Exception as e:
                            self.errors[f"access line {node.line_no}: {e}"] += 1
                    continue

            if self.op_mode in ("all", "copy"):
                cp = parse_frisk_copy(line)
                if cp:
                    src, dst, at_values, map_text = cp
                    src_shape, dst_shape = parse_copy_memref_shapes(line)
                    mapped_buffer = choose_copy_mapped_buffer(src, dst, src_shape, dst_shape)
                    mapped_shape = src_shape if mapped_buffer == src else dst_shape
                    kind = "read" if mapped_buffer == src else "write"
                    site_id = self.record_site(
                        node.line_no, f"frisk.copy.{kind}", mapped_buffer, line, mapped_shape
                    )
                    if self.may_record(site_id, mapped_buffer):
                        if map_text is None or not at_values:
                            # Whole-buffer copy: no meaningful elementwise coordinate available.
                            continue
                        try:
                            # frisk.copy 的坐标由 at(...) 实参经过 affine_map 映射得到。
                            vals = [get_value(x, env) for x in at_values]
                            coord = eval_map(map_text, vals)
                            self.append_point(
                                site_id, node.line_no, kind, mapped_buffer, coord, env
                            )
                        except PointLimitReached:
                            raise
                        except Exception as e:
                            self.errors[f"copy line {node.line_no}: {e}"] += 1

    def run(self):
        self.exec_nodes(self.roots, dict(self.initial_env))


# -----------------------------------------------------------------------------
# Visualization / reporting
# -----------------------------------------------------------------------------

def iter_statements(nodes: Sequence[Node]) -> Iterable[Statement]:
    for node in nodes:
        if isinstance(node, Statement):
            yield node
        elif isinstance(node, AffineFor):
            yield from iter_statements(node.body)

def find_last_relevant_access_line(
    roots: Sequence[Node],
    shapes: Dict[str, Tuple[int, ...]],
    selected_buffer: Optional[str],
    op_mode: str,
) -> Optional[int]:
    if not selected_buffer:
        return None
    selected = normalize_name(selected_buffer)
    last_line = None
    for stmt in iter_statements(roots):
        line = stmt.text
        if op_mode in ("all", "affine"):
            ma = parse_affine_mem_access(line)
            if ma:
                _, buffer, _ = ma
                if buffer == selected:
                    last_line = stmt.line_no if last_line is None else max(last_line, stmt.line_no)
        if op_mode in ("all", "copy"):
            cp = parse_frisk_copy(line)
            if cp:
                src, dst, _, _ = cp
                src_shape, dst_shape = parse_copy_memref_shapes(line)
                mapped_buffer = choose_copy_mapped_buffer(src, dst, src_shape, dst_shape)
                if mapped_buffer == selected:
                    last_line = stmt.line_no if last_line is None else max(last_line, stmt.line_no)
    return last_line

def print_sites(sites: List[AccessSite], buffer_filter: Optional[str] = None):
    bf = normalize_name(buffer_filter) if buffer_filter else None
    print("Discovered access sites:")
    for s in sites:
        if bf and s.buffer != bf:
            continue
        shape = "x".join(map(str, s.shape)) if s.shape else "?"
        print(f"  [{s.site_id:3d}] line {s.line_no:4d}  {s.kind:18s} "
              f"%{s.buffer:24s} shape={shape}")
        print(f"        {s.text[:150]}")

def summarize(points: List[Point]):
    print(f"\nCollected dynamic accesses: {len(points)}")
    if not points:
        return
    unique = {p.coord for p in points}
    print(f"Unique coordinates:         {len(unique)}")
    by_kind = Counter(p.kind for p in points)
    print("Kinds:                      " + ", ".join(f"{k}={v}" for k, v in by_kind.items()))
    dims = len(points[0].coord)
    for d in range(dims):
        vals = [p.coord[d] for p in points]
        print(f"dim{d} range:                [{min(vals)}, {max(vals)}]")

def choose_tick_step(span: int, target_ticks: int = 12) -> int:
    if span <= target_ticks:
        return 1
    rough = max(1, span / target_ticks)
    base = 10 ** int(math.floor(math.log10(rough)))
    for mul in (1, 2, 5, 10):
        step = base * mul
        if rough <= step:
            return int(step)
    return int(base * 10)

def apply_coordinate_grid(ax, xs: Sequence[int], ys: Sequence[int]):
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan = xmax - xmin
    yspan = ymax - ymin
    xstep = choose_tick_step(xspan)
    ystep = choose_tick_step(yspan)

    # 坐标轴边界放在整数点外侧半格，点位会落在整数刻度/网格线上。
    ax.set_xlim(xmin - 0.5, xmax + 0.5)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)
    ax.set_xticks(list(range(xmin, xmax + 1, xstep)))
    ax.set_yticks(list(range(ymin, ymax + 1, ystep)))

    # 范围不大时显示每个整数坐标的细网格，方便直接读出具体点位。
    if xspan <= 256:
        ax.set_xticks(list(range(xmin, xmax + 1)), minor=True)
    if yspan <= 256:
        ax.set_yticks(list(range(ymin, ymax + 1)), minor=True)

    ax.grid(True, which="major", linewidth=0.45, alpha=0.45)
    ax.grid(True, which="minor", linewidth=0.18, alpha=0.18)

@dataclass
class Region:
    top_y: int
    top_x: int
    min_y: int
    max_y: int
    min_x: int
    max_x: int
    size: int

def find_connected_regions(points: Iterable[Tuple[int, int]]) -> List[Region]:
    # points 使用 buffer 坐标顺序 (dim0/y, dim1/x)，按 4 邻接划分离散区域。
    remaining = set(points)
    regions: List[Region] = []
    while remaining:
        start = remaining.pop()
        stack = [start]
        comp = [start]
        while stack:
            y, x = stack.pop()
            for nxt in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
                    comp.append(nxt)

        min_y = min(y for y, _ in comp)
        max_y = max(y for y, _ in comp)
        min_x = min(x for _, x in comp)
        max_x = max(x for _, x in comp)
        top_row_xs = [x for y, x in comp if y == min_y]
        regions.append(
            Region(
                top_y=min_y,
                top_x=min(top_row_xs),
                min_y=min_y,
                max_y=max_y,
                min_x=min_x,
                max_x=max_x,
                size=len(comp),
            )
        )
    return sorted(regions, key=lambda r: (r.top_y, r.top_x, r.min_x, r.min_y))

def print_regions(regions: Sequence[Region]):
    print(f"Connected regions:          {len(regions)}")
    for i, r in enumerate(regions):
        print(
            f"  region {i:3d}: top-left=(dim0={r.top_y}, dim1={r.top_x}) "
            f"bbox=dim0[{r.min_y},{r.max_y}] dim1[{r.min_x},{r.max_x}] points={r.size}"
        )

def annotate_regions(ax, regions: Sequence[Region], max_labels: int):
    for r in regions[:max_labels]:
        ax.annotate(
            f"({r.top_y},{r.top_x})",
            xy=(r.top_x, r.top_y),
            xytext=(4, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=8,
            color="black",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="black", lw=0.35, alpha=0.78),
        )
    if len(regions) > max_labels:
        print(f"warning: only labeled first {max_labels} of {len(regions)} regions", file=sys.stderr)

def apply_region_ticks(ax, regions: Sequence[Region]):
    ax.set_xticks(sorted({r.top_x for r in regions}))
    ax.set_yticks(sorted({r.top_y for r in regions}))
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)
    ax.grid(True, which="major", linewidth=0.45, alpha=0.45)

def visualize(
    points: List[Point],
    color_by: Optional[str],
    output: Optional[str],
    title: Optional[str],
    invert_y: bool,
    marker_size: float,
    coordinate_grid: bool,
    label_regions: bool,
    region_ticks_only: bool,
    max_region_labels: int,
):
    if plt is None:
        raise SystemExit("matplotlib is not installed. Run: pip install matplotlib numpy")
    if not points:
        raise SystemExit("No points to plot.")

    pts = [p for p in points if len(p.coord) >= 2]
    if not pts:
        raise SystemExit("Selected accesses are not rank >= 2.")

    # 画图时沿用矩阵/图像习惯：dim0 是 y 轴，dim1 是 x 轴。
    xs = [p.coord[1] for p in pts]
    ys = [p.coord[0] for p in pts]
    regions = find_connected_regions({(p.coord[0], p.coord[1]) for p in pts})
    if label_regions or region_ticks_only:
        print_regions(regions)

    fig, ax = plt.subplots(figsize=(10, 9))

    if color_by:
        key = normalize_name(color_by)
        vals = []
        missing = 0
        for p in pts:
            # color-by 可以选择静态 site、读写类型，也可以选择某个循环标签/SSA 变量。
            if key in ("site", "site_id"):
                vals.append(p.site_id)
            elif key in ("kind",):
                vals.append(0 if p.kind == "read" else 1)
            elif key in p.env:
                vals.append(p.env[key])
            else:
                vals.append(0)
                missing += 1

        sc = ax.scatter(xs, ys, c=vals, s=marker_size, alpha=0.85, linewidths=0)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(color_by)
        if missing == len(pts):
            print(f"warning: --color-by {color_by!r} not found in point environments", file=sys.stderr)
    else:
        ax.scatter(xs, ys, s=marker_size, alpha=0.85, linewidths=0)

    ax.set_xlabel("buffer dim1 / x")
    ax.set_ylabel("buffer dim0 / y")
    ax.set_aspect("equal", adjustable="box")
    if coordinate_grid:
        apply_coordinate_grid(ax, xs, ys)
    else:
        ax.grid(True, linewidth=0.35, alpha=0.35)
    if region_ticks_only:
        apply_region_ticks(ax, regions)
    if invert_y:
        ax.invert_yaxis()
    if label_regions:
        annotate_regions(ax, regions, max_region_labels)

    if title:
        ax.set_title(title)
    else:
        buffers = sorted({p.buffer for p in pts})
        ax.set_title("MLIR buffer access: " + ", ".join("%" + b for b in buffers))

    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=180, bbox_inches="tight")
        print(f"Saved plot to: {output}")
    else:
        plt.show()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_set(values: Sequence[str]) -> Dict[str, int]:
    env = {}
    for item in values:
        # --set block_id_y=0 这类外部值会作为解释器的初始 SSA 环境。
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--set expects NAME=VALUE, got: {item}")
        k, v = item.split("=", 1)
        env[normalize_name(k.strip())] = int(v.strip(), 0)
    return env

def parse_where(values: Sequence[str]) -> List[EnvFilter]:
    filters: List[EnvFilter] = []
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"--where expects NAME=VALUE or NAME=START:END, got: {item}")
        k, v = item.split("=", 1)
        name = normalize_name(k.strip())
        value = v.strip()
        if ":" in value:
            a, b = value.split(":", 1)
            filters.append((name, int(a, 0), int(b, 0)))
        else:
            filters.append((name, int(value, 0), None))
    return filters

def apply_thread_range(envs: List[Dict[str, int]], spec: Optional[str]) -> List[Dict[str, int]]:
    if not spec:
        return envs
    if ":" in spec:
        a, b = spec.split(":", 1)
        values = range(int(a), int(b))
    else:
        values = [int(spec)]
    out = []
    for env in envs:
        for tid in values:
            # 为每个 thread_id_x 生成一份独立环境，相当于重复解释同一段 IR。
            e = dict(env)
            e["thread_id_x"] = tid
            out.append(e)
    return out

def main():
    ap = argparse.ArgumentParser(
        description="Interpret affine loop/access patterns and visualize rank-2 buffer IO points."
    )
    ap.add_argument("mlir", type=Path, default="/data2/xsl/DeepGenGraph/3rd/deepgengraph/build/frisk.mlir")
    ap.add_argument("--list", action="store_true", help="list access sites and exit")
    ap.add_argument("--buffer", help="SSA buffer name, with or without %%")
    ap.add_argument("--op", choices=["all", "affine", "copy"], default="all",
                    help="access operation class to interpret")
    ap.add_argument("--site", type=int, help="only collect one listed site id")
    ap.add_argument("--thread", help="thread_id_x value or half-open range, e.g. 0 or 0:128", default="0:1")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="set an external SSA/index value; repeatable")
    ap.add_argument("--where", action="append", default=[], metavar="NAME=VALUE",
                    help="only collect dynamic points whose env matches; VALUE may be START:END")
    ap.add_argument("--color-by", help="thread_id_x, loop iterLabel (e.g. br0), SSA IV, site")
    ap.add_argument("--output", help="save plot to PNG/PDF instead of opening a window")
    ap.add_argument("--title")
    ap.add_argument("--no-invert-y", action="store_true")
    ap.add_argument("--no-coordinate-grid", action="store_true",
                    help="use matplotlib's automatic axis/grid instead of integer coordinate grid")
    ap.add_argument("--label-regions", action="store_true",
                    help="label each connected region's top-left coordinate as (dim0,dim1)")
    ap.add_argument("--region-ticks-only", action="store_true",
                    help="show axis ticks only at connected region top-left coordinates")
    ap.add_argument("--max-region-labels", type=int, default=200)
    ap.add_argument("--dedup-points", action="store_true",
                    help="store one point per site/kind/coordinate/color value instead of every dynamic access")
    ap.add_argument("--marker-size", type=float, default=12.0)
    ap.add_argument("--max-points", type=int, default=1_000_000)
    ap.add_argument("--verbose-errors", action="store_true")
    args = ap.parse_args()

    text = args.mlir.read_text(encoding="utf-8")
    roots = parse_ir(text)
    shapes = discover_shapes(text)

    base_env = parse_set(args.set)
    env_filters = parse_where(args.where)
    envs = apply_thread_range([base_env], args.thread)
    stop_after_line = None
    if args.site is None and not args.list:
        stop_after_line = find_last_relevant_access_line(roots, shapes, args.buffer, args.op)
    keep_env_keys: List[str] = []
    if args.color_by:
        color_key = normalize_name(args.color_by)
        if color_key not in ("site", "site_id", "kind"):
            keep_env_keys.append(color_key)

    # First pass with no buffer selection is useful for --list.
    # 注意：每个线程环境都会单独跑一遍解释器，所以 site 需要在所有线程结果中再合并。
    combined_sites: Dict[Tuple[int, str, str], AccessSite] = {}
    all_points: List[Point] = []
    errors = Counter()

    for env in envs:
        it = Interpreter(
            roots=roots,
            shapes=shapes,
            selected_buffer=args.buffer,
            op_mode=args.op,
            selected_site=args.site,
            initial_env=env,
            env_filters=env_filters,
            max_points=max(1, args.max_points - len(all_points)),
            stop_after_line=stop_after_line,
            keep_env_keys=keep_env_keys,
            dedup_points=args.dedup_points,
        )
        try:
            it.run()
        except PointLimitReached as e:
            print(str(e), file=sys.stderr)
            all_points.extend(it.points)
            errors.update(it.errors)
            for s in it.sites:
                combined_sites[(s.line_no, s.kind, s.buffer)] = s
            break
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            break
        all_points.extend(it.points)
        errors.update(it.errors)
        for s in it.sites:
            combined_sites[(s.line_no, s.kind, s.buffer)] = s

    # Renumber sites deterministically because each thread interpretation creates its own list.
    # 合并后按源码位置重新编号，保证 --list 看到的 site id 稳定可复现。
    sites = sorted(combined_sites.values(), key=lambda s: (s.line_no, s.kind, s.buffer))
    renumber = {(s.line_no, s.kind, s.buffer): i for i, s in enumerate(sites)}
    for i, s in enumerate(sites):
        s.site_id = i

    # Re-map dynamic point site ids by line/kind/buffer.
    # Old site numbers are local to each interpreter, so use static key.
    # 动态 Point 里原来的 site_id 是单个 Interpreter 内部编号，需要映射到合并后的全局编号。
    for p in all_points:
        static_kind = f"affine.{p.kind}"
        # frisk.copy points need alternate static kind.
        candidates = [
            (p.line_no, static_kind, p.buffer),
            (p.line_no, f"frisk.copy.{p.kind}", p.buffer),
        ]
        for key in candidates:
            if key in renumber:
                p.site_id = renumber[key]
                break

    if args.list:
        print_sites(sites, args.buffer)
        if errors and args.verbose_errors:
            print("\nSkipped/unsupported evaluations:")
            for msg, count in errors.most_common():
                print(f"  {count:6d}  {msg}")
        return

    if not args.buffer and args.site is None:
        print_sites(sites)
        print("\nChoose one buffer with --buffer NAME or one access with --site N.")
        return

    summarize(all_points)

    if errors:
        print(f"\nSkipped/failed evaluations: {sum(errors.values())}")
        if args.verbose_errors:
            for msg, count in errors.most_common():
                print(f"  {count:6d}  {msg}")
        else:
            print("Use --verbose-errors to see details.")

    visualize(
        all_points,
        color_by=args.color_by,
        output=args.output,
        title=args.title,
        invert_y=not args.no_invert_y,
        marker_size=args.marker_size,
        coordinate_grid=not args.no_coordinate_grid,
        label_regions=args.label_regions,
        region_ticks_only=args.region_ticks_only,
        max_region_labels=args.max_region_labels,
    )

if __name__ == "__main__":
    main()
