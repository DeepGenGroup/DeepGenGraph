#!/usr/bin/env python3
"""基于 test_input.mlir 生成不同 BHSD 尺寸和 BM/BN/BK 的 attention IR。

用法：
  python3 generate_attention_ir.py --bm 128 --bn 64
  python3 generate_attention_ir.py --bm 32 64 128 --bn 32 64 --output-dir generated_ir
  python3 generate_attention_ir.py --bm 64 --bn 32 --bk 32 --output 3rd/deepgengraph/test/test_input_spidD.mlir
  python3 generate_attention_ir.py --bm 64 --bn 32 --bk 16 32 64 --output-dir generated_ir
  python3 generate_attention_ir.py --b 2 --h 8 --s 1024 --d 64 --bm 64 --bn 32 --output-dir generated_ir

标准输出为每行一个生成文件的绝对路径，可直接交给 TuneAttention：
  set -o pipefail
  python3 generate_attention_ir.py --bm 64 --bn 32 --bk 16 32 --output-dir generated_ir/tiles |
    python3 TuneAttention.py --input-list - --case baseline
未指定 --bk 时，可使用 TuneAttention 的 --case legacy-baseline。
在目标 GPU 环境给 TuneAttention 加 --benchmark，可继续生成 HSACO 并校验、测时。

输出名：test_input_{B}x{H}x{S}x{D}-{BM}-{BN}.mlir。
模板的输入布局为 BSHD；参数和文件名使用 BHSD。
--b/--h/--s/--d 必须为正整数，未指定的维度沿用模板（默认 1/32/4096/128）。
尺寸改变时同步更新高层 Attn 和 kernel 的类型、寻址、grid 及 D 相关缩放；
默认生成 online softmax。
默认 causal；--non-causal 移除高层及 kernel 的 mask，并遍历全部 S 个 key。
BM、BN 必须为 S 的正整数因子，因为模板没有尾块越界保护。
生成 IR 不代表所有分块都满足后端的硬件指令和共享内存限制。
指定 BK 时输出名增加 -{BK}；BK 必须整除 D。不指定则保留原 BM/BN 模式。
split-D 的 QK 使用 f32 累加，完成整个 D 的规约后才计算 mask/exp2。
V/O 按输出列分块（不沿输出 D 求和），每个输出块重算 QK/分母。
默认维护跨 key tile 的行最大值，并同步重缩放分母和输出累加器。
--softmax legacy 可复现历史非稳定 exp2 版本；online 文件名增加 -online。
"""

import argparse
import itertools
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
# Portable bundles keep the template beside the scripts; source checkouts
# retain the original test directory. Resolve against the script, not cwd.
TEMPLATE = ROOT / "test_input.mlir"
if not TEMPLATE.is_file():
    TEMPLATE = ROOT / "3rd/deepgengraph/test/test_input.mlir"


def replace_once(text, pattern, replacement):
    result, count = re.subn(pattern, replacement, text)
    if count != 1:
        raise ValueError(f"模板结构不符合预期：{pattern!r} 匹配了 {count} 次")
    return result


def generate_ir(source, bm, bn, bk=None, softmax="online", *,
                b=None, h=None, s=None, d=None, causal=True):
    """返回 (IR 文本, BHSD)，针对仓库中 BM=64、BN=32 的模板。"""
    if softmax not in ("online", "legacy"):
        raise ValueError("softmax 必须为 online 或 legacy")
    if type(causal) is not bool:
        raise ValueError("causal 必须为布尔值")
    # 分块替换仅作用于 kernel；尺寸和 online softmax 同时更新高层 Attn。
    kernels = list(re.finditer(r"^  deepgengraph\.kernel @Attn_p2\(", source, re.M))
    if len(kernels) != 1:
        raise ValueError("模板必须包含一个 deepgengraph.kernel @Attn_p2")
    start = kernels[0].start()
    end_match = re.search(r"^  }[^\n]*(?:\n|$)", source[kernels[0].end():], re.M)
    if end_match is None:
        raise ValueError("找不到 kernel 结束位置")
    end = kernels[0].end() + end_match.end()
    kernel = source[start:end]
    shape = re.search(r"%Q: tensor<(\d+)x(\d+)x(\d+)x(\d+)xf16>", kernel)
    if shape is None:
        raise ValueError("模板 Q 必须为静态 BSHD f16 tensor")
    old_b, old_s, old_h, old_d = map(int, shape.groups())
    b = old_b if b is None else b
    h = old_h if h is None else h
    s = old_s if s is None else s
    d = old_d if d is None else d
    for name, value in (("B", b), ("H", h), ("S", s), ("D", d)):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name}={value} 必须为正整数")
    if bm <= 0 or bn <= 0 or s % bm or s % bn:
        raise ValueError(f"BM={bm}、BN={bn} 必须为正整数且均能整除 S={s}")
    if bk is not None and (bk <= 0 or d % bk):
        raise ValueError(f"BK={bk} 必须为正整数且能整除 D={d}")
    # 以下映射专用于给定模板。若基础模板改变，明确报错，避免误改 H/D。
    if old_d != 128 or not re.search(r"sizes = \[64, 32\]", kernel):
        raise ValueError("此脚本要求模板 D=128、BM=64、BN=32")

    # 模板中的调试注释含有旧尺寸和已注释代码，生成时去掉 kernel 注释。
    kernel = re.sub(r"[ \t]*//[^\n]*", "", kernel)
    kernel = re.sub(r"^[ \t]+\n", "", kernel, flags=re.M)

    # 按完整形状一次性替换，避免新维度与旧维度相等时发生串联替换。
    full_shapes = {
        (old_b, old_s, old_h, old_d): (b, s, h, d),
        (old_b, old_h, old_s, old_d): (b, h, s, d),
        (old_b, old_h, old_d, old_s): (b, h, d, s),
        (old_b, old_h, old_s, old_s): (b, h, s, s),
        (old_b, old_h, old_s, 1): (b, h, s, 1),
        (old_s, old_s): (s, s),
    }

    def full_type(match):
        dims = tuple(map(int, match[1].split("x")))
        return "tensor<" + "x".join(map(str, full_shapes.get(dims, dims))) + "x"

    prefix = source[:start]
    if (b, h, s, d) != (old_b, old_h, old_s, old_d):
        prefix = re.sub(r"tensor<(\d+(?:x\d+)+)x(?=f\d+>)", full_type, prefix)
        kernel = re.sub(r"tensor<(\d+x\d+x\d+x\d+)x(?=f\d+>)", full_type, kernel)
        prefix = replace_once(prefix, r"(deepgengraph.trilu[^\n]*shape = )\[\d+, \d+\]",
                              lambda m: f"{m[1]}[{s}, {s}]")
    if d != old_d:
        prefix = replace_once(prefix, r"(?m)(%cst = arith.constant dense<)[^>]+(>[^\n]*?)(?:[ \t]*//[^\n]*)?$",
                              lambda m: f"{m[1]}{math.sqrt(d):.9e}{m[2]}")
        kernel = replace_once(kernel, r"(%cst = arith.constant dense<)[^>]+>",
                              lambda m: f"{m[1]}{math.log2(math.e) / math.sqrt(d):.9e}>")

    # 同时替换二维 tile 类型，避免 BM=32/128、BN=64/128 时串联替换。
    tiles = {
        (64, 128): (bm, d),
        (64, 1): (bm, 1),
        (128, 32): (d, bn),
        (32, 128): (bn, d),
        (64, 32): (bm, bn),
    }
    pointer_shapes = {**tiles, (128, old_s): (d, s), (old_s, 128): (s, d)}

    def tile_type(match):
        dims = tuple(map(int, match.group(1, 2)))
        rows, cols = tiles.get(dims, dims)
        return f"tensor<{rows}x{cols}x"

    def tile_shape(match):
        dims = tuple(map(int, match.group(2, 3)))
        rows, cols = pointer_shapes.get(dims, dims)
        return f"{match[1]} = [{rows}, {cols}]"

    kernel = re.sub(r"tensor<(\d+)x(\d+)x(?=f\d+>)", tile_type, kernel)
    kernel = re.sub(r"\b(shape|block_shape|sizes) = \[(\d+), (\d+)\]",
                    tile_shape, kernel)
    kernel = re.sub(r"stride = \[(\d+), (\d+)\]",
                    lambda m: f"stride = [1, {h*d}]" if m[1] == "1"
                    else f"stride = [{h*d}, 1]", kernel)
    for dim, units in ((0, b), (2, h)):
        kernel = replace_once(
            kernel,
            rf"(arg_dims = \[{dim}, {dim}, {dim}\], res_dims = \[{dim}\], "
            r"size_per_unit = 1 : i64, unit_num = )\d+ : i64",
            lambda m: f"{m[1]}{units} : i64",
        )
    kernel = replace_once(
        kernel,
        r"(arg_dims = \[1, -1, -1\], res_dims = \[1\], )"
        r"size_per_unit = 64 : i64, unit_num = \d+ : i64",
        lambda m: f"{m[1]}size_per_unit = {bm} : i64, unit_num = {s // bm} : i64",
    )
    # Grid follows entry argument order [z, y, x]; bx denotes the query tile.
    axes = re.search(r"\^bb0\(%bz: index, %(bx|by): index, %(bx|by): index,", kernel)
    if axes is None or axes[1] == axes[2]:
        raise ValueError("模板入口必须包含 bz 和不同的 bx/by 轴")
    extents = {"bx": s // bm, "by": h}
    kernel = replace_once(kernel, r"grid = \[\d+, \d+, \d+\]",
                          f"grid = [{b}, {extents[axes[1]]}, {extents[axes[2]]}]")

    # 用语义化 SSA 名称，BM/BN 与 D/H 相等时也不会出现重复定义。
    kernel = re.sub(r"%c(?:64|32)\b",
                    lambda m: "%cBM" if m[0] == "%c64" else "%cBN", kernel)
    kernel = replace_once(kernel, r"%cBM = arith.constant 64 : index",
                          f"%cBM = arith.constant {bm} : index")
    kernel = replace_once(kernel, r"%cBN = arith.constant 32 : index",
                          f"%cBN = arith.constant {bn} : index")
    kernel = re.sub(rf"%c(?:{old_s}|128)\b",
                    lambda m: "%cD" if m[0] == "%c128" else "%cS", kernel)
    kernel = replace_once(kernel, rf"%cS = arith.constant {old_s} : index",
                          f"%cS = arith.constant {s} : index")
    kernel = replace_once(kernel, r"%cD = arith.constant 128 : index",
                          f"%cD = arith.constant {d} : index")
    if b > 1:
        # base_offset 使用 permute 后的 BHSD/BHDS 布局，batch 跨 H*S*D。
        kernel = replace_once(
            kernel, r"%6 = arith.muli %5, %cS : index",
            f"%cHSD = arith.constant {h*s*d} : index\n"
            "      %batchBase = arith.muli %bz, %cHSD : index\n"
            "      %headOffset = arith.muli %5, %cS : index\n"
            "      %6 = arith.addi %batchBase, %headOffset : index",
        )
    kernel = replace_once(kernel, r"(block_advance %tempK, offsets = )\[0, 32\]",
                          lambda m: f"{m[1]}[0, {bn}]")
    kernel = replace_once(kernel, r"(block_advance %tempV, offsets = )\[32, 0\]",
                          lambda m: f"{m[1]}[{bn}, 0]")
    if bk is not None:
        kernel = split_d_kernel(kernel, b, h, s, d, bm, bn, bk)
    if softmax == "online":
        kernel = online_softmax_kernel(kernel, bm, bn, d, bk)
    # causal attention 需覆盖当前 Q tile 的全部有效 key：(bx + 1) * BM。
    # BN > BM 也可用；多加载的未来 key 由原有 causal mask 排除。
    if softmax == "online":
        prefix = stable_reference_function(prefix)
    if not causal:
        # Remove both masks after the shape/softmax transformations, whose
        # template contracts deliberately still describe causal attention.
        prefix = replace_once(prefix, r"(?m)^    %0 = deepgengraph.trilu[^\n]*\n", "")
        prefix = replace_once(prefix, r"(?m)^    %6 = deepgengraph.add %5, %0[^\n]*\n", "")
        prefix = re.sub(r"%6\b", "%5", prefix)
        mask, masked_score, score = (("%mask", "%masked", "%qk") if bk is not None
                                     else ("%25", "%26", "%24"))
        kernel = replace_once(
            kernel, rf"(?ms)^[ \t]*{mask} = deepgengraph.mask starts = .*?"
            r"^[ \t]*} : \(index, index\) -> tensor<[^>]+>\n", "")
        kernel = replace_once(
            kernel, rf"(?m)^[ \t]*{masked_score} = deepgengraph.add {score}, {mask}[^\n]*\n", "")
        kernel = re.sub(rf"{masked_score}\b", score, kernel)
        kernel = replace_once(kernel, r"to %(?:keyEnd|loopUb) step %cBN", "to %cS step %cBN")
    return prefix + kernel + source[end:], (b, h, s, d)


def stable_reference_function(text):
    """Keep the accompanying high-level function stable, including FP32 QK."""
    text = replace_once(text, r"(%4 = )deepgengraph.dot (%1, %3) : (.*?) -> (tensor<[^>]+)xf16>",
                        r"\1deepgengraph.precise_dot_op \2, acc = f32 : \3 -> \4xf32>")
    text = replace_once(text, r"(%cst = arith.constant dense<[^>]+> : tensor<1x)f16>", r"\1f32>")
    for name in ("5", "6"):
        text = replace_once(text, rf"(%{name} = [^\n]+)",
                            lambda m: m[0].replace("xf16>", "xf32>"))
    text = replace_once(text, r"(val = )0xFC00 : f16", r"\g<1>0xFF800000 : f32")
    # Conversion is now an identity; use the masked FP32 score directly.
    text = replace_once(text, r"    %7 = deepgengraph.convert %6[^\n]*\n", "")
    match = re.search(r"    %8 = deepgengraph.exp %7 : \((tensor<([^>]+)xf32>)\) -> [^\n]+", text)
    if match is None:
        raise ValueError("模板 Attn 缺少预期的 exp softmax")
    score = match[1]
    dims = match[2].split("x")
    row = 'tensor<' + 'x'.join(dims[:-1] + ['1']) + 'xf32>'
    replacement = (f"    %rowMax = deepgengraph.reduce(%6), dim = -1, op = MAX, keep_dim = true : ({score}) -> {row}\n"
                   f"    %shifted = deepgengraph.sub %6, %rowMax : ({score}, {row}) -> {score}\n"
                   f"    %8 = deepgengraph.exp %shifted : ({score}) -> {score}")
    return text[:match.start()] + replacement + text[match.end():]


def online_softmax_kernel(kernel, bm, bn, d, bk):
    """m'=max(m, rowmax(s)); alpha=exp2(m-m'); (l,o)'=alpha*(l,o)+(sum(p),pV).

    Scores already use log2(e)/sqrt(D). Every causal row includes key zero in
    the first tile; later fully masked row/tile pairs keep the finite old max.
    Consequently neither initial -inf - -inf nor division by zero is needed
    for finite Q/K/V in this template's valid launch geometry.
    """
    row = f"tensor<{bm}x1xf32>"
    score = f"tensor<{bm}x{bn}xf32>"
    out = f"tensor<{bm}x{bk or d}xf32>"
    split = bk is not None
    zero, old_m = ("%rowZero", "%accM") if split else ("%15", "%accM")
    old_l, old_o = ("%accL", "%accO") if split else ("%arg14", "%arg13")
    masked, weights = ("%masked", "%weights") if split else ("%26", "%27")
    init = (f"\n        %maxInitValue = arith.constant dense<0xFF800000> : tensor<1xf32>\n"
            f"        %rowNegInf = deepgengraph.add {zero}, %maxInitValue : ({row}, tensor<1xf32>) -> {row}")
    kernel = replace_once(kernel, rf"({re.escape(zero)} = deepgengraph.zero[^\n]+)", lambda m: m[0] + init)
    if split:
        kernel = replace_once(kernel, r"%attention:2 = ([^\n]+)%accL = %rowZero\) -> \(([^\n]+)\) \{",
                              r"%attention:3 = \1%accL = %rowZero, %accM = %rowNegInf) -> (\2, " + row + ") {")
    else:
        kernel = replace_once(kernel, r"%19:4 = ([^\n]+)%arg14 = %15\) -> \(([^\n]+)\) \{",
                              r"%19:5 = \1%arg14 = %15, %accM = %rowNegInf) -> (\2, " + row + ") {")
    stable = (f"%nextM = deepgengraph.reduce({masked}, init = {old_m}), dim = 1, op = MAX, keep_dim = true : ({score}, {row}) -> {row}\n"
              f"          %shifted = deepgengraph.sub {masked}, %nextM : ({score}, {row}) -> {score}\n"
              f"          %maxDelta = deepgengraph.sub {old_m}, %nextM : ({row}, {row}) -> {row}\n"
              f"          %alpha = deepgengraph.exp2 %maxDelta : ({row}) -> {row}\n"
              f"          %scaledL = deepgengraph.mul {old_l}, %alpha : ({row}, {row}) -> {row}\n"
              f"          {weights} = deepgengraph.exp2 %shifted : ({score}) -> {score}")
    kernel = replace_once(kernel, rf"{re.escape(weights)} = deepgengraph.exp2 {re.escape(masked)}[^\n]+", stable)
    kernel = replace_once(kernel, rf"(deepgengraph.reduce\({re.escape(weights)}, init = ){re.escape(old_l)}", r"\1%scaledL")
    next_o = "%nextO" if split else "%31"
    kernel = replace_once(kernel, rf"({re.escape(next_o)} = deepgengraph.add ){re.escape(old_o)}",
                          f"%scaledO = deepgengraph.mul {old_o}, %alpha : ({out}, {row}) -> {out}\n          " + r"\1%scaledO")
    yields = "%nextO, %nextL" if split else "%32, %33, %31, %28"
    kernel = replace_once(kernel, rf"scf.yield {re.escape(yields)} : ([^\n]+)",
                          f"scf.yield {yields}, %nextM : " + r"\1, " + row)
    return kernel


def split_d_kernel(kernel, b, h, s, d, bm, bn, bk):
    """Replace only the device body, retaining the template's signature/ABI.

    The existing compiler interprets base_offset in each argument's permuted
    layout (Q/V/O: BHSD; K: BHDS), and remaps the original BSHD strides.
    In particular K's reduction offset is dk*S, not dk. The subsequent
    NormalizeAttentionKLayout pass exposes contiguous BHSD K at launch time.
    Use explicit offsets: nested block_advance currently flattens parent loops
    in the lowering and would incorrectly carry Q/K pointers across key tiles.
    """
    full = f"tensor<{b}x{s}x{h}x{d}xf16>"
    ptr = f"!deepgengraph_triton.ptr<{full}>"

    def tile(rows, cols, dtype="f16"):
        return f"tensor<{rows}x{cols}x{dtype}>"

    def block(rows, cols):
        return f"!deepgengraph_triton<block_ptr{{{tile(rows, cols)}}}>"

    def pointer(name, arg, offset, rows, cols, shape, stride, order):
        return (f"%{name} = deepgengraph_triton.block_ptr_of base = %{arg}, "
                f"base_offset = %{offset}, shape = [{shape[0]}, {shape[1]}], "
                f"stride = [{stride[0]}, {stride[1]}], offset = [0, 0], "
                f"block_shape = [{rows}, {cols}], order = [{order[0]}, {order[1]}] "
                f": ({ptr}, index) -> {block(rows, cols)}")

    q, k, v = tile(bm, bk), tile(bk, bn), tile(bn, bk)
    score, prob = tile(bm, bn, "f32"), tile(bm, bn)
    out, row = tile(bm, bk, "f32"), tile(bm, 1, "f32")
    # Preserve the template's exp2 scale, including its f16 conversion.
    scale = re.search(r"%cst = arith.constant dense<([^>]+)>", kernel)[1]
    body = f"""
      // BK splits QK's reduction dimension and V/O's output columns.
      %scale32 = arith.constant dense<{scale}> : tensor<1xf32>
      %scale = deepgengraph.convert %scale32, type = f16 : (tensor<1xf32>) -> tensor<1xf16>
      %negInf = arith.constant 0xFF800000 : f32
      %zero = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cBM = arith.constant {bm} : index
      %cBN = arith.constant {bn} : index
      %cBK = arith.constant {bk} : index
      %cD = arith.constant {d} : index
      %cS = arith.constant {s} : index
      %cSD = arith.constant {s*d} : index
      %cHSD = arith.constant {h*s*d} : index
      %batchBase = arith.muli %bz, %cHSD : index
      %headOffset = arith.muli %by, %cSD : index
      %headBase = arith.addi %batchBase, %headOffset : index
      %qStart = arith.muli %bx, %cBM : index
      %qOffset = arith.muli %qStart, %cD : index
      %qBase = arith.addi %headBase, %qOffset : index
      %keyEnd = arith.addi %qStart, %cBM : index
      // Output D is NOT a reduction axis of PV. Store each BK-column slice.
      affine.for %od = 0 to {d} step {bk} {{
        %outOffset = arith.addi %qBase, %od : index
        {pointer('outPtr', 'argO', 'outOffset', bm, bk, (bm, d), (h*d, 1), (1, 0))}
        %outZero = deepgengraph.zero shape = [{bm}, {bk}], type = f32 : () -> {out}
        %rowZero = deepgengraph.zero shape = [{bm}, 1], type = f32 : () -> {row}
        %attention:2 = scf.for %n = %c0 to %keyEnd step %cBN iter_args(%accO = %outZero, %accL = %rowZero) -> ({out}, {row}) {{
          %scoreZero = deepgengraph.zero shape = [{bm}, {bn}], type = f32 : () -> {score}
          // Reset Q/K and the f32 score accumulator for EVERY key tile.
          %qk = scf.for %dk = %c0 to %cD step %cBK iter_args(%accQK = %scoreZero) -> ({score}) {{
            %qTileOffset = arith.addi %qBase, %dk : index
            %kDOffset = arith.muli %dk, %cS : index
            %kHeadOffset = arith.addi %headBase, %kDOffset : index
            %kTileOffset = arith.addi %kHeadOffset, %n : index
            {pointer('qPtr', 'argQ', 'qTileOffset', bm, bk, (bm, d), (h*d, 1), (1, 0))}
            {pointer('kPtr', 'argK', 'kTileOffset', bk, bn, (d, s), (1, h*d), (0, 1))}
            %qTile = deepgengraph_triton.block_load %qPtr : ({block(bm, bk)}) -> {q}
            %kTile = deepgengraph_triton.block_load %kPtr : ({block(bk, bn)}) -> {k}
            %qScaled = deepgengraph.mul %qTile, %scale : ({q}, tensor<1xf16>) -> {q}
            %partialQK = deepgengraph.precise_dot_op %qScaled, %kTile, acc = f32 : ({q}, {k}) -> {score}
            %nextQK = deepgengraph.add %accQK, %partialQK : ({score}, {score}) -> {score}
            scf.yield %nextQK : {score}
          }}
          %mask = deepgengraph.mask starts = [%qStart, %n], sizes = [{bm}, {bn}], type = f32 {{
          ^bb0(%qi: index, %ki: index):
            %nextRow = arith.addi %qi, %c1 : index
            %future = arith.cmpi ule, %nextRow, %ki : index
            %maskValue = scf.if %future -> (f32) {{
              scf.yield %negInf : f32
            }} else {{
              scf.yield %zero : f32
            }}
            deepgengraph.mask_yield %maskValue : f32
          }} : (index, index) -> {score}
          %masked = deepgengraph.add %qk, %mask : ({score}, {score}) -> {score}
          %weights = deepgengraph.exp2 %masked : ({score}) -> {score}
          %nextL = deepgengraph.reduce(%weights, init = %accL), dim = 1, op = ADD, keep_dim = true : ({score}, {row}) -> {row}
          %p = deepgengraph.convert %weights, type = f16 : ({score}) -> {prob}
          %vRowOffset = arith.muli %n, %cD : index
          %vHeadOffset = arith.addi %headBase, %vRowOffset : index
          %vTileOffset = arith.addi %vHeadOffset, %od : index
          {pointer('vPtr', 'argV', 'vTileOffset', bn, bk, (s, d), (h*d, 1), (1, 0))}
          %vTile = deepgengraph_triton.block_load %vPtr : ({block(bn, bk)}) -> {v}
          %partialO = deepgengraph.precise_dot_op %p, %vTile, acc = f32 : ({prob}, {v}) -> {out}
          %nextO = deepgengraph.add %accO, %partialO : ({out}, {out}) -> {out}
          scf.yield %nextO, %nextL : {out}, {row}
        }}
        %normalized = deepgengraph.div %attention#0, %attention#1 : ({out}, {row}) -> {out}
        %result = deepgengraph.convert %normalized, type = f16 : ({out}) -> {q}
        deepgengraph_triton.block_store %outPtr, %result : ({block(bm, bk)}, {q}) -> ()
      }}
"""
    start = kernel.index("\n", kernel.index("    ^bb0("))
    end = kernel.index("\n    } :", start) + 1
    return kernel[:start] + body + kernel[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for name, description in (("b", "batch 大小"), ("h", "head 数量"),
                              ("s", "序列长度"), ("d", "每个 head 的维度")):
        parser.add_argument(f"--{name}", f"--{name.upper()}", type=int,
                            help=f"{description}，必须为正整数；默认沿用模板")
    parser.add_argument("--bm", "--BM", type=int, nargs="+", required=True,
                        help="Q 行分块大小，可输入多个值")
    parser.add_argument("--bn", "--BN", type=int, nargs="+", required=True,
                        help="K/V 序列分块大小，可输入多个值；与 BM 取笛卡尔积")
    parser.add_argument("--bk", "--BK", type=int, nargs="+",
                        help="D 轴分块大小，可输入多个值；必须整除 D")
    parser.add_argument("--softmax", choices=("online", "legacy"), default="online",
                        help="默认 online：MAX 归约及跨 key tile 重缩放；legacy 复现旧非稳定版本")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--causal", "--casual", dest="causal", action="store_true", default=True)
    mode.add_argument("--non-causal", dest="causal", action="store_false",
                      help="生成无 mask、遍历全部 key 的 full attention")
    parser.add_argument("--output", type=Path,
                        help="单组参数的指定输出文件（例如 test_input_spidD.mlir）")
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="输出目录，默认脚本所在目录")
    args = parser.parse_args()
    try:
        source = TEMPLATE.read_bytes().decode("utf-8")
        outputs = []
        for bm, bn, bk in itertools.product(dict.fromkeys(args.bm), dict.fromkeys(args.bn),
                                           dict.fromkeys(args.bk or [None])):
            ir, bhsd = generate_ir(source, bm, bn, bk, softmax=args.softmax,
                                  b=args.b, h=args.h, s=args.s, d=args.d, causal=args.causal)
            suffix = f"-{bk}" if bk is not None else ""
            if args.softmax == "online":
                suffix += "-online"
            if not args.causal:
                suffix += "-noncausal"
            filename = f"test_input_{'x'.join(map(str, bhsd))}-{bm}-{bn}{suffix}.mlir"
            outputs.append((args.output or args.output_dir / filename, ir))
        if args.output and len(outputs) != 1:
            raise ValueError("--output 只能用于一组 BM/BN/BK 参数")
        # 所有参数先检查通过，再开始输出；模板永远不会作为输出目标。
        for path, ir in outputs:
            if path.resolve() == TEMPLATE.resolve():
                raise ValueError("输出路径不能指向模板")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(ir.encode("utf-8"))
            print(path.resolve())
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
