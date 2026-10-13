#!/usr/bin/env python3
"""基于 test_input.mlir 生成不同 BM/BN/BK 的 masked attention IR。

用法：
  python3 generate_attention_ir.py --bm 128 --bn 64
  python3 generate_attention_ir.py --bm 32 64 128 --bn 32 64 --output-dir generated_ir
  python3 generate_attention_ir.py --bm 64 --bn 32 --bk 32 --output 3rd/deepgengraph/test/test_input_spidD.mlir
  python3 generate_attention_ir.py --bm 64 --bn 32 --bk 16 32 64 --output-dir generated_ir

输出名：test_input_{B}x{H}x{S}x{D}-{BM}-{BN}.mlir。
模板的输入布局为 BSHD；文件名使用 BHSD。只改变 kernel 的分块参数，
保留模板中的寻址、计算及缩放方式，func.func @Attn 原文保持不变。
BM、BN 必须为 S 的正整数因子，因为模板没有尾块越界保护。
生成 IR 不代表所有分块都满足后端的硬件指令和共享内存限制。
指定 BK 时输出名增加 -{BK}；BK 必须整除 D。不指定则保留原 BM/BN 模式。
split-D 的 QK 使用 f32 累加，完成整个 D 的规约后才计算 mask/exp2。
V/O 按输出列分块（不沿输出 D 求和），每个输出块重算 QK/分母。
沿用模板的预缩放与非稳定 exp2 softmax，不改变其数值范围限制。
"""

import argparse
import itertools
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "3rd/deepgengraph/test/test_input.mlir"


def replace_once(text, pattern, replacement):
    result, count = re.subn(pattern, replacement, text)
    if count != 1:
        raise ValueError(f"模板结构不符合预期：{pattern!r} 匹配了 {count} 次")
    return result


def generate_ir(source, bm, bn, bk=None):
    """返回 (IR 文本, BHSD)，针对仓库中 BM=64、BN=32 的模板。"""
    # 单独截取 kernel；其前后的字节（包括完整 Attn）不做任何替换。
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
    b, s, h, d = map(int, shape.groups())
    if bm <= 0 or bn <= 0 or s % bm or s % bn:
        raise ValueError(f"BM={bm}、BN={bn} 必须为正整数且均能整除 S={s}")
    if bk is not None and (bk <= 0 or d % bk):
        raise ValueError(f"BK={bk} 必须为正整数且能整除 D={d}")
    # 以下映射专用于给定模板。若基础模板改变，明确报错，避免误改 H/D。
    if d != 128 or not re.search(r"sizes = \[64, 32\]", kernel):
        raise ValueError("此脚本要求模板 D=128、BM=64、BN=32")

    # 模板中的调试注释含有旧尺寸和已注释代码，生成时去掉 kernel 注释。
    kernel = re.sub(r"[ \t]*//[^\n]*", "", kernel)
    kernel = re.sub(r"^[ \t]+\n", "", kernel, flags=re.M)

    # 同时替换二维 tile 类型，避免 BM=32/128、BN=64/128 时串联替换。
    # 四维完整 tensor、H、D、全局 stride 和 scale 均保持原值。
    tiles = {
        (64, 128): (bm, d),
        (64, 1): (bm, 1),
        (128, 32): (d, bn),
        (32, 128): (bn, d),
        (64, 32): (bm, bn),
    }

    def tile_type(match):
        dims = tuple(map(int, match.group(1, 2)))
        rows, cols = tiles.get(dims, dims)
        return f"tensor<{rows}x{cols}x"

    def tile_shape(match):
        dims = tuple(map(int, match.group(2, 3)))
        rows, cols = tiles.get(dims, dims)
        return f"{match[1]} = [{rows}, {cols}]"

    kernel = re.sub(r"tensor<(\d+)x(\d+)x(?=f\d+>)", tile_type, kernel)
    kernel = re.sub(r"\b(shape|block_shape|sizes) = \[(\d+), (\d+)\]",
                    tile_shape, kernel)
    kernel = replace_once(
        kernel,
        r"(arg_dims = \[1, -1, -1\], res_dims = \[1\], )"
        r"size_per_unit = 64 : i64, unit_num = \d+ : i64",
        lambda m: f"{m[1]}size_per_unit = {bm} : i64, unit_num = {s // bm} : i64",
    )
    kernel = replace_once(kernel, r"grid = \[\d+, \d+, \d+\]",
                          f"grid = [{b}, {s // bm}, {h}]")

    # 用语义化 SSA 名称，BM/BN 与 D/H 相等时也不会出现重复定义。
    kernel = re.sub(r"%c(?:64|32)\b",
                    lambda m: "%cBM" if m[0] == "%c64" else "%cBN", kernel)
    kernel = replace_once(kernel, r"%cBM = arith.constant 64 : index",
                          f"%cBM = arith.constant {bm} : index")
    kernel = replace_once(kernel, r"%cBN = arith.constant 32 : index",
                          f"%cBN = arith.constant {bn} : index")
    kernel = replace_once(kernel, r"(block_advance %tempK, offsets = )\[0, 32\]",
                          lambda m: f"{m[1]}[0, {bn}]")
    kernel = replace_once(kernel, r"(block_advance %tempV, offsets = )\[32, 0\]",
                          lambda m: f"{m[1]}[{bn}, 0]")
    if bk is not None:
        kernel = split_d_kernel(kernel, b, h, s, d, bm, bn, bk)
    # causal attention 需覆盖当前 Q tile 的全部有效 key：(bx + 1) * BM。
    # BN > BM 也可用；多加载的未来 key 由原有 causal mask 排除。
    return source[:start] + kernel + source[end:], (b, h, s, d)


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
    parser.add_argument("--bm", "--BM", type=int, nargs="+", required=True,
                        help="Q 行分块大小，可输入多个值")
    parser.add_argument("--bn", "--BN", type=int, nargs="+", required=True,
                        help="K/V 序列分块大小，可输入多个值；与 BM 取笛卡尔积")
    parser.add_argument("--bk", "--BK", type=int, nargs="+",
                        help="D 轴分块大小，可输入多个值；必须整除 D")
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
            ir, bhsd = generate_ir(source, bm, bn, bk)
            suffix = f"-{bk}" if bk is not None else ""
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
