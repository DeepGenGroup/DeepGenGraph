#!/usr/bin/env python3
"""基于 test_input.mlir 生成不同 BM/BN 的 masked attention IR。

用法：
  python3 generate_attention_ir.py --bm 128 --bn 64
  python3 generate_attention_ir.py --bm 32 64 128 --bn 32 64 --output-dir generated_ir

输出名：test_input_{B}x{H}x{S}x{D}-{BM}-{BN}.mlir。
模板的输入布局为 BSHD；文件名使用 BHSD。只改变 kernel 的分块参数，
保留模板中的寻址、计算及缩放方式，func.func @Attn 原文保持不变。
BM、BN 必须为 S 的正整数因子，因为模板没有尾块越界保护。
生成 IR 不代表所有分块都满足后端的硬件指令和共享内存限制。
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


def generate_ir(source, bm, bn):
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
    # causal attention 需覆盖当前 Q tile 的全部有效 key：(bx + 1) * BM。
    # BN > BM 也可用；多加载的未来 key 由原有 causal mask 排除。
    return source[:start] + kernel + source[end:], (b, h, s, d)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bm", "--BM", type=int, nargs="+", required=True,
                        help="Q 行分块大小，可输入多个值")
    parser.add_argument("--bn", "--BN", type=int, nargs="+", required=True,
                        help="K/V 序列分块大小，可输入多个值；与 BM 取笛卡尔积")
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="输出目录，默认脚本所在目录")
    args = parser.parse_args()
    try:
        source = TEMPLATE.read_bytes().decode("utf-8")
        outputs = []
        for bm, bn in itertools.product(dict.fromkeys(args.bm), dict.fromkeys(args.bn)):
            ir, bhsd = generate_ir(source, bm, bn)
            filename = f"test_input_{'x'.join(map(str, bhsd))}-{bm}-{bn}.mlir"
            outputs.append((args.output_dir / filename, ir))
        # 所有参数先检查通过，再开始输出；模板永远不会作为输出目标。
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path, ir in outputs:
            if path.resolve() == TEMPLATE.resolve():
                raise ValueError("输出路径不能指向模板")
            path.write_bytes(ir.encode("utf-8"))
            print(path.resolve())
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
