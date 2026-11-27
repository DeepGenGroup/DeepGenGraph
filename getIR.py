#!/usr/bin/env python3
"""直接调用 MyTest，导出 non-causal attention LLVM IR。

运行：python3 getIR.py
只打印命令：python3 getIR.py --dry-run

对齐 pmc_test.log 1–28 行的 non-causal 基线（median 1.957201 ms），
仅将 workgroup-group-size: 4 -> 1、workgroup-reverse: 1 -> 0。
完整参数来源：
generated_ir/spidD_noncasual/run-20261027-162059-vbwzllhc/online-baseline/config.json
参数含义核对：3rd/deepgengraph/test/test_src.cpp、调优参数说明.md。
这里对齐的是编译配置；新配置的性能需要另行在 GPU 上测量。
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parent
# MyTest 可执行文件路径；使用与基线相同的本地构建，不自动重新编译。
MYTEST = ROOT / "3rd/deepgengraph/build/test/MyTest"
# 输入路径：必须是 MyTest 支持的 MLIR。non-causal / online softmax 由输入算法决定，
# 不是 masks=0/1 决定。本例 BHSD=(1,32,4096,128)，BM=BN=128、BK=32。
# 输入已带 frisk.simplify_loop 标记且无 frisk.pipeline_hint，与归档输入一致。
INPUT = ROOT / "3rd/deepgengraph/test/test_input_spidD_noncasual.mlir"
# 输出目录/文件路径：可改为任意可写路径；重复运行覆盖该目录中的同名结果。
OUTPUT_DIR = ROOT / "generated_ir/noncausal_group1_forward"
OUTPUT = OUTPUT_DIR / "kernel.ll"
LOG = OUTPUT_DIR / "lowering.log"

# 以下两项是环境变量，不是 MyTest 位置参数；显式设置以覆盖 shell 同名值。
COPY_ENV = {
    # global -> LDS 搬运模式；legacy / async / sync / hybrid。
    # useKPack、noKPack 要求 async。
    "FRISK_GM_TO_SHM_MODE": "async",
    # 地址对齐保证（字节）；0=自动推导，或正的 2 次幂。
    # 这是调用者保证，不会自动修正地址；当前 asm128 + swizzle 保持 16。
    "FRISK_GM_TO_SHM_ALIGNMENT": "16",
}

# 位置参数顺序必须与 MyTest 一致；注释中的 argv 编号包含 argv[0]=MyTest。
COMMAND = [
    str(MYTEST),
    str(INPUT),             # argv[1]：输入 MLIR 路径。
    "0",                    # argv[2] pipeline：0/1，旧手工 seqLen 流水调度。
    str(OUTPUT),            # argv[3]：导出 LLVM IR 路径。
    "1",                    # argv[4] reorder：0/1，展开后的 fragment 指令重排。
    "1",                    # argv[5] pack：0/1，packed shared operands。
    "3.3",                  # argv[6] layout：3.1=shared、3.2=shuffle、3.3=b-permute。
    "34816",                # argv[7] carrier：uint64，循环 tensor carrier 预算；
                            # 单位 32-bit words/block，0=不限；不是 VGPR 上限。
    "approx",               # argv[8] math：strict / mmac / mmac-exp2 /
                            # mmac-recip / approx。strict 不启用这些近似变换；
                            # mmac=累加融合，mmac-exp2=再加 native exp2，
                            # mmac-recip=融合+倒数归一化，approx=近似数学组合。
    "256",                  # argv[9] threads：64..1024，须为 64 的倍数。
    "4,1",                  # argv[10] waves：M,N，各 1..16；M*N=threads/64。
                            # 当前 wave64，4,1 表示四个 wave 沿 M 方向分布。
    "1",                    # argv[11] lifetimes：0/1，寄存器/地址存活期优化。
    "1",                    # argv[12] masks：0/1，完整有效 tile 的 mask 快速路径；
                            # 不改变输入的 causal/non-causal 语义。
    "1",                    # argv[13] prefetch：0/1，普通 global packet 寄存器预取；
                            # 不等于 LDS fragment 预取，direct DMA 下可能无匹配。
    "0",                    # argv[14] fuse：0/1，mask/add 融合。
    "0",                    # argv[15] reuse：uint32，LDS fragment 寄存器复用预算，
                            # words/thread；0=关闭，不是 LDS 空间复用开关。
    "useKPack",             # argv[16] dma-pack：legacy / useKPack / noKPack。
    "asm128",               # argv[17] dma-export：split32 / hcu128 / asm128。
                            # asm128=128-bit buffer inline asm；MyTest 不接受 auto。
                            # 当前 gfx936 DTK 不支持 hcu128 对应的宽 HCU intrinsic。
    "32",                   # argv[18] qk-tile：非负 int64，QK 的 D 规约分块；
                            # 0=自动；有效分块为 16 的倍数且整除 D（本例 128）。
    "32",                   # argv[19] vo-tile：同样的范围/约束，控制 V/O 输出列分块；
                            # 不沿输出 D 做规约。本次两个 tile 参数均保持 32。

    # 0/1：提前 MMAC fragment 的 LDS 读取和地址计算。本次保持基线关闭。
    # uint32，跨循环保留不变 operand 的预算，words/thread；0=关闭。
    # 本例 32 保留完整 Q fragments；不是所有情况下都能用满预算。
    "--retain-fragment-words=32",
    # 0/1：循环简化；是否生效还取决于输入标记、trip count 等。
    "--simplify-loop-control=1",
    # 0/1：合并可证明冗余的 wait/barrier。
    "--coalesce-waits=0",
    # off/legal/approx：零 mask 分支跳过加法；approx 要求 math=approx，
    # 允许忽略匹配加零的 signed-zero 差异。
    "--mask-add-bypass=approx",
    # off/legal/approx：符合条件的跨 lane 求和移到循环出口；
    # legal 当前处理 i32，浮点用 approx 且要求 math=approx。
    "--defer-lane-sum=approx",
    # off/legal/approx：线程内长归约拆成多条独立链；approx 要求 math=approx。
    "--local-reduce=approx",
    # 2..16：线程内独立部分累加链数，不是 wave/lane 数。
    "--local-reduce-chains=4",
    # 4..2048 且 >= 2*chains：触发优化的最少输入数。
    "--local-reduce-min-inputs=16",
    # all，或 add,mul,and,or,xor,min,max 的逗号组合，例如 add,max。
    "--local-reduce-ops=all",

    # 三个 swizzle 参数均可取 off/on/源位:目标位；0=off，1/on=8:6。
    # 对 allocation 相对字节地址执行 bit[目标] ^= bit[源]；
    # 两个位不同且均在 4..30，保留完整 16B DMA 包。
    # 启用要求 pack=1、async/hybrid、alignment>=16、export!=split32；
    # 实际布局和地址范围仍须通过编译器校验。
    "--swizzle-q=off",       # Q LDS swizzle。
    "--swizzle-k=8:6",       # K LDS swizzle，保持基线。
    "--swizzle-v=off",       # V LDS swizzle。
    # 以下各取 -1 或非负 int32 ABI 参数编号；-1=使用角色注解自动识别。
    # 不是 block 轴编号；显式绑定必须指向对应 operand。
    "--swizzle-q-arg=-1",
    "--swizzle-k-arg=-1",
    "--swizzle-v-arg=-1",

    # 非负 int64：0=禁用分组重排，正数=每个 head 的 query 分组大小。
    # 本次 4 -> 1；group-size=1 仍启用分组映射，不等于 group-size=0。
    "--workgroup-group-size=4",
    # 0/1/false/true：是否反转逻辑 query 顺序。本次 1 -> 0。
    "--workgroup-reverse=0",
    # 两项均为 x/y/z，必须不同；MyTest 不接受 tuner 的 auto。
    "--workgroup-query-axis=x",  # 输入原始 query tile 的 block 轴。
    "--workgroup-head-axis=y",   # 输入原始 head 的 block 轴。
]

# pipeline-stages、loop-mark、lds-bytes 是 TuneAttention.py 的输入/输出改写项，
# 不是 MyTest argv。本脚本沿用已对齐输入：无 pipeline hint、有 simplify_loop，
# 不做 LDS 容量后处理；不加入 K0 预取等额外实验变换。


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令，不生成文件")
    args = parser.parse_args()
    command_text = shlex.join(["env", *(f"{k}={v}" for k, v in COPY_ENV.items()), *COMMAND])
    print(command_text, flush=True)
    if args.dry_run:
        return
    if not MYTEST.is_file() or not os.access(MYTEST, os.X_OK):
        parser.error(f"MyTest 不存在或不可执行：{MYTEST}")
    if not INPUT.is_file():
        parser.error(f"输入不存在：{INPUT}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 在独立目录运行，保存实际命令及完整日志，便于复现和查看 Pass 输出。
    (OUTPUT_DIR / "command.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd -- {shlex.quote(str(OUTPUT_DIR))}\n{command_text}\n", encoding="utf-8"
    )
    with LOG.open("w", encoding="utf-8") as log:
        result = subprocess.run(COMMAND, cwd=OUTPUT_DIR, env={**os.environ, **COPY_ENV},
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(f"MyTest 失败（退出码 {result.returncode}），详见：{LOG}")
    if not OUTPUT.is_file() or not OUTPUT.stat().st_size:
        raise SystemExit(f"MyTest 未产生有效输出，详见：{LOG}")
    print(f"LLVM IR: {OUTPUT}\n日志: {LOG}")


if __name__ == "__main__":
    main()
