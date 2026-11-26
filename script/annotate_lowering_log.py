#!/usr/bin/env python3
"""Annotate a MyTest lowering log with what each pass changed.

The log must come from MyTest with --dump-llvm-passes=1 so that LLVM stages are
printed too. For each stage, the output contains the pass explanation, a change
count, and the full IR with a comment before every changed region showing what
was removed and what was added. Comments use // for MLIR and ; for LLVM IR.

    python3 script/annotate_lowering_log.py lowering.log -o lowering.annotated.log
"""
import argparse
import difflib
import re
import sys

MARKER = re.compile(r'^-{4,}\s*(.+?)\s*-{4,}\s*$')
SSA_NUMBER = re.compile(r'([%#!])\d+')
MAX_REMOVED_SHOWN = 40
MAX_LINE_WIDTH = 400

# Keyed by a substring of the stage name. First match wins, so specific names
# must precede general ones.
EXPLANATIONS = [
    ('reorder-workgroups', '工作组分组：只改 block_id 到 (query, head) 的映射，让同一 head 的 block 连续运行。group=32 时等价于 q=bx, head=by。'),
    ('NormalizeAttentionKLayout', '规范化 K 的布局：K 通过 transpose 视图访问。'),
    ('promote-gm-to-shm-async', '把 GM 到 LDS 的搬运改成异步：async_copy 返回 task，wait_buffer_ready 消费 task。'),
    ('promote-gemm-matrix-loads', 'V 改用矩阵读指令：一条读取供两条 MMAC 使用。'),
    ('convert-friskbase-to-fragment', '块操作展开成 fragment 循环：gemm 变成 gemm_frag，操作数按 fragment 读取。'),
    ('fuse-fragment-accumulator', '把逐片段 add 融入 MMAC 所在循环，删除额外的整 tile add 循环。'),
    ('frisk-frag-ir-reorder', '融合相邻 fragment 循环；load_frag 提前一片（frisk.pipeline_stage = "prefetch"）。'),
    ('simplify-loop-control', '展开已标记的小循环，例如 4 个 DK 分片变成 4 段直线代码。'),
    ('retain-invariant-fragments', '跨 BN 保留 Q：Q 的 copy、wait、load、缩放全部移到 BN 循环外，结果留在寄存器。'),
    ('fragment-level optimization', 'ScheduleAsyncCopies：提前发 DMA、推迟 wait。AnalyzeShmLiveness 只输出分析结果。'),
    ('convert-fragment-to-thread', 'fragment 语义操作展开为线程级寄存器操作。'),
    ('createFinalizeThreadTilingPass', '完成线程展开：reduce 和 convert_layout 变成 shuffle 通信；O 的布局转换移到 BN 循环外。'),
    ('lower-frisk-fragments', '确定 LDS 物理地址：按生命周期复用，把多个逻辑分配压进 16 KiB；K 的 8:6 swizzle 在此落实。'),
    ('threadIR legalize', '线程级 IR 合法化。'),
    ('createIRDeepOptimizePass', '展开能静态化索引的循环，并在展开后按依赖调度。'),
    ('simplify-bounded-affine', '【未实测】用循环边界消去 floordiv/mod，结果变为循环不变后可被外提。'),
    ('hoist-wave-invariants', '【未实测】把 readfirstlane 链、mbcnt 和 bpermute 索引提到循环外。'),
    ('llvm translateToLLVM', 'MLIR 翻译成 LLVM IR，后续阶段都在此基础上比较。'),
    ('llvm optimizeRegisterMath', '原生 exp2、共享分母的倒数后乘法，以及累加器融合。'),
    ('llvm bypassMaskAdds', '无 mask 路径跳过 S+0。本 case 没有 mask，预期无变化。'),
    ('llvm optimizeLocalReductions', '线程内 max/sum 拆成四条链后树形合并，运算次数不变，依赖深度变浅。'),
    ('llvm deferLaneSums', '分母的跨 lane 求和从每轮一次改到循环结束一次；max 仍每轮通信。'),
    ('llvm prefetchGlobalPackets', '普通 global load 的寄存器预取。direct DMA 路径可能无匹配。'),
    ('llvm optimizeRegisterLifetimes', '缩短地址和 exp2 中间值的寿命，尽早打包。'),
    ('llvm canonicalizeLDSGEPs', 'LDS 地址规范化为公共基址 + 常量偏移，保留合并成 ds_read2_b64 的机会。'),
    ('llvm lowerAsyncDMA', '异步 DMA 降为 buffer_load ... lds 内联汇编；复用已外提的 readfirstlane。'),
    ('llvm fuseMMACAccumulators', 'alpha*O 直接作为 MMAC 的 C 初值，省去临时值和链尾加法。'),
    ('llvm interleavePVAccumulators', 'PV 四链交错：A0 B0 C0 D0 A1 B1 ...，每条链内部顺序不变。'),
    ('llvm doubleBufferV', 'V 双缓冲：V 交替使用两个 8 KiB slot，下一片 DMA 提前，删除 3 个中间覆盖 barrier。'),
    ('llvm narrowKDMA', 'K 的 16 B DMA 拆成两条 8 B DMA，地址增加 bit3 异或 bit9，消除 bank 冲突。'),
    ('llvm prefetchPairedLDSTail', 'QK 尾部读对提前：允许越过局部 max/sum，最多越过 4 条 MMAC。'),
    ('llvm prefetchPairedLDS', 'K 读对整对提前，最多越过 8 条独立 MMAC，并用 sched.barrier(0) 保护。'),
    ('llvm scheduleSoftmaxFragments', 'softmax 按需生成：每个 P fragment 首次用于 PV 前才算，分母加法在输入就绪时推进。'),
    ('llvm prefetchNextK', '跨 BN 预取 K0：最后一片 PV 前发下一轮 K0；分母加法加 identity 防止下沉造成 spill。'),
    ('llvm elideBackedgeBarrier', '【未实测】删除循环尾 reader barrier，下一轮开头的 wait+barrier 已保证同样的顺序。'),
]

# Line-level hints for markers the passes leave on the IR they rewrite.
LINE_HINTS = [
    ('frisk.pv_interleave', 'PV 交错移动过的 MMAC'),
    ('frisk.k_tail_pair', '尾部预取提前的 K 读'),
    ('frisk.k_lds_pair', '读对预取提前的 K 读'),
    ('frisk.cross_bn_sum_anchor', '分母加法锚定，防止下沉'),
    ('frisk.pipeline_stage = "prefetch"', '提前一片的 load'),
    ('s_barrier', '硬件 barrier'),
    ('s_waitcnt', '硬件 wait'),
    ('buffer_load_dwordx2', '8 B DMA'),
    ('buffer_load_dwordx4', '16 B DMA'),
    ('readfirstlane', '读取 wave 的 LDS 基址'),
    ('sched.barrier', '调度边界，运行时无代价'),
]


def explanation(stage):
    for key, text in EXPLANATIONS:
        if key in stage:
            return text
    return '（无预设说明）'


def split_stages(lines):
    """Return [(stage name, IR lines)]. Text before the first marker is dropped."""
    stages = []
    for line in lines:
        match = MARKER.match(line)
        if match:
            stages.append((match.group(1), []))
        elif stages:
            stages[-1][1].append(line)
    return stages


def normalize(line):
    # SSA numbering shifts when anything earlier changes, which is not a real change.
    return SSA_NUMBER.sub(r'\1N', line.strip())


def shorten(line):
    return line if len(line) <= MAX_LINE_WIDTH else line[:MAX_LINE_WIDTH] + ' ...'


def hints_for(lines):
    found = []
    for key, text in LINE_HINTS:
        if any(key in line for line in lines):
            found.append(text)
    return found


def diff_opcodes(before, after):
    """Opcodes over normalized lines, trimming the common prefix and suffix first."""
    a, b = [normalize(x) for x in before], [normalize(x) for x in after]
    start = 0
    while start < len(a) and start < len(b) and a[start] == b[start]:
        start += 1
    end = 0
    while end < len(a) - start and end < len(b) - start and a[-1 - end] == b[-1 - end]:
        end += 1
    matcher = difflib.SequenceMatcher(None, a[start:len(a) - end], b[start:len(b) - end], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != 'equal':
            yield i1 + start, i2 + start, j1 + start, j2 + start


def annotate(name, previous, current, out):
    is_llvm = name.startswith('after llvm')
    comment = ';' if is_llvm else '//'
    print(f'\n{comment} {"=" * 76}', file=out)
    print(f'{comment} 阶段：{name}', file=out)
    print(f'{comment} 作用：{explanation(name)}', file=out)
    if previous is None:
        print(f'{comment} 变化：与上一阶段的 IR 形式不同，不做比较', file=out)
        print(f'{comment} {"=" * 76}', file=out)
        out.writelines(line + '\n' for line in current)
        return
    hunks = list(diff_opcodes(previous, current))
    added = sum(j2 - j1 for _, _, j1, j2 in hunks)
    removed = sum(i2 - i1 for i1, i2, _, _ in hunks)
    print(f'{comment} 变化：{len(hunks)} 处，新增 {added} 行，删除 {removed} 行'
          + ('（本 Pass 在此 case 上没有改动 IR）' if not hunks else ''), file=out)
    print(f'{comment} {"=" * 76}', file=out)
    by_position = {j1: (i1, i2, j2) for i1, i2, j1, j2 in hunks}
    index = 0
    while index <= len(current):
        if index in by_position:
            i1, i2, j2 = by_position[index]
            gone, new = previous[i1:i2], current[index:j2]
            tags = hints_for(gone + new)
            print(f'{comment} >>> [{name}] 此处删除 {len(gone)} 行，新增 {len(new)} 行'
                  + (f'；涉及：{"、".join(tags)}' if tags else ''), file=out)
            for line in gone[:MAX_REMOVED_SHOWN]:
                print(f'{comment} -  {shorten(line.strip())}', file=out)
            if len(gone) > MAX_REMOVED_SHOWN:
                print(f'{comment} -  ...（还有 {len(gone) - MAX_REMOVED_SHOWN} 行）', file=out)
            for line in new:
                print(line, file=out)
            print(f'{comment} <<< 变化结束', file=out)
            if new:
                index = j2
                continue
            # Pure deletion: the unchanged line at this position still has to be printed.
        if index < len(current):
            print(current[index], file=out)
        index += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log')
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    with open(args.log, encoding='utf-8', errors='replace') as f:
        stages = split_stages(f.read().split('\n'))
    if not any(name.startswith('after llvm') for name, _ in stages):
        print('warning: no LLVM stages found; run MyTest with --dump-llvm-passes=1', file=sys.stderr)

    with open(args.output, 'w', encoding='utf-8') as out:
        previous, previous_is_llvm = None, None
        for name, lines in stages:
            if not any(line.strip() for line in lines):
                continue  # A marker with no IR, such as "secondLowering done".
            is_llvm = name.startswith('after llvm')
            comparable = previous if previous_is_llvm == is_llvm else None
            annotate(name, comparable, lines, out)
            previous, previous_is_llvm = lines, is_llvm


if __name__ == '__main__':
    main()
