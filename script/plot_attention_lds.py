#!/usr/bin/env python3
"""Plot the concrete k5 hybrid DMA-guard attention layout (not a generic GPU layout).

Run from any directory:
  MPLCONFIGDIR=/tmp/mpl-attention python3 script/plot_attention_lds.py

Requires numpy and matplotlib. Uses this repository's symbolic LLVM address
tracer; checks all 256 copy lanes against the synchronous k5 export and checks
the layout formulas before plotting. No GPU execution or timing simulation.
Figure text is English so the script works without a Chinese font installation.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-attention")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "3rd/deepgengraph/test"))
from check_direct_shared_copy import Trace
from check_gemm_pv_addresses import split_args

COLORS = ["#247BA0", "#E6A23C", "#8B62AD", "#32A887"]
INK = "#203047"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "text.color": INK, "axes.labelcolor": INK,
                     "savefig.facecolor": "white", "pdf.fonttype": 42})


class AddressTrace(Trace):
    def value(self, name):
        if name not in self.cache and name in self.definitions:
            expr = self.definitions[name]
            if expr.startswith("ptrtoint "):
                p = self.typed(expr.split(" to ")[0][9:])
                self.cache[name] = p[1] if p[0] == "@shm_0" else p
            elif expr.startswith("inttoptr "):
                p = self.typed(expr.split(" to ")[0][9:])
                self.cache[name] = p if isinstance(p, tuple) else ("@shm_0", p)
            elif "@llvm.amdgcn.readfirstlane.i32(" in expr:
                n = re.search(r"\(i32 (%[\w.]+)\)", expr)[1]
                first = self.peers[self.thread // 64 * 64].value(n)
                assert self.value(n) == first, "Nonuniform DMA destination base"
                self.cache[name] = first
            elif expr.startswith("fmul "):
                # Q scaling changes the value, but preserves its LDS provenance.
                self.cache[name] = self.typed(split_args(expr[5:])[0])
        return super().value(name)


def traces(text):
    defs = {n: re.sub(r", ![\w.]+ !\d+", "", e)
            for n, e in re.findall(r"^  (%[\w.]+) = (.*)$", text, re.M)}
    tid = next(n for n, e in defs.items() if "@llvm.amdgcn.workitem.id.x()" in e)
    context = {f"%{i}": (f"%{i}", 0) for i in range(4)}
    context.update({n: 0 for n, e in defs.items()
                    if "@llvm.amdgcn.workgroup.id." in e or e.startswith("phi i64 ")})
    peers = []
    peers.extend(AddressTrace(defs, {**context, tid: i}, peers) for i in range(256))
    return defs, peers


def copy_map(text, peers):
    result = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        store = re.search(r"^  store <(\d+) x half> (%[\w.]+), ptr addrspace\(3\) (%[\w.]+)", line)
        dma = re.search(r"call void @llvm.amdgcn.global.load.async.lds\((.*)\)", line)
        if not (store or dma):
            continue
        for lane, t in enumerate(peers):
            if store:
                data = t.value(store[2])
                base, offset = t.value(store[3])
                mode = "VGPR"
            else:
                src, dst, size, imm, aux = split_args(dma[1])
                assert (size, imm, aux) == ("i32 4", "i32 0", "i32 0")
                g, start = t.typed(src)
                data = [(g, start), (g, start + 2)]
                base, offset = t.typed(dst)
                offset += lane % 64 * 4
                mode = "DMA"
            assert base == "@shm_0"
            for j, key in enumerate(data):
                assert key[0] in ("%0", "%1", "%2") and key not in result
                result[key] = {"lds": offset + 2 * j, "copy_thread": lane,
                               "ir_line": lineno, "mode": mode}
    assert len(result) == 24576
    return result


def q_addr(m, d):
    return 1024 * (d // 4) + 8 * m + 2 * (d % 4)


def k_addr(n, d):
    g = d % 16 // 4
    cg = n // 8
    np_ = cg * 8 + ((n // 4 + cg // 4 + g // 2) % 2) * 4 + (n + cg) % 4
    return 8192 + 2 * ((g * 128 + np_) * 8 + (d // 16) * 4 + d % 4)


def v_addr(n, c):
    g, cg = n // 8, c // 8
    cp = cg * 8 + ((c // 4 + cg // 4 + g) % 2) * 4 + (c + cg) % 4
    return 2 * ((g * 32 + cp) * 8 + (n // 4 % 2) * 4 + n % 4)


def validate(current, reference):
    assert {k: v["lds"] for k, v in current.items()} == {k: v["lds"] for k, v in reference.items()}
    inverses = {name: {} for name in "QKV"}
    for (arg, gm), data in current.items():
        name = {"%0": "Q", "%1": "K", "%2": "V"}[arg]
        row, col = divmod(gm // 2, 128)
        expected = {"Q": q_addr, "K": k_addr, "V": v_addr}[name](row, col % 32)
        assert expected == data["lds"], (name, row, col, expected, data)
        # V has four separate output-column stages which reuse the same space.
        if name != "V" or col < 32:
            assert data["lds"] not in inverses[name]
            inverses[name][data["lds"]] = (row, col)
    assert all(len(v) == 4096 for v in inverses.values())
    return inverses


def mmac_operands(defs, peers, inv):
    mma = [e for e in defs.values() if "asm sideeffect" in e and "v_mmac_" in e]
    assert len(mma) == 160
    rows = []
    for index, tensor, operand in [(0, "Q", 0), (0, "K", 1), (1, "Q", 0),
                                   (1, "K", 1), (32, "V", 1), (33, "V", 1)]:
        args = split_args(re.search(r'"\((.*)\) #', mma[index])[1])
        for lane in range(64):
            values = peers[lane].typed(args[operand])
            assert len(values) == 4
            for reg, (base, address) in enumerate(values):
                assert base == "@shm_0"
                row, col = inv[tensor][address]
                rows.append(dict(mmac_index=index, tensor=tensor, lane=lane,
                                 element=reg, row=row, col=col, lds=address))
    return rows


def page(title, subtitle):
    fig = plt.figure(figsize=(16, 10), facecolor="white")
    fig.text(.045, .954, title, fontsize=23, weight="bold")
    fig.text(.045, .918, subtitle, fontsize=11, color="#58677D")
    fig.text(.045, .025, "Actual export: k5_hybrid_dma_guard | FP16 | 256 threads / 4 waves | Addresses in bytes | Not a timing simulation",
             fontsize=9, color="#69778A")
    return fig


def canvas(fig, rect=(.04, .08, .92, .8)):
    ax = fig.add_axes(rect)
    ax.set(xlim=(0, 100), ylim=(0, 100))
    ax.axis("off")
    return ax


def box(ax, x, y, w, h, label, color, size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1",
                              linewidth=1.2, edgecolor=color, facecolor=color + "16"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size, color=INK)


def arrow(ax, x1, y1, x2, y2, color=INK):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6))


def overview():
    fig = page("01  Global -> LDS -> lane registers -> MMAC", "Read left to right. Q/K and V use the same LDS bytes at different times.")
    ax = canvas(fig)
    ax.text(2, 95, "GLOBAL (BHSD)", weight="bold", fontsize=13)
    ax.text(30, 95, "SHARED MEMORY / LDS", weight="bold", fontsize=13)
    ax.text(67, 95, "COMPUTE", weight="bold", fontsize=13)
    box(ax, 1, 73, 18, 15, "Q [128,32]\nD is contiguous", COLORS[0])
    box(ax, 1, 52, 18, 15, "K [128,32]\nD is contiguous", COLORS[1])
    box(ax, 30, 73, 27, 15, "Q: 0..8191\n4-half column packets", COLORS[0])
    box(ax, 30, 52, 27, 15, "K: 8192..16383\npaired D fragments + swizzle", COLORS[1])
    for y in (80, 59):
        arrow(ax, 19.5, y, 29, y)
        ax.text(24.5, y + 3, "DMA", ha="center", fontsize=10)
    box(ax, 67, 60, 31, 22, "LDS reads -> VGPR operands\nQ scaled in registers\nQ x K^T -> score\nMMAC 16 x 16 x 16", COLORS[2])
    arrow(ax, 58, 80, 66, 76); arrow(ax, 58, 59, 66, 66)
    box(ax, 67, 40, 31, 12, "mask / exp2 / reduction\nP stays in registers", COLORS[2])
    arrow(ax, 82, 59, 82, 53)
    ax.text(2, 44, "AFTER QK: reuse LDS (synchronized)", fontsize=12, weight="bold")
    box(ax, 1, 17, 18, 17, "V [128,32]\noutput D contiguous", COLORS[3])
    box(ax, 30, 17, 27, 17, "V: 0..8191\nreuses Q storage\npaired token fragments", COLORS[3])
    arrow(ax, 19.5, 25, 29, 25)
    ax.text(24.5, 30, "load", ha="center", fontsize=9)
    ax.text(24.5, 20, "VGPR\npermute/store", ha="center", fontsize=8)
    box(ax, 67, 17, 31, 17, "LDS V -> VGPR operand B\nregister P -> operand A\nP x V -> output accumulator", COLORS[3])
    arrow(ax, 58, 25, 66, 25); arrow(ax, 82, 39, 82, 35)
    ax.text(2, 5, "A wave cooperates on each MMAC; each lane supplies 4 half values per A/B operand.\n"
                    "The lane that copied a value need not be the lane that later consumes it.", fontsize=12)
    return fig


def q_copy(current, ref):
    fig = page("02  Q: contiguous global load, separated LDS stores", "Example: m=0, d=0..15. Same values and final LDS addresses; different copy-lane ownership.")
    ax = canvas(fig)
    ax.text(1, 95, "GLOBAL: Q[m,d], byte offset = 256*m + 2*d", weight="bold", fontsize=13)
    for g in range(4):
        box(ax, 2 + 24 * g, 78, 21, 12, f"Q[0,{4*g}:{4*g+4}]\nGM {8*g}..{8*g+7}", COLORS[g])
    ax.text(1, 69, "k5 / VGPR: copy thread 0 loads two 16-byte vectors", weight="bold", fontsize=13,
            bbox=dict(facecolor="white", edgecolor="none", pad=3), zorder=5)
    box(ax, 2, 52, 45, 12, "load Q[0,0:8] -> registers", COLORS[0])
    box(ax, 50, 52, 45, 12, "load Q[0,8:16] -> registers", COLORS[2])
    for g in range(4):
        arrow(ax, 12.5 + 24*g, 77, 12.5 + 24*g, 65, COLORS[g])
        arrow(ax, 12.5 + 24*g, 51, 12.5 + 24*g, 40, COLORS[g])
        assert ref[("%0", 8*g)]["copy_thread"] == 0
        box(ax, 2 + 24*g, 25, 21, 14, f"Q[0,{4*g}:{4*g+4}]\nLDS {1024*g}..{1024*g+7}\n8-byte store", COLORS[g])
    ax.text(1, 43, "Take register slices; Q needs no cross-lane shuffle here", fontsize=11,
            bbox=dict(facecolor="white", edgecolor="none", pad=3), zorder=5)
    ax.text(1, 18, "dma_guard: each 8-byte packet is written by TWO DMA lanes", weight="bold", fontsize=13)
    for g in range(4):
        a, b = current[("%0", 8*g)], current[("%0", 8*g+4)]
        ax.text(12.5+24*g, 7, f"DMA round {g}, m0={1024*g}\nthreads {a['copy_thread']}, {b['copy_thread']}\n4 bytes each; no data VGPR", ha="center", fontsize=10)
    ax.text(1, 1, "Q LDS address = 1024*(d//4) + 8*m + 2*(d%4)   |   Physical order: [d//4][m][d%4]", fontsize=12)
    return fig


def packet_panel(ax, tensor, inv, title, start, rows, step, color_axis):
    ax.axis("off")
    ax.set(xlim=(-.9, 8), ylim=(-1.15, rows + 1.5))
    ax.text(-.85, rows + 1, title, fontsize=13, weight="bold")
    for r in range(rows):
        addr = start + step*r
        y = rows - r - 1
        ax.text(-.12, y + .5, str(addr), ha="right", va="center", fontsize=9)
        for j in range(8):
            coord = inv[tensor][addr + 2*j]
            g = coord[1] // 16 if tensor == "K" else (coord[color_axis] // 4) % 4
            ax.add_patch(Rectangle((j, y), 1, 1, facecolor=COLORS[g]+"42", edgecolor="white", lw=1.2))
            ax.text(j+.5, y+.5, f"{coord[0]},{coord[1]}", ha="center", va="center", fontsize=8)
    ax.text(0, -.75, "Cell = half (2 B); row = 16 consecutive LDS bytes", fontsize=9)


def packed(inv):
    fig = page("03  Q / K / V: what is physically adjacent in LDS?", "Labels are logical coordinates, NOT lane IDs. Packet samples below include the column swizzles from the actual IR.")
    ax = canvas(fig, (.04, .69, .92, .18))
    box(ax, 1, 5, 30, 86, "Q[m,d]\n[4 consecutive d] per row\nQ[0,0:4], Q[1,0:4], ...\nnext d-group starts at +1024 B", COLORS[0], 12)
    box(ax, 35, 5, 30, 86, "K[n,d] (used as K^T)\n[4 d] + [4 d at +16]\n8 half per physical packet\npermuted token-column order", COLORS[1], 12)
    box(ax, 69, 5, 30, 86, "V[n,c]\n[4 tokens] + [next 4 tokens]\n8 half at one output column\npermuted output-column order", COLORS[3], 12)
    packet_panel(fig.add_axes((.06, .36, .41, .29)), "K", inv, "K packets: labels (token n, D d)", 8192, 12, 16, 1)
    packet_panel(fig.add_axes((.55, .36, .41, .29)), "V", inv, "V packets: labels (token n, output c)", 0, 12, 16, 0)
    ax = canvas(fig, (.04, .075, .92, .25))
    ax.text(1, 89, "Why V uses a register permutation", fontsize=14, weight="bold")
    box(ax, 1, 26, 30, 48, "GLOBAL rows (contiguous D)\nV[0,0:4], V[1,0:4],\nV[2,0:4], V[3,0:4]", COLORS[3], 11)
    box(ax, 38, 26, 23, 48, "lane-local registers\nselect / permute\ncolumns out of rows", COLORS[2], 11)
    box(ax, 68, 26, 31, 48, "LDS 8-byte stores\nV[0:4,0], V[0:4,1],\nV[0:4,2], V[0:4,3]", COLORS[3], 11)
    arrow(ax, 32, 50, 37, 50); arrow(ax, 62, 50, 67, 50)
    ax.text(1, 5, "The consumer reads paired 16-byte K/V packets, then selects a 4-half operand for each MMAC.\n"
                    "K pairs D fragments 16 apart; V pairs token fragments 4 apart. Global order and LDS order differ.", fontsize=11)
    return fig


def lane_grid(ax, rows, name, index):
    selected = [x for x in rows if x["tensor"] == name and x["mmac_index"] == index]
    rr = sorted({x["row"] for x in selected}); cc = sorted({x["col"] for x in selected})
    assert len(rr) == len(cc) == 16 and len(selected) == 256
    arr = np.full((16, 16), -1)
    for x in selected:
        r, c = rr.index(x["row"]), cc.index(x["col"])
        assert arr[r,c] == -1
        arr[r,c] = x["lane"]
    ax.imshow(arr//16, cmap=matplotlib.colors.ListedColormap(COLORS), vmin=0, vmax=3, alpha=.42, aspect="equal")
    for r in range(16):
        for c in range(16):
            ax.text(c, r, str(arr[r,c]), ha="center", va="center", fontsize=6.2)
    ax.set_xticks(range(16), cc, rotation=90, fontsize=8)
    ax.set_yticks(range(16), rr, fontsize=8)
    ax.set_xticks(np.arange(-.5,16,1), minor=True); ax.set_yticks(np.arange(-.5,16,1), minor=True)
    ax.grid(which="minor", color="white", lw=.6); ax.tick_params(which="minor", length=0)
    labels = {"Q": ("D coordinate d", "query row m"), "K": ("D coordinate d", "key token n"), "V": ("output column c", "key token n")}
    ax.set_xlabel(labels[name][0], fontsize=10); ax.set_ylabel(labels[name][1], fontsize=10)
    ax.set_title(f"{name}: {'first QK' if index==0 else 'first PV'} MMAC\ncell number = consuming lane", fontsize=12, pad=12)


def consumers(rows):
    fig = page("04  Which lane supplies each MMAC operand?", "Wave 0, first QK and PV instructions, traced from LLVM operands. Four half values per lane; 64 lanes together supply a 16 x 16 tile.")
    for i,(name,index) in enumerate([("Q",0),("K",0),("V",32)]):
        lane_grid(fig.add_axes((.065+i*.317, .41, .265, .42)), rows, name, index)
    ax = canvas(fig, (.04, .065, .92, .27))
    ax.text(1, 96, "Follow lane 0 through two consecutive instructions", weight="bold", fontsize=14)
    data=[]
    for name, index in [("Q",0),("K",0),("Q",1),("K",1),("V",32),("V",33)]:
        vals=[r for r in rows if r['tensor']==name and r['mmac_index']==index and r['lane']==0]
        coords=" ".join(f"({r['row']},{r['col']})" for r in vals)
        data.append([f"{'QK' if index<32 else 'PV'} #{index if index<32 else index-32}", name, coords,
                     f"{vals[0]['lds']}..{vals[-1]['lds']+1}"])
    tab=ax.table(cellText=data, colLabels=["Instruction", "Operand", "Lane 0 logical coordinates", "LDS bytes"],
                 colWidths=[.14,.12,.5,.24], bbox=[.01,.23,.98,.63], cellLoc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(10)
    for (r,c),cell in tab.get_celld().items():
        cell.set_edgecolor('white');cell.set_facecolor('#E9EEF5' if r==0 else '#F5F7FA')
    ax.text(1, 5, "K's selected token columns are permuted by the producer layout: read the axis coordinates, not just the grid position.\n"
                    "P (PV operand A) comes from registers. MMAC consumes VGPR operands; it does not read LDS directly.", fontsize=11)
    return fig


GUIDE = """当前 k5_hybrid_dma_guard 的 Q/K/V 布局图

阅读顺序：
01_overview.png：整体链路。Q/K 用 DMA，V 经 VGPR 重排写入。前 8 KiB 先存 Q，后复用为 V；P 保持在寄存器。
02_q_copy.png：同一行 Q 的 16 个 half 如何从连续 GM 地址分成四组，写到间隔 1024 字节的 LDS 地址。
  上半部分是旧 k5 的 VGPR 路径，下半部分是当前 DMA 的搬运线程。thread 是 block 内编号，lane=thread%64。
03_packed_qkv.png：K/V 每个连续 16 字节包中的实际逻辑坐标，以及 V 从“行”到“列”的寄存器重排。
04_mmac_lanes.png：格子内数字为计算时的 lane ID，坐标轴是真实逻辑坐标。颜色按 lane//16 分组。
  三张矩阵均描述某一条指令的输入集合。K 的 token 坐标不连续，是当前 producer layout 的列置换，不是绘图省略。
  下方表格对比同一 lane 在前两条指令使用的坐标和地址，解释为何 K/V 要做配对存储。

图中约定：Q[m,d]、K[n,d]、V[n,c]；0:4 表示 0,1,2,3。地址单位均为字节；每个 half 占 2 B。
选择 head=0、query block=0、key tile=0、QK D chunk=0、V output chunk=0。GM 公共基址视为 0。
GM 的 Q/K/V 使用当前运行器的 BHSD ABI，每行 D=128，因此下一行相差 256 字节。
图中的寄存器只描述元素分组，不声称某个具体 vN 寄存器编号。
指令实际为 v_mmac_f32_16x16x16_f16；图中 MMAC 对应提问中的 WMMA。
Q 的 scaling 保留地址来源，但会改变数值。所有图展示地址/数据来源，不展示性能或执行时序。

验证：
从当前 .ll 提取 256 线程的 24576 个 global->LDS half 元素映射，与旧同步 k5 对照。
Q/K 各 4096，四个 V 输出段各 4096；V 输出段依次复用相同地址，不能将四段视为同时驻留。
再逐元素验证 Q/K/V 公式。提取首两条 QK/PV 指令的 Q/K/V 操作数（wave 0）。
未模拟真实 DMA 时序、MMAC hazards 或硬件性能。

精确字节地址公式（tile 内坐标；// 为整除，% 为非负取模）：
Q: 1024*(d//4) + 8*m + 2*(d%4)
K: g=(d%16)//4; cg=n//8
   np=cg*8 + ((n//4 + cg//4 + g//2)%2)*4 + (n+cg)%4
   addr=8192 + 2*((g*128+np)*8 + (d//16)*4 + d%4)
V: g=n//8; cg=c//8
   cp=cg*8 + ((c//4 + cg//4 + g)%2)*4 + (c+cg)%4
   addr=2*((g*32+cp)*8 + ((n//4)%2)*4 + n%4)

复现：python3 script/plot_attention_lds.py
依赖：numpy、matplotlib，以及仓库 3rd/deepgengraph/test 中的符号地址解析器。
可选参数：--ir、--reference、--output-dir。脚本专用于当前 128/128/32、256 线程布局；不匹配时验证报错。
PDF 为四页矢量输出；PNG 用于直接查看；CSV 可用于搜索具体逻辑坐标/线程。
"""


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ir', type=Path, default=ROOT/'generated_ir/test_input_spidD_best_k5_hybrid_dma_guard.ll')
    parser.add_argument('--reference', type=Path, default=ROOT/'generated_ir/test_input_spidD_best_k5.ll')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'generated_ir/layout_visualization')
    args=parser.parse_args()
    text=args.ir.read_text(); reftext=args.reference.read_text()
    defs, peers=traces(text); _, refpeers=traces(reftext)
    current=copy_map(text, peers); reference=copy_map(reftext, refpeers)
    inv=validate(current, reference); operands=mmac_operands(defs, peers, inv)
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True)
    with (out/'copy_mapping.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['tensor','row','global_D','global_byte','lds_byte','current_copy_thread','reference_copy_thread','mode','ir_line'])
        for (arg,gm), item in sorted(current.items()):
            r,c=divmod(gm//2,128)
            writer.writerow([{'%0':'Q','%1':'K','%2':'V'}[arg],r,c,gm,item['lds'],item['copy_thread'],reference[(arg,gm)]['copy_thread'],item['mode'],item['ir_line']])
    with (out/'mmac_operands.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(operands[0]));writer.writeheader();writer.writerows(operands)
    pages=[('01_overview',overview),('02_q_copy',lambda:q_copy(current,reference)),
           ('03_packed_qkv',lambda:packed(inv)),('04_mmac_lanes',lambda:consumers(operands))]
    with PdfPages(out/'attention_lds_explained.pdf') as pdf:
        for name, draw in pages:
            fig=draw();fig.savefig(out/(name+'.png'),dpi=170);pdf.savefig(fig);plt.close(fig)
    (out/'阅读说明.txt').write_text(GUIDE,encoding='utf-8')
    report={'ir':str(args.ir.resolve()),'ir_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'reference':str(args.reference.resolve()),'reference_sha256':hashlib.sha256(reftext.encode()).hexdigest(),
            'copy_elements_checked':len(current),'all_copy_destinations_match_reference':True,
            'Q_K_V_formulas_checked':True,'mmac_operand_elements_traced':len(operands),
            'context':{'head':0,'query_block':0,'key_tile':0,'QK_D_chunk':0,'wave':0},
            'gpu_executed':False}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));print('Figures written to',out)


if __name__=='__main__':
    main()
