# =============================================================================
# Attention 优化手段说明（2026-11-06 更新，主线为 non-causal）
# =============================================================================
#
# 这是用 Python 语法写的“说明书”，不是可执行的 kernel。
#
# 怎么读：
#   第 1 部分  术语：先把 wave / lane / tile / fragment / LDS / DMA 对齐
#   第 2 部分  算法本身：不含任何优化，先弄懂要算什么
#   第 3 部分  优化后的完整程序：优化都叠上以后长什么样
#   第 4 部分  Pass 总表：每个 Pass 的顺序、所在阶段、开关
#   第 5 部分  Pass 前后对比：每个 Pass 一组 before / after，这部分最重要
#
# 配置（取自 generated_ir/wave_invariant_noncausal/command.sh）：
#   输入 (B,H,S,D) = (1, 32, 4096, 128)，FP16，无 mask。
#   BM = BN = 128，D = 128，BK = 32；256 线程，每个 workgroup 4 个 wave。
#
# 原有实测数字来自 测试数据分析/1104.md；新增项核对了 pmc_test.log 和实验目录。
# “单步”= 上一版耗时 / 这一版耗时。组合实测不能拆成单个 Pass 的收益。
# 本次新增：第 34 项滞后 softmax 基准、第 35 项转换打包；第 6/13 项补充 causal mask lowering。
# 第 34/35 项默认关闭，表中 8 / rtz 是实验取值，不是编译器默认值。
#
# 证据程度：第 5 部分的 before / after 是对 IR 改动的概括，来自 Pass 说明和源码；
# “为什么快”里写“推断”的部分，没有经过单变量实验证实。
# 注明“概括自 1014.md”的 Pass，没有逐行核对过源码。


# =============================================================================
# 第 1 部分：术语
# =============================================================================
#
# block（workgroup）  一次处理一个 query tile + 一个 head。4096/128 = 32 个 query tile，
#                     32 个 head，所以共 32*32 = 1024 个 block。
#                     硬件编号是 bx、by（各 0~31），线性编号 linear = bx + 32 * by。
#                     编号如何对应到 (query tile, head)，由第 2 项 ReorderWorkgroups 决定。
# wave                一个 block 有 4 个 wave，每个 wave 64 个 lane。
# lane（线程）        4*64 = 256 个线程。
#
# tile                一块矩阵。Q tile = 128行 x 128列（BM x D）。
# 分片 BK             D=128 切成 4 片，每片 32，所以 QK 和 PV 都有 4 个分片。
#                     QK 的分片叫 DK（K 的 D 轴）；PV 的分片叫 OD（输出 D 轴）。
# fragment            一个 tile 中，某个线程在寄存器里持有的一小块。
#                     本例 S 每线程持有 2行 x 8个 1x4 的 FP32 fragment。
#
# GM（global）        显存，慢。
# LDS（shared memory）一个 workgroup 共享的 16 KiB 高速内存。数据从 GM 搬到 LDS，再读进寄存器。
# DMA                 一条异步指令：把 GM 的数据直接写进 LDS。发出去后可以继续计算。
#                     这里是 buffer_load_dwordx4 / dwordx2 ... lds。
#
# MMAC                矩阵乘加指令 v_mmac_f32_16x16x16_f16：C = C + A @ B。
#                     一条指令由整个 wave 协作完成 16x16x16 的计算。
#                     每个 lane 提供 4 个 half 的 A、4 个 half 的 B、4 个 float 的 C。
#
# wait                s_waitcnt vmcnt(0)：等本 wave 发出的 DMA 全部完成。
# barrier             s_barrier：全部 wave 到齐才继续。
#                     数据搬运完成后要 barrier，才能让别的 wave 安全地读。
#                     某块 LDS 要被覆盖前也要 barrier，确保旧数据的读者都读完了。
# sched.barrier(0)    只是告诉编译器调度器别跨过这里，不是真 barrier，运行时没有代价。
#
# 性能的三类损失（后面所有优化都是在减少其中一种）：
#   1. 等数据：MMAC 要用的数据还在路上（GM → LDS 的 DMA，或 LDS → 寄存器的 load）。
#   2. 等同步：wave 之间互相等（barrier）。
#   3. 空转：指令之间有依赖，下一条要等上一条的结果。


# =============================================================================
# 第 2 部分：算法本身（不含任何优化）
# =============================================================================

def attention_textbook(Q, K, V):
    """要算的东西，一共 4 行。"""
    S = Q @ K.T / sqrt(D)                  # 每个 query 对每个 key 的分数
    W = exp(S - S.max(axis=1, keepdims=True))   # 减最大值防止溢出
    P = W / W.sum(axis=1, keepdims=True)   # softmax
    return P @ V


def attention_online_softmax(Q, K, V, query_tile):
    """safe online softmax：key 一块一块（每块 BN 个）处理，不用一次拿到全部 S。

    难点：减的最大值 M 会随着看到新 key 而变大，之前累积的 L、O 是按旧 M 算的，
    所以 M 变大时要乘 alpha 把旧值换算到新基准。
    """
    M = -inf     # 到目前为止每行的最大分数
    L = 0        # 分母：sum(exp(S - M))
    O = 0        # 分子：sum(exp(S - M) @ V)
    for n in range(0, 4096, BN):                  # 每一轮是 1 个 key tile，共 32 轮
        S = Q[query_tile] @ K[n:n + BN].T          # 这一轮的分数（实际 Q 已预乘 log2(e)/sqrt(D)）
        new_M = max(M, S.max(axis=1))
        alpha = exp2(M - new_M)                    # 把旧基准换算到新基准的系数
        W = exp2(S - new_M)                        # FP32，用来算分母
        P = to_fp16(W)                             # FP16，MMAC 的 A 操作数
        L = alpha * L + W.sum(axis=1)
        O = alpha * O + P @ V[n:n + BN]
        M = new_M
    return to_fp16(O / L)


# =============================================================================
# 第 3 部分：原有调度主线（第 1~33 项）的完整程序
# =============================================================================
# 方括号是优化名，对应第 5 部分的同名 Pass。这里关注“一个 block 的执行顺序”。
# 开启第 34/35 项后的 softmax 算法见 lazy_base_after：保留这里的 DMA/LDS 调度，
# 将 QK 初值和 softmax 状态更新换成滞后基准，并可把 P 转换换成 softmax_cvt_pack_after。

def final_program_one_block(bx, by):
    q, head = bx, by                        # [工作组分组] group=32 时恰好就是这样

    # ---- 循环外：Q 只准备一次 --------------------------------------------
    Q_reg = load_and_scale_Q_into_registers()   # [Q 保留] 4 个 DK 分片，每线程 32 words
    slot0, slot1 = LDS[0:8192], LDS[8192:16384]  # [LDS 复用] 全程只用这两块，共 16 KiB
    M, L_local, O = -inf, 0, 0

    k_task = dma_K(n=0, dk=0, dst=slot0)

    for n in range(0, 4096, 128):            # 32 轮，每轮一个 key tile
        # ---- QK：4 个 DK 分片，K 轮流放 slot0 / slot1 ----------------------
        # K分片大小=DK*BN=32*128*f16 = 8KiB
        # V分片大小=OD*D=32*128*f16 = 8KiB
        # Q 总体大小= BM*D = 128*128*f16, 每个线程=128*128/(4*64)=64*f16
        S = 0
        for dk_index in range(4):
            wait_and_barrier(k_task)         # 这一片 K 到了
            if dk_index < 3:
                k_task = dma_K(n, dk_index + 1, dst=slot[(dk_index + 1) % 2])
            else:
                v_task = dma_V(n, od=0, dst=slot0)  # K3 在 slot1，slot0 已空，提前搬 V0
            # 每片 32 条 MMAC；K 读对提前发射 [K 读对预取] [尾部读对预取]
            S = qk_mmac(Q_reg[dk_index], read_K(slot[dk_index % 2]), S)

        # ---- softmax 需要的 max：每轮都要跨 lane 通信 -----------------------
        new_M = max(M, cross_lane_max(local_max4(S)))   # [局部四链归约]
        alpha = exp2(M - new_M)
        M = new_M

        # ---- PV：4 个 OD 分片，V 轮流放 slot0 / slot1 [V 双缓冲] -----------
        for od_index in range(4):
            wait_and_barrier(v_task)
            if od_index < 3:
                v_task = dma_V(n, od_index + 1, dst=slot[(od_index + 1) % 2])
            elif n + 128 < 4096:
                k_task = dma_K(n + 128, 0, dst=slot0)  # [跨 BN K0] 下一轮的 K0 现在就发
            packets = read_V_matrix(slot[od_index % 2])
            # 第一片 PV 的同时才生成 P 和分母更新 [softmax 交错]
            # 4 条累加链交错执行 [PV 四链交错]；alpha*O 直接作为 C 初值 [MMAC 融合]
            O[od_index] = pv_mmac(lazy_P(S, M), packets, seed=alpha * O[od_index])

        # 循环尾原有一个 reader barrier，已删除 [elide barrier，已有组合实测]

    # ---- 循环外：只做一次的收尾 ------------------------------------------
    L = cross_lane_sum(L_local)              # [延迟 lane sum]
    O_store = convert_layout(O)              # [布局转换移出循环]
    store(to_fp16(O_store / L))


# =============================================================================
# 第 4 部分：Pass 总表（实际执行顺序）
# =============================================================================
# 位置依据 3rd/deepgengraph/test/test_src.cpp。
# 字段：序号, Pass, 阶段, 开关（主线/新增实验取值）, 实测单步或验证状态

PIPELINE = [
    # ---- tensor / block IR：算法结构 -------------------------------------
    (1,  "OptimizeTensorLoops",              "tensor",   "carrier=34816",                 "-"),
    (2,  "ReorderWorkgroups",                "block",    "--workgroup-group-size=32",     "1.0645x"),
    (3,  "NormalizeAttentionKLayout",        "block",    "默认",                          "-"),
    (4,  "PromoteGlobalToSharedAsync",       "block",    "FRISK_GM_TO_SHM_MODE=async",    "-"),
    (5,  "PromoteGemmMatrixLoads",           "block",    "默认",                          "-"),
    (6,  "ConvertFriskBaseToFragmentIR",     "转 fragment", "layout=3.3, --swizzle-k=8:6", "-"),
    # ---- fragment IR：计算还是语义操作，物理存储未定 ----------------------
    (7,  "FuseFragmentAccumulator",          "fragment", "默认",                          "-"),
    (8,  "FriskFragIRReorder",               "fragment", "默认",                          "-"),
    (9,  "SimplifyLoopControl",              "fragment", "--simplify-loop-control=1",     "-"),
    (10, "RetainInvariantFragments",         "fragment", "--retain-fragment-words=32",    "causal: 1.08~1.10x/档"),
    (11, "ScheduleAsyncCopies",              "fragment", "默认",                          "-"),
    (12, "AnalyzeShmLiveness",               "fragment", "默认（只分析，不改 IR）",        "-"),
    # ---- thread IR：reduce/布局变换展开为通信，确定物理 LDS ---------------
    (13, "ConvertFragmentToThreadIR",        "thread",   "默认",                          "-"),
    (14, "FinalizeThreadTiling",             "thread",   "默认",                          "causal: 1.1655x（含 25）"),
    (15, "LowerFriskFragments",              "thread",   "默认",                          "-"),
    (16, "IRDeepOptimize",                   "thread",   "reorder-after-unroll=true",     "-"),
    (17, "SimplifyBoundedAffine",            "thread",   "--simplify-bounded-affine=1",   "组合实测，见表后说明"),
    (18, "HoistWaveInvariants",              "LLVM dialect", "--hoist-lds-bases=1 --hoist-lane-shuffles=1", "同上"),
    # ---- LLVM IR：最终调度 ------------------------------------------------
    (19, "optimizeApproximateRegisterMath",  "LLVM",     "math=approx",                   "-"),
    (20, "bypassMaskAdds",                   "LLVM",     "--mask-add-bypass=approx（本 case 无 mask）", "-"),
    (21, "optimizeLocalReductions",          "LLVM",     "--local-reduce=approx",         "causal: 1.0347x（与 20、22 合计）"),
    (22, "deferLaneSums",                    "LLVM",     "--defer-lane-sum=approx",       "同上"),
    (23, "optimizeRegisterLifetimes",        "LLVM",     "默认",                          "-"),
    (24, "canonicalizeLDSGEPs",              "LLVM",     "开启 swizzle 后自动打开",        "causal: 1.0233x"),
    (25, "fuseMMACAccumulators",             "LLVM",     "math!=strict",                  "见 14"),
    (26, "interleavePVAccumulators",         "LLVM",     "--pv-interleave=4",             "1.0043x（接近噪声）"),
    (27, "doubleBufferV",                    "LLVM",     "--v-double-buffer=1",           "1.0306x"),
    (28, "narrowKDMA",                       "LLVM",     "--k-dma=64-swizzle",            "1.0149x"),
    (29, "prefetchPairedLDS",                "LLVM",     "--k-lds-prefetch=8",            "MyTest 路径 1.0272x"),
    (30, "scheduleSoftmaxFragments",         "LLVM",     "--softmax-fragments=pv",        "1.0162x"),
    (31, "prefetchPairedLDSTail",            "LLVM",     "--k-tail-prefetch=4",           "1.0098x"),
    (32, "prefetchNextK",                    "LLVM",     "--cross-bn-k0=1",               "1.0168x"),
    (33, "elideBackedgeBarrier",             "LLVM",     "--elide-backedge-barrier=1",    "组合实测，见表后说明"),
    (34, "lazySoftmaxBase",                  "LLVM",     "--lazy-softmax-base=8",        "实验版 1.0582x；见 34"),
    (35, "softmaxCvtPack",                   "LLVM",     "--softmax-cvt-pack=rtz",       "实验版 1.0439x；Pass 复测见 35"),
]
# 第 34/35 项入口分别对应 LazySoftmaxBasePass / SoftmaxCvtPackPass。
# 新增 causal 分支能力：第 6 项传递 --causal-mask=off|i32|finite（默认 off），
# 第 13 项展开 fragment mask 时落实；直接 thread lowering 路径也支持。详见第 6/13 项对比。
# 它是已有转换 Pass 内的优化，不是排在 LLVM 调度之后的新 Pass；non-causal 主线不使用它。
# 第 17/18/33 项的组合产物 wave_invariant_noncausal/kernel.o3.ll 已记录 1.545679 ms，
# VGPR 228、scratch 0（pmc_test.log）；没有各项单独开关的同版对照，不填写单项加速比。
# 未列入的是纯 lowering 或规范化步骤，不算优化：ConvertKernelOpToFrisk、ConvertMemAndCalcOp、
# ConvertFriskToBase、ThreadLevelIRLegalize、first/secondLowering、lowerWideAsyncDMAToBufferAsm 等。
# 第 1、3、12、16 项和 AutoPipeline（本配置 pipeline=0）在第 5 部分没有单独对比。
# 第 6/13 项只补充新增的 causal mask 能力，不代表这两个转换 Pass 的全部工作。

# 优化方式 ：性能优化Pass 匹配的粒度能更细一点吗？是否有机会做的更加通用（一个Pass=拆成若干个工具组合）
# 每个op的属性？—— elementwise reduce mma （是否有机会从Layout的角度进行？ 某一类Layout 做成opInterface ）
# - 针对每个 interface 的 Op 做特定优化 （elemetnwise的特定优化，reduce类的优化 ）
# - 每类Op有一组优化方案的集合。调优工具能选特定几个
# 切到for循环之后，再加一些标记 
# - review 一下每个Pass的实现（ 每个部分的功能，匹配的Pattern， 变换的方式 ）
# - 检查下优化措施的出现位置 是否适当？是否冗余
# - 是否有重复Pass？（两个层用了同样的优化） 
# - 先让AI分析，然后人工read，是否对的上AI的解释
# 排序一下  IR设计合理性 
# 保留最简单的Pass（必要的）
# 后续 ： 还得考虑 attn 变体 
# 具体方案，分工，时间安排（ 2周内 ）（ 1周测试 ）（ 1周开题报告 ）


# =============================================================================
# 第 5 部分：Pass 前后对比
# =============================================================================
# 每个 Pass 一组 *_before / *_after，末尾注明：
#   省了什么：对应第 1 部分的三类损失
#   限制：什么时候不生效，或者有什么代价


# ---- 2. ReorderWorkgroups：工作组分组 ---------------------------------------
# 实测：L2 命中 74% -> 94%，耗时 1.957 -> 1.839 ms，是 non-causal 收益最大的一项。

def reorder_workgroups_before_group4(linear):
    # 对照配置 group=4：连续 4 个 block 是 head0 的 q0~q3，接着是 head1 的 q0~q3 ...
    # 线性编号 0,1,2,3 -> (q0..q3, head0)；4,5,6,7 -> (q0..q3, head1)；...
    # 32 个 head 都做完 q0~q3，才轮到 q4~q7。
    # 所以一个 head 的 32 个 q 被拆成 8 段，每段只连续 4 个 block。
    start = (linear // (4 * 32)) * 4
    within = linear % (4 * 32)
    return start + within % 4, within // 4          # (q, head)

def reorder_workgroups_after_group32(linear):
    # group=32：连续 32 个 block 是 head0 的全部 q0~q31，再到 head1。
    # 本形状下公式恰好化简为 q = bx, head = by（linear = bx + 32*by）。
    start = (linear // (32 * 32)) * 32              # linear < 1024，所以恒为 0
    within = linear % (32 * 32)                     # 就是 linear
    return start + within % 32, within // 32        # (bx, by)

# group 数 G 的含义：同一个 head 连续处理多少个 query tile。只改执行顺序，不改计算内容。
# reverse：再做一次 q = 31 - q，翻转的是 q，不影响 head 的分组。non-causal 的最佳配置不开。
#
# 省了什么：等数据。每个 block 读的数据量：
#     Q：只读自己的 tile，128 * 128 * 2 B = 32 KiB
#     K、V：读该 head 的全部 token，各 4096 * 128 * 2 B = 1 MiB
#   同一 head 的 block 读同一份 2 MiB 的 K/V，紧接着运行的 block 可以直接命中 L2。
#   关键是时间上相邻，不要求这些 block 同时运行。
#
# 简单模型：一个 head 被拆成 32/G 段，每段连续 G 个 block。段与段之间其他 31 个 head 的
#   约 62 MiB 数据流过 L2，这份 K/V 很可能已被挤出。所以每段只有第一个 block 不命中：
#     L2 命中率约为 1 - 1/G
#     G     模型      实测      耗时
#     4     75.0%     74.07%    1.939 ms
#     8     87.5%     85.37%    1.876 ms
#     16    93.75%    91.22%    1.850 ms
#     32    96.9%     93.71%    1.839 ms
#   这个模型是推断，表中实测值和耗时是实测。它依赖两个未确认的前提：硬件大致按线性编号
#   派发 block，L2 容量远小于全部 K/V（64 MiB）。
#
# 限制：只改编号，不保证硬件严格按编号调度。这个模型不完全适用于 causal：causal 的 query
#   只读部分 K/V、工作量也不同，最优值是 group4，且 L2 命中率更低时反而更快，不能照搬。


# ---- 4. PromoteGlobalToSharedAsync：把搬运变成异步 DMA ------------------------

def promote_async_before():
    for i in range(128):
        LDS[i] = load_global(K, i)          # 同步：线程自己 load 再 store，全程占着指令
    gemm(Q_reg, LDS)

def promote_async_after():
    task = async_copy_gm_to_shm(K, dst=LDS) # DMA 发出去就走，数据在路上时可以做别的
    # ... 这里可以穿插其他计算 ...
    wait_buffer_ready(task)                 # 用之前再等
    gemm(Q_reg, LDS)

# 省了什么：等数据。这是后面所有“提前发 DMA”优化的前提。


# ---- 5. PromoteGemmMatrixLoads：V 用矩阵读指令 --------------------------------

def matrix_load_before():
    # 每个 fragment 分别从 LDS 读一次，每次 8 B
    for f in range(8):
        v_low, v_high = ds_read_b64(addr(f)), ds_read_b64(addr(f) + 8)

def matrix_load_after():
    # 一条指令读 16 B：llvm.hcu.ds.read.m32x16.b16.v4i32，返回的 4 个 i32 = 8 个 half
    # 低 4 个 half 给一条 MMAC，高 4 个 half 给另一条 MMAC
    for p in range(8):
        packet = ds_read_matrix_m32x16(addr(p))
        v_low, v_high = packet[:4], packet[4:]

# 省了什么：一次读取供两条 MMAC 使用，读取次数减半。
# 限制：只有形状和布局能被该指令整除时才选用，否则保留普通 load。


# ---- 7. FuseFragmentAccumulator：把整块 add 融入 fragment 循环 ------------------

def fuse_fragment_acc_before(S, Q, K):
    # 第 1 个循环：算这一片 DK 的 QK 结果
    for f in fragments:
        part[f] = 0
        for kk in range(2):
            part[f] = mmac(Q[f, kk], K[f, kk], part[f])
    # 第 2 个循环：把结果加到 S 上
    for f in fragments:
        S[f] = S[f] + part[f]

def fuse_fragment_acc_after(S, Q, K):
    # 两个循环合成一个，算完一个 fragment 立刻加，不用保存整块 part
    for f in fragments:
        part = 0
        for kk in range(2):
            part = mmac(Q[f, kk], K[f, kk], part)
        S[f] = S[f] + part

# 省了什么：中间结果活得更短，为后面的调度和累加融合创造条件。
# 限制：不把 S 作为 MMAC 的初值，保持精度和操作数顺序。


# ---- 8. FriskFragIRReorder：循环融合 + 下一片提前读 ----------------------------

def frag_reorder_before():
    for i in range(N):
        a[i] = load(i)
    for i in range(N):
        acc = compute(a[i], acc)

def frag_reorder_after():
    # 先融合成一个循环；再让“下一片的 load”排在“当前片的计算”之前（一级 lookahead）
    carried = load(0)
    for i in range(N - 1):
        next_value = load(i + 1)            # load 发出去后，下面的 compute 掩盖它的延迟
        acc = compute(carried, acc)
        carried = next_value
    acc = compute(carried, acc)             # 最后一次，不会越界多读

# 省了什么：等数据。累加顺序不变。
# 限制：循环中有写、barrier、未知副作用时跳过。


# ---- 9. SimplifyLoopControl：展开小循环 ----------------------------------------

def simplify_loop_before():
    for dk in range(0, 128, 32):            # 只循环 4 次
        ...                                 # 同一段循环体

def simplify_loop_after():
    # 展开成 4 段直线代码。Q 保留和“K0、K1、K2、K3 走不同路径”都需要这样才能处理
    ...                                     # dk = 0
    ...                                     # dk = 32
    ...                                     # dk = 64
    ...                                     # dk = 96


# ---- 10. RetainInvariantFragments：跨 BN 保留 Q --------------------------------
# 实测（causal）：每多保留一个 DK 分片少 0.12~0.14 ms，累计约占 causal 总节省的一半。

def retain_q_before():
    for n in range(0, 4096, 128):
        for dk in range(4):
            task = async_copy(Q[dk], dst=lds_q)  # Q 根本没变，却每轮都重新搬、重新读
            wait_buffer_ready(task)
            q = load_frag(lds_q) * scale
            task_k = async_copy(K[n, dk], dst=lds_k)
            wait_buffer_ready(task_k)
            S = mmac(q, load_frag(lds_k), S)

def retain_q_after():
    # Q 的搬运、读取、缩放全部提到 BN 循环外，结果留在寄存器里
    # 真实 IR：06-optimized-fragment.mlir 的 %20、%26、%32、%36
    q_reg = []
    for dk in range(4):
        task = async_copy(Q[dk], dst=lds_q)
        wait_buffer_ready(task)
        q_reg.append(load_frag(lds_q) * scale)
    # 从这里开始 Q 占用的 LDS 可以释放给 K/V
    for n in range(0, 4096, 128):
        for dk in range(4):
            task_k = async_copy(K[n, dk], dst=lds_k)
            wait_buffer_ready(task_k)
            S = mmac(q_reg[dk], load_frag(lds_k), S)

# 省了什么：等数据、等同步。循环里 Q 的 DMA、wait、barrier、LDS 读全部消失。
# 代价：VGPR 228 -> 244（32 words/thread）。这是用寄存器换搬运。


# ---- 11. ScheduleAsyncCopies：提前发 DMA，推迟 wait -----------------------------

def schedule_copies_before():
    task = async_copy(K1, dst=b)
    wait_buffer_ready(task)                 # 发完马上等，异步等于没用
    x = compute_something_unrelated()
    y = use(b)

def schedule_copies_after():
    task = async_copy(K1, dst=b)
    x = compute_something_unrelated()       # 不依赖 b 的计算排到 wait 前面
    wait_buffer_ready(task)                 # 推到第一次真正用 b 之前
    y = use(b)

# 省了什么：等数据。
# 限制：只根据逻辑 LDS 生命周期做保守调度；不把 wait 移过另一条 DMA。


# ---- 14. FinalizeThreadTiling：布局转换移出循环 --------------------------------
# 实测（causal，与第 25 项合计）：1.449 -> 1.243 ms，smem 9.22M -> 5.16M。

def layout_sink_before():
    for n in tiles:
        pv_native = pv_mmac(P, V, c=0)
        pv_store = convert_layout(pv_native)  # 每轮跨 lane 重排（ds.bpermute）
        O_store = alpha * O_store + pv_store
    return O_store

def layout_sink_after():
    for n in tiles:
        pv_native = pv_mmac(P, V, c=0)
        O_native = alpha * O_native + pv_native  # 循环内全程用 MMAC 原生布局
    return convert_layout(O_native)              # 循环结束才转换一次

# 省了什么：跨 lane 通信。ds.bpermute 是 DS 指令，计入 smem；循环内静态调用点 72 -> 8。
# 代码：FriskBaseToTheadlevelIR_plan2.cpp 的 sinkLoopLayoutConversions。


# ---- 15. LowerFriskFragments：确定 LDS 物理地址并复用 --------------------------

def lds_plan_before():
    # 12 个逻辑分配，每个独立的话共需 98304 字节，远超 16 KiB
    q0, q1, q2, q3 = alloc(8192), alloc(8192), alloc(8192), alloc(8192)
    k0, k1, k2, k3 = alloc(8192), alloc(8192), alloc(8192), alloc(8192)
    v0, v1, v2, v3 = alloc(8192), alloc(8192), alloc(8192), alloc(8192)

def lds_plan_after():
    # 根据生命周期判断谁和谁不会同时使用，复用地址。最终共 16384 字节
    slot0, slot1 = LDS[0:8192], LDS[8192:16384]
    k0, k2 = slot0, slot0                         # K 沿 DK 交替，当前片读、下一片写，互不覆盖
    k1, k3 = slot1, slot1
    v0, v1, v2, v3 = slot0, slot0, slot0, slot0   # 此时 V 还是单缓冲，第 27 项才把 V1、V3 换到 slot1
    # Q 在 BN 循环前就读完了（第 10 项），它占用的空间也复用同一个池
    # K 的 8:6 swizzle（第 6 项打的属性）也在这个阶段落实：a ^= ((a >> 8) & 1) << 6


# ---- 17. SimplifyBoundedAffine：用循环范围消掉除法【仅组合实测】---------------

def simplify_affine_before(iv, s0, s1):             # iv 在 [0, 32) 内
    return ((iv + s0 * 4096 + s1 * 131072 + 1024) // 4096) % 32
    # 编译器不知道 iv < 32，这个有符号除法和取模会一路留到底层

def simplify_affine_after(iv, s0, s1):
    # iv + 1024 < 4096，所以除以 4096 后这部分贡献为 0；131072 = 32*4096，mod 32 后消失
    return s0 % 32
    # 结果不依赖 iv，于是可以被 LICM 提到循环外

# 预期（推断）：减少每轮的地址计算。只省地址计算，预计收益有限。
# 限制：线程坐标表达式刻意不折叠，否则 LLVM 阶段的 V/K 地址分析会失败。


# ---- 18. HoistWaveInvariants：循环不变的 wave 级计算外提【仅组合实测】---------

def hoist_wave_before():
    for n in tiles:
        for dma in dmas_in_this_iteration:
            base = readfirstlane(lds_address)        # 每次 DMA 都读一遍，值其实不变
            issue_dma(m0=base)
        lane = mbcnt()                               # 每轮重算 lane 号
        index = lane ^ 16                            # 每轮重算 bpermute 索引
        x = bpermute(index, value)

def hoist_wave_after():
    base = readfirstlane(lds_address)                # 提到入口，相同的合并为一个
    lane = mbcnt()
    index = lane ^ 16
    for n in tiles:
        for dma in dmas_in_this_iteration:
            issue_dma(m0=base)                       # lowering 时直接复用（reuseLeaderRead）
        x = bpermute(index, value)

# 为什么普通 LICM 做不到：readfirstlane 依赖 exec 掩码；而且这里已经是 CFG 形式。
# 这个 Pass 自己证明循环是入口后的两块计数循环、分支 uniform，所以 exec 不变。
# 预期（推断）：每轮少一些 ALU 指令。
# 代价：索引跨循环常驻 VGPR。K0 路径曾因 VGPR 到顶发生 spill，实测先看 scratch。


# ---- 19. optimizeApproximateRegisterMath：数学优化（概括自 1014.md）-------------

def approx_math_before(x, L):
    return libm_exp2(x), a / L, b / L, c / L, d / L

def approx_math_after(x, L):
    inverse = 1.0 / L                                # 共享分母的除法只算一次倒数
    return amdgcn_exp2(x), a * inverse, b * inverse, c * inverse, d * inverse

# 省了什么：空转。原生 exp2 指令比通用实现快；一次倒数代替多次除法。


# ---- 20. bypassMaskAdds：无 mask 路径跳过 S+0（本 case 无 mask，无效果）--------

def mask_bypass_before(S):
    mask = 0 if fully_visible else causal_mask()
    return S + mask                                  # 完全可见时白白加一次 0

def mask_bypass_after(S):
    return S if fully_visible else S + causal_mask()


# ---- 21. optimizeLocalReductions：局部四链归约 ----------------------------------

def local_sum_before(x):                             # x 是本线程的 32 个值
    total = x[0]
    for i in range(1, 32):
        total = total + x[i]                         # 每次加法都要等上一次的结果
    return total                                     # 31 次加法，依赖深度 31

def local_sum_after(x):
    a, b, c, d = x[0], x[1], x[2], x[3]              # 4 条链互不依赖，可以连续发射
    for i in range(4, 32, 4):
        a, b, c, d = a + x[i], b + x[i + 1], c + x[i + 2], d + x[i + 3]
    return (a + b) + (c + d)                         # 仍是 31 次加法，依赖深度 9

# 省了什么：空转。max 同理，依赖深度 32 -> 10。
# 代价：同时存活的值变多；sum 的浮点结合顺序变了（approx 约定）。


# ---- 22. deferLaneSums：分母的跨 lane 求和延后 ----------------------------------

def defer_sum_before():
    for n in tiles:
        # 每轮每行 2 次 shuffle，每线程 2 行，所以每轮 4 次跨 lane 通信
        L = alpha * L + cross_lane_sum(local_sum(W))
    return L

def defer_sum_after():
    for n in tiles:
        L_local = alpha * L_local + local_sum(W)     # 循环内只累加本线程的部分
    return cross_lane_sum(L_local)                   # 循环结束才通信一次

# 为什么合法：同一行各 lane 的 alpha 相同，所以
#   sum(alpha * L_local + s_local) == alpha * sum(L_local) + sum(s_local)
# 省了什么：跨 lane 通信。32 轮时分母通信从 128 次降到 4 次。
# 限制：max 的跨 lane 通信不能延后，alpha 每轮都依赖完整的行最大值。


# ---- 23. optimizeRegisterLifetimes：缩短寄存器生命周期（概括自 1014.md）---------

def lifetimes_before():
    addr = compute_address()                         # 算得早
    x = exp2(y)                                      # 算完很久才用，一直占着寄存器
    many_other_instructions()
    use(addr), use(to_fp16(x))

def lifetimes_after():
    many_other_instructions()
    addr = compute_address()                         # 用之前才算
    p = to_fp16(exp2(y))                             # 尽早打包成 half，占用的寄存器减半
    use(addr), use(p)

# 省了什么：寄存器压力。不 spill 才能让其他优化的提前量有空间。


# ---- 24. canonicalizeLDSGEPs：公共基址 + 常量偏移 -------------------------------

def lds_gep_before(lane):
    # 每个 load 单独算地址；编译器看不出它们相邻，只能生成两条读
    a = load_b64(complicated_address_1(lane))
    b = load_b64(complicated_address_2(lane))
    return a, b

def lds_gep_after(lane):
    base = common_base(lane)
    a = load_b64(base + 0)                           # 差值是常量，且标注了对齐
    b = load_b64(base + 8)                           # 后端可合并成一条 ds_read2_b64
    return a, b

# 省了什么：LDS 读次数。实测对照很清楚：K-swizzle 不做 GEP 时 smem +45%、VGPR 244，
#   做了 GEP 后 smem 和 VGPR 都恢复（L272 -> L214）。


# ---- 25. fuseMMACAccumulators：alpha*O 直接作为 MMAC 初值 -----------------------

def fuse_mmac_before(alpha, O_old):
    c = 0
    for kn in range(8):
        c = mmac(P[kn], V[kn], c)
    return alpha * O_old + c                         # 多一个临时值、多一次向量加法

def fuse_mmac_after(alpha, O_old):
    c = alpha * O_old                                # 从缩放后的旧值开始累加
    for kn in range(8):
        c = mmac(P[kn], V[kn], c)
    return c

# 省了什么：一次加法和一个临时值。需要第 14 项先让 O 和 C 布局一致。
# 限制：改变浮点结合顺序（approx 约定）。


# ---- 26. interleavePVAccumulators：PV 四链交错 ----------------------------------

def pv_interleave_before(a, b, c, d):
    # 一条链做完再做下一条。同链相邻 MMAC 有累加器依赖，下一条要等上一条的 C
    for kn in range(8): a = mmac(Pa[kn], V[kn], a)
    for kn in range(8): b = mmac(Pb[kn], V[kn], b)
    for kn in range(8): c = mmac(Pc[kn], V[kn], c)
    for kn in range(8): d = mmac(Pd[kn], V[kn], d)

def pv_interleave_after(a, b, c, d):
    # 相邻 4 条 MMAC 来自不同的链，互不依赖
    for kn in range(8):
        a = mmac(Pa[kn], V[kn], a)
        b = mmac(Pb[kn], V[kn], b)
        c = mmac(Pc[kn], V[kn], c)
        d = mmac(Pd[kn], V[kn], d)

# 省了什么：空转（推断）。每条链内部顺序不变，数值完全相同。
# 实测：宽度 1/2/4 为 1.830/1.831/1.822 ms，只有 4 变快，且接近噪声。


# ---- 27. doubleBufferV：V 双缓冲 ------------------------------------------------
# 实测：1.822 -> 1.768 ms，是最终调度中收益最大的一项。

def v_double_buffer_before():
    # V0~V3 都放 slot0。下一片要覆盖当前片，所以必须等当前片读完
    for s in range(4):
        wait_and_barrier(v_task)                     # V[s] 到了
        packets = read(slot0)
        mmac_x4(packets)                             # 第 0~3 条
        barrier()                                    # 全部 wave 都读完了，才能覆盖
        if s < 3:
            v_task = dma_V(s + 1, dst=slot0)         # 第 4 条 MMAC 时才发下一片
        mmac_x28(packets)                            # 第 4~31 条

def v_double_buffer_after():
    # V0/V2 用 slot0，V1/V3 用 slot1，下一片不再覆盖当前片
    for s in range(4):
        wait_and_barrier(v_task)                     # 这个 barrier 同时释放另一个 slot
        if s < 3:
            v_task = dma_V(s + 1, dst=slot[(s + 1) % 2])  # 第 0 条 MMAC 之前就发
        packets = read(slot[s % 2])
        mmac_x32(packets)

# 省了什么：
#   等同步：每个 key tile 少 3 个 barrier。
#   等数据：DMA 现在能覆盖当前片的 LDS 读和全部 32 条 MMAC，原来只覆盖 28 条。
#   推断主要来自去 barrier：单缓冲下调整 DMA 时机（after2/4/8）几乎没有收益。
# 限制：不增加 LDS。需要先证明 K 已读完、Q 已不占 LDS，否则跳过。


# ---- 28. narrowKDMA：K 的 64-bit DMA + swizzle ----------------------------------
# 实测：bank 冲突 15.90% -> 0.125%，smem 不变，1.768 -> 1.742 ms。

def k_dma_before():
    # 一个 lane 一次搬 16 B，写到连续地址
    buffer_load_dwordx4(src=k_src(lane), lds=m0 + 16 * lane)

def k_dma_after():
    # 拆成两半，每半 8 B，每半都重新计算该 lane 负责哪段数据
    for h in range(2):
        buffer_load_dwordx2(src=k_src_half(lane, h), lds=m0 + 512 * h + 8 * lane)

def k_address_swizzle(a):
    # 读写都走这个映射。按 1014.md 的模型（32 bank，每 bank 4 B），bank 只由地址的 bit2~6 决定，
    # 所以只有高位不同、低位相同的两个地址会撞 bank。把高位异或到低位，就把它们分开了。
    a = a ^ (((a >> 8) & 1) << 6)                    # --swizzle-k=8:6（第 6、15 项已做）
    a = a ^ (((a >> 9) & 1) << 3)                    # --k-dma=64-swizzle（这里新增）
    return a

# 省了什么：等数据。同时读的 lane 不再撞 bank。
# 为什么要拆 64-bit：16 B 的包内不能做 bit3 的置换，8 B 的包才行。
# 实测对照：只拆不 swizzle，冲突仍是 15.20%（L243），所以消除冲突靠的是 swizzle。
# 限制：Q/V 仍是 128-bit。冲突几乎清零却只提速 1.5%，代价与计数不成比例。


# ---- 29. prefetchPairedLDS：K 读对提前 ------------------------------------------

def paired_prefetch_before():
    unrelated_mmac_1()
    unrelated_mmac_2()
    b0 = load_b64(base + 0)                          # 紧贴着消费者才读
    b1 = load_b64(base + 8)
    mmac(a, b0, c)                                   # 马上要用，只能干等
    mmac(a2, b1, c2)

def paired_prefetch_after():
    b0 = load_b64(base + 0)                          # 整对提前，最多越过 8 条独立 MMAC
    b1 = load_b64(base + 8)                          # 两个 load 不拆开，才能合并成 read2_b64
    sched_barrier_0()                                # 防止调度器把它们移回去
    unrelated_mmac_1()                               # load 在路上时 MMAC 照常执行
    unrelated_mmac_2()
    mmac(a, b0, c)                                   # 到这里数据已经到了
    mmac(a2, b1, c2)

# 省了什么：等数据。MMAC 顺序不变。
# 限制：不跨 DMA、wait、barrier、基本块。距离 8 是上限，不是预取 8 个 tile。


# ---- 30. scheduleSoftmaxFragments：softmax 与 PV 交错 ----------------------------

def softmax_fragments_before():
    # 先把所有 P 和分母一次算完，这段全是 ALU，然后才开始 PV
    for f in all_fragments:
        w = exp2(S[f] - M)
        update_denominator(w)
        P[f] = to_fp16(w)
    for f in all_fragments:
        mmac(P[f], V[f], c)

def softmax_fragments_after():
    # 第一次要用某个 P 时才算它；V1~V3 的 PV 直接复用
    for f in all_fragments:
        w = exp2(S[f] - M)
        update_denominator_if_inputs_ready(w)
        P[f] = to_fp16(w)
        mmac(P[f], V[f], c)

# 省了什么（推断）：把集中的一段 ALU 分散到 MMAC 之间，并和 V1 的在途 DMA 重叠。
# 不是靠省寄存器：VGPR 反而从 220 升到 228。
# 限制：算术表达式和分母加法顺序都不变。


# ---- 31. prefetchPairedLDSTail：QK 尾部的读对提前 --------------------------------

def tail_prefetch_before():
    mmac_1()
    x = local_max(y)                                 # 第 29 项把这种归约当作边界，不跨过去
    z = local_sum(w)
    b0, b1 = load_pair()                             # 这对没被提前
    mmac_2(b0, b1)

def tail_prefetch_after():
    b0, b1 = load_pair()                             # 允许越过局部 max/sum，最多越过 4 条 MMAC
    sched_barrier_0()
    mmac_1()
    x = local_max(y)                                 # 归约自身不动
    z = local_sum(w)
    mmac_2(b0, b1)

# 省了什么：等数据，补上第 29 项够不到的区域。
# 实测：距离 2/4/8 为 1.693/1.691/1.693 ms，差异很小，说明能藏的延迟有限。


# ---- 32. prefetchNextK：跨 BN 预取 K0 + 分母锚定 ----------------------------------

def next_k_before():
    for n in tiles:
        k_task = dma_K(n, 0, dst=slot0)              # 每轮开头才发，只能干等
        wait_and_barrier(k_task)
        qk_and_pv(n)
        reader_barrier()

def next_k_after():
    k_task = dma_K(0, 0, dst=slot0)                  # 第一轮在进入循环前发
    for n in tiles:
        wait_and_barrier(k_task)                     # 数据大概率已经到了
        qk_and_pv_except_last_slice(n)
        wait_and_barrier(v3_task)                    # V3 只读 slot1，slot0 已空
        if n + 128 < 4096:
            k_task = dma_K(n + 128, 0, dst=slot0)    # 现在就发下一轮的 K0
        last_pv_slice(n)                             # 这段计算掩盖 K0 的延迟
        reader_barrier()

def sum_anchor_why():
    # 上面的改写新增了一个 if 分支。LLVM 发现分母加法的结果只在后面用，
    # 就把 64 条加法全沉到分支之后，导致输入一直存活：VGPR 256、scratch 20，
    # 反而变慢（1.752 ms）。所以每条加法后面要加一个 identity 把它钉在原地：
    s = a + b
    s = asm_identity(s)                              # tied-register 空 asm，不生成指令
    return s

# 省了什么：等数据。加锚定后 1.691 -> 1.663 ms。
# 限制：必须和锚定同时使用；CPU 模型记录 k0_overlap_mmac = 992。


# ---- 33. elideBackedgeBarrier：删除循环尾 barrier【仅组合实测】-----------------

def elide_barrier_before():
    for n in tiles:
        wait_and_barrier(k_task)                     # 第二个 barrier
        qk_and_pv(n)
        reader_barrier()                             # 第一个 barrier，之后没有任何 LDS 访问

def elide_barrier_after():
    for n in tiles:
        wait_and_barrier(k_task)                     # 下一轮开头这个 barrier 已能保证同样的顺序
        qk_and_pv(n)
                                                     # 循环尾的 barrier 删除

# 预期（推断）：每轮少一次全 wave 同步。两个 barrier 本来就相邻，主要省 barrier 自身的开销。
# CPU 模型：barrier 次数 293 -> 261（每轮少 1 次，共 32 次），输出与基线逐位相同。
# 限制：必须在第 32 项之后。之前 K0 的 DMA 夹在两个 barrier 之间、会写 V3 正在读的 slot，
#   那时这个 barrier 不能删。负向对照也验证了这点：没有 slot 划分时删除会被检出。
#   现有组合产物已有 GPU 计时；CPU 模型本身不模拟 wave 独立时序，也没有单项性能对照。


# ---- 34. lazySoftmaxBase：滞后更新 softmax 基准 --------------------------------
# 开关：--lazy-softmax-base=off|<T>，默认 off；开启时要求 0 < T <= 15，实验取 T=8。
# 作用：每轮不必都把基准推进到最新最大值。把 -B 放进 QK 的累加器初值，
#   直接算 S-B；仅在首轮或峰值超过阈值时重缩放，减少常走路径上的减法和乘法。

def lazy_base_before():
    M, L_local, O = -inf, 0, 0
    for n in tiles:
        S = qk_mmac_tile(n, seed=0)
        peak = cross_lane_max(local_max4(S))
        new_M = maximum(M, peak)                     # 每行独立更新
        alpha = exp2(M - new_M)
        W = exp2(S - new_M)                          # 每个 score 都要减基准
        L_local = alpha * L_local + local_sum(W)
        O = pv_mmac_tile(to_fp16_rne(W), n, seed=alpha * O)
        M = new_M
    return to_fp16_rne(O / cross_lane_sum(L_local))


def lazy_base_after(T=8, use_cvt_pack=False):
    N, L_local, O = 0, 0, 0                          # N=-B，基准初值 B=0
    for iteration, n in enumerate(tiles):
        shifted = qk_mmac_tile(n, seed=N)            # MMAC 直接得到 QK-B
        peak = cross_lane_max(local_max4(shifted))   # 行最大值仍每轮通信，不能省掉
        if iteration == 0 or wave_any(peak > T):     # ballot：同一 wave 一起走慢路径
            delta = peak if iteration == 0 else maximum(peak, 0)
            shifted = shifted - delta
            alpha = 1 if iteration == 0 else exp2(-delta)
            L_local, O = alpha * L_local, alpha * O
            N = N - delta                           # B 向真实行最大值推进
        # 快路径直接到这里：不减新的基准，不缩放旧 L/O
        W = exp2(shifted)
        P = softmax_cvt_pack_after(W) if use_cvt_pack else to_fp16_rne(W)
        L_local = L_local + local_sum(W)             # 分母继续用 FP32
        O = pv_mmac_tile(P, n, seed=O)
    return to_fp16_rne(O / cross_lane_sum(L_local))

# 上面按整 tile 表示算法；实际 W/P 按四元素 fragment 生成，仍可与 PV 交错。
# 为什么可行：O、L 用同一个基准缩放，最终 O/L 的比例不变；浮点运算顺序改变，
#   不保证逐位相同。快路径 shifted <= T，故 P <= 2^T，T<=15 给 FP16 留出范围。
# 首轮必须进慢路径：若初始分数很负，直接 exp2 可能全下溢；首轮 alpha=1，
#   避免旧状态为 0 时出现 0 * inf。后续由 ballot 带入慢路径的行可取 delta=0。
# 省了什么：常走路径省逐元素减法、旧 O/L 的乘法，以及 alpha 的 exp2 等 ALU 工作。
#   历史配置的快路径移走 64 条 score 减法和 32 条 packed O 乘法；慢路径仍保留它们。
# 代价：新增 ballot、分支和 PHI，触发慢路径过于频繁时可能得不偿失。
# 限制：当前只支持 non-causal、可匹配的 wave64 online-softmax 循环；不支持 causal mask。
#   分支仅 wave-uniform，必须在 prefetchNextK / elideBackedgeBarrier 等依赖
#   workgroup-uniform 证明的优化之后执行；慢路径仅有 ALU，不增加 DMA、wait 或 barrier。
# 历史实验：control 1.6633 -> lazy_t8 1.571839 ms（约 1.0582x）；
#   T=4 为 1.572239 ms，每轮强制慢路径 lazy_always 为 1.691279 ms。
#   这些是文本改写实验版的计时，不冒充当前 MyTest 完整流水线的单项复测。
# 验证记录：noncausal_lazy_base/pass_validation.json 比较 Pass 版与实验版的结构、
#   数值、访存和调度；CPU 模型不模拟目标寄存器分配、MMAC 内部舍入和 GPU 时序。
# 代码：3rd/deepgengraph/lib/Conversion/ConvertToLLVM/LazySoftmaxBase.cpp。


# ---- 35. softmaxCvtPack：FP32 转 FP16 与打包合成一条指令 -----------------------
# 开关：--softmax-cvt-pack=off|rtz，默认 off；rtz 显式允许改变 P 的舍入方式。

def softmax_cvt_pack_before(w):                      # 一个 fragment，4 个 FP32 exp2 结果
    h = [to_fp16_rne(w[i]) for i in range(4)]         # 最近偶数舍入，各自转换
    return as_half4(pack_half2(h[0], h[1]), pack_half2(h[2], h[3]))


def softmax_cvt_pack_after(w):
    lo = v_cvt_pkrtz_f16_f32(w[0], w[1])            # 向零舍入，一条指令转换并打包两个值
    hi = v_cvt_pkrtz_f16_f32(w[2], w[3])
    return as_half4(lo, hi)                          # bitcast 成原 MMAC A 的 <4 x half>

# 省了什么：转换/打包指令。典型每片由 4 次转换 + 2 次打包变成 2 次转换打包，
#   16 个 fragment 的该段预计从 96 条变成 32 条；这不是整个 kernel 的指令数。
# 限制：只匹配带 register_lifetimes 标记和两端 identity 锚点的四元素 exp2 转换，
#   且 half 结果仅用作 MMAC 的 A 操作数；任意 cast、额外 store/use、错误宽度都会跳过。
#   分母仍累计 FP32 W，最终输出转换保留原舍入；RNE -> RTZ 会改变数值，不能声称逐位等价。
#   保留调度锚点和 MMAC 顺序；可独立开启，也能用于匹配的 causal 路径。
#   排在第 34 项之后，但不要求开启它，也不依赖 workgroup-uniform 控制流。
# 实测：实验版 lazy_t8 1.571839 -> lazy_t8_pk_rtz 1.505759 ms（1.0439x，下降约 4.2%）；
#   Pass 固化版 pass_pk_rtz.o3.ll 复测 1.505119 ms，VGPR 228、scratch 0。
#   固化版以原 lazy_t8.ll 为输入；不能把该计时标成当前 MyTest 全流水线重导出的成绩。
#   保留 RNE 的 SDWA 实验版为 1.538879 ms；日志注明 PK_RNE 指令不支持，均非此 Pass 模式。
# 代码：3rd/deepgengraph/lib/Conversion/ConvertToLLVM/SoftmaxCvtPack.cpp。
# 验证：3rd/deepgengraph/test/check_softmax_cvt_pack.py；
#   generated_ir/noncausal_cvt_pack_pass/{validation.json,integration_validation.json}。


# ---- 6/13 补充. causal mask lowering：窄化比较并融合 mask 加法 -----------------
# 这是已有 ConvertFriskBaseToFragmentIR / ConvertFragmentToThreadIR 的新增能力。
# 开关：--causal-mask=off|i32|finite，默认 off；以下是 causal 分支，不用于上面的无 mask 主线。

def causal_mask_before(score, row, col, diagonal=1):
    masked = row + diagonal <= col                  # 通用 index / 宽整数比较
    mask = -inf if masked else 0.0
    return score + mask


def causal_mask_i32_after(score, row, col, diagonal=1):
    # 编译期已证明 row、col、row+diagonal 在非负 signed-i32 范围内
    masked = i32(row) + i32(diagonal) <= i32(col)
    return score + (-inf if masked else 0.0)         # 保留原 mask 和浮点加法语义


def causal_mask_finite_after(score, row, col, diagonal=1):
    masked = i32(row) + i32(diagonal) <= i32(col)
    return -inf if masked else score                # 直接选结果，省去 score+mask

# 省了什么：范围证明允许用 i32 比较，去掉通用比较路径的额外计算/锚点；
#   finite 进一步消掉本例 16 个 vec4 mask add，降低 ALU 工作和寄存器压力。
# 限制：范围未知、负坐标或可能溢出时回退；finite 还要求 -inf/+0 mask、
#   单个紧邻 add 消费者，并显式假定 pre-mask score 有限、允许 signed-zero 差异。
#   NaN/+inf 的传播不在 finite 的等价约定内；i32 模式没有这个浮点假设。
# 配套：K-tail 预取允许跨过纯寄存器 icmp/select，DMA/wait/barrier 仍是边界。
# 实测：同一 causal 配置的 off+PKRTZ 1.787039 ms、scratch 100，
#   finite+PKRTZ 1.199839 ms、scratch 0，二者 VGPR 均为 256（pmc_test.log）。
#   包含 mask lowering 与配套预取变化，不能把总收益全部归给少做一次加法。
# 代码：FriskBaseToTheadlevelIR_plan2.cpp / CausalMaskLowering.h（FriskToBase 目录）。
# 验证记录：generated_ir/causal_mask_lowering/README.txt 与 validation.json；
#   27 次 LLVM CPU 执行通过，测试中的 i32/finite 与 off 输出逐位相同，不等于任意输入等价。


# =============================================================================
# 附：MMAC 16x16x16 的坐标（lane 号 l，寄存器号 r，详见 1014.md 第 4 节）
# =============================================================================
#   A 的第 r 个 half：A[l % 16, 4 * (l // 16) + r]
#   B 的第 r 个 half：B[4 * (l // 16) + r, l % 16]
#   C 的第 r 个 float：C[l % 16, l // 16 + 4 * r]
#
# 相关说明：
#   3rd/deepgengraph/lib/Conversion/ConvertToLLVM/{PVInterleave,VDoubleBuffer,KDMA,PairedLDS}.md
#   3rd/deepgengraph/lib/Conversion/FriskToBase/FriskFragmentIR.md
#   SafeSoftmax最后两项优化说明.md
#   测试数据分析/1014.md（布局）、测试数据分析/1104.md（收益）
#   generated_ir/noncausal_lazy_base/README.txt（第 34 项）
#   generated_ir/noncausal_cvt_pack_pass/README.txt（第 35 项，GPU 状态以最新 pmc_test.log 为准）
#   generated_ir/causal_mask_lowering/README.txt（第 6/13 项新增能力）
