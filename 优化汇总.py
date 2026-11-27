# =============================================================================
# Attention 优化手段说明（2026-11-27 更新，主线为 non-causal，另列 causal 扩展）
# Attention 优化手段说明（2026-11-26 更新，主线为 non-causal，另列 causal 扩展）
# =============================================================================
#
# 这是用 Python 语法写的“说明书”，不是可执行的 kernel。
#
# 怎么读：
#   第 1 部分  术语：先把 wave / lane / tile / fragment / LDS / DMA 对齐
#   第 2 部分  算法本身：不含任何优化，先弄懂要算什么
#   第 3 部分  优化后的完整程序：优化都叠上以后长什么样
#   第 4 部分  优化步骤总表：实际顺序、当前归属、开关和迁移关系
#   第 5 部分  Pass 前后对比：每个 Pass 一组 before / after，这部分最重要
#   第 6 部分  近期验证记录：区分结构检查、CPU 模型与 GPU 实测
#
# 说明用配置（沿用 generated_ir/wave_invariant_noncausal/command.sh）：
#   输入 (B,H,S,D) = (1, 32, 4096, 128)，FP16，无 mask。
#   BM = BN = 128，D = 128，BK = 32；256 线程，每个 workgroup 4 个 wave。
#
# 原优化记录核对至 de99ebb7；本次按当前工作区源码补齐 Frisk 全流程和调度分支；
# 入口为 3rd/deepgengraph/test/test_src.cpp，后端同步清理的调用位置见 LowerPipe.cpp。
# 历史实测数字来自 测试数据分析/1104.md、pmc_test.log 和对应实验目录。
# “单步”= 上一版耗时 / 这一版耗时。组合实测不能拆成单个 Pass 的收益。
# 历史计时不代表当前实现的单项收益。
# 本次更新：PIPELINE 补齐 Frisk 入口、LowerInfo 布局分析、AutoPipeline、
# Fragment CP-SAT 调度、线程合法化和 LLVM 导出，并在第 4 部分补充白话说明。
# 补充 V 双缓冲逻辑规划前移至 AutoPipeline、V0 首次 DMA 提前发射的回退修复；
# 后续更新：提前展开有标记的内层 QK，跨 bn K0 逻辑规划也前移至 AutoPipeline；
# SharedMemoryPlan.execution 保留物理验证与落实，尚无本次修改后的 GPU 实测。
# 以下保留已有优化记录：第 25 项为独立的通用 LLVM 清理，支持 --llvm-cleanup 四种模式；
# MMA / convert_pack 锚点的契约在 LLVM translation 时补齐，不再由主路径后置扫描。
# 补充 ExecutionOpInterface、最新 BK32 实测与清理/调度消融记录，修正测试迁移路径。
# 第 34/35 项仍在 ThreadIR；转换两端的生命周期锚点由 ROCm lowering 生成，
# FriskLLVMTranslation 校验并补齐属性，保留锚点的 sideeffect。
# 原优化编号 2~41 保留；新增流程编号从 42 起，编号不是执行序号。
# 同编号重复出现表示多次调用或多个内部阶段；标为“内部”的行不额外运行一个 Pass。
# 26/41 已合为通用 MMA 链交错；29/31 已合为通用 LDS 配对预取。
# 表中编号保留，分别描述通用 Pass 的不同来源/窗口配置；旧命令行参数作为兼容映射。
# 表中 32 / 4 / 8 / rtz 等是说明用调优值；各开关默认值在第 4 部分另列。
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
# QK tile / VO tile   归约轴与输出轴独立选择；本配置各为 32。tensor IR 不分 D，
#                     block 层才生成 QK 的 4 轮归约和 PV 的 4 个输出分片。
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
# 第 3 部分：当前调度主线的完整程序（未开启第 34/35/41 项）
# =============================================================================
# 方括号是优化名，对应第 5 部分的同名步骤。这里关注“一个 block 的执行顺序”。
# V 的逻辑 stage / slot / 发射锚点先由 AutoPipeline 规划；LDS/DMA 布局在
# block → fragment 时联合选择。LowerFriskFragments 内的 SharedMemoryPlan.execution
# 校验并落实物理槽位、发射顺序及早期跨 bn K0 计划；LLVM 发射已定方案。
# 下面写的是本配置规划成功的结果；不能把两个固定 8 KiB 槽当作所有输入的默认布局。
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
        # V分片大小=BN*OD=128*32*f16 = 8KiB
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

        # [elide barrier] 规划证明下一轮 publication / 退出 drain 足够时删除 reader barrier

    # ---- 循环外：只做一次的收尾 ------------------------------------------
    drain_async_and_barrier()                 # [跨轮预取] 等待退出任务，保护后续 LDS 使用
    L = cross_lane_sum(L_local)              # [延迟 lane sum]
    O_store = convert_layout(O)              # [布局转换移出循环]
    store(to_fp16(O_store / L))


# =============================================================================
# 第 4 部分：Frisk 完整流程总表（执行顺序、可选分支和 Pass 内部阶段）
# =============================================================================
# 入口见 test/test_src.cpp；fragment 子流水见 lib/Pipeline/FragmentScheduling.cpp；
# 第 6 项内的顺序见 FriskBaseToTheadlevelIR_plan2.cpp；secondLowering 见 LowerPipe.cpp。
# 字段：稳定编号, 当前 Pass / 内部步骤, 阶段 / 所属 Pass, 调优取值, 历史收益 / 说明。
# 范围：MyTest 中从进入 Frisk 前的准备，到 LLVM 导出的全部 Frisk 相关环节。
# 表中既有优化 Pass，也有转换、分析、驱动函数；它们不全是“让代码更快”的优化。
# 标为“内部”的行展开上一层 Pass 的工作；通用清理按阶段合并列出，避免淹没主线。
# 同一行中的多个编号表示这些步骤由同一次 Pass 调用处理。
# 默认介绍非 CP-SAT 路径；“可选”只在相应开关/标注存在时生效，不是全部串行叠加。
# 第 6 项下的 52/55/14/28/37 和第 15 项下的 78/11/27/32 都是内部阶段。
# 第 27 项分两段：第 51 项内部做逻辑规划，第 15 项内部做物理校验与落实。
# 第 32 项也按相同边界拆分，逻辑跨轮关系已前移至第 51 项。
# 第 52 项在第 5、6 项中分别调用；第 57 项分别观察 fragment 和 thread。
# 白话主线：整理输入 → 分块 → 分配线程任务和数据布局 → 安排流水 → 展开线程代码 → 导出。
# 注意：block 级流水可先安排“整块数据”的搬运；精确的线程布局随后才确定。

PIPELINE = [
    # ---- 进入 Frisk：把前端表达整理成统一的 block tile 操作 ----------------
    (42, "DeepgenGraphSimplify",              "Frisk 前准备", "默认，Frisk 转换层版本",      "内联 device kernel，整理输出参数和指针"),
    (43, "AddKernelargPermuteInfo",           "Frisk 前准备", "默认",                      "记录参数的维度排列，避免把转置读错"),
    (44, "ConvertScfForOp",                  "Frisk 前准备", "默认，转换可处理的循环",      "把循环写成更便于分析下标的 affine 形式"),
    (45, "MemspaceAnalyze",                  "Frisk 前准备", "默认，分析/标注",            "分清数据放显存、共享内存还是寄存器"),
    (46, "ConvertKernelOpToFrisk",           "Frisk kernel", "默认",                      "转换 kernel 外壳、参数与并行结构"),
    (47, "ConvertMemAndCalcOp",              "Frisk kernel", "默认",                      "转换内部 copy/gemm/reduce/block 等操作"),
    (48, "ReconcileUnrealizedCasts",         "通用清理", "转换后；导出末尾还会重跑",        "清掉转换过程中临时搭建的类型桥梁"),
    (49, "FriskFuseBlockOps",                "Frisk kernel", "默认",                      "按数据流合并相连 block，减少中间搬运"),
    (50, "ConvertFriskToBase",               "转 block", "默认",                          "展开 kernel/parallel/block 等外壳，形成 func 和显式循环"),
    # 至此已全部接入 frisk
    # ---- tensor / block IR：算法结构 -------------------------------------
    # DeepgenGraphSimplify / AddKernelargPermuteInfo 各运行一次，随后运行 ConvertScfForOp、
    # MemspaceAnalyze，再接 ConvertKernelOpToFrisk → ConvertMemAndCalcOp
    # → ReconcileUnrealizedCasts → FriskFuseBlockOps → ConvertFriskToBase。
    (36, "FriskOpFuse（分块前）",            "block",    "allow-reassociation=(math!=strict)", "先融合完整 PV 的 alpha*O 初值"),
    (40, "TileAttention",                    "block",    "QK=32，VO=32；math!=strict 可切 QK", "真实切加载/分配；无 GPU 单项计时"),
    # --pipeline=1 在此标注手动 stage/order，第 5 项后才运行 PipelineSchedule。
    (36, "FriskOpFuse（分块后）",            "block",    "同上",                          "补充分块暴露的融合机会"),
    (2,  "ReorderWorkgroups",                "block",    "--workgroup-group-size=32",     "1.0645x"),
    (3,  "NormalizeAttentionKLayout",        "block",    "Attn_p2 缺失或无 frisk.arg_layouts", "显式布局保留输入 ABI"),
    (4,  "PromoteGlobalToSharedAsync",       "block",    "FRISK_GM_TO_SHM_MODE 非 legacy", "-"),
    # ---- 至此，attention 算法结构确定；下面开始规划数据搬运
    # 启发式路径调用一次 AutoPipeline；27/32 任一开启时先尝试展开标记的内循环，
    # 然后处理显式 pipeline_hint，再按开关生成下面两种 DMA 计划。
    (51, "AutoPipeline",                     "block", "非 CP-SAT 时调用；hint / 27 / 32 按需生效", "有界内层展开、N 槽循环预取、V 流及跨轮逻辑规划"),
    ((27, 32), "V 双缓冲与跨 bn K0 逻辑计划", "block，第 51 项内部", "--v-double-buffer=1 / --cross-bn-k0=1，各自可选",
     "27 记录 stage / slot / issue-after；32 指定首 DMA、末片 readiness 与跨轮关系，均待后端校验"),
    (5,  "PromoteGemmMatrixLoads",           "block",    "默认",                          "-"),
    # CSE → PipelineSchedule（--pipeline=1）→ LowerFriskAsyncToSync（copy mode 为 sync）。
    (52, "LowerInfoAnalysis（矩阵读选型）",    "block，第 5 项内部", "有未分配线程的 GEMM 时运行", "先推断各线程拿哪些元素，才能选择矩阵读"),
    (53, "PipelineSchedule",                 "block，可选", "--pipeline=1；非 CP-SAT",        "按 stage/order 排好预热、稳态和收尾"),
    (54, "LowerFriskAsyncToSync",            "block，可选", "FRISK_GM_TO_SHM_MODE=sync",      "把异步任务改回同步 copy，便于对照和回退"),
    # -- 至此，block级别的pipeline已确定
    # ---- asynccopy / 矩阵读语义已明确；开始完整的计算布局与存储布局规划
    (6,  "ConvertFriskBaseToFragmentIR",     "转 fragment", "--wmma-layout=b-permute --swizzle-k=8:6", "-"),
    (52, "LowerInfoAnalysis（完整线程布局）",  "block，第 6 项内部", "需要布局的计算运行；重新分析当前 IR", "从 GEMM 向前后传播，得到元素到线程/寄存器的映射"),
    (55, "布局衔接 / PlanWmmaLayoutConversions", "block，第 6 项内部", "wmma-layout；跨 52/14 前后调用", "推断前选布局，推断后插入/折叠转换，14 后补 shuffle 计划"),
    (14, "sinkLoopLayoutConversions",        "block，第 6 项内部", "自动",                    "历史 causal: 1.1655x（含 25）"),
    (28, "SharedMemoryPlan.layout",          "block，第 6 项内部", "--shared-copy-constraints=1:8:9:3", "历史 K DMA/swizzle: 1.0149x"),
    (37, "SinkWaitBufferReady",              "block，第 6 项内部", "自动",                    "延迟 readiness；无单项计时"),
    # ---- fragment IR：计算还是语义操作，物理存储未定 ----------------------
    # 转换后先做 VectorSafeCanonicalizer，再融合剩余的 fragment 累加。
    (7,  "FuseFragmentAccumulator",          "fragment", "默认",                          "-"),
    (80, "FragmentScheduling（frisk-fragment-scheduling）", "fragment，组合流水", "默认；CP-SAT 关闭 heuristic-scheduling", "统一入口 buildFragmentSchedulingPipeline；包含下列 8/9/10/29/31/11"),
    # 统一入口已收拢为 FragmentScheduling；FriskFragIRReorder 仍是内部 Pass 的源码名。
    (8,  "FriskFragIRReorder",               "fragment，第 80 项内部", "非 CP-SAT 路径",       "循环融合 + fragment 一级 lookahead"),
    (9,  "SimplifyLoopControl",              "fragment，第 80 项内部", "--simplify-loop-control=1", "-"),
    # 不变 fragment 保留会改变工作图；启发式和 CP-SAT 路径都在导出调度前执行。
    (10, "RetainInvariantFragments",         "fragment，第 80 项内部", "--retain-fragment-words=32", "causal: 1.08~1.10x/档"),
    # 非 CP-SAT 路径上任一距离非零即运行。
    ((29, 31), "PrefetchFragmentLDS",        "fragment，第 80 项内部", "--k-lds-prefetch=8 / --k-tail-prefetch=4，各自可选",
     "迁移前：普通窗口 1.0272x（MyTest 路径）；归约窗口 1.0098x"),
    (11, "ScheduleAsyncCopies",              "fragment，第 80 项内部", "启发式；子流水末尾", "-"),
    # CP-SAT 在此调用 processScheduleJSON；--cpsat-stop-after=1 成功后直接返回。
    (56, "ScheduleJSON / CP-SAT",             "fragment，可选驱动步骤", "--cpsat-export/import/baseline", "第 10 项后导出/导入调度；替代第 11 项"),
    # shm liveliness 分析
    (12, "AnalyzeShmLiveness",               "fragment", "默认（只分析，不改 IR）",        "-"),
    (57, "AnalyzeRegisterPressure（fragment）", "fragment，可选分析", "--analyze-register-pressure=1", "估算同时活跃的寄存器数据，查找压力峰值"),
    # ---- thread IR：reduce/布局变换展开为通信，确定物理 LDS ---------------
    (13, "ConvertFragmentToThreadIR",        "thread",   "默认",                          "-"),
    # FinalizeThreadTiling 移除 to/from_threadtile 桥接，随后 VectorSafeCanonicalizer → CSE。
    (58, "FinalizeThreadTiling",              "thread", "默认",                            "清除 to/from_threadtile 桥接，落实线程数据形状"),
    (15, "LowerFriskFragments",              "thread",   "默认；启发式验证 27/32，CP-SAT 保留调度", "落实物理 LDS 分配、同步及有标记循环的有界展开"),
    (78, "PackSharedMemory",                 "thread，第 15 项内部", "使用第 28 项发布的布局",       "将打包/swizzle 方案落实到分配和所有访问地址"),
    # ScheduleAsyncCopies（物理访问可见后）
    (11, "ScheduleAsyncCopies（物理访问可见后）", "thread，第 15 项内部", "非 CP-SAT 路径",        "重查搬运/等待顺序，再按最终生命周期分配槽位"),
    (27, "SharedMemoryPlan.execution（V 物理落实）", "thread，第 15 项内部", "--v-double-buffer=1；非 CP-SAT；消费早期计划", "历史: 1.0306x"),
    (32, "SharedMemoryPlan.execution（跨轮物理落实）", "thread，第 15 项内部", "--cross-bn-k0=1；非 CP-SAT；消费早期计划", "历史: 1.0168x（含 sum anchor）"),
    # 第 15 项始终落实共享存储：启发式路径在重排前后补循环回边同步，按最终生命周期分配。
    # 27/32 开启时先有界展开再物理落实；均关闭时先落实再展开。
    # CP-SAT 有 pool_bytes 时使用其槽位计划，否则只做保序分配；两者均补必要同步。
    (60, "通用循环/存储清理（前段）",         "thread，多个标准 Pass", "默认；CP-SAT 跳过循环融合", "外提不变分配和计算，规范循环、消除冗余读写"),
    (59, "ThreadLevelIRLegalize",             "thread，穿插在第 60 项中", "默认",              "整理 kernel 属性、内存分配和目标矩阵指令"),
    # 第 16 项前保留两轮 CSE → VectorSafeCanonicalizer：第一轮折叠地址或算术后，
    # 第二轮 CSE 仍可合并新出现的相同表达式，随后规范化还能继续折叠其使用者。
    (16, "IRDeepOptimize",                   "thread",   "reorder-after-unroll=(reorder && 非 CP-SAT)", "-"),
    (60, "通用循环/存储清理（后段）",         "thread，多个标准 Pass", "默认",              "规范循环，用 SSA 值替代可消除的临时内存"),
    # 算术优化。简化affine的ub indice计算
    (17, "SimplifyBoundedAffine",            "thread",   "--simplify-bounded-affine=1",   "组合实测，见表后说明"),
    (24, "CanonicalizeThreadLDSAddresses",    "thread",   "swizzle 自动开启", "历史 causal: 1.0233x"),
    # 算术优化
    (19, "ApproximateThreadMath",            "thread",   "math=approx 或 mmac-recip",    "倒数复用；native exp2 属于 lowering"),
    # 计算图变化。暂时视为数学算法优化
    (20, "ThreadMaskAddBypass",              "thread",   "--mask-add-bypass=approx（本 case 无 mask）", "-"),
    # 计算图变化。暂时视为数学算法优化
    ((21, 22), "ThreadReductions",            "thread",   "--local-reduce 和 --defer-lane-sum 各选 off|legal|approx；非 CP-SAT",
     "先拆局部归约链，再延后跨 lane 求和；历史 causal: 1.0347x（20、21、22 合计）"),
    # LDS 提前读取由 fragment 第 29/31 项安排；这里仅落实物理读对和锚点。
    (24, "CanonicalizeThreadLDSAddresses（预取落实前）", "thread", "29/31 任一距离非零；非 CP-SAT", "首次执行，或仍有普通 LDS load 时补证"),
    (29, "ThreadPrefetchPairedLDS（落实 29/31）", "thread", "29/31 任一距离非零；非 CP-SAT；{0, fragment}", "完成物理读对和 MMA 锚点，不重新选择预取距离"),
    (26, "ThreadInterleaveMMAChains（matrix）", "thread", "--pv-interleave=4 --pv-guard-group=-1；非 CP-SAT", "历史: 1.0043x（接近噪声）"),
    (24, "CanonicalizeThreadLDSAddresses（QK 交错前）", "thread", "qk-interleave!=0（包括 CP-SAT）", "首次执行，或仍有普通 LDS load 时补证"),
    (41, "ThreadInterleaveMMAChains（lds）", "thread，可选", "--qk-interleave=2 --qk-guard-group=2 --qk-prefetch-words=32；非 CP-SAT", "CPU 验证；无 GPU 计时"),
    (30, "ThreadScheduleSoftmaxFragments",    "thread",   "--softmax-fragments=pv；非 CP-SAT", "历史: 1.0162x"),
    (33, "ThreadElideBackedgeBarrier",        "thread",   "--elide-backedge-barrier=1；非 CP-SAT", "历史组合实测；当前无迁移单项计时"),
    # LowerFriskAsyncCopy → LowerGPUShuffleToROCDL → VectorSafeCanonicalizer → CSE，随后外提 wave setup。
    (61, "LowerFriskAsyncCopy",               "thread → 目标操作", "默认；sync 模式禁用异步发射", "落实 DMA、task、wait 和跨 wave 可见性"),
    (62, "LowerGPUShuffleToROCDL",            "thread → ROCDL", "默认",                     "把 lane 间交换数据变成目标 GPU 操作"),
    (18, "HoistWaveInvariants",              "thread + ROCDL/LLVM dialect", "--hoist-lds-bases=1 --hoist-lane-shuffles=1", "组合实测"),
    # 计算图变化。暂时视为数学算法优化
    (34, "ThreadLazySoftmaxBase",             "thread",   "--lazy-softmax-base=8；非 CP-SAT", "实验版 1.0582x；见 34"),
    # 指令优化
    (35, "ThreadSoftmaxCvtPack",              "thread",   "--softmax-cvt-pack=rtz",       "历史 LLVM 版 1.0439x；见 35"),
    (57, "AnalyzeRegisterPressure（thread）",  "thread，可选分析", "--analyze-register-pressure=1", "观察展开和调度后寄存器压力如何变化"),
    # ---- MLIR 后端：first/secondLowering 是组合流程，不是单个 Pass ---------
    (63, "firstLowering",                    "MLIR lowering 驱动", "默认",                  "降低 affine 循环，清理重复计算和无用符号；仅跳过已确认重复的入口 CSE"),
    (64, "secondLowering",                   "MLIR lowering 驱动", "默认；以下 65/60/38/39/66~69/79 为其内部阶段", "先 LICM，再把控制流、内存和 GPU 操作译成 LLVM dialect"),
    (65, "AmendAllocaOpAddrSpace",            "第 64 项内部", "ROCm private 地址空间为 5",      "为线程私有分配填写正确地址空间"),
    (60, "通用结构清理",                     "第 64 项内部", "默认",                         "展开 memref 步长信息、vector 循环，再清理"),
    ((38, 39), "CoalesceWaitsAndBarriers",   "第 64 项内部", "38：--coalesce-waits=1 或 CP-SAT 保序；39：非 CP-SAT",
     "先合并同 block 重复同步，再按跨线程内存依赖删 barrier；无单项 GPU 计时"),
    (66, "控制流/内存/函数转 LLVM",           "第 64 项内部，标准 Pass", "默认",              "把循环分支、内存访问和函数接口变成低层表示"),
    (79, "LowerFriskSchedBarrier",           "第 64 项内部", "默认，函数转 LLVM 后",       "将 Frisk 调度边界转换为目标调度屏障"),
    (67, "LLVMFuncOpAddGPUAttr",              "第 64 项内部", "默认",                         "标明 GPU kernel、workgroup 大小及目标信息"),
    (68, "GPUToROCDLOrNVVM",                 "第 64 项内部", "MyTest 目标为 ROCm",            "翻译线程编号、barrier 和目标相关数学操作"),
    (69, "VectorToLLVM",                     "第 64 项内部", "项目封装的 vector lowering",    "展开动态多维向量访问，转换为 LLVM 向量操作"),
    (66, "标量转 LLVM 与转换清理",            "第 64 项内部，标准 Pass", "默认",              "转换 arith/UB/index，清理临时 cast 和无用代码"),
    # ---- LLVM IR：寄存器处理与最终导出 ------------------------------------
    (70, "translateToLLVM / FriskLLVMTranslation", "LLVM 导出入口", "默认，包括 strict/cleanup=none", "导出真正的 LLVM IR，并补齐 MMA/锚点属性契约"),
    (25, "cleanupLLVMInstructions（early）",  "LLVM",     "非 CP-SAT；math!=strict；llvm-cleanup=both/early", "两次 InstCombine；不补齐导出契约"),
    (71, "prefetchGlobalPackets",             "LLVM，可选", "--prefetch-packets=1；非 CP-SAT", "提前读取后续计算所需的显存数据包"),
    (72, "anchorNextKSums（首次）",            "LLVM", "非 CP-SAT，匹配跨轮预取结构时生效",    "保护分母求和输入，避免被下沉到跨轮预取之后"),
    (23, "optimizeRegisterLifetimes",        "LLVM",     "--register-lifetimes=1（默认）", "32 的 sum anchor 在它之前；CP-SAT 走保序模式"),
    (73, "reuseLDSFragments",                 "LLVM，可选", "--lds-reuse-words=N；非 CP-SAT",  "在预算内复用已读 fragment，减少重复 LDS 读取"),
    (74, "prefetchMMACFragments",             "LLVM，可选", "--frag-prefetch=1；非 CP-SAT",    "按 frag-prefetch-words 预算提前准备 MMAC 操作数"),
    (75, "lowerAsyncDMA（最终导出）",          "LLVM 导出内部", "--dma-export 选择实现",          "发射目标 DMA 表达，并保护源地址的生命周期"),
    (25, "cleanupLLVMInstructions（late）",   "LLVM，DMA 导出后", "非 CP-SAT；math!=strict；llvm-cleanup=both/late", "SimplifyCFG + 两次 InstCombine"),
    (76, "canonicalizeLLVMExport",            "LLVM", "k-lds-prefetch!=0；非 CP-SAT",         "合并最终 DMA 导出带来的重复描述符和表达式"),
    (72, "anchorNextKSums（再次检查）",        "LLVM", "非 CP-SAT；重复调用不重复加锚点",      "导出清理之后再次保护跨轮求和顺序"),
    (77, "default<O3>",                      "LLVM，标准优化管线", "--llvm-opt=3；非 CP-SAT",   "常规 LLVM 优化后输出 .ll，尚未生成机器码"),
]
# 13 后运行第 58 项 FinalizeThreadTiling：去掉 to/from_threadtile 桥接；第 14 项的
# 布局转换外提在第 6 项内已经完成，不能再把它归给 FinalizeThreadTiling。
# 启发式 fragment 子流水先由 29/31 合用一次 Pass 规划预取，再执行 11。
# thread 调度顺序为 24 → 29（落实 fragment 的 29/31）→ 26 → 24 → 41 → 30 → 33，
# 各项按开关执行；预取落实和 QK 交错沿用第 24 项已证明的地址，仅在仍有候选时重跑。
# 31 不再是 30 之后独立运行的 thread Pass；CP-SAT 跳过这些启发式调度。
# 33 之后先局部 lower async copy / GPU shuffle，再执行 18，使 wave setup 可见并保留结构化循环。
# wave lowering 和可选的 18 都以 CSE 收尾。若最后一次 CSE 未改动 IR，且之后未执行
# 34/35，MyTest 跳过 firstLowering 入口 CSE；独立调用 firstLowering 默认仍执行入口 CSE。
# translateToLLVM 已完成 MMA/锚点契约；25 的 early 在其后、23 之前执行，
# late 在最终 DMA 导出后带 simplifyCFG=true 执行，两轮由 --llvm-cleanup 独立选择。
# 23 后可选 prefetch/reuse 见下；DMA 导出及 late 之后，k-lds-prefetch!=0 时
# canonicalizeLLVMExport 做通用 CSE/InstCombine 清理；anchorNextKSums 再调用一次
# （幂等），最后默认 O3。LLVM 不再运行 33，none 也不关闭这些其他步骤。
# 第 34/35 项在 18 之后、first/secondLowering 之前依次执行，入口分别为
# ThreadLazySoftmaxBasePass / ThreadSoftmaxCvtPackPass；各自可独立开启。
# 第 35 项在 ThreadIR 选择转换边，ROCm lowering 落实指令和两端锚点；LLVM 不再匹配转换。
# --llvm-opt=0 可关闭最终 O3，--llvm-opt=3 为默认；CP-SAT 跳过最终 O3。
# --llvm-cleanup=both|early|late|none 默认 both；strict 或 CP-SAT 跳过两轮清理。
# 该选项不控制导出契约或最终 O3；strict / none 也必须得到完整 MMA/锚点属性。
# 新增 causal 分支能力：第 6 项传递 --causal-mask=off|i32|finite（默认 off），
# 第 13 项展开 fragment mask 时落实；直接 thread lowering 路径也支持。详见第 6/13 项对比。
# 它是已有转换 Pass 内的优化，不是排在 LLVM 调度之后的新 Pass；non-causal 主线不使用它。
# 第 17/18/33 项的历史组合产物 wave_invariant_noncausal/kernel.o3.ll 已记录 1.545679 ms，
# VGPR 228、scratch 0（pmc_test.log）；没有各项单独开关的同版对照，不填写单项加速比。
# 纯 lowering 与导出也已列入表；通用规范化按第 48/60/63/66 项归组，细节见下方介绍。
# 第 3、12、16 项在第 5 部分没有单独对比。
# 第 40 项替代旧 tensor split-D 和 driver retile：BK 现在只是 lowering hint，
# QK/softmax 在生成时就共享；旧 tensor jam Pass 和 carrier budget 参数已删除。
# 第 6/13 项只补充新增的 causal mask 能力，不代表这两个转换 Pass 的全部工作。
#
# ---- 4A. 先分清：布局、流水、存储复用分别在解决什么 -------------------------
#
# 想象一个 workgroup 是一组工人，tile 是一批货，LDS 是共同使用的临时货架：
#   LowerInfo（52）：分工表。某个矩阵元素交给哪个 wave、哪个 lane、哪个寄存器位置？
#   SharedMemoryPlan.layout（28）：货架排布。元素放哪一格，怎样搬得宽、读得顺？
#   AutoPipeline / PipelineSchedule / CP-SAT（51/53/56）：作业时间表。
#       当前批在计算时，下一批能不能先搬？什么操作必须等前面的结果？
#   SharedMemoryPlan.execution（15 内部）：货架使用时间表。
#       哪批货已经用完，空位能否给下一批；覆盖旧数据之前是否需要同步？
#       V 的逻辑时间表来自 AutoPipeline；这里证明真实访问安全后再落实。
#       跨 bn K0 的逻辑时间表也来自 AutoPipeline；这里验证循环生命周期并落实。
# 这几件事互相约束：提前搬运能减少等待，却可能多占 LDS 和寄存器，未必总是更快。
#
# ---- 4B. 进入 Frisk 的准备（42~50）---------------------------------------
#
# 前端写的是“对矩阵做计算”；后端需要知道“哪个线程读写哪个地址”。
# 42~45 先整理函数、参数维度、循环和存储空间；46 换 kernel 外壳，47 转换里面的算子。
# 49 把可以衔接的 block 合在一起；50 再展开外壳，形成普通函数、循环和 block tile。
# 此时 GEMM 仍表示一块矩阵的乘加，还没有把工作拆成每个 lane 的具体指令。
# 49 与 36 不同：49 融合 block 的数据流；36 把 GEMM 后的加法并入累加器。
# 第 3 项处理旧 attention 输入的 K 轴顺序；已有明确参数布局的输入无需再规范一次。
#
# ---- 4C. Layout 分析 / LowerInfo 推断（52、55，重点）----------------------
#
# Layout 不只是“行优先还是列优先”，还包括“矩阵元素由哪个线程持有”。
# LowerInfo 可以理解为某个值在某个操作处的线程分工表，主要记录：
#   1. 矩阵乘指令要求的基础布局：一条 MMA 里每个 lane 提供哪些 A/B/C 元素。
#   2. block 内 wave 如何分工；一条指令覆盖不完时，重复多少次才能算完 tile。
#   3. 每个线程持有多少数据，以及线程编号/循环下标如何换算成矩阵的行列坐标。
#   4. 归约只在本线程完成，还是需要同 wave 的 shuffle，或跨 wave 的 LDS 通信。
#
# 推断过程：
#   GEMM 的硬件布局作为起点
#       → 沿 copy、逐元素计算、reduce、循环携带值等向前/向后传播
#       → 收集并协调不同使用点的布局要求
#       → 必要时插入 convert_layout，让前一步的输出接得上后一步的输入。
# 举例：QK 产生 S → softmax 产生 P → P 作为 PV 的 A 输入。
# QK 输出布局和 PV 输入布局可能不同，即使都是同样大小的矩阵，也不能直接当作同一种布局。
# 若数据只需在同 wave 内重排，可规划 shuffle；不能这样完成的转换可能需要 LDS 和同步。
# 55 包含 insertConvertLayoutOps、WMMA C 布局衔接、逆转换消除和
# planWmmaLayoutConversions；后者会把可证明的 wave 内重排记录为 shuffle_plan。
# 14 接着尝试把循环里反复发生的布局转换移到循环外，减少重复通信。
#
# 为什么表中出现两次 LowerInfo？
#   第一次在 PromoteGemmMatrixLoads 内：先知道 GEMM 需要的线程分工，才能选矩阵读。
#   第二次在 ConvertFriskBaseToFragmentIR 内：重新分析当前 IR，生成完整 fragment 布局。
#   第 6 项还会在推断前设置 WMMA 布局偏好，推断后协调归约布局和实际转换。
#   两次分析服务于不同阶段；不能把第一次的分析结果当作后续一直有效的缓存。
#
# LowerInfoAnalysis 是内部分析组件，不是当前独立运行的 frisk-layout-infer Pass。
# 分析负责给出布局信息，插入转换和展开代码由调用它的转换流程完成。
# 它也不是全局最优布局搜索器：要满足硬件规则、线程数、使用点约束和所选布局方案。
# 需要推断的计算通常要求 GEMM 起点；单纯显式搬运等路径可以跳过这项推断。
#
# ---- 4D. AutoPipeline 与其他调度（51、53、56）-----------------------------
#
# AutoPipeline 的直观目标是“算这一块时，先搬下一块”。当前有两条独立路径：
#   循环 hint 路径：生成 N 槽循环预取；直线 V 流路径：生成双缓冲逻辑计划。
# 后者由 --v-double-buffer=1 开启，不要求循环 hint；--cross-bn-k0=1 另启跨轮计划。
# 两者都可触发有标记的内层循环提前展开，详见 4D.2。
# 例如 N=2，两块 LDS 缓冲交替使用（下面省略边界检查和同步细节）：
#   原来：搬 tile0 → 等待 → 算 tile0 → 搬 tile1 → 等待 → 算 tile1。
#   改后：先搬 tile0；等 tile0 到达后，发出 tile1 搬运并计算 tile0；
#         等 tile1 到达后，向已空闲的槽发出 tile2 搬运并计算 tile1……
# 真正的重叠发生在硬件执行期间；提前发出 DMA 不代表数据已经到达。
#
# 循环路径的开关是 frisk.pipeline_hint=N：N>=2 使用 N 个轮换槽；
# 该路径无 hint 不改写，只有标记而没有数值时按 2 处理，N=1 关闭。
# 变换会生成受边界条件保护的前 N-1 次预取；稳态中等待 tile i，发出 tile i+N-1，
# 再计算 tile i；退出时等完剩余任务，确保缓冲区可安全交给后续使用者。
# 只有证明整块搬运与循环携带计算、外部写入等不冲突时才改写；证明不了就保留原循环。
# N 更大意味着能看得更远，也需要更多同时存活的缓冲，不能简单认为越大越好。
#
# ---- 4D.1 AutoPipeline 的循环 hint 路径 ---------------------------------
# 对照 lib/Pipeline/AutoPipeline.cpp：runOnOperation → analyze → build。
# 它做的是“整块 GM→LDS 数据的提前搬运”，不会搜索最优 stage 数，也不会重新排列
# GEMM/softmax 的计算依赖；N 由 hint 指定，原循环的计算及递推关系仍留在原迭代。
#
# 循环 hint 是什么时候加上的？当前有两条输入路径，这个分支只消费标记：
#   路径 A：输入里已经有要流水化的循环。
#     手动在 scf.for/affine.for 上写 frisk.pipeline_hint=N，或由 TuneAttention.py
#     的 prepare_input 按 pipeline-stages=N 给选定的 scf.for 添加它。
#     ConvertScfForOp 等循环转换会把已有 hint 复制到新 affine.for；不是在这里自动选 N。
#   路径 B：输入还是 full-D Attention，QK 分块循环尚未生成。
#     先在 Attn_p2 kernel 上指定 frisk.qk_pipeline_hint=N（注意多了 qk_）。
#     TuneAttention.py 的共享 QK 输入路径也把 pipeline-stages=N 写到这个 kernel 属性。
#     MyTest 在转换 kernel 前暂存该属性，ConvertFriskToBase 后写回 func。
#     第 40 项 TileAttention 的 tileQK 创建 QK 归约分块 affine.for 时，才将它写成
#     循环属性 frisk.pipeline_hint=N；成功分块后删除函数上的 frisk.qk_pipeline_hint。
#     若 QK tile 等于完整归约维度，tileQK 不创建新循环，也就不会通过此路径添加循环 hint。
#
# 因此共享 QK 主线的传递顺序是：
#   输入/调优脚本：kernel.frisk.qk_pipeline_hint=N
#     → MyTest 保留到 Frisk Base func
#     → TileAttention：新 QK 循环.frisk.pipeline_hint=N
#     → PromoteGlobalToSharedAsync
#     → AutoPipeline：读取循环 hint，检查并尝试改写。
# 没有输入 hint 时，TileAttention 不会自行猜测预取深度，AutoPipeline 也不自动标注所有循环。
# TuneAttention 的 pipeline-stages=keep 保留输入、none 删除标记、1 保留关闭标记；
# --pipeline=1 是另一套显式 stage/order 调度入口，不负责添加这里的 hint。
# 源码定位：TuneAttention.py::prepare_input；test_src.cpp 的 qkPipelineHint 保存/恢复；
# TileAttention.cpp::tileQK；DeepgengraphToFrisk.cpp 中的循环 hint 复制。
#
# 第一步：找出用户指定的循环（AutoPipelinePass::runOnOperation）。
#   从内向外遍历函数中的操作，收集 frisk.pipeline_hint，并先统一检查 hint 格式。
#   正整数 N 表示槽数；unit 标记等于 2；N=1 不处理；没有 hint 的操作不处理。
#   0、负数或不支持的属性格式会报错并让 Pass 失败，不是悄悄按默认值处理。
#   N>=2 时，目标必须是 Frisk Base 的 affine.for；例如只给 scf.for 加 hint 会跳过。
#   MyTest 在 PromoteGlobalToSharedAsync 之后、PromoteGemmMatrixLoads 之前调用它；
#   即使输入还是符合条件的同步 frisk.copy，它也能在改写时生成异步搬运。
#   CP-SAT 模式跳过整个 AutoPipeline；hint 和 --pipeline=1 是两个独立入口。
#
# 第二步：先检查循环本身能不能处理（PrefetchLoop::analyze）。
#   a. 循环自身或祖先不能已有 pipeline.stage/order/scheduled，避免重复流水化。
#   b. 循环边界和所在控制流必须能证明对整个 block/workgroup 一致，不只是 warp/wave 内一致。
#      所有线程必须参与同一轮的集体等待/同步；不能只有部分 wave 进入或迭代次数不同。
#      这要求同步点的参与和顺序一致，不要求每个线程的数据、地址或所有计算路径都相同。
#
#      当前检查有三个不同层次（isInBlockUniformControlFlow / isCollectiveLoop）：
#        循环边界：lb/ub（scf 外层循环还包括 step）必须通过 isBlockUniform。
#          常量、标量函数入口参数、block_id/block_dim/grid_dim，以及由这些值构成的
#          受支持 arith/affine.apply 表达式可通过；共同执行的外层循环 iv 也可通过。
#          thread_id、load 结果、循环携带的计算值不被证明为一致，即使实际数据恰好相同。
#        外层控制流：目标循环必须直接位于单 block 函数入口，或仅嵌在通过检查的
#          scf.for/affine.for 中。当前任何外层 if 都不接受，不会再分析 if 条件；
#          所以 if tid<64 包住循环会跳过，连 if block_id==0 包住循环也会保守跳过。
#        循环体内部：这个检查不递归禁止所有 if tid，也不禁止使用 thread_id。
#          例如所有线程执行同样的循环，体内按 tid 分支做纯算术，再回到共同同步点，
#          不会仅因分支条件含 tid 就被这一项拒绝；仍需通过 safeEffects 等其余检查。
#          特别是预取缓冲的直接读者必须在循环体顶层，不能移到这个 if 内，
#          也不能在分歧路径插入本 Pass 不支持的同步或外部写入。
#      因此“warp 内不分歧”不够，但“出现任何 tid 相关 op 就拒绝”也不符合当前实现。
#   c. N 不能超过 64，且 N*step 不能溢出；这里的 64 是代码体积上限，非性能建议。
#   d. 不支持 memref 类型的循环携带结果，避免跨轮指针别名难以证明。
#      普通标量/vector 的累加状态可以保留，例如每轮更新的 sum、acc。
#   边界不要求全是编译期常量，但需要满足上面的 workgroup 一致性要求。
#
# 第三步：找出可预取的整块搬运，并检查地址（analyze / address）。
#   只扫描循环体顶层的 async_copy_gm_to_shm 或表示完整拷贝的 frisk.copy。
#   候选必须满足：源是 GM（地址空间 1），目的地是 LDS（地址空间 3）；
#   tile 形状静态、源/目的形状及元素类型相同；目的地由循环体内的 alloc_buffer 分配。
#   同一目的缓冲不能登记两次；已有 frisk.source_map 的异步搬运拒绝，
#   因为此时已涉及物理 packing，AutoPipeline 应在那之前安排逻辑 tile 的流水。
#   没有找到合格搬运，就不改写这个循环。
#
#   地址可以随当前循环下标 iv 变化，也可以依赖循环外的值；
#   例如 K[iv * stride] 可以提前算出 K[(iv + (N-1)*step) * stride]。
#   地址不能依赖本循环上一轮算出来的 iter_arg，也不能依赖有副作用或不能安全提前
#   执行的操作。循环内地址表达式还必须在顶层、没有 region，且其所有输入都通过检查。
#   所以“可提前计算地址”不等于“地址完全不随循环变化”。
#
# 第四步：确认缓冲读写和异步任务不会逃出当前迭代（analyze / safeEffects）。
#   每个预取缓冲及 copy 返回的句柄，只能被同一循环体顶层、位于 copy 之后的
#   直接读者使用，例如 GEMM、部分 Frisk 算术、memref/affine load、作为源的 copy。
#   不能把它作为循环结果传出去、生成未受支持的别名、再次写入，或放到嵌套分支里读。
#   每块缓冲至少要有一个直接读者；异步 task 只能交给当前循环体内的 wait_buffer_ready。
#   对每个异步读者，都必须已有位于“发射之后、读取之前”的匹配 wait。
#
#   接着检查整个循环体：允许已知的计算、只读操作和能证明安全的局部写入；
#   拒绝写外部内存、未知副作用，以及无法处理的同步/region。
#   普通 scf/affine 的 if/for 会递归检查，但其中不能藏着不受支持的预取缓冲使用。
#   causal MaskOp 可以保留，前提是其内部所有操作都没有内存副作用；mask 仍在原轮计算。
#   这些是当前实现的保守规则，不是完整的通用别名分析或任意循环流水调度器。
#
#   以上 analyze 阶段只收集信息，不修改 IR。任何一项证明失败，就输出
#   “automatic pipeline skipped: 原因”并保留该循环；通过后才进入 build。
#   这与第一步的非法 hint 不同：非法 hint 是报错，合法但不支持的循环是跳过。
#
# 第五步：在循环前创建 N 个槽，并发出预热搬运（build / issue）。
#   对每一个被识别的搬运，都把它的 alloc_buffer 复制成循环外的 N 份，
#   分别加 pipeline_slot=0..N-1；多个输入各有自己的 N 个槽，使用相同的轮换相位。
#   若有 T 个搬运，便有 T*N 个逻辑缓冲；这一步不负责最终 LDS 字节偏移或跨区域复用。
#   若每份 tile 为 B 字节，单个搬运的逻辑容量就是 N*B，还未计入对齐和后续物理规划。
#
#   在进入循环前，尝试发出 tile 0..N-2 的搬运，分别放入槽 0..N-2。
#   每个预热搬运都有独立边界检查，连 tile 0 也检查：零次循环不发 DMA，短循环不越界。
#   issue 会复制必要的源地址计算，并把 iv 替换成目标 tile 的下标；
#   原来是 async copy 就复制它，原来是同步 copy 就生成 async_copy_gm_to_shm，返回 task。
#   越界的分支返回 async_task_completed，表示“这项任务无需等待”，而不是读取越界数据。
#
# 第六步：重建稳态循环，使用任务 FIFO 和环形槽（build / selectSlots）。
#   新 affine.for 保留原上下界、step 和原 iter_args，在后面追加 (N-1)*T 个 task。
#   FIFO 就是“先进先出的任务队列”：每组 T 个 task 对应同一轮的所有输入搬运。
#   队首对应当前轮的数据，后面是尚未消费的未来轮；队列深度是 N-1，缓冲槽数是 N。
#   令 j=(iv-lb)/step，j 是从 0 开始的迭代编号，不是原始 iv：
#       当前读取槽 = j % N
#       未来写入槽 = (j + N - 1) % N
#       未来数据下标 = iv + (N - 1) * step
#   因此非零/负的 lb、step>1 都不能直接拿 iv%N 选槽。
#   selectSlots 用 scf.if 返回所选缓冲的描述符；不会为每个槽复制一份 GEMM。
#
#   每轮严格按下面的顺序生成：
#     ① wait 队首的 T 个 task，确保当前 tile 可读。
#     ② sync_threads_in_block，让所有线程都结束上一轮对旧槽的读取，才允许覆盖旧槽。
#     ③ 若未来 tile 存在，向未来槽发出它的搬运；否则返回 completed task。
#     ④ 执行原循环体的计算，把原缓冲/句柄替换成“当前槽”，原 iter_args 接到新循环上。
#        原预取 copy、相应 alloc 和旧 wait 不再复制，其他计算仍按原顺序复制。
#     ⑤ yield 原计算结果，同时将 task 队列左移一组，尾部追加本轮新发出的 T 个 task。
#
#   为什么一定先 wait，再发未来搬运？
#   底层 wait 可能实现为“等待该 wave 所有在途搬运”。如果先发未来任务再 wait，
#   就可能把刚发出的未来任务也等完，再去计算，导致本想隐藏的搬运延迟重新暴露出来。
#   显式的 block 同步还保护旧槽的读者；不能只因当前 task 完成就认为所有旧读者都已退出。
#
#   未来 tile 的检查用 remaining=ub-iv，再比较 remaining>(N-1)*step；
#   只有条件成立的分支内才生成 futureIv，避免先算最后几轮的未来下标而溢出。
#   预热也使用剩余跨度和距离检查，支持少于 N-1 轮、尾部不足一个 step 等情况。
#
# 一个 N=3、总共 5 个 tile、只有 1 个搬运的例子（tX 表示 tile X 的 task）：
#   阶段        当前读取/等待       提前发出的搬运       计算后任务队列
#   预热        —                   tile0→槽0、tile1→槽1 [t0,t1]
#   第 0 轮     槽0，wait t0        tile2→槽2             [t1,t2]
#   第 1 轮     槽1，wait t1        tile3→槽0             [t2,t3]
#   第 2 轮     槽2，wait t2        tile4→槽1             [t3,t4]
#   第 3 轮     槽0，wait t3        不发射，返回 done      [t4,done]
#   第 4 轮     槽1，wait t4        不发射，返回 done      [done,done]
#   每一轮 wait 后都有上述 block 同步；第 1 轮重用槽0时，tile0 的读取已经结束。
#
# 用伪代码把 build 的核心串起来（仅说明顺序，边界检查使用上面所述的安全算法）：
#   slots = 为每个搬运分配 N 个槽
#   queue = [带边界检查地搬第 j 块到槽 j，j=0..N-2]
#   state = 原循环初值
#   for iv in 原来的循环范围:
#       j = (iv-lb)//step
#       wait(queue[0])                         # 一组 T 个任务
#       block_barrier()                       # 保护将被覆盖的旧槽
#       future = guarded_issue(iv+(N-1)*step, slots[(j+N-1)%N])
#       state = 原计算(state, slots[j%N])     # 例如原来的累加/softmax 递推
#       queue = queue[1:] + [future]
#   wait(queue 中全部剩余任务)
#   return state
#
# 第七步：明确收尾、替换旧循环（build 尾部）。
#   在新循环后对全部 (N-1)*T 个出口 task 分别生成 wait_buffer_ready。
#   即使边界分析可推断它们通常已经是 completed，也仍保留显式完成关系，
#   让后续共享内存活跃性分析能证明任务已经结束，进而判断前后区域能否复用存储。
#   这不代表 N 个轮换槽可彼此合并：流水内部它们的生命周期仍可能同时重叠。
#   新循环增加 pipeline.scheduled、pipeline.num_stages、pipeline.slots，移除原 hint；
#   对外仅用原数量的计算结果替换旧结果，再删除旧循环，新增 task 只用于内部收尾。
#   再跑一次本 Pass 不会重复改写成功生成的循环。
#   尾部通过条件发射和队列完成来处理，没有另外复制一套 GEMM/softmax 收尾循环。
#
# 如何对照代码或排查没有生效：
#   runOnOperation：hint 收集、格式错误、仅接受 affine.for，以及跳过原因输出。
#   analyze / address / safeEffects：候选搬运、地址依赖、消费者和副作用检查。
#   build / issue / selectSlots：预热、克隆地址、选槽、稳态 task FIFO 和出口 wait。
#   已有用例：test/unittest/check_auto_pipeline.py、check_auto_pipeline_shm_reuse.py，
#   覆盖多 stage、边界、槽位轮换、递推值、拒绝路径和退出任务对存储复用的影响；
#   CPU 模型不能说明 GPU 实际重叠程度或加速比，本次仅核对源码并补充说明。
#
# ---- 4D.2 AutoPipeline 的直线 V 流逻辑计划与后端分工 ---------------------
# 本例 vo-tile=32 将 V/O 拆为 4 个直线 slice，并没有创建一个 V 循环。
# AutoPipeline 先展开有 frisk.loopUnrollFull=true 的内层 QK 循环，保留外层 bn。
# 先将 legacy copy-to-vector iter_arg 更新恢复为显式 SSA，再展开并清理临时 carrier；
# 否则会丢失累加递推，或让人工引入的 cast 边界影响布局推断。
# 单循环最多 64 次、累计最多新增 65536 个操作；有 hint/显式调度归属的循环保留。
# Q0/K0～Q3/K3 与 V0～V3 在同一 block 内可见，候选搬运流按源存储分别分析；
# 当前双缓冲契约每 block 选择一个合法末尾流，其余搬运仍走依赖/别名检查的调度。
# MyTest 将 --v-double-buffer=1 传给 AutoPipeline 的 double-buffer 选项；
# --pipeline=0、没有循环 hint 时，这条路径仍可工作。CP-SAT 模式跳过此 Pass。
#
# planSharedCopyStream 在同一 block 内识别 2~32 个 GM→shared async copy stage：
# 要求同源、形状匹配、独立 owned allocation、直接 GEMM B 消费者，
# 并检查控制流一致性、地址依赖和副作用；不满足条件就不生成逻辑计划。
# 成功时给 DMA 写 frisk.shared_pipeline_plan，字段为
#   [stage, count, logicalSlot, issueAfter]，stage 编号仅在当前 block 内有效。
# 本例依次为 [0,4,0,-1]、[1,4,1,0]、[2,4,0,1]、[3,4,1,2]。
# 这里保留串行流顺序及独立分配，只声明逻辑槽和发射锚点，不直接分配物理 LDS。
#
# 随后 Fragment/Thread 调度保护这个契约；其中 stage 0 的 issueAfter=-1
# 表示没有前一 V stage 的就绪锚点，仍允许既有 alias-checked 调度器安全前移首次 DMA。
# 所有 wait 和其余 stage 的 DMA 都继续受保护；显式调度、同步、其他搬运、
# 未知副作用、源写入或目的访问仍限制前移，不能把“首发无锚点”理解成任意移动。
# 将 V0 也完全冻结会丢掉最后一段 QK 对 V0 DMA 的延迟掩盖；本次修复见第 6 部分第 10 条。
#
# SharedMemoryPlan.execution 消费上述契约，结合真实 DMA、matrix reader、
# task readiness、别名活跃期和原 LDS 预算验证后，才建物理池、落实 handoff 和 DMA 移动。
# 本例 4 个 V stage 使用两个 8192 B 槽，整个 kernel LDS 仍为 16384 B。
# 证明失败时保留普通分配/安全调度；不重新发现没有早期标记的 V 流。
# 因此单独启用 lower-frisk-fragments 的 double-buffer=true 不等价于完整 MyTest 路径。
#
# --cross-bn-k0=1 同时启用 AutoPipeline / execution 的 cross-iteration 选项。
# 早期用 frisk.cross_iteration_plan 标记外层循环，cross_iteration_issue 标记首 DMA，
# cross_iteration_anchor 标记末片 DMA（后两者同样以 frisk. 为前缀）；等待点由 task 恢复。
# execution 的 collectCrossIteration 消费并验证契约，不再自行寻找未标记候选；
# materializeCrossIteration 落实首次/下一 bn 的 K0、原 bounds/step guard、
# loop-carried task、inactive completed task、exit drain 和跨轮生命周期，详见第 32 项。
# 代码：Pipeline/AutoPipeline.cpp；Pipeline/SharedPipelinePlan.h（include 目录）；
# Analysis/SharedMemorySchedule.cpp；FriskToBase/FriskFragIRReorder.cpp。
#
# ---- 4D.3 三种调度入口的区别 --------------------------------------------
#   51 AutoPipeline：按循环 hint 生成轮换流水，或按 double-buffer 选项规划直线 V 流。
#   53 PipelineSchedule：给 pipeline.stage/order，按已有分级和顺序展开流水。
#       stage 表示提前多少级安排，order 表示稳态内发射顺序；生成预热/稳态/收尾。
#       MyTest 的 --pipeline=1 会调用已有 attention 标注逻辑，再运行这个 Pass。
#       它和 AutoPipeline 的 hint 是独立开关，不应当成同一个选项。
#   56 ScheduleJSON / CP-SAT：在 Fragment 阶段建立依赖/资源约束调度图，导出给外部求解器；
#       导入时先校验再应用。它是驱动流程，不能只在 Pass 列表中加一个名字就完成求解。
#       当前在第 10 项之后，替代第 11 项；与 --pipeline=1 互斥，并跳过 AutoPipeline。
#       --cpsat-stop-after=1 可在该阶段结束；否则继续 lowering，并保护导入的顺序。
#
# ---- 4E. Fragment、共享内存和线程代码如何接起来（6~16、57~60、78）---------
#
# 6：把“算整块矩阵”拆成带类型、布局的小片段（fragment），此时仍方便调整计算顺序。
# 28：联合考虑所有读者/写者，选择 LDS 元素排布、打包方式、swizzle 和合法 DMA 宽度。
#     swizzle 是有规则地重排地址，目的是减少多个 lane 同时挤向同一 LDS bank。
# 37：把“等数据准备好”尽量放到第一次真正使用附近，让等待前还能做别的工作。
# 7~11：融合累加器、调整片段顺序、简化小循环、保留不变片段、提前搬运。
# 12：分析共享数据“从何时开始到何时结束仍可能被使用”，不是这里就直接分配物理槽。
# 57：在 fragment/thread 两处报告同时活跃的数据量；用于找寄存器压力峰值。
#     这是估算和诊断，不会自动减少寄存器，也不等于机器码最终的 VGPR 数或 spill 结果。
# 13：将 fragment 计算、归约和布局转换展开为每个线程的向量计算与通信。
# 58：去掉连接 block/线程两种表示的临时桥接；要求线程形状和布局都能对上。
# 15：根据已经展开的真实访问，落实物理 LDS 与任务生命周期；其中：
#     78 先把已选的 packing/swizzle 应用到分配及所有别名访问，保证搬入和读出一致；
#     非 CP-SAT 路径还会重排 async copy、检查循环回边同步，再规划最终存储复用。
#     27/32 消费 AutoPipeline 的 V / 跨轮逻辑计划，做物理验证与落实。
#     两项均关闭时仍要落实基本存储；V 逻辑规划前移不意味着整个 execution 前移。
#     CP-SAT 路径落实导入的存储安排或按最终活跃期分配，并补充必要的交接同步。
# 59：把共享分配、矩阵乘和 block 同步变成后端能接收的形式，并整理 kernel 信息。
# 16：进一步展开/整理线程 IR；是否在展开后重排由 reorder-after-unroll 控制。
#
# 第 60 项归组的标准步骤包括 BufferLoopHoisting、BufferHoisting、LICM、
# AffineLoopNormalize、BufferDeallocationSimplification、LoopFusion、
# AffineScalarReplacement、Mem2Reg，以及转换后反复运行的 canonicalizer / CSE。
# 白话说：把不变的事移出循环、消掉不必要的临时存储、把同一个表达式只算一次。
# 项目实际使用 VectorSafeCanonicalizer：关闭一个已知不安全的向量转换规则，
# 其余规范化仍保留。CSE 是消除重复表达式，不是新的 Attention 算法优化。
#
# ---- 4F. 最后如何变成 LLVM IR（61~77）------------------------------------
#
# 61/62 先落实异步任务与 lane 通信，让 18 能看到其中不随循环变化的准备工作。
# 63/64 是两段 PassManager 流程：先展开 affine，再逐步处理控制流、内存、GPU 和向量。
# 64 内的 65 修正私有地址空间，38/39 清理同步，66 转换通用操作，
# 67 填 GPU 属性，68 选择 ROCm/NVVM 表达，69 完成向量 lowering。
# 这里得到的 LLVM dialect 仍在 MLIR 里；70 才导出真正的 llvm::Module。
# FriskLLVMTranslation 同时说明哪些 MMA 是寄存器计算、哪些锚点必须保留，
# 避免 LLVM 误删或错误移动；这类正确性信息不受 llvm-cleanup=none 影响。
#
# 71/73/74 是可选的 LLVM 预取/复用，受各自开关和预算控制；不能算作默认必经优化。
# 72 保护跨轮预取前的分母求和，23 缩短值需要被保留的时间，以降低寄存器压力。
# 75 按 dma-export 选择 lowerWideAsyncDMAToBufferAsm、splitFullWaveAsyncDMAToDwords
# 或 lowerWideAsyncDMAToHCU 等导出方式，并保护异步搬运源地址。
# 然后做 25 的 late 清理、按需执行 76、再次检查 72，最后由 77 运行常规 O3。
# 输出 .ll 只完成了 LLVM IR 导出；生成 GPU 机器码和运行 kernel 是之后的工作。
#
# ---- 4G. 独立入口与未接入主线的名字，避免误读源码 --------------------------
#
# ConvertFriskBaseToThreadLevelIR：可直接从 block 降到 thread 的另一条路径；
#     MyTest 主线采用第 6 → 13 项的 fragment 中间层，不会再串行运行此直接路径。
# PlanWmmaLayoutConversions、SinkWaitBufferReady、PackSharedMemory 有独立 Pass 入口；
#     主线分别在 55、37、78 所属流程内调用对应实现，不额外再跑一遍独立 Pass。
# FriskCanonicalize / FuseBlockOpWithDTypeConvertOp：不是 MyTest 当前调用的 49；
#     前者 dtype 融合仍未完成，后者有实现但没有接入此主线。
# frisk-layout-infer：Pass 定义已注释；现在应查 LowerInfoAnalysis 和第 5/6 项。
# frisk-overlap：目前是占位；不能把它算作已经实现的流水优化。
# ExecutionOpInterface：操作共同提供的执行语义查询，辅助判断能否移动/融合，
#     是基础设施而非单独 Pass；更详细说明见本部分末尾。
#
# 源码索引（以下相对 3rd/deepgengraph/）：
#   实际执行顺序：test/test_src.cpp；lib/Conversion/ConvertToLLVM/LowerPipe.cpp。
#   LowerInfo：include/deepgengraph/Analysis/LowerInfo.h；lib/Analysis/LowerInfo.cpp。
#   布局与线程转换：lib/Conversion/FriskToBase/FriskBaseToTheadlevelIR_plan2.cpp。
#   自动/手动/外部调度：lib/Pipeline/{AutoPipeline,PipelineSchedule,ScheduleJSON}.cpp。
#   存储落实：lib/Conversion/FriskToBase/{FriskFragIRReorder,PackSharedMemory}.cpp。
#
# MyTest 默认值（不是 ThreadTilingTest 独立 Pass 的默认值）：
#   retain-fragment-words=0；pv-interleave=0；v-double-buffer=0；k-dma=128；
#   qk-interleave=0；pv-guard-group=-1；qk-guard-group=2；qk-prefetch-words=32。
#   k-lds-prefetch=0；k-tail-prefetch=0；softmax-fragments=off；cross-bn-k0=0；
#   mask-add-bypass=off；local-reduce=off；defer-lane-sum=off；causal-mask=off；
#   lazy-softmax-base=off；softmax-cvt-pack=off；math=strict；llvm-cleanup=both；llvm-opt=3；llvm-cleanup=both。
#   simplify-loop-control / simplify-bounded-affine / hoist-lds-bases /
#   hoist-lane-shuffles / coalesce-waits / elide-backedge-barrier 默认为 1。
#   local-reduce、defer-lane-sum 支持 off|legal|approx；mask-add-bypass 仅 off|approx。
#   mask bypass / local-reduce / defer-lane-sum 的 approx 均要求 math=approx；
#   math=mmac-recip 开启倒数优化，不许可这些 approx 改写。
#   softmax-cvt-pack=rtz 还要求 native exp2 路径（math=approx 或 mmac-exp2）；
#   strict / mmac / mmac-recip 下不改写，也不依赖 --register-lifetimes 开启。
#   未设置 FRISK_GM_TO_SHM_MODE 时通常为 legacy；启用 CP-SAT 时默认 async。
#   本说明的 DMA 调度要求显式选 async，不能把表中调优值当作裸 MyTest 默认行为。
#
# MyTest 已支持 --名称=值；旧位置参数及 <input> <output.ll> 简写仍兼容。
#   名称：input / pipeline / output / reorder / pack-shared / wmma-layout /
#   math / threads / block-warps / register-lifetimes / uniform-masks /
#   prefetch-packets / fuse-mask-add / lds-reuse-words / dma-pack / dma-export /
#   qk-tile / vo-tile。例：--wmma-layout=b-permute 等价于旧值 3.3。
#   命名/位置参数混用时，位置编号不因命名项而前移；同一项重复指定会报错。
#   未指定 --threads 时保留输入线程属性；--qk-tile=0 / --vo-tile=0 各自选择自动值。
#   命名参数本身不改变优化或默认值；完整例见 causal_kv_shm_reuse_20261119/command.sh。
#
# MyTest 的三种流水入口：
#   AutoPipeline（启发式路径，第 4/5 项之间）：frisk.pipeline_hint=N（N≥2，unit 取 2）
#     在可分析的 affine.for 上生成环形预取，并在出口等待所有 FIFO task。另有独立的
#     double-buffer / cross-iteration 开关；27/32 可在没有 hint 的循环上规划。
#   手动 --pipeline=1：第 40 项后标注 stage/order，第 5 项及 CSE 后由 PipelineSchedule
#     展开序言、稳态、收尾。AutoPipeline 尊重已有调度标记。
#   CP-SAT（--pipeline=0）：--cpsat-export / --cpsat-baseline / --cpsat-import 任一启用，
#     关闭 AutoPipeline。第 7 项后仍跑 fragment 子流水的规范化、可选 9/10；
#     跳过 8、29/31、11。在第 12 项前设置 cpsat.preserve_schedule，调用 processScheduleJSON，
#     展开所选外循环内的静态循环，在 FragmentIR 导出问题/基线或校验并应用外部解。
#     --cpsat-stop-after=1 到此为止。继续 lowering 时，15 依据解里的存储偏移或实际
#     生命周期落实 LDS 与交接同步；禁用 16 的展开后重排以及 ThreadIR 的 21/22/42/26/41/30/33/34。
#     secondLowering 仍执行 38，跳过 39。LLVM 在开启 23 时走保序的索引简化和地址重算，
#     跳过启发式预取、两次 25、分母锚定和最终 O3。
# LLVM 可选预取参数：prefetchGlobalPackets 用 --prefetch-packets=1，reuseLDSFragments 用
# --lds-reuse-words=N。--frag-prefetch=1 开启 prefetchMMACFragments，--frag-prefetch-words=N 设预算。
# 可选分支补充（已列入 PIPELINE；按各自开关启用，不与主线混作默认链）：
#   AutoPipeline：在第 4 项之后、第 5 项之前运行。循环路径读取 frisk.pipeline_hint=N，
#     生成环形 tile 预取；无 hint 不走循环改写，与旧位置参数 pipeline=0|1 独立。
#     本例 --v-double-buffer=1 启用另一条直线 V 流逻辑规划路径，不依赖循环 hint。
#     以下关于 FIFO 和出口的说明仅针对循环 hint 路径：
#     允许内部全为无内存副作用操作的 causal MaskOp；有副作用/未知嵌套操作仍拒绝。
#     退出时显式 wait 所有 FIFO task，闭合生命周期证明，使 Q 暂存可与后续 KV 复用。
#     N 个未来/当前槽仍跨迭代互相冲突；不能仅凭 completed token 省略出口交接。
#   ScheduleJSON / CP-SAT：--cpsat-export=graph.json / --cpsat-import=solution.json
#     或 --cpsat-baseline=baseline.json，在第 10 项后处理 typed Fragment 调度图。
#     求解器在外部生成 JSON，编译器校验后导入；启用时跳过 AutoPipeline、第 8/11 项，
#     以及会改变既定顺序的多项 Thread/LLVM 调度、两轮 25 和最终 O3。
#     LowerFriskFragments 仍落实存储，但关闭其双缓冲/跨轮预取选项；不再次另排一套流水。
#     secondLowering 仍做局部重复同步合并（38），跳过 BarrierOptimize（39）。
#   PipelineSchedule：--pipeline=1（也兼容旧位置参数）时，按 pipeline.stage/order 展开
#     序言、稳态、收尾；在第 6 项之前，与 cpsat 启用模式互斥。
#   prefetchGlobalPackets：LLVM 中、25 首次尝试之后、23 之前（--prefetch-packets=1）。
#   reuseLDSFragments：LLVM 中、23 之后，由 --lds-reuse-words=N 或旧位置参数开启。
#   prefetchMMACFragments：随后由 --frag-prefetch=1 / --frag-prefetch-words=N 开启。
# 具体选项与约束以 test_src.cpp、lib/Pipeline/ScheduleJSON.cpp 为准。

# 已前移 / 合并关系（旧名字仅用于查历史，不是当前 MyTest 的执行入口）：
#   optimizeApproximateRegisterMath → 19 + native exp2 lowering。
#   bypassMaskAdds → 20；optimizeLocalReductions → 21；deferLaneSums → 22。
#   canonicalizeLDSGEPs → 24；interleavePVAccumulators → 26。
#   doubleBufferV → 51 内的 27 逻辑计划 + 15 内的 27 物理落实；独立 Thread V Pass 已删除。
#   narrowKDMA → 28（已删除；--k-dma=64-swizzle 兼容翻译为 argument:8:9:3 约束）。
#   prefetchPairedLDS / prefetchPairedLDSTail → 29/31 的逻辑窗口 + 42 的物理配对。
#   scheduleSoftmaxFragments → 30。prefetchNextK → 32 规划、15 落实，LLVM 保留 anchorNextKSums。
#   lazySoftmaxBase → 34；softmaxCvtPack → 35（ThreadIR 选择 + ROCm lowering）。
#   elideBackedgeBarrier → 33（ThreadIR，最终 LDS 调度之后、async lowering 之前）。
#   fuseMMACAccumulators → 36 的 block 融合；25 改由 cleanupLLVMInstructions 清理。
#   cleanupRegisterMMA → prepareLLVMExportContracts + cleanupLLVMInstructions，
#     只保留为 LLVM-only 兼容组合入口；prepareRegisterMMAForLLVM 是前者的旧名别名。
#     MyTest / 库导出直接在 FriskLLVMTranslation 中补齐契约，不再后置扫描 Module。
#   LegacyLLVMExportTest 仍保留部分历史转换用于回归，且拒绝已删除的调度选项；
#   第 33/34/35 项也拒绝在旧 LLVM 入口重放，需使用 MyTest / ThreadTilingTest。
#   不能把旧 .ll 重放入口等同于当前 MyTest 流水线。

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

# 上述接口方向的当前落地：ExecutionOpInterface（基础设施，不新增优化编号）
#   Block / Fragment / Thread 的 Frisk op 共用执行语义查询：scope、kind、registerOnly、
#   collective、schedulingBoundary，由 op/类型/已验证属性推导；未知 op 保守视为边界。
#   registerOnly 表示 SSA 寄存器计算，不保证物理寄存器分配无 spill。
#   普通局部移动同时检查执行语义、内存效应和可投机执行性；MMA、跨 lane reduce、
#   布局转换、DMA、wait/barrier 仍受集合参与及调度边界约束。无内存效果不等于任意可移动。
#   算术/cast 只对具体 memref operand 标记 Read；scalar/vector operand 不产生内存读取。
#   matrix read 去掉重复的泛化读效应，仍保留 source 上的读取及 collective 约束。
#   26/41 的 MMA 识别共用 ThreadScheduleUtils，不依赖 inline asm 的 has_side_effects；
#   MLIR 阶段保留调度锚点，到 LLVM translation 才对严格匹配的寄存器 MMA 清除 sideeffect。
#   这提供通用合法性查询，不代表所有优化已统一成可任意组合的接口调度器。
#   源码：include/deepgengraph/Dialect/Frisk/IR/FriskInterfaces.{h,td}、
#         lib/Dialect/Frisk/IR/FriskInterfaces.cpp（均在 3rd/deepgengraph 下）。


# 第 40 项：TileAttention（frisk-tile-attention）
#   输入：每个 key tile 一次 full-D QK、一次 softmax、一次 full-D PV；O/L/M 共享。
#   输出：QK 用 qk_tile 沿归约轴切 Q/K，PV 用 vo_tile 沿输出列切 V/O。
#   切片沿 SSA producer 图推到 global buffer_view、copy 和 alloc_buffer，
#   因而 K/V 的 128x128 FP16 暂存可变为 32x128 / 128x32（8 KiB），
#   而不是在 32 KiB 的原分配上只加 reader view。输出分片共享同一份 P 和 L/M。
#   分块不减少总乘加次数，也不保证降低全 Q 或全 O 常驻寄存器总量；
#   它改变 LDS 暂存大小、搬运批次和可调度的片段。16x16x16 MMAC 不强制 BK=32。
#   strict 保留原 QK 分组（QK=D）；切 QK 需要 allow-reassociation。
#   新 QK 循环默认带预算展开标记；frisk.qk_simplify_loop=false 可关闭，
#   frisk.qk_pipeline_hint 传给循环的 frisk.pipeline_hint，TuneAttention 可独立控制。
#   运行在 ConvertFriskToBase 后、两次 FriskOpFuse 之间，早于布局/DMA/流水规划。
#   分块前融合得到的 alpha*O 显式累加器随 V/O 一起切片，不需要跨过下一片 V 的写入再融合。
#   源码：3rd/deepgengraph/lib/Conversion/FriskToBase/TileAttention.cpp。

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
    # 若该循环仍在 FragmentIR 且满足展开条件，会得到 4 段直线代码；27/32 也可在 Base 提前展开它。
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
# 当前阶段：fragment IR。第 37 项先在 block tiling 内推迟逻辑 readiness，
# 本项再结合 fragment 读写和任务完成做保守调度；第 15 项最终按物理执行计划重算生命周期。


# ---- 14. sinkLoopLayoutConversions：布局转换移出循环（第 6 项内部）-----------
# 历史实测（causal，与第 25 项合计）：1.449 -> 1.243 ms，smem 9.22M -> 5.16M。
# 当前在 block IR 的布局推断和转换插入之后、shared planning 和 fragment tiling 之前。

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
# WMMA producer 布局还支持 3.1/shared、3.2/shuffle、3.3/b-permute、
# 3.4/inner-inst-b-permute；3.4 在每条指令内排列 B 列，形状不支持时回退到 shuffle。
# terminal C 输出保留 shuffle。planWmmaLayoutConversions 在 tiling 内生成可序列化
# shuffle plan，不能把 producer 的布局选择当作额外 LLVM 调度 Pass。


# ---- 15. LowerFriskFragments：联合规划 LDS 槽位、生命周期与执行时序 -----------
# 物理布局 / DMA 映射已由第 28 项确定；这里规划 pool offset、发射点及同步。
# 第 27/32 项在此消费早期 V / 跨轮 K0 计划并做物理验证；
# 两项均不在 LLVM 再改槽位或移动 DMA。

def lds_plan_before():
    # 12 个逻辑分配，每个独立的话共需 98304 字节，远超 16 KiB
    q0, q1, q2, q3 = alloc(8192), alloc(8192), alloc(8192), alloc(8192)
    k0, k1, k2, k3 = alloc(8192), alloc(8192), alloc(8192), alloc(8192)
    v0, v1, v2, v3 = alloc(8192), alloc(8192), alloc(8192), alloc(8192)

def lds_plan_after():
    # 本配置：消费 AutoPipeline 的 V 与跨轮 K0 计划；
    # 按拟定发射顺序投影生命周期，再分配物理偏移
    # 只有互不干扰的生命周期能复用；先只读规划、验证输入快照，再统一落实
    slot0, slot1 = LDS[0:8192], LDS[8192:16384]
    k0, k2 = slot0, slot0                         # K 沿 DK 交替，当前片读、下一片写，互不覆盖
    k1, k3 = slot1, slot1
    v0, v2 = slot0, slot0                        # 开启 27 时，V 槽位在同一规划中确定
    v1, v3 = slot1, slot1
    # Q 在 BN 循环前就读完了（第 10 项），它占用的空间也复用同一个池
    # XOR / DMA 宽度沿用第 28 项发布的方案，本阶段不再重新选 swizzle

# 限制：双缓冲 / 跨轮计划须满足原 LDS 预算、所有别名读写及 collective handoff 证明。
# 不能满足时保留普通调度；选项开启不等于实际命中。展开 fragment 循环后再做后续 Thread 调度。
# 代码：FriskToBase/FriskFragIRReorder.cpp 中的 LowerFriskFragments 调用
#   lib/Analysis/SharedMemorySchedule.cpp 的 planSharedMemoryExecution / materializeExecution。


# ---- 17. SimplifyBoundedAffine：用循环范围消掉除法【仅组合实测】---------------

def simplify_affine_before(iv, s0, s1):             # iv 在 [0, 32) 内
    return ((iv + s0 * 4096 + s1 * 131072 + 1024) // 4096) % 32
    # 编译器不知道 iv < 32，这个有符号除法和取模会一路留到底层

def simplify_affine_after(iv, s0, s1):
    # iv + 1024 < 4096，所以除以 4096 后这部分贡献为 0；131072 = 32*4096，mod 32 后消失
    return s0 % 32
    # 结果不依赖 iv，于是可以被 LICM 提到循环外

# 预期（推断）：减少每轮的地址计算。只省地址计算，预计收益有限。
# 限制：只改能证明范围的 affine.apply，纯线程坐标表达式仍刻意不折叠。
# 当前线程范围统一由 ThreadBounds 查询 gpu.thread_id 的 range / 函数 thread_num；
# 原 exposeThreadBounds 已删除，不再把额外 LLVM 范围暴露作为必需优化步骤。


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

# 当前在 async / shuffle 局部 lowering 后、firstLowering 前运行，循环仍是 affine/scf。
# 为什么普通 LICM 做不到：readfirstlane 依赖 exec 掩码，需额外证明控制流 uniform。
# 本项证明循环和跨越的条件分支是 block-uniform，再外提基址；固定全 1 mask 的
# mbcnt 提供物理 lane 号。已有 LLVM CFG 的旧两块计数循环路径仍作为兼容路径保留。
# 预期（推断）：每轮少一些 ALU 指令。
# 代价：索引跨循环常驻 VGPR。K0 路径曾因 VGPR 到顶发生 spill，实测先看 scratch。


# ---- 19. ApproximateThreadMath：Thread IR 倒数复用 + native exp2 lowering ------

def approx_math_before(x, L):
    return libm_exp2(x), a / L, b / L, c / L, d / L

def approx_math_after(x, L):
    inverse = 1.0 / L                                # 共享分母的除法只算一次倒数
    return amdgcn_exp2(x), a * inverse, b * inverse, c * inverse, d * inverse

# 上面合并展示两件事：本 Pass 只把 arith.divf 改为共享倒数的乘法；
# native exp2 在已有 math.exp2 → rocdl.exp2 lowering 中由 approx / mmac-exp2 选择。
# MyTest 在 math=approx / mmac-recip 下运行本 Pass，位置在展开和 SSA 清理之后。
# 同一基本块、相同 SSA 分母、FP32 标量/向量才复用；常量分母和 strictfp/optnone 跳过。
# 只有倒数获 arcp，乘法不额外获得 reassoc/nnan/ninf 等假设。
# 省了什么：空转。一次倒数代替多次除法；近似倒数改变数值约定。
# 代码：ConvertToLLVM/ApproximateThreadMath.cpp；记录：generated_ir/thread_approximate_math/README.md。


# ---- 20. ThreadMaskAddBypass：causal 完全可见分支跳过 S+0（无 mask 时无效果）--

def mask_bypass_before(S):
    mask = 0 if fully_visible else causal_mask()
    return S + mask                                  # 完全可见时白白加一次 0

def mask_bypass_after(S):
    return S if fully_visible else S + causal_mask()

# 当前处理 Thread SSA 中的 Frisk mask、单结果 scf.if / arith.select，
# 支持静态 extract/shuffle/broadcast 等 vector 包装及多个 add 消费者。
# 分支形式把 add 放进 masked 分支；select 形式保留 select，不保证硬件跳过 add。
# 限制：--mask-add-bypass 仅 off|approx；允许 signed-zero 差异，不保证 NaN 位模式。
# 动态索引、逃逸消费者、内存/同步边界、不能证明 clear lane 为 +0 时不改写。
# 代码：ConvertToLLVM/ThreadMaskAddBypass.cpp；记录：generated_ir/thread_mask_add_bypass/README.md。

# ---- 21. ThreadReductions 第一阶段：局部四链归约 -------------------------------

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
# 当前在 Thread 展开后匹配带 frisk.try_multichain 的同块标量链，不任意重排所有算术。
# 支持 add/mul/minnum/maxnum，默认 chains=4、min-inputs=16；上限和保护函数按源码检查。
# legal 模式需原节点提供相应 fast-math 许可；approx 模式需显式开启 math=approx。
# 21/22 共用一个函数 Pass，两个开关独立。MyTest 在两者均关闭或 CP-SAT 时跳过；
# 开启 22 时，在 Pass 之后运行 VectorSafeCanonicalizer 和 CSE。两个旧独立入口仍可用。
# ThreadTilingTest 可用 --frisk-thread-reductions='local-reduce=legal defer-lane-sum=approx wave-size=64'。
# 两阶段都开启时，MyTest 只在第二阶段清理后输出一次归约 IR，仍分别打印两个阶段的统计。
# 代码：ConvertToLLVM/ThreadLocalReduction.cpp。


# ---- 22. ThreadReductions 第二阶段：分母的跨 lane 求和延后 ---------------------

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
# 当前在 affine/scf 循环和显式 gpu.shuffle XOR 仍保留时证明零初值递推、
# 完整 lane sum 和 wave-uniform 缩放；在 42 落实预取 guard、26 交错 PV 之前执行。
# MyTest 要求 wave-size=64；独立 Pass 可处理显式声明的 32/64，不能只由 shuffle width 猜 ABI。
# legal 与 approx 保留不同数值许可；浮点分配律不是任意浮点输入的逐位等价证明。
# 代码：ConvertToLLVM/ThreadDeferLaneSum.cpp。


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
# 本项仍在 LLVM。若 Thread softmax 调度已有标记，则保留该顺序；
# 第 35 项已经在 ThreadIR 选好转换，转换两端 identity 由 ROCm lowering 生成，
# FriskLLVMTranslation 在翻译时补齐 memory(none)、convergent、nounwind、willreturn
# 和 !frisk.register_lifetimes；锚点保留 sideeffect，不再依赖后置 Module 扫描。
# 不再要求本项先生成 identity 才能匹配第 35 项；第 32 项的分母锚定仍须在本项之前完成。
# lazy 慢路径引入纯计算分支后，地址重算器会在合法汇合点复用支配它的 header 地址缓存，
# 避免快路径每轮重算公共 DMA/LDS 地址；内存/副作用/汇合约束不满足时不复用。
# 验证记录：generated_ir/thread_lazy_softmax_base_fix_20261118/README.txt。


# ---- 24. CanonicalizeThreadLDSAddresses：公共地址锚点 + 常量偏移 ---------------

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
# 上述数字来自历史 LLVM GEP 版本；当前在 packing、槽位分配和展开之后证明
# Thread LDS load 的 alignment 与常量差，记录为 frisk.lds_load 的 storage/anchor/offset。
# 使用符号 affine 差或完整线程范围的精确求值；未知地址不采样、不假定相邻。
# LLVM lowering 机械生成带对齐的公共 GEP，后端仍决定是否选 ds_read2_b64。
# 预取配对和 QK 交错需要这些地址。首次请求执行完整证明，以后的请求检查当前 IR：
# 若仍存在地址空间 3 的 memref.load、vector.load 或 affine.load，则重新运行完整 Pass。
# 代码：ConvertToLLVM/CanonicalizeThreadLDSAddresses.cpp。


# ---- 25. cleanupLLVMInstructions：通用清理与导出契约分离 ---------------------
# 整数表达式示意，涵盖 late 的常量分支与指令折叠；不表示浮点 add 可以任意删掉。

def llvm_cleanup_before(x):
    y = x + 0                                      # x 为 i32
    if True:
        return y
    return 0

def llvm_cleanup_after(x):
    return x

# 省了什么：可证明冗余的指令和常量控制流；GPU 收益需单独测量。
# early：translateToLLVM 后、预取及生命周期优化前，连续两次 InstCombine。
# late：DMA 导出改写后，SimplifyCFG + 连续两次 InstCombine。
# --llvm-cleanup=both/early/late/none 分别选择两轮/仅早期/仅晚期/均关闭，默认 both；
# math=strict 跳过两轮。它不控制 --llvm-opt，也不关闭其他专用 Pass 内部的清理。
# 本项不匹配 MMA 来推导属性，不做累加融合；完整 PV 融合见 36，剩余 fragment add 见 7。
#
# 导出契约由 FriskLLVMTranslation 独立完成，在 strict / cleanup=none 下也生效：
#   frisk.register_mma：校验指令、寄存器约束和 v4f16/v4f32 ABI，清除 asm sideeffect，
#     设置 memory(none)、nounwind、willreturn、convergent。
#   frisk.register_anchor：校验空 asm、=v,0、单参数、同型 v4f32/v4f16 输入输出，
#     补齐同样的四个属性及 !frisk.register_lifetimes，但保留 sideeffect 固定转换边界。
#   标记不能代替结构验证，非法标记报错；不把任意 asm 当成无内存效果。
#   无 Frisk translation 扩展的外部导出仍保守；LLVM-only 兼容输入可显式执行
#     prepareLLVMExportContracts，旧 cleanupRegisterMMA 是它与通用清理的组合。
# 源码：ConvertToLLVM/{CleanupRegisterMMA,FriskLLVMTranslation,TranslateToLLVM}.cpp。
# 回归：test/unittest/check_llvm_cleanup_contracts.py、check_register_anchor_export.py。
# 同配置四模式的静态计数与输出一致性见第 6 部分；不能按指令数直接选最快模式。


# ---- 26. ThreadInterleavePV：PV 四链交错 ---------------------------------------

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
# 当前在 Thread IR 识别寄存器 MMAC（Frisk op 或校验过的 inline asm）及 B 的
# matrix-read packet，先建完整拓扑计划，再交错独立链；不改变各链 K 顺序或 FP 属性。
# 等深链允许来自不同 matrix packet（如 causal 的不同输出列）；所有 packet 必须已在
# 候选 MMAC 之前，不提前内存读取。合并分组不合法时再尝试相同有序 packet 的小组。
# 动态 extract 的 SSA 值原样保留，不依赖 LLVM 先将其折叠为固定低/高半片。
# 0 关闭，1 是串行调度对照，2/4 是交错宽度；guard 独立由 --pv-guard-group 控制。
# guard=-1（默认）在每条 MMAC 前后加 sched.barrier(0)；0 不加；1..32 每 N 条后加一次，
# 包含不足 N 条的末组，不加首部 guard。这是编译器调度提示，不是 GPU 同步 barrier。
# 内存/同步原序、SSA 依赖和矩阵 packet 必须可证明；任意 asm / 未知效果不参与。
# 第 42 项先完成 LDS 配对。本项之后按需执行 24 → 41、30、33，再 lower async/GPU。
# 代码：ConvertToLLVM/ThreadInterleaveMMAChains.cpp；说明：ConvertToLLVM/PVInterleave.md。


# ---- 27. V 双缓冲：AutoPipeline 逻辑计划 + execution 物理落实 -----------------
# 历史实测：1.822 -> 1.768 ms；不代表本次迁移或 V0 修复的单项收益。
# 当前第 51 项先写 stage / slot / issue-after 契约，第 15 项再验证并落实物理分配、
# handoff 和最终发射点，细节见 4D.2。跨 bn K0 的逻辑规划也已随后前移。

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
    # 进入本段前，V0 首次 DMA 可在安全点提前发出，由最后一段 QK 掩盖延迟；
    # 它的 wait 仍留在消费点。下面只展示 V stage 之间的双缓冲顺序。
    for s in range(4):
        wait_and_barrier(v_task)                     # 这个 barrier 同时释放另一个 slot
        if s < 3:
            v_task = dma_V(s + 1, dst=slot[(s + 1) % 2])  # 第 0 条 MMAC 之前就发
        packets = read(slot[s % 2])
        mmac_x32(packets)

# 历史单缓冲 → 双缓冲对照省了什么（本次 V0 修复未减少 barrier 总数）：
#   等同步：每个 key tile 少 3 个 barrier。
#   等数据：DMA 现在能覆盖当前片的 LDS 读和全部 32 条 MMAC，原来只覆盖 28 条。
#   推断主要来自去 barrier：单缓冲下调整 DMA 时机（after2/4/8）几乎没有收益。
# 当前限制：候选 matrix-copy stage 的别名、完成事件和所有读者须完整可追踪，
# 地址须证明合法的 4/8/16 B wave transfer，控制流 uniform；在原 LDS 预算内选择两个槽。
# 相邻阶段按更早发射时间形成干扰，隔两阶段可复用；还须检查与其他 Q/K 缓冲的冲突。
# 无法规划时保留普通分配和调度，不在 LLVM 另找备用槽；落实前再次核验输入快照。
# 代码：Pipeline/AutoPipeline.cpp；Analysis/SharedMemorySchedule.cpp；
# FriskToBase/FriskFragIRReorder.cpp；说明：FriskToBase/VDoubleBuffer.md。


# ---- 28. SharedMemoryPlan.layout：联合选择 packing / DMA / swizzle ------------
# 历史 K 优化实测：bank 冲突 15.90% -> 0.125%，smem 不变，1.768 -> 1.742 ms。
# 当前在第 6 项的布局推断后收集 owned allocation 的全部访问和别名，统一选择
# pack axis、XOR 组合、DMA 宽度、参与线程与源/目的映射，再把唯一方案发布给所有访问。

def k_dma_before():
    # 一个 lane 一次搬 16 B，写到连续地址
    buffer_load_dwordx4(src=k_src(lane), lds=m0 + 16 * lane)

def k_dma_after():
    # 规划直接生成 8 B 包及其源/目的映射，不在 LLVM 把 16 B 指令二次拆分
    for packet in planned_8B_packets(lane):
        buffer_load_dwordx2(src=packet.src, lds=packet.dst)

def k_address_swizzle(a):
    # 读写都走这个映射。按 1014.md 的模型（32 bank，每 bank 4 B），bank 只由地址的 bit2~6 决定，
    # 所以只有高位不同、低位相同的两个地址会撞 bank。把高位异或到低位，就把它们分开了。
    a = a ^ (((a >> 8) & 1) << 6)                    # --swizzle-k=8:6（第 6、15 项已做）
    a = a ^ (((a >> 9) & 1) << 3)                    # 通用约束 1:8:9:3 的第二级 XOR
    return a

# 省了什么：等数据。同时读的 lane 不再撞 bank。
# 为什么要拆 64-bit：16 B 的包内不能做 bit3 的置换，8 B 的包才行。
# 实测对照：只拆不 swizzle，冲突仍是 15.20%（L243），所以消除冲突靠的是 swizzle。
# 限制：Q/V 仍是 128-bit。冲突几乎清零却只提速 1.5%，代价与计数不成比例。
# 上述 Q/V 宽度属于本历史配置，不是规划器只支持 K 的限制；通用规划不依赖
# attention 名称或固定参数位置。--shared-copy-constraints=argument:bytes[:xor-src:xor-dst]
# 显式绑定 ABI 参数；旧 --k-dma=64-swizzle --k-dma-arg=1 翻译为 1:8:9:3。
# swizzle-k=8:6 与它组合；8 B 包内可保留 bit3 XOR，读写必须使用相同物理映射。
# 代码：Analysis/SharedMemoryPlan.cpp、FriskToBase/FriskBaseToTheadlevelIR_plan2.cpp；
# 记录：generated_ir/shared_memory_plan/README.txt。NarrowKDMA 已删除。


# ---- 29/42. PrefetchFragmentLDS 提前读取，ThreadPrefetchPairedLDS 落实物理读对 --

def paired_prefetch_before():
    unrelated_mmac_1()
    unrelated_mmac_2()
    b0 = load_b64(base + 0)                          # 紧贴着消费者才读
    b1 = load_b64(base + 8)
    mmac(a, b0, c)                                   # 马上要用，只能干等
    mmac(a2, b1, c2)

def paired_prefetch_after():
    b0 = load_b64(base + 0)                          # 第 29 项已在 fragment 层提前了逻辑读取
    b1 = load_b64(base + 8)                          # 第 42 项在合法窗口配对，供后端选择 read2_b64
    sched_barrier_0()                                # 保护选定的 MMA 锚点
    unrelated_mmac_1()                               # load 在路上时 MMAC 照常执行
    sched_barrier_0()
    unrelated_mmac_2()
    mmac(a, b0, c)                                   # 到这里数据已经到了
    mmac(a2, b1, c2)

# 省了什么：等数据。MMAC 顺序不变。
# 限制：不跨 DMA、wait、barrier、基本块。距离 8 是上限，不是预取 8 个 tile。
# FragmentIR 可先展开小型静态 MMA 循环、折叠坐标并去重。共享 load_frag 的全部用途须为
# 同块 gemm_frag 的 A/B 输入。按首个消费者和可移动的地址依赖选发射位置，记录
# frisk.lds_prefetch_anchor。新累加链的 MMA 0 前最多额外预读下一条 MMA 的操作数；
# 可追溯到前序 MMA 的累加器保留完整窗口。第 8 项的一级 lookahead 独立运行。
# ThreadIR 在第 24 项证明物理地址后用 window=fragment distance=0 配对可编码的 b64 读，
# 可以跨过独立 MMA 补齐读对，边界是较早读取的首个消费者。仍有读取的锚点保留 guard；
# 后端决定是否生成 ds_read2_b64。第 42 项在 PV/QK 链交错之前完成。
# 代码：FriskToBase/PrefetchFragmentLDS.cpp、ConvertToLLVM/ThreadPrefetchPairedLDS.cpp。


# ---- 30. ThreadScheduleSoftmaxFragments：softmax 与 PV 交错 ---------------------

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
# 当前在 Thread 中匹配 arith.subf → math.exp2 → arith.truncf 及真实 MMAC A 使用，
# 不依赖第 23 项的 LLVM identity；保留既有 FP32 sum DAG，并在输入就绪时调度
# broadcast / shape_cast / vector add 等包装，避免它们把整行 exp2 强迫留在 PV 之前。
# 模式 off|fragment|pv 默认 off；不跨未知效果或嵌套 region 的使用。
# 记录已有 softmax schedule，供第 23 项保持顺序；第 35 项随后只替换转换边，
# 其两端生命周期锚点由 ROCm lowering 生成，不依赖第 23 项提供匹配模式。
# 代码：ConvertToLLVM/ThreadScheduleSoftmaxFragments.cpp。


# ---- 31. PrefetchFragmentLDS：局部归约窗口 -------------------------------------

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
    sched_barrier_0()
    x = local_max(y)                                 # 归约自身不动
    z = local_sum(w)
    mmac_2(b0, b1)

# 省了什么：等数据，补上第 29 项够不到的区域。
# 历史实测：距离 2/4/8 为 1.693/1.691/1.693 ms。
# 与第 29 项在同一次 FragmentIR Pass 中执行。若窗口末端的 reduce_frag 消费本窗口的 MMA
# 结果，普通扫描先取两种距离的较小值；随后归约扫描允许跨 lane-local 归约。跨 lane 归约、
# DMA、wait 和写入仍划分窗口。物理读对及 guard 与第 29 项一起交给第 42 项。
# 代码：FriskToBase/PrefetchFragmentLDS.cpp 的 runWindow(distance, false) / runWindow(reductionDistance, true)。


# ---- 32. SharedMemoryPlan.execution：跨轮首 DMA 预取 + LLVM 分母锚定 ------------
# --cross-bn-k0=1 同时映射为 AutoPipeline / lower-frisk-fragments 的 cross-iteration=true。
# 逻辑候选与锚点已前移到 AutoPipeline；SharedMemoryPlan.execution 消费契约并验证物理可行性。
# 调用链：planSharedMemoryExecution → collectCrossIteration，验证后由
# materializeCrossIteration 落实；首次发射、跨轮生命周期、槽位和 handoff
# 与第 27 项的物理计划协调。MyTest 不在 LLVM 重试预取。

def next_k_before():
    for n in tiles:
        k_task = dma_K(n, 0, dst=slot0)              # 每轮开头才发，只能干等
        wait_and_barrier(k_task)
        qk_and_pv(n)
        reader_barrier()

def next_k_after():
    k_task = async_task_completed()
    if has_iterations():                            # 原上下界/步长证明，零次循环不读 GM
        reader_barrier()                            # 首轮槽位 handoff
        k_task = dma_K(first_tile(), 0, dst=slot0)
    for n in tiles:
        wait_and_barrier(k_task)                     # 数据大概率已经到了
        qk_and_pv_except_last_slice(n)
        wait_and_barrier(v3_task)                    # V3 只读 slot1，slot0 已空
        if next_tile_is_in_original_bounds(n):
            k_task = dma_K(next_tile(n), 0, dst=slot0)
        else:
            k_task = async_task_completed()         # inactive path 仍有合法任务 token
        last_pv_slice(n)                             # 这段计算掩盖 K0 的延迟
        reader_barrier()
    wait_and_barrier(k_task)                         # 显式 exit drain，保护退出后的 LDS 使用

def sum_anchor_why():
    # 上面的改写新增了一个 if 分支。LLVM 发现分母加法的结果只在后面用，
    # 就把 64 条加法全沉到分支之后，导致输入一直存活：VGPR 256、scratch 20，
    # 反而变慢（1.752 ms）。所以每条加法后面要加一个 identity 把它钉在原地：
    s = a + b
    s = asm_identity(s)                              # tied-register 空 asm，不生成指令
    return s

# 省了什么：等数据。加锚定后 1.691 -> 1.663 ms。
# 历史 CPU 模型记录 k0_overlap_mmac = 992，不能当作当前所有形状的 overlap 数。
# 当前 task token 跨 affine/scf loop 传递；下一轮地址替换原 IV，guard 沿用原 bounds/step。
# 生命周期包括循环前缀的当前读者，以及预取发射点到回边的后缀；最后 PV 读区必须与它分离。
# 未知内存效果、全局写、非 uniform bound、不可用源地址或冲突布局都会保守回退。
# LLVM 的 anchorNextKSums 只为 shared-plan-v1 标记的递推加法加锚，且在 23 前完成；
# 分母锚定不选择槽位、不移动 DMA。LegacyLLVMExportTest 的旧 PrefetchNextKPass 仅保留作兼容。
# 代码：Analysis/SharedMemorySchedule.cpp；ConvertToLLVM/PrefetchNextK.cpp（锚定 / 旧路径）。


# ---- 33. ThreadElideBackedgeBarrier：删除循环尾 barrier【历史组合实测】--------
# 位置：第 30 项之后、async/GPU 局部 lowering 之前。

def elide_barrier_before():
    k_task = issue_first_k()
    for n in tiles:
        wait_and_barrier(k_task)                     # 第二个 barrier
        k_task = qk_and_pv_with_next_k(n)             # 已由 32 证明下一轮 DMA 槽安全
        reader_barrier()                             # 第一个 barrier，之后没有任何 LDS 访问
    wait_and_barrier(k_task)                         # 匹配循环结果的 exit drain

def elide_barrier_after():
    k_task = issue_first_k()
    for n in tiles:
        wait_and_barrier(k_task)                     # 下一轮开头这个 barrier 已能保证同样的顺序
        k_task = qk_and_pv_with_next_k(n)
                                                     # 循环尾的 barrier 删除
    wait_and_barrier(k_task)                         # 最后一轮/零轮退出的保护仍保留

# 预期（推断）：每轮少一次全 wave 同步。两个 barrier 本来就相邻，主要省 barrier 自身的开销。
# 历史 CPU 模型：barrier 次数 293 -> 261（每轮少 1 次，共 32 次），输出与基线逐位相同。
# 限制：必须在第 32 项之后。之前 K0 的 DMA 夹在两个 barrier 之间、会写 V3 正在读的 slot，
#   那时这个 barrier 不能删。负向对照也验证了这点：没有 slot 划分时删除会被检出。
#   现有组合产物已有 GPU 计时；CPU 模型本身不模拟 wave 独立时序，也没有单项性能对照。
# 当前 shared execution plan 允许循环后的 LDS 用户，因此还需证明 exit drain 在
# 退出后的任何内存效果之前；不能沿用“最后一轮没有读者”来无条件删除同步。
# 当前只处理带 shared-plan-v1 函数标记和跨轮循环标记的顶层 affine/scf 循环；
# 要求 block-uniform 控制流，开头 readiness 消费循环携带 task，出口等待对应循环结果。
# 两端仅可跨纯寄存器运算 / GPU shuffle，内存、调用、嵌套控制和其他同步都会阻断。
# 只删尾部 reader barrier，保留循环头 readiness 及出口 drain；strictfp/optnone 跳过。
# 代码：ConvertToLLVM/ThreadElideBackedgeBarrier.cpp；
# 当前回归：3rd/deepgengraph/test/unittest/check_thread_backedge_barrier.py。
# 当前回归：3rd/deepgengraph/test/unittest/check_thread_backedge_barrier.py。


# ---- 34. ThreadLazySoftmaxBase：滞后更新 softmax 基准 --------------------------------
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
#   在第 32 项跨轮规划、寄存器/LDS 调度及第 18 项 wave setup 外提之后执行。
#   ThreadIR 的慢路径只含 ALU，不增加 DMA、wait、barrier 或跨 lane 通信；随后才做
#   first/secondLowering 和 LLVM 处理。分支仅 wave-uniform，后续同步优化仍须保守证明安全。
#   支持 affine/scf 循环与 typed MMA/inline asm；拒绝 strictfp、wave32、分数外逃等不安全形态。
# 历史实验：control 1.6633 -> lazy_t8 1.571839 ms（约 1.0582x）；
#   T=4 为 1.572239 ms，每轮强制慢路径 lazy_always 为 1.691279 ms。
#   这些是文本改写实验版的计时，不冒充当前 MyTest 完整流水线的单项复测。
# 验证记录：noncausal_lazy_base/pass_validation.json 比较 Pass 版与实验版的结构、
#   数值、访存和调度；CPU 模型不模拟目标寄存器分配、MMAC 内部舍入和 GPU 时序。
# 当前验证：check_thread_lazy_softmax_base.py 覆盖合法性、慢路径纯 ALU、幂等及 LLVM/O3；
#   上述 pass_validation.json 为旧 LLVM 版记录，不能代替当前 ThreadIR 版验证。
#   generated_ir/thread_lazy_softmax_base_20261118/noncausal/README.txt 记录迁移时的
#   9 类输入、21 次 CPU 模型执行及访存/同步对比；地址生命周期修复另见第 23 项。
# 代码：3rd/deepgengraph/lib/Conversion/ConvertToLLVM/ThreadLazySoftmaxBase.cpp。


# ---- 35. ThreadSoftmaxCvtPack：每两个 FP32 的转换与打包合成一条指令 -----------
# 开关：--softmax-cvt-pack=off|rtz，默认 off；rtz 显式允许改变 P 的舍入方式。
# 位置：第 34 项之后、firstLowering 之前；匹配 ThreadIR 的类型和 SSA 使用关系。

def softmax_cvt_pack_before(w):                      # 一个 fragment，4 个 FP32 exp2 结果
    h = [to_fp16_rne(w[i]) for i in range(4)]         # 最近偶数舍入，各自转换
    return as_half4(pack_half2(h[0], h[1]), pack_half2(h[2], h[3]))


def softmax_cvt_pack_after(w):
    # ThreadIR: frisk.convert_pack %w {rounding = "rtz"} : vector<4xf32> -> vector<4xf16>
    return frisk_convert_pack(w, rounding="rtz")


def softmax_cvt_pack_rocm_lowering(w):
    w = register_identity(w)                       # FP32 输入生命周期锚点，空的 tied-register asm
    lo = v_cvt_pkrtz_f16_f32(w[0], w[1])            # 向零舍入，一条指令转换并打包两个值
    hi = v_cvt_pkrtz_f16_f32(w[2], w[3])
    return register_identity(as_half4(lo, hi))      # FP16 输出锚点；每对前一个元素占低 16 位

# 省了什么：转换/打包指令。典型每片由 4 次转换 + 2 次打包变成 2 次转换打包，
#   16 个 fragment 的该段预计从 96 条变成 32 条；这不是整个 kernel 的指令数。
# 匹配：math.exp2 直接产生 vector<4xf32>，经未显式指定舍入模式的 arith.truncf
#   变为 vector<4xf16>，且所有 half 使用都必须是可识别 MMAC 的 A 操作数。
#   支持 typed frisk.warp_mma_rr 或校验过的 MMAC inline asm；只替换这条 narrowing 边。
# 限制：需显式 rtz + native exp2（MyTest 的 math=approx 或 mmac-exp2），
#   函数需有正数 thread_num；strictfp/optnone、非 exp2、错误宽度、显式舍入、
#   half 外逃或同时作为 B 操作数的转换都跳过；安全转换仍可在同函数中单独改写。
#   分母仍累计 FP32 W，最终输出转换保留原舍入；RNE -> RTZ 会改变数值，不能声称逐位等价。
#   不要求已有 register_lifetimes 标记或 identity；lowering 新建两端锚点以保持调度。
#   frisk.convert_pack 只接受固定 vector<4xf32> -> vector<4xf16> 与 rounding="rtz"；
#   目前仅有 ROCm lowering，CUDA 会拒绝，不能静默换回其他舍入模式。
#   保留 MMAC 顺序；可独立开启，也能用于匹配的 causal 路径。
#   排在第 34 项之后，但不要求开启它，也不依赖 workgroup-uniform 控制流。
# 历史实测：实验版 lazy_t8 1.571839 -> lazy_t8_pk_rtz 1.505759 ms（1.0439x，下降约 4.2%）；
#   旧 LLVM Pass 固化版 pass_pk_rtz.o3.ll 复测 1.505119 ms，VGPR 228、scratch 0。
#   固化版以原 lazy_t8.ll 为输入；不能把该计时标成当前 MyTest 全流水线重导出的成绩。
#   保留 RNE 的 SDWA 实验版为 1.538879 ms；日志注明 PK_RNE 指令不支持，均非此 Pass 模式。
# 代码：3rd/deepgengraph/lib/Conversion/ConvertToLLVM/SoftmaxCvtPack.cpp。
#   FriskOps.td / FriskOps.cpp 定义及验证 convert_pack；GPUToNvvmRocdl.cpp 实现 lowering；
#   FriskLLVMTranslation.cpp 在翻译时补齐锚点属性，LLVM 级 softmaxCvtPack 改写已移除。
# 当前验证：3rd/deepgengraph/test/unittest/check_softmax_cvt_pack.py 覆盖匹配边界、幂等、
#   FP32 sum DAG、类型验证、元素打包顺序、生命周期锚点、LLVM/O3 及旧入口拒绝。
# 历史验证：generated_ir/noncausal_cvt_pack_pass/{validation.json,integration_validation.json}。
# 迁移产物：generated_ir/thread_softmax_cvt_pack_20261118/noncausal/generation.json
#   记录 16 个 convert_pack -> 32 条 packed conversion 及 LLVM verify 通过；
#   assembly_audit/diagnosis.json 未发现结构性回退，但缺同基线汇编与成对 GPU 重测，
#   不能据此宣称 ThreadIR 迁移有独立性能收益。


# ---- 36. FriskOpFuse：在 block IR 给 GEMM 放入显式累加器 ----------------------
# 当前在 ConvertFriskToBase 之后运行两次：先融合完整 PV，再 TileAttention，最后再尝试融合。

def block_gemm_fuse_before(A, B, old, alpha):
    partial = gemm(A, B, acc=zero_tile())
    return partial + alpha * old


def block_gemm_fuse_after(A, B, old, alpha):
    return gemm(A, B, acc=alpha * old)

# 省了什么：整块临时结果和后续 add，让布局推断直接看到累加器，避免依赖 LLVM 兜底。
# 限制：GEMM 是单用户、零初值，add 与 seed 同形同元素类型；支持 add 两种操作数顺序。
# 不匹配广播、非零 seed、多用户、已调度/已 tiled 的 op 或中间写/同步/未知效果。
# 浮点累加顺序改变，需 allow-reassociation=true；MyTest 仅在 math!=strict 时许可。
# 分块前融合避免后续 V slice 的 alloc/copy 隔开 GEMM 与 add；tiler 将 seed 随 V/O 切分，
# 不放宽跨写、跨同步的合法性检查。分块后的第二次尝试保留新暴露的机会。
# 第 7 项仍处理 strict / block 融合回退后剩余的 fragment add；25 只清理 LLVM 指令。
# 代码：Dialect/Frisk/Transforms/FriskOpFuse.cpp。


# ---- 37. SinkWaitBufferReady：block tiling 内延迟 readiness --------------------

def sink_ready_before():
    task = async_copy_gm_to_shm(src, dst)
    wait_buffer_ready(task)
    independent_register_work()
    consume(dst)


def sink_ready_after():
    task = async_copy_gm_to_shm(src, dst)
    independent_register_work()
    wait_buffer_ready(task)
    consume(dst)

# 第 6 项先完成布局和 async/sync 选择，再推迟 wait，最后才展开消费者为 fragment。
# 省了什么：用独立计算覆盖等待；这里只移动 readiness，第 11 项负责后续 fragment 调度。
# 限制：跟踪逻辑 view 与物理 alias，不越过源/目的访问冲突、同步、另一 DMA 或已有流水安排。
# 代码：FriskToBase/SinkWaitBufferReady.cpp。


# ---- 38. CoalesceWaitsAndBarriers：合并同块重复 wait-all / publication ---------
# secondLowering 中，任务 readiness 已变成 target wait-all 后、SCF → CFG 之前运行。

def coalesce_sync_before():
    wait_all()
    barrier()
    pure_index_work()
    wait_all()
    barrier()


def coalesce_sync_after():
    wait_all()
    barrier()
    pure_index_work()

# 省了什么：多余等待和全 workgroup 同步。不能在 Base 层把不同任务 token 的 wait 当作相同。
# 限制：状态只在同一 block 内有效；内存、调用、opaque asm、新 async、partial wait、
# 未知 target op、region / 控制流边界均重置状态。barrier → wait-all → barrier 保留后一个
# publication，因为前一个 barrier 不能发布随后 wait 才完成的写。
# 代码：ConvertToLLVM/CoalesceWaitsAndBarriers.cpp；说明：lib/Pipeline/README.md。


# ---- 39. CoalesceWaitsAndBarriers 的依赖分析阶段 -------------------------------
# 启发式路径在同一次 Pass 调用中分析结构化 loop、回边和 zero-trip 路径。
# --coalesce-waits=0 时只做此阶段。CP-SAT 保序时只做第 38 项。

def barrier_dependency_before():
    a = read_shared(buffer_a)
    barrier()
    b = read_shared(buffer_b)                        # 两侧都只有读，不需要写的可见性
    return register_compute(a, b)


def barrier_dependency_after():
    a = read_shared(buffer_a)
    b = read_shared(buffer_b)
    return register_compute(a, b)

# 省了什么：无跨线程写依赖时的全 workgroup 同步；它与 38 的重复序列合并、
# 33 的 ThreadIR 回边特例分别匹配不同证据，不是三个位置无条件重复删同一个 barrier。
# 限制：只有能证明两侧读写范围不冲突才删，不能用不同 lane 的索引直接证明无跨线程别名。
# 未知效果 / 条件 barrier 保守保留；每删一个 barrier 就更新图，避免同时删掉唯一保护。
# 代码：ConvertToLLVM/IROptimize.cpp 中的 BarrierAnalysis。独立命令 barrier-optimize 仍只执行此阶段。


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


# ---- 41. ThreadInterleaveQK：QK 独立累加链交错与受预算约束的 K 预读 ----------
# 与第 26 项共用通用入口 frisk-thread-interleave-mma-chains；
# 本项使用 operand-source=lds load-motion=bounded；旧 qk 名称仅为兼容别名。
# 默认关闭；--qk-interleave=0|1|2|4，guard 默认 2，prefetch-words 默认 32。
# 位置：第 42、26 项之后、30 之前；单独开启也会先执行 LDS 地址规范化。

def qk_interleave_before(a, b):
    # 示意两条等深链；实际从 SSA 的累加关系识别，不依赖 attention 的变量名。
    packets = []
    for k in range(8):
        packet = load_lds_K(k)
        packets.append(packet)                      # 表示已加载的 SSA 值供第二条链共享
        a = mmac(Qa[k], packet, a)
    for k in range(8):
        b = mmac(Qb[k], packets[k], b)
    return a, b


def qk_interleave_after(a, b):
    # 仅在整个窗口可安全预读且预算足够时；已有公共 packet 复用同一 SSA 值。
    packets = [load_lds_K(k) for k in range(8)]
    for k in range(8):
        a = mmac(Qa[k], packets[k], a)
        b = mmac(Qb[k], packets[k], b)
        sched_barrier(0)                            # guard-group=2：两条 MMAC 后一个提示
    return a, b

# 省了什么：希望用独立累加链覆盖 MMAC 依赖延迟，并提前准备 K（推断，无 GPU 单项计时）。
# 第 26 项要求已就绪的 matrix-read packet；本项接受对齐的 frisk.lds_load vector<4xf16>，
# 可将候选晚到的 B load 提到首个 MMAC 前，保持 load 顺序和每条累加链的 K 顺序。
# 每个不同 B packet 计 2 个 32-bit words/thread；共享 packet 只计一次，已有 packet 也计入。
# 预算限制的是 B 操作数 footprint，不是总 VGPR；0 拒绝所有候选。提前读取可能增加寄存器压力。
# 窗口仅允许选中的读取、MMAC 和可推测执行的纯寄存器操作；store、其他读取、调用、
# 嵌套控制、DMA、wait、硬件 barrier、已有 guard / 已保护的 read pair 都会拒绝该窗口。
# load 地址依赖尚未调度的 MMAC 也拒绝；绝不越过分支或同步投机读取。
# width=1 使用同样预读策略但串行排链，可与 width=2/4 对照；guard 规则与第 26 项相同。
# guard=0 只是去掉本 Pass 的提示，不保证后端仍保持交错，也不删除其他 Pass 的 guard。
# 保留 FP 属性，不要求 math=approx；strictfp/optnone 函数仍跳过。
# 计划成功后才修改 IR；通用 Pass 用 frisk.mma_interleave 标记已调度的 MMA。
# 代码：ConvertToLLVM/ThreadInterleaveMMAChains.cpp；回归：test/unittest/check_mma_interleave.py。


# =============================================================================
# 第 6 部分：近期验证记录
# =============================================================================
# 1. 分块前融合完整 PV（第 36 项）
#    generated_ir/best_noncausal_early_wave_upstream_pv_fusion_20261119/README.txt：
#    BHSD=1/32/4096/128，BM=BN=128，QK/VO=32/32。导出 256 条 MMAC、32 次 matrix read、
#    32 次 RTZ pack、36 次 DMA，额外 float4 add 为 0；两个随机输入和单-wave 重缩放用例
#    与恢复融合对照的 CPU 输出逐位一致。记录没有 GPU 性能结果。
#
# 2. causal 自动流水 + Q/KV LDS 复用（第 15 项与可选 AutoPipeline）
#    generated_ir/causal_kv_shm_reuse_20261119/README.txt、validation.json：
#    BHSD=1/32/4096/128，BM=64，BN=32，QK/VO=128/128，2-stage KV 流水。
#    出口等待闭合完成证明后，静态 LDS 48 -> 32 KiB；Q 初始化用 [0,16384)，
#    Q 读完并经交接 barrier 后，K0/K1 复用该区，V0/V1 使用 [16384,32768)。
#    输入与参数相同的旧流水产物作对照，4 个 query tile 共 8 次 CPU 执行逐位一致，
#    最大绝对误差约 0.001030902；每 workgroup 增加 1 个交接和 2 个出口 barrier。
#    LDS 降低不等于已证实提速，GPU 正确性/耗时仍未测。此配置不是第 3 部分的 16 KiB 布局。
#    更早的 test_input -> KV 双缓冲实验同时改了 QK/VO=32 -> 128，不能作为单变量对照。
#
# 3. QK/PV 链调度（第 26/41 项，已纳入当前源码）
#    generated_ir/causal_mma_chain_pass_20261119/validation_summary.json、pass_tests.json：
#    BM128/D128，BN16 比较 QK width=2 的 guard=2/0，BN32 比较 PV width=4、guard=4。
#    每个变体 4 个数值用例、16 个 wave 调度检查，均与各自同版本基线逐位一致；
#    最大绝对误差约 0.001030901。三个回归脚本记录共 236 项通过，gpu_tested=false。
#    BN16 guard=2 的 schedule_check.json 确认 A0 B0 ... A7 B7、8 个 QK guard，
#    K 在首个 MMA 前加载并供每对 MMA 共享。这是 IR/CPU 证据，不是目标 ISA 性能结论。
#
# 4. 命名参数兼容
#    generated_ir/causal_kv_shm_reuse_20261119/named_options_validation.json：
#    相同配置下命名参数与旧位置参数生成的原始 LLVM IR 逐字节一致，并通过 llvm-as。
#    当前回归入口：3rd/deepgengraph/test/unittest/check_mytest_named_options.py。
#
# 5. 完整 D 输入经 block tiling 的最新 GPU 记录（组合配置）
#    generated_ir/best_early_wave_full_dim_bk32_20261120/test_input.mlir 保留完整 D=128，
#    MyTest 用 --qk-tile=32 --vo-tile=32 真正切分；BHSD=1/32/4096/128，BM=BN=128。
#    pmc_test.log 开头记录对应 kernel.o3.ll：HSACO median=1.474885 ms，
#    PyTorch SDPA median=1.554646 ms，speedup=1.054079x；这是组合结果，不能拆成单项收益。
#    命令启用 lazy-softmax-base=8、softmax-cvt-pack=rtz、prefetch-packets=1，
#    coalesce-waits=0；并非第 3 部分未开启 34/35 的示意或裸 MyTest 默认配置。
#    命令使用 --llvm-opt=0，随后外部 default<O3>,verify 生成测试 IR。
#    目录 README 的“GPU 未测试”描述生成时状态；后补的 GPU 耗时以 pmc_test.log 为据。
#    日志另有 RNE + 去掉转换两端空 asm 的实验；两处同时变化，不能归因于单个开关。
#
# 6. 执行语义与 MMA 导出契约（基础设施）
#    generated_ir/frisk_execution_semantics_bk32_20261120/README.txt：
#    13 组回归通过；256 条 MMA 在 translateToLLVM 时已有完整调用属性。
#    raw/O3 IR 经 llvm-diff 与上述实测基线的函数体一致；未重新执行 GPU，不能记作新提速。
#
# 7. 第 25 项两轮通用清理消融
#    generated_ir/llvm_cleanup_split_bk32_20261120/README.txt、validation/analysis.json：
#    四模式使用相同输入和调优参数，translation 产物逐字节一致，均保留完整导出契约。
#       模式      pre-O3 静态指令数     O3 静态指令数
#       both              2985                 3024
#       early             2988                 3018
#       late              3014                 3044
#       none              3036                 3058
#    raw/O3 及保存的中间模块通过 LLVM verifier；both 与实测基线经 llvm-diff 一致。
#    去掉任一轮均改变最终 IR，两轮在此配置下并非完全重复，当前默认保留 both。
#    early 的静态指令更少不等于更快；四模式没有成对 GPU 正确性/性能测试。
#    回归记录中，生命周期核心检查通过；可选 LLVM 15 解析器不接受 range 返回属性，
#    这是工具版本限制，不能把该回归的完整命令记为无条件通过。
#
# 8. 第 35 项生命周期锚点直接导出
#    generated_ir/register_anchor_export_bk32_20261120/README.txt、validation/comparison.json：
#    32 个向量锚点与 256 条 MMA 在翻译时已补齐契约，6 组回归通过（含 7 项 cvt-pack 测试）。
#    新 translation 等于旧 translation + 后置补齐；后续各阶段及 raw/O3 输出一致，
#    比较忽略调试信息、文件名和 metadata 编号，仍核对调用属性及实际生命周期 metadata。
#    删除 MyTest、库 pre-O3/最终导出和 fragment prefetch 共四处冗余扫描；兼容入口保留。
#    这是职责迁移的一致性验证，没有新增 GPU 耗时。
#
# 9. BK32 的 LLVM 文本调度实验（没有纳入编译器 Pass）
#    generated_ir/bk32_schedule_experiments_20261120/README.txt、numerical_validation.json：
#    baseline / 仅改名 control，加 QK 交错 4/8 链、PV guard 间隔 4/8、V 读取窗口 2/4，
#    共 8 份 IR。它们是独立文本实验，不扩展第 41 项 --qk-interleave=0|1|2|4 的取值。
#    所有变体保留 256 条 MMAC、32 次 V read、14 个硬件 barrier、13 个显式 VM wait 站点；
#    QK 相邻累加依赖可由 64 降为 0，PV 编译器 barrier 可由 272 降为 36/20，
#    V 的 IR 读取窗口可由 8 降为 2/4；均为静态指标，不代表最终机器调度或周期数。
#    llvm-as/verify 及 SSA、内存/同步顺序检查通过；指定 head=1、query block=31 的
#    完整 S=4096 CPU 执行 8/8 通过，与基线逐位一致，参考最大绝对误差约 6.12773e-05。
#    CPU 模型不模拟 gfx936 机器 hazard 或独立 wave 时序；未做目标代码生成和 GPU 测量。
#    build.sh 直接 llc -O3 编译既有 O3 IR；再次运行通用 O3 可能改变实验调度，
#    比较时需同路线重编 baseline/control，并在最终汇编中确认调度是否保留。
#
# 10. V 逻辑规划前移与 V0 首次预取回退修复（2026-11-26）
#    输入：generated_ir/best_early_wave_full_dim_bk32_20261120/test_input.mlir，noncausal。
#    保持用户原生成参数，包括 --pipeline=0 --v-double-buffer=1 --cross-bn-k0=1。
#    首次迁移产物 auto_pipeline_v_double_buffer_noncausal_20261126/after/kernel.ll
#    是出现回退的版本；用户提供 kernel.amdgcn.bad.log，报告加速比降至 1.036。
#    参考 k_test.gcn.best.log，仅比较主 Attn_p2 kernel：两者均为 227 VGPR、
#    16384 B LDS、无 scratch spill；不能把参考文件内其他函数的 spill 算到主 kernel。
#    历史最佳汇编中，V0 最后一包 DMA 到 wait 之间有 229 条机器指令、32 条 MMAC；
#    回退汇编中为 0 条，DMA 后紧跟 s_waitcnt vmcnt(0)，丢失了预取重叠窗口。
#
#    原因是调度保护连无前驱锚点的 stage 0 也冻结在 QK/softmax 之后。
#    修复仅允许合法 [0,count,0,-1] 首次 DMA 参与已有别名检查的安全前移；
#    中间 stage、全部 wait 和显式调度继续保护，V 两槽与跨 bn K0 均保留。
#    修复后 O3 LLVM 中 V0→wait 恢复 32 条 MMAC、447 条 LLVM 指令；
#    历史最佳 IR 对应为 32 / 424。LLVM 指令数不能与上面的 229 条机器指令直接比较。
#    修复前后总量均为 256 MMAC、13 个显式 VM wait、14 个 block barrier，
#    DMA 为 16 个 dwordx4 包、20 个 dwordx2 包；LDS 仍为 16384 B。
#
#    当前修复产物（以下路径相对 generated_ir/）：
#      auto_pipeline_v_regression_fix_20261126/fixed/kernel.ll
#      同目录 command.sh、after_auto_pipeline.mlir、after_fragment_scheduling.mlir、
#      after_lower_frisk_fragments.mlir 和 lowering.log 保存复现命令与各阶段 IR。
#      上级 README.txt、summary.json、assembly_analysis.json 和 verify/ 保存证据。
#    已有验证：首发前移 24 + 早期 V 流 12 + 物理 V 双缓冲 24 + 跨轮 22 = 82 项；
#    另有 task/liveness、C++ 快照/循环生命周期、构建及 MLIR/LLVM verifier 检查通过。
#    本次文档更新核对这些记录，未重新跑 GPU；修复环境无 gfx936 GPU / DTK 代码生成器。
#    尚不能声称加速比恢复，修复后的目标汇编、GPU 正确性和性能仍需在目标机器复测。
#
# 11. 提前展开 QK，并将跨 bn K0 逻辑规划前移（2026-11-26）
#    generated_ir/auto_pipeline_early_unroll_cross_bn_20261126/README.txt：
#    保持同一 noncausal 输入和完整参数，当前产物为该目录 after/kernel.ll。
#    AutoPipeline 先把内层 0..128 step 32 展开为四段，外层 bn 仍保持循环；
#    先显式化 legacy vector iter_arg 的更新，保留 QK0→QK1→QK2→QK3 的累加链。
#    布局选择沿 GEMM 的 acc 边传播，使四段采用相同 C 排列，避免整块累加器落 LDS。
#    早期 IR 中有四个 V stage 契约，以及唯一的 cross_iteration_plan / issue / anchor；
#    execution 消费这些标记并验证/落实，未标记的跨轮候选不再由后端重新发现。
#    最终 LDS 为 16384 B，V0→wait 保留 32 MMAC / 447 条 LLVM 指令；
#    总计 256 MMAC、13 VM wait、14 block barrier、16 个 x4 和 20 个 x2 DMA 包。
#    llvm-diff 对上条 V0 修复版未报告差异；这是职责迁移及生成一致性检查，不是 GPU 计时。
#    102 项聚焦回归通过，另有循环流水、task/liveness、布局数值检查及 MLIR/LLVM 验证。
#
# 12. 第 38/39 项合并与 MyTest 重复调用清理
#    原合并提交 76a1f719 以 e2ae5ef0 为对照，用相同工具链和输入导出六组 --llvm-opt=0 的 LLVM IR：
#    causal 的 coalesce-waits=1/0、CP-SAT 保序、strict，以及 non-causal 的
#    K swizzle 和关闭两项地址/索引规范化的 mmac-recip。六组逐字节一致，
#    且各自通过 opt -passes='default<O3>,verify'。本次未测 GPU 耗时。


# =============================================================================
# 附：MMAC 16x16x16 的坐标（lane 号 l，寄存器号 r，详见 1014.md 第 4 节）
# =============================================================================
#   A 的第 r 个 half：A[l % 16, 4 * (l // 16) + r]
#   B 的第 r 个 half：B[4 * (l // 16) + r, l % 16]
#   C 的第 r 个 float：C[l % 16, l // 16 + 4 * r]
#
# 相关说明：
#   3rd/deepgengraph/test/unittest/（Python 回归脚本已从 test/ 根目录迁入）
#   3rd/deepgengraph/test/unittest/check_frisk_semantics.py（执行语义 / MMA 即时导出）
#   3rd/deepgengraph/test/unittest/check_llvm_cleanup_contracts.py（契约与通用清理拆分）
#   3rd/deepgengraph/test/unittest/check_register_anchor_export.py（锚点直接导出及兼容性）
#   3rd/deepgengraph/lib/Conversion/ConvertToLLVM/{PVInterleave,PairedLDS}.md（当前 Thread 调度）
#   3rd/deepgengraph/lib/Conversion/FriskToBase/{VDoubleBuffer,SharedSwizzle}.md（联合规划）
#   3rd/deepgengraph/lib/Conversion/FriskToBase/FriskFragmentIR.md
#   3rd/deepgengraph/lib/Pipeline/README.md（自动流水和同步清理；执行位置以当前源码为准）
#   3rd/deepgengraph/test/unittest/check_shared_cross_iteration.py（跨轮 token、guard、drain 回归）
#   3rd/deepgengraph/test/unittest/check_thread_fragment_schedules.py（29/30/31 当前调度回归）
#   3rd/deepgengraph/test/unittest/check_thread_lds_addresses.py（24 当前地址证明回归）
#   3rd/deepgengraph/test/unittest/check_thread_backedge_barrier.py（33 ThreadIR 回归）
#   3rd/deepgengraph/test/unittest/check_mma_interleave.py（26/41 调度、预算和 guard 回归）
#   3rd/deepgengraph/test/unittest/check_auto_pipeline_shm_reuse.py（自动流水的出口等待和复用）
#   3rd/deepgengraph/test/unittest/check_auto_pipeline_v_stream.py（V 逻辑契约及拒绝路径）
#   3rd/deepgengraph/test/unittest/check_auto_pipeline_v_head_prefetch.py（V0 安全前移与边界）
#   3rd/deepgengraph/test/unittest/check_v_double_buffer.py（消费早期计划、物理验证与落实）
#   generated_ir/auto_pipeline_v_regression_fix_20261126/README.txt（本次修复及复测入口）
#   3rd/deepgengraph/test/unittest/check_auto_pipeline_cross_iteration.py（早期展开、递推及跨轮契约）
#   generated_ir/auto_pipeline_early_unroll_cross_bn_20261126/README.txt（当前生成产物与验证）
#   generated_ir/shared_memory_plan/README.txt（28 联合布局 / DMA 的历史验证记录）
#   generated_ir/thread_approximate_math/README.md（19 前移验证记录）
#   generated_ir/thread_mask_add_bypass/README.md（20 前移验证记录）
#   SafeSoftmax最后两项优化说明.md
#   测试数据分析/1014.md（布局）、测试数据分析/1104.md（收益）
#   generated_ir/noncausal_lazy_base/README.txt（第 34 项历史实验）
#   generated_ir/thread_lazy_softmax_base_20261118/noncausal/README.txt（第 34 项迁移验证）
#   generated_ir/thread_lazy_softmax_base_fix_20261118/README.txt（第 23/34 项地址生命周期修复）
#   3rd/deepgengraph/test/unittest/check_softmax_cvt_pack.py（第 35 项当前 ThreadIR 回归）
#   3rd/deepgengraph/test/unittest/check_softmax_cvt_pack.py（第 35 项当前 ThreadIR 回归）
#   generated_ir/thread_softmax_cvt_pack_20261118/assembly_audit/diagnosis.json（迁移产物审计）
#   generated_ir/noncausal_cvt_pack_pass/README.txt（第 35 项历史 LLVM 版实测）
#   generated_ir/causal_mask_lowering/README.txt（第 6/13 项新增能力）
