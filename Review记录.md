# Pass 人工review记录

## Pass
RetainInvariantFragments  
## 功能
跨 BN 保留 Q  
```py
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
```
本质上是完美嵌套的两个for 交换位置，拆成不完美循环（在内层循环做LICM）

## 作用位置（时机）

fragmentIR 形成后

## 匹配Pattern
1. 如果已有 frisk.retained_words 属性：说明之前已做过保留优化，避免重复处理。直接跳过
2. 必须证明循环至少执行一次：nonempty() 用区间分析证明“下界最大可能值 < 上界最小可能值”。范围未知也会跳过，避免将 DMA 外提后，在原本执行零次的循环外发生非法读取。
3. 必须证明 block 内所有线程控制流一致：主要检查循环边界的输入是否对整个 block 一致。常量、标量函数参数、block ID 及其纯算术派生值可以通过；thread ID、内存 load 等不能直接证明一致。
4. 循环执行大于一次

作用在 affineFor + 内部DMA （frisk.AsyncCopyGmToShmOp）+ barrier 上
| 专用逻辑 | 为什么需要 |
|---|---|
| `nonempty()`、uniform 检查 | 保证搬运和集体同步可以安全外提 |
| `immutableSource()` | 证明源内存不会被修改 |
| 专用 LDS、唯一 wait、读取顺序检查 | 证明整条链可一起移动 |
| 依赖切片和白名单 | 确定外提边界 |
| `maxWords` 预算 | 限制跨循环常驻的寄存器数据 |

## 依赖的输入信息
 暂无

## 其他
通用性一般。

---------------------------------------------------------------------------------------------------

## Pass
ScheduleAsyncCopies
## 功能
提前发 DMA，推迟 wait   

## 作用位置（时机）
asyncCopy出现后。fragment级别的优化

## 匹配Pattern
匹配 WaitBufferReadyOp  AsyncCopyGmToShmOp ，将wait向后移动，DMA向前移动

这个函数做的是 **异步拷贝的指令调度：尽量提前发起拷贝，尽量推迟等待，让数据搬运和计算重叠，减少等待时间。**

代码位置：[FriskFragIRReorder.cpp:791](/data2/xsl/DeepGenGraph/3rd/deepgengraph/lib/Conversion/FriskToBase/FriskFragIRReorder.cpp:791)。

例如，原来的执行顺序是：

```text
计算 A（与拷贝无关）
token = async_copy GM → SHM
wait(token)
计算 B（与拷贝无关）
读取 SHM 并计算 C
```

调度后可以变成：

```text
token = async_copy GM → SHM
计算 A
计算 B
wait(token)
读取 SHM 并计算 C
```

这样执行 A、B 的同时，硬件可以搬运数据；到了 C 真正需要数据时才等待。这里 GM 是全局内存，SHM/LDS 是共享内存。

具体分成以下几步：

1. **判断当前阶段和是否允许调度。**  
   如果 IR 中还有 `FragmentType` 或 `ToThreadTileOp/FromThreadTileOp`，视为逻辑阶段，采用更保守的条件：目的 buffer 必须被生命周期分析认定为 `reusable`，拷贝还必须位于线程块内一致的控制流中。  
   如果发现 `frisk.shm_pool`，直接返回，因为共享内存已经安排了物理复用，再改变顺序可能破坏已有的生命周期关系。

2. **判断哪些操作可以跨过。**  
   `independent(op, issue)` 检查操作是否与当前拷贝独立：
   - 不能读写拷贝的目的 buffer。
   - 不能写拷贝的源 buffer。
   - 通常不能跨过 barrier、wait、其他异步拷贝、显式流水调度等。
   - 副作用或内存别名关系无法证明安全时，也停止移动。

   因此，“看起来用了不同变量”还不够，必须通过别名分析证明它们不冲突。

3. **把拷贝及其地址准备操作向前移动。**  
   从 `src`、`dst` 递归收集可移动的描述符/地址计算，放入 `setup`；不能移动的依赖放入 `anchors`。随后在**同一个基本块内**向前扫描，找到最早的安全位置，先移动准备操作，再移动拷贝，保留依赖顺序。

4. **找到对应的 wait，并向后移动。**  
   按相同的 token 找到 `WaitBufferReadyOp`，逐条跨过无关操作，直到遇到数据访问、同步或其他不能跨越的边界。这样 wait 就尽量接近真正需要等待完成的位置。

5. **物理阶段允许一种特殊优化。**  
   逻辑阶段，wait 不跨过其他 DMA 发起操作；物理阶段，通过 `DirectDMAWaitMotion::canCross()` 严格检查后，可以允许：
   ```text
   issue A → wait A → issue B
   ```
   变成：
   ```text
   issue A → issue B → wait A
   ```
   检查不仅考虑两个拷贝是否冲突，还考虑 wait 降低后包含的 barrier 是否在保护旧的共享内存读者。

它**直接修改 `kernel` 内的 IR 顺序**，不返回结果，也不负责把普通拷贝转换成异步拷贝。它既用于独立的 `frisk-schedule-async-copies` pass，也在 `LowerFriskFragments` 中、共享内存复用分配之前再次执行。

## 依赖的输入信息
需要执行 shmLiveliness 分析（静态）


## 其他 
什么是shm的别名？
这里的 **SHM alias（共享内存别名）是指：多个不同的 SSA 值，可能指向同一块共享内存，或者其中重叠的部分。**
比如：
```mlir
%a = memref.alloc() : memref<128xf32, 3>
%b = memref.subview %a[0] [64] [1] ...
```

可以理解成：

```text
%a ──→ 整块共享内存：[0 ................ 127]
%b ──→ 其中一部分：  [0 ...... 63]
```

`%b` 只是一个新的视图，没有复制数据，也没有分配新的存储。通过 `%b` 写入，会改变 `%a` 对应位置的数据。因此 `%b` 是 `%a` 的别名。

**为什么生命周期分析必须追踪 alias？**

看这个例子：

```text
a = alloc SHM
b = view(a)
写入 a
读取 b
```

如果只检查变量 `a` 的直接使用，会漏掉最后的“读取 b”，误以为那时 `a` 的存储已经不用了。如果提前把这块存储分配给另一个 buffer，`b` 就可能读到被覆盖的数据。

所以需要把所有别名的访问汇总：

```text
a 的存储生命周期
    = a 以及所有可能指向它的别名的访问范围
```

**这份代码里的 alias 还包括“可能指向”。**

例如：

```text
c = select(condition, a, b)
读取 c
```

`c` 运行时可能指向 `a`，也可能指向 `b`。因此分析 `a` 时会把 `c` 加入它的 `aliases`；分析 `b` 时也会加入。读取 `c` 会被保守地计入两块 buffer 的生命周期。

循环参数同理：

```text
for (..., iter_arg = a) {
    读取 iter_arg
    yield iter_arg
}
```

这里 `iter_arg` 传递的是内存描述符，不是把 `a` 的内容复制一份，因此也需要沿它追踪访问。

对应实现是 [analyzeUses](/data2/xsl/DeepGenGraph/3rd/deepgengraph/lib/Analysis/LivelinessAnalyze.cpp:213)：从 `alloc` 的结果出发，把 view、cast、select、循环参数等传播得到的值收集进 `info.aliases`。**这个集合包含原始的 alloc 结果本身**，所以代码中的 `aliases.size() == 1` 表示目前只发现原始值，没有额外的别名。

---------------------------------------------------------------------------------------------------

## Pass
AnalyzeShmLiveness

## 功能
分析shm以及其别名（view等）的生命周期。直接调 analyzeShmLiveness 函数，返回一个 ShmLivenessResult

## 作用位置（时机）
createScheduleAsyncCopiesPass 之后 （DMA尽早发射，wait尽量贴近consumer）

## 匹配Pattern
检查 memref.alloc 分配 shm的情况，根据消费链条分析（保守逻辑下，可能用到它的都算 如无法直接确定的 if 分支等）

## 依赖的输入信息
无

---------------------------------------------------------------------------------------------------

## Pass
ConvertFragmentToThreadIR

## 功能
将fragmentIR 转到具体的线程级别操作。用vector代替 fragmentType。对于shm，使用 from_threadtile 和 to_threadtile 临时桥接。
这个Pass 正式引入vector类型表达寄存器，将任务切到thread上

## 作用位置（时机）
fragmentIR 优化完，定型以后。
Pass 作用完后，还剩一些针对全局reg和shm的 to_threadTile 没显式地生成 thread级别实现，以及buffer_view 没转化（ blockTile和 threadTile 之间的残存桥接）

## 匹配Pattern
Lowering Pass

## 依赖的输入信息
LowerInfo规划的布局

---------------------------------------------------------------------------------------------------

## Pass
FinalizeThreadTiling

## 功能
是 thread tiling 的收尾 lowering Pass：消除 block tile 与 thread tile 之间的临时桥接，把剩余逻辑访问变成实际的寄存器操作、地址计算和访存。（for循环的loop-carried 的blockTile也替换为 threadTile）

## 作用位置（时机）
ConvertFragmentToThreadIR 后，做一下收尾，彻底消除fragment类型

## 匹配Pattern
Lowering Pass。 先折叠再物化 （to_tt(from_tt(thread_vector))）-> thread_vector

## 依赖的输入信息
无

---------------------------------------------------------------------------------------------------

## Pass
LowerFriskFragments
## 功能
核心作用是：在线程级访存已经明确之后，完成共享内存（LDS）的物理布局、异步拷贝调度和空间复用，最后展开指定的 fragment 循环。
barrier补全

| 步骤 | 做什么 | 目的 |
|---|---|---|
| LDS packing | 将符合条件的二维共享内存改成一维物理布局，重写访问地址 | 落实打包、swizzle，并合并连续访问 |
| 回边同步 | 必要时在循环末尾插入 `gpu.barrier` | 防止下一轮写覆盖本轮尚未读完的数据 |
| DMA 调度 | 尽量提前异步拷贝、推迟对应 wait | 增加拷贝与计算重叠的机会 |
| 再检查回边同步 | 根据调度后的访问位置补同步 | 保证最终调度的跨迭代安全 |
| LDS 复用 | 根据最终生命周期，让不同 buffer 共用物理空间 | 降低共享内存占用 |
| 循环展开 | 展开带指定属性的小常量循环 | 将前面保留的 fragment 循环落实为直线操作 |

## 作用位置（时机）
FinalizeThreadTiling 之后。此时fragment类型已经没了，只有vector和shm

## 匹配Pattern
展开fragment循环。

## 依赖的输入信息

## 其他
需要思考先 multi-version-buffer 还是先 DMA调度

最关键的是：先固定访问布局和 DMA 调度，再按最终生命周期分配复用空间。
提前发起 DMA 会让目的 buffer 更早进入活跃状态。如果先确定两个 buffer 共用地址，再把 DMA 上提，就可能提前覆盖仍在使用的数据。因此代码明确将 reuseSharedMemory 放在最终调度之后，并在形成 frisk.shm_pool 后停止这类 DMA 重调度。

11/10 修改：考虑将其功能整合到 PipelineSchedule （流水线调度的功能 + LDS 复用）

---------------------------------------------------------------------------------------------------

## Pass
IRDeepOptimize

## 功能
1. 展开循环以静态化 vector 下标、
2. 合并 LDS 读写
它扫描 vector.insert、vector.extract 的动态位置参数，判断下标能否通过展开循环变成常量。例如：
affine.for %i = 0 to 4 {
  %x = vector.extract %v[%i] : f32 from vector<4xf32>
  ...
}
展开后，下标成为 [0]、[1]、[2]、[3]，便于后端处理寄存器 vector 的元素访问。
判断规则包括：
- 常量直接接受。
- 支持沿 affine.apply、整数加减乘除、取余、index cast 等操作继续追踪。
- 如果遇到 affine.for 的归纳变量，就检查该循环是否可静态展开。
- 如果遇到循环携带的 iter_arg，同时检查初始值和 yield 回来的递推值。例如从常量 0 开始、每轮 +2 的索引也能处理。
- 如果依赖线程号、函数运行期参数等无法静态确定的值，这条下标依赖链就不触发展开。


合并 LDS 访问：把相邻小访问变成宽访问：
标量合并见 [coalesceUnrolledSharedScalars (line 529)](/data2/xsl/DeepGenGraph/3rd/deepgengraph/lib/Conversion/ConvertToLLVM/IROptimize.cpp:529)：
- 两个 memref.load 的下标差能证明为 ±1，就变成一次两元素 vector.load，再分别 extract。
- 对同一 buffer 的一组标量 store，证明它们的相对地址固定且不重复后，按地址排序，把连续部分组成向量写，最大 128 bit。
- 合并后的 store 放在最后一条原 store 的位置，确保待写值都已经产生。
向量读合并见 [coalesceUnrolledSharedReads (line 383)](/data2/xsl/DeepGenGraph/3rd/deepgengraph/lib/Conversion/ConvertToLLVM/IROptimize.cpp:383)。例如：
load vector<2xf32> from LDS[i]
load vector<2xf32> from LDS[i+2]

→ load vector<4xf32> from LDS[i]
  再拆出前后两半
它要求相同 buffer、相同向量类型，且第二次访问紧接第一次的地址范围。每次宽度翻倍，执行三轮，最多到 128 bit。
“地址相邻”不是靠采样猜测，而是通过：
1. affine 表达式化简得到常量差；
2. 特定线程号场景下枚举全部合法线程坐标；
3. 使用 Presburger 整数约束证明不存在不相邻的情况。
遇到写操作、barrier、控制流或未知副作用时，会截断读合并窗口，避免移动访问改变语义。

3. 重新安排访存与计算的顺序（访存计算交错——这部分是否和之前的Block级别IR调度 重合了？—— 不冲突。其定位是降级到fragment后再优化，而之前的blockIR级别的调度为高层优化。

## 作用位置（时机）
fragmentIR 变成 threadIR后。

## 匹配Pattern

vector.insert extract 展开，用常量索引
vector.load vector.store 做指令合并，尽可能用宽指令
reorder 在Block内进行，根据 Issue Boundary 确定边界，在窗口内做重排。基本适用于affine vector memref 以及gpu 的常用op


## 依赖的输入信息

## 其他
通用性强

---------------------------------------------------------------------------------------------------

## Pass
SimplifyBoundedAffine

## 功能
利用循环变量的取值范围，消除 affine.apply 中可以证明冗余的 floordiv 和 mod，减少地址计算，并为后续循环不变量外提（LICM）创造条件。
用一个实际例子说明
文件开头给出的典型表达式是：
((iv + s0 * 4096 + s1 * 131072 + 1024) floordiv 4096) mod 32
已知循环范围为 0 ≤ iv < 32，那么：
1024 ≤ iv + 1024 ≤ 1055
这一部分除以 4096 向下取整，结果始终为 0。因此：
原表达式
= (s0 + s1 * 32 + 0) mod 32
= s0 mod 32
这里有两个收益：
- 消除内层 floordiv 及相关算术。
- 消除对 iv 的依赖；如果其他操作数对该循环不变，后续 LICM 就可以把计算移到循环外。
注意，这个例子仍然保留 s0 mod 32，因为代码没有证明 s0 的范围。

## 作用位置（时机）
循环已经定型、归一化之后，lowering 之前

## 匹配Pattern
affine.apply 

## 依赖的输入信息

## 其他
代数化简，通用性强

---------------------------------------------------------------------------------------------------

## Pass
HoistWaveInvariants
## 功能
针对 AMD GPU wave 相关计算的循环不变量外提 pass：把循环中每轮重复计算的 LDS 基址、lane ID 和 shuffle 索引移到函数入口，减少循环内的指令
## 作用位置（时机）
secondLowering 后，到llvm之前
发生在gpu.shuffle 和 DMA指令被翻译成 rocdl::readfirstlane / lane = mbcnt(...); index = shuffle_index(lane, constants); 之后 （）

## 匹配Pattern
for循环+ rocdl.readfirstlane/ mbcnt 计算地址操作

## 依赖的输入信息
无
## 其他
已经调整 DMA和 shuffle 的lowering时机，并将 HoistWaveInvariants 提前到 firstLowering之前。
