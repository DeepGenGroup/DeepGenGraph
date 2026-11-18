# TileAttention 伪代码对比

配置：BM=BN=D=128，QK tile=VO tile=32，online softmax，noncausal。
`GEMM(A, B, acc)` 表示用 FP32 累加矩阵乘积。
`load_Q/load_K/load_V` 包含 global view、shared allocation 和 copy；K 为转置方向的输入。
`scale_Q` 保留原 IR 的 FP16 缩放。下面描述 TileAttention 本身，尚未进行后续融合或流水调度。

## 优化前

```python
O = zeros_fp32(128, 128)
L = zeros_fp32(128, 1)
M = negative_inf_fp32(128, 1)

for n in range(0, 4096, 128):
    Q = load_Q(query_rows, columns=0:128)       # shared: 128×128 FP16
    K = load_K(rows=0:128, key_columns=n:n+128) # shared: 128×128 FP16
    S = GEMM(scale_Q(Q), K, zeros_fp32(128, 128))

    P, alpha, L, M = online_softmax_update(S, L, M)

    V = load_V(key_rows=n:n+128, columns=0:128) # shared: 128×128 FP16
    PV = GEMM(P, V, zeros_fp32(128, 128))
    O = alpha * O + PV

store_output(columns=0:128, value=fp16(O / L))
```

## 优化后

```python
O0, O1, O2, O3 = four_zeros_fp32(128, 32)
L = zeros_fp32(128, 1)
M = negative_inf_fp32(128, 1)

for n in range(0, 4096, 128):
    S = zeros_fp32(128, 128)
    for dk in range(0, 128, 32):
        Q = load_Q(query_rows, columns=dk:dk+32)       # shared: 128×32 FP16
        K = load_K(rows=dk:dk+32, key_columns=n:n+128) # shared: 32×128 FP16
        S = GEMM(scale_Q(Q), K, S)

    # 在四段 QK 全部累加后执行一次；四份输出共享 P、alpha、L、M。
    P, alpha, L, M = online_softmax_update(S, L, M)

    V0 = load_V(key_rows=n:n+128, columns=0:32)     # shared: 128×32 FP16
    PV0 = GEMM(P, V0, zeros_fp32(128, 32))
    V1 = load_V(key_rows=n:n+128, columns=32:64)
    PV1 = GEMM(P, V1, zeros_fp32(128, 32))
    V2 = load_V(key_rows=n:n+128, columns=64:96)
    PV2 = GEMM(P, V2, zeros_fp32(128, 32))
    V3 = load_V(key_rows=n:n+128, columns=96:128)
    PV3 = GEMM(P, V3, zeros_fp32(128, 32))

    O0 = alpha * O0 + PV0
    O1 = alpha * O1 + PV1
    O2 = alpha * O2 + PV2
    O3 = alpha * O3 + PV3

store_output(columns=0:32,   value=fp16(O0 / L))
store_output(columns=32:64,  value=fp16(O1 / L))
store_output(columns=64:96,  value=fp16(O2 / L))
store_output(columns=96:128, value=fp16(O3 / L))
```

## 两边使用相同的 online softmax 更新

```python
def online_softmax_update(S, old_L, old_M):
    new_M = maximum(old_M, row_max(S))
    alpha = exp2(old_M - new_M)
    P32 = exp2(S - new_M)
    new_L = alpha * old_L + row_sum(P32)
    return fp16(P32), alpha, new_L, new_M
```

单段 Q/K/V 暂存从32 KiB变为8 KiB；后续 Pass 可利用较小的分片安排物理槽复用、双缓冲和搬运重叠。总乘加量不变，四份 O 的总元素数也不变。
QK 改变浮点归约分组需要允许 reassociation；strict 模式保留完整 D 的 QK，VO 可独立切列。

[完整实际 IR 与 diff](comparison.md)
