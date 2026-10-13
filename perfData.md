# 性能分析与kernel优化
## 定义
问题输入 ： bhsd QKV
大体计算过程：
copy QTile 从global -> shm
Q = scale * Q0
for BN : seq:
  copy Ktile 从global -> shm
  copy Vtile 从global -> shm
  P0 = gemm(Q,K)
  P = softmax(mask + P0) （online softmax）
  O = gemm(P,V)
out = O/reduce_acc 
写回out到 global O

## 已有优化手段汇总
1. splitD（attentionIR层， dim切分。即qtile=<BM*BK>, ktile=<BK*BN> vtile=<BN*BK>, dim 轴多一个reduce（for BK : d））
2. copy QTile Ktile Vtile 时，采用向量化。向量化宽度尽可能打满（最大128bit，即f16*8）
3. P layout重整：
  3.1 采用普通策略（reg-shm-reg 传输）/ 
  3.2 采用shuffle方式：通过shuffle 直接交换warp内thread的数据，使得thread持有位置连续的四个结果 / 
  3.3 或者定制化方法（Q不变，改变K的数据排布，使得4次wmma后，thread自然持有位置连续的4个数据，符合gemmPV的要求，已加入）
4. 使用fragment为粒度，组织op以及for循环。方便进行 fragment 加载-计算的交叠（用Pass 做 reorder）
5. 写回global O 时直接用reg，不经过shm中转
6. 异步拷贝KV（未实现）（rocdl.global.load.async.lds 或者其向量化版本）

## 师兄的已有高效kernel尺寸参照：
BM=128 BN=128 BK=32 , shm= 16384, 一个CU 能跑4block。无double-buffer ，异步copy QKV
