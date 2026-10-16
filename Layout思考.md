buffer : 数据
op ： 计算任务
ops 通过 buffer 耦合

op的计算任务性质不同 ：是否需要跨thread通信。哪些轴能并行，哪些只能串行

```c++
[BM,D] @ [D,BN] = [BM,BN] -> [BM,BN] = gemm([BM,D], [D,BN])

for tileD in D :
    [BM,BN] += gemm([BM,tileD], [tileD,BN])
    

for tileD in D :
    for tileM in BM :
        for tileN in BN :
            // [BM,BN] += gemm([BM,tileD], [tileD,BN])
            C[m,n] += gemm([tileM, tileD], [tileD, tileN])
    

```
shm 本质作用 ： 跨warp数据共享。提高访问效率，将 membound 变为 calcbound 问题
如果没有跨warp数据共享（如block中仅一个warp，或 warp对这个data的访问次数太少， gm->rf->shm 成本不划算 ）


## Layout & fragment：
wgmma / wmma 中，对thread持有的buffer数据有要求。详细见指令手册。对于指令 wmma_mxnxk, 给定 warp_id lane_id, 能知道其在bufferA B C 中持有的元素坐标集合 {(i,j) | {i,j} ∈ Buffer 给定 warpId, laneId} 。
该映射关系 f(warpId,lane_id, regId) = buffer[i,j] 叫做 buffer的 layout
线程t持有的buffer 元素集合 叫做 buffer在给定tid下的 fragment

下面例子描述一个问题：
```
copy(smD, D)
copy(smE, E)
A = add(D,E)
B = copy(smB)
C = alloc_buffer(<m*n>)
gemm(A,B,C)
copy(C,smC)
```

如果串行化表达，则为
```cpp
// stage1
for(<bm*bk> : <m*k>){
    copy(smD, blocktile_D)
}
for(<bm*bk> : <m*k>){
    copy(smE, blocktile_E)
}
for(<bm*bk> : <m*k>){
    blocktile_A = add(blocktile_D,blocktile_E)
}
for(<bk*bn> : <k*n>){
    blocktile_B = copy(smB)
}
for(bm : m){
    for(bn : n){
        fill(blocktile_C, 0)
        for(bk : k){
            blocktile_C += gemm(blocktile_A,blocktile_B)
        }
    }
}
for(<bm*bn> : m*n){
    copy(blocktile_C,smC)
}


```


切分到blockTile后，假设 LAyout已经协商完毕，无冲突。那么：
```cpp
// stage1
copy(smD, blocktile_D)
copy(smE, blocktile_E)
blocktile_A = add(blocktile_D,blocktile_E)
blocktile_B = copy(smB)
fill(blocktile_C, 0)
for(bk : k){
    blocktile_C += gemm(blocktile_A,blocktile_B)
}
copy(blocktile_C,smC)

```

切到threadTile上 ：
```cpp
// stage2
for(<tm*tk> : <bm*bk>){
    copy(smD, threadtile_D)
}
for(<tm*tk> : <bm*bk>){
    copy(smE, threadtile_E)
}
for(<tm*tk> : <bm*bk>){
    threadtile_A = add(threadtile_D,threadtile_E)
}
for(<tk*tn> : <bk*bn>){
    threadtile_B = copy(smB)
}
for(tm : bm){
    for(tn : bn){
        fill(threadtile_C, 0)
        for(bk : k){
            for(tk : bk){
                threadtile_C += gemm(threadtile_A, threadtile_B)
            }
        }
    }
}
for(<tm*tn> : <bm*bn>){
    copy(threadtile_C,smC)
}

```
上述代码中有 threadtile_C += gemm(threadtile_A, threadtile_B)。 假设该gemm 尺寸大，需要多次 wmma 才能完成, 则视为:
threadtile_C，threadtile_A，threadtile_B 含多个 fragment。

于是 gemm 结构：
```cpp
// stage3
    for(tm : bm){
        for(tn : bn){
            fill(threadtile_C, 0)
            for(bk : k){
                for(tk : bk){
                    threadtile_C += gemm(threadtile_A, threadtile_B)
                }
            }
        }
    }
```
进一步切分到fragment级别：
```cpp
// stage4
    for(tm : bm){
        for(fm : tm){
            for(tn : bn){
                for(fn : tn){
                    fill(frag_C, 0)
                    for(bk : k){
                        for(tk : bk){
                            for(fk : tk){
                                frag_C += wmma(frag_A, frag_B)
                            }
                        }
                    }
                    save_frag_to_reg(frag_C)
                }
            }
        }
    }
```

对所有op：

切到threadTile上 ：
```cpp
// stage5
for(<tm*tk> : <bm*bk>){
    for(<fm * fk> : <tm*tk>){
        copy(smD, frag_D)  // [m,k]
    }
}
for(<tm*tk> : <bm*bk>){
    for(<fm * fk> : <tm*tk>){
        copy(smE, frag_E)  // [m,k]
    }
}
for(<tm*tk> : <bm*bk>){
    for(<fm * fk> : <tm*tk>){
        frag_A = add(frag_D, frag_E)  // frag_A=[m,k]
    }
}
for(<tk*tn> : <bk*bn>){
    for(<fk*fn> : <tk*tn>){
        frag_B = copy(smB)  // frag_B=[k,n]
    }
}
threadtile_C = alloc_threadtile(<tm/fm * tn/fn * fm * fn>)
for(tm : bm){
    for(fm : tm){
        for(tn : bn){
            for(fn : tn){
                frag_C = get_frag_from(threadtile_C)
                fill(frag_C, 0)  // [m,n]
                for(bk : k){
                    for(tk : bk){
                        for(fk : tk){
                            frag_C += wmma(frag_A, frag_B)  // fragA=[m,k]  fragB[k,n] frag_C[m,n]
                        }
                    }
                }
                save_frag_to(frag_C, threadtile_C)
            }
        }
    }
}
for(<tm*tn> : <bm*bn>){
    copy(threadtile_C,smC)
}


```
所有op都按fragment统一粒度
此时，可观察到许多for循环重复，因此经过reorder后可合并
（注意：该步骤需要知道每个buffer 两个轴的语义(二维shape上 对应mnk的哪两个维度)—— 这可通过Layout推定时自动传播得到）
```cpp
// stage 6 : reorder 调整loop为 kmn 结构（或其他更合理顺序，mnk mkn 等等，方便融合和内部reorder）
// 注意：代码仅作为例子表达思想。实际操作过程里，gm->sm 的拷贝一般不按照 fragment来，因为会效率低下. 一般直接用连续 packet 传输；sm->reg 的拷贝是否按frag组织待探索
// 注意：loop次序会影响每个fragment的liveliness、保有数量、读写计算次数。需综合评价（计算+ 访存/同步代价 + reg压力）


for(<tm*tk> : <bm*bk>){
    for(<fm * fk> : <tm*tk>){
        copy(smD, frag_D)  // [m,k]
        copy(smE, frag_E)  // [m,k]
        frag_A = add(frag_D, frag_E)  // frag_A=[m,k]
    }
}

for(<tk*tn> : <bk*bn>){
    for(<fk*fn> : <tk*tn>){
        frag_B = copy(smB)  // frag_B=[k,n]
    }
}
threadtile_C = alloc_threadtile(<tm/fm * tn/fn * fm * fn>)
for(tm : bm){
    for(tn : bn){
        for(fm : tm){
            for(fn : tn){
                frag_C = get_frag_from(threadtile_C)
                fill(frag_C, 0)  // [m,n]
                for(bk : k){
                    for(tk : bk){
                        for(fk : tk){
                            frag_C += wmma(frag_A, frag_B)  // fragA=[m,k]  fragB[k,n] frag_C[m,n]
                        }
                    }
                }
                save_frag_to(frag_C, threadtile_C)
            }
        }
        copy(threadtile_C,smC)
    }
}

```

| 候选结构 | A 准备次数 | B load 次数 | 同时存活的 C fragment |
|---|---:|---:|---:|
| `m→n→k`，每次完成一个 C | `MNK` | `MNK` | 1 |
| `m→k→n`，保留一行 C，A 在 n 外准备 | `MK` | `MNK` | N |
| 你的 `k→m→n`，每个 k 先缓存整行 B | `MK` | `KN` | MN |

## 一个可能的评价思路（性能模型）
对于没Layout的kernel，每个op间简单地用shm做界面做传输。同时live的reg压力最小
此时可根据blockTile级别的IR得到 访存&计算次数 (F,J), 也能知道每个线程最多持有多少reg (R)
做了Layout推定后，不冲突的op间没了shm界面。省掉了 reg->shm->reg 的过程。但同时live的reg变多。此时可计算出访存&计算次数& reg峰值 (F1, J1,R1)
综合评价 (F1,J1,R1) & (F,J,R). 方法：根据寄存器文件大小 & shm，根据sm用量 & R计算占有率。进而计算出完整kernel需要的调度轮数 S
每轮进行计算&访存次数=(F,J).总代价= S*(F,J)   
F的代价一般比J高。访存比计算慢。具体处理上可以简单乘以一个系数。最后计算出 TotalCost = S*(lambda*F+J), lambda>1
想办法令 TotalCost 尽可能小



## 其他思考

deepgengraphIR ： 表示kernel要算什么，总体任务
friskIR-block阶段： 表达切分到block层面 每个block要算啥。这个是 asuka 定了的
Layout推定 ： 尽可能减少 shm 中转。但代价是访存碎片化。需更多的访存指令。确定每个线程，每个op 需要哪些数据输入，输出什么数据（索引关系）
friskIR-thread阶段： Layout定了后，切分到 thread层面，每个thread要算什么。 
friskIR-fragment 阶段 ： friskIR-thread阶段细化为 fragment粒度， 更好地做指令reorder（计算访存指令交错）

其他优化手段：用不用 multiversionbuffer？怎么排流水？向量化访存宽度多少，和已有Layout方案冲突怎么办如何取舍？—— 回答“怎么算”的问题

计算op：数据怎么算
访存op：op的数据从哪来？shm还是global——图IR只有global 没有shm层。shm 是 friskIR-block阶段自己加上去的。

——需要做的：
1.设计数据链路（op的变量，需要从gm搬到shm吗？还是gm直接到reg？结果写到shm还是留在reg？）
2.根据数据链路设计，得到 friskIR-block阶段 memref的具体memspace


## Layout的其他关联

Layout：本质为索引关系。表达一个 buffer的数据由block中的thread以F规律持有。
    F = thread排列+ thread_creg 
    thread的排布按照 warp_layout + 行列优先 + warp_repeat 
    连续元素用 thread_creg 表达

buffer的数据来源 ：上一个op的输出
    上一op：Layout沿着op调用链条 上下游进行传播
    来自shm or global 怎么办？还需要考虑别的因素：
        shm/gm 一般尺寸大于buffer shape —— 新增 buffer_repeat 描述？
        shm : 还需考虑 thread 从当中read / write时的性能 —— vecLoad / bank冲突布局。此时可能要调Layout
        gm ：考虑数据read的效率 —— vecLoad，指令尽可能宽 
    
如何描述一个shm buffer的数据排列（访问模式）
    假定已知：
    bankline_size=32*4B=128B
    thread_copy_bytes=8bytes (4xf16)
    此时能计算： threads_per_bankline = 128 / 8 = 16 threads, 每bankline 16 threads
    又因为：thread_copy_bytes 决定phase数（参与bankconflict计量的lane数目） -> phase=16 
    threads_per_bankline <= phase，故没有冲突
    还需要得知其元素的逻辑坐标（在GM里的位置）—— thread_creg + warp_layout + major 可共同算出
    元素的逻辑坐标（在GM里的位置）决定了怎么从GM把数据写进 shm —— 
        thread_creg 的形状 + major 能得知 数据在GM中在行方向连续的宽度。
    
    假定读取指令=vecLoad_64b 一条可读4个f16.此时 bankconflictphase = 16
    
    图a:
    [t0,0 - t0,3][t1,0 - t1,3][t2,0 - t2,3] ... [t15,0 - t15,3]    // 一个bank行，能放下16thread*（4*f16）
    [t16,0 - t16,3] ... ...                     [t31,0 - t31,3]
    [t32,0 - t32,3] ... ...
    [t48,0 - t48,3] ... ...                     [t63,0 - t64,3]
    此布局下，由于一个bank行的threads刚好等于phase，所以没bank冲突

    设 x,y 表示 GM中的元素在x行y列，元素在GM中的逻辑坐标为固定的。据上述布局，能得到 shm里的元素的逻辑坐标 ：
    图b:
    [0,0 - 0,3][1,0 - 1,3] ... ... [15,0 - 15,3]  // 各个小块负责的tid见图a。即：连续的两个lane读取的GM元素在不同行
    [0,4 - 0,7][1,4 - 1,7] ... ... [15,4 - 15,7]
    [0,8 - 0,11][1,8 - 1,11] ...   
    [0,12 - 0,15] ... ...          [15,12-15,15]

    若DMA能从GM中按照图b方式读取数据，再按照图a方式写入shm。则可实现GM的 向量化读，也能保证shm无 bank conflict
    amdgpu的DMA为 GM可离散读取，但写只能按lane连续写。所以a形式写入可行。只需要改一下GM读取的addr计算 
    根据图b 能求出per-lane的地址计算公式。
    各个小块负责的tid见图a。即：连续的两个lane读取的GM元素在不同行
    放到GM视角，应该为：
    图c：
    [t0:0-3][t16:4-7] ... ... [t48:12-15]
    [t1:0-3][t17:4-7]  ...
    [t2:0-3][t18:4-7]  ... 
    ...
    [t15:0-3][t31:4-7] ...    [t63:12-15]

    per-lane的地址：行=lane // 16, 列=lane% 16*4 (按元素。如果要字节，还需乘元素宽度)
    好处：thread_creg 和 GM 拷贝的宽度联系起来了。  坏处：没打满GM 搬运指令的宽度。最大能达到4*f32，这才用了一半，不够高效率
    改进：可尝试拉满DMA宽度到 4*f32 即 8*f16，此时由于一个wmma只能吃4*f16，意味着一次DMA搬了两次wmma所需的数
    但需要注意：对于连续的两次wmma，在GM上其数据为离散的两个4*f16， 而不i是一个连续的 8*f16.一旦这样DMA了，则shm中的布局：
    [0,0 - 0,3 0,4- 0,7][1,0 - 1,3 1,4 - 1,7] ... ... [15,0 - 15,3 15,4 - 15,7]
    [0,12 - 0,15 0,16-0,19] ... ...          
    本质 ： f16*8 写shm，f16*4 消费shm。
    
    但对于wmma来说，其还是读4个data。读shm的方式还是不变的。又因为DMA不用 vgpr中转，所以只考虑最后的shm就行 坏处：shm翻倍了（只让一半的线程干活呢？是否就不用翻倍shm了。读shm时还按照wmma需要的 f16*4 读）
    
    师兄的做法：DMA宽度打满128bit，shm翻倍（取决于creg和DMA宽度的比值，得到 kpack=2，即当前shm的数据供两次wmma用）
    我们的做法： 可将 kpack 作为一个可选参数 . 先通过HWSpecification 查看硬件能力，选择最大DMA宽度。
        之后选最大宽度的DMA，将gm数据拷贝到shm。数据排布根据 shmio 定（每个lane从global的哪里读数据，一次读多少。根据lane*宽度写入到shm指定位置）
        如果DMA宽度大于 wmma thread所用数据宽度, 则：kpack=DMA width / wmma 输入的元素宽。之后
        策略1 useKPack :. shm据 kpack 尺寸在行方向翻倍。shape修改体现在IR上
        2. noKPack : 不采用翻倍策略。将 GM-shm async DMA搬运线程数减少
        根据输入参数选择策略





## attention 
定义：Q=Q0*scale, P0=gemm(QK), P1=P0+mask, P=softmax(P1), O=gemm(PV)
应作优化： 
1. Q K V 从gm -> shm ，向量化copy，拉满向量化指令宽度（4个f32 or 8个f16）。每个线程连续
2. gemmQK  gemmPV  使用fragment 组织for循环。将fragment加载和 wmma计算穿插进行
3. Op做Layout推定。shmK  shmV 的数据根据Layout交给不同线程。尽可能用向量化 copy
4. 其他op根据Layout的结果，也按照fragment 组织for循环生成IR
5. fragment级别的代码做 访存和计算的交替。掩盖
6. O的写回 reg->gm 尽可能向量化copy 
7. pipeline（shm尺寸限制，可能引入负面效果）
帮我看下上述优化都有了吗？做了这些之后，性能还有哪些进步空间

8. buffer的规约轴不能跨wave。

## 更好的图IR表达？
```python
# Q: [B, H, M, D]
# K: [B, H, N, D]
# V: [B, H, N, Dv]

# 1. QK^T
k0 = linalg.transpose(K, [0, 1, 3, 2])
p0 = linalg.matmul(Q, k0)                # [B,H,M,N]

# 2. Scale
scale = 1.0 / sqrt(D)
p1 = linalg.mul(p0, scale)

# 3. Mask
mask = frisk.mask(shape=[M,N])           # 0 or -inf
p2 = linalg.add(p1, mask)

# 4. Softmax
p3 = linalg.reduce_max(p2, axis=-1, keepdims=True)
p4 = linalg.sub(p2, p3)
p5 = linalg.exp(p4)
p6 = linalg.reduce_sum(p5, axis=-1, keepdims=True)
p7 = linalg.div(p5, p6)

# 5. Attention × V
O = linalg.matmul(p7, V)                 # [B,H,M,Dv]

return O
```

进一步：
```python
# 串行化表达 —— 为了做并行化做铺垫。便于分析哪些for能切，哪些不能
# Q: [B,H,S,D]
# K: [B,H,S,D]
# V: [B,H,S,D]
#
# BM, BN, BK

for b in range(B):
  for h in range(H):
    for m in range(0, S, BM):

      # Online Softmax state
      row_max = full([BM, 1], -inf)
      row_sum = zeros([BM, 1])
      acc     = zeros([BM, D])

      for n in range(0, S, BN):
        # ----------------------------------
        # 1. QK^T with D-dimension tiling
        # ----------------------------------
        p = zeros([BM, BN])
        pv1 = zeros([BM, BK])

        for k in range(0, D, BK):
          qtile = tile_from(Q, [b,h,m,k], [1,1,BM,BK])
          ktile = tile_from(K, [b,h,n,k], [1,1,BN,BK])

          kt = transpose(ktile)
          p0 = matmul(qtile, kt)  # [BM,BN]
          p  = add(p, p0)   # [BM,BN]

        # ----------------------------------
        # 2. Scale and causal mask
        # ----------------------------------

        p1 = mul(p, 1.0 / sqrt(D))
        mask = frisk.mask(m, n, BM, BN)  # [BM,BN]
        p2 = add(p1, mask)

        # ----------------------------------
        # 3. Online Softmax
        # ----------------------------------

        block_max = reduce_max(p2, axis=1)  # [BM,1]
        new_max = maximum(row_max, block_max) # [BM,1]

        alpha = exp(row_max - new_max)  # [BM,1]
        p3 = exp(p2 - new_max)   # [BM,BN] - [BM,1]  braodcast-semantic 

        block_sum = reduce_sum(p3, axis=1)
        new_sum = row_sum * alpha + block_sum

        # ----------------------------------
        # 4. Accumulate PV
        # ----------------------------------

        for k in range(0, D, BK):
          vtile = tile_from(V, [b,h,n,k], [1,1,BN,BK])  # [BN,BK]
          pv0 = matmul(p3, vtile)  # [BM,BK]
          pv  = add(pv, pv0)

        acc = acc * alpha + pv

        row_max = new_max
        row_sum = new_sum

      # ----------------------------------
      # 5. Normalize and store
      # ----------------------------------

      out = acc / row_sum
      tile_store(O, [b,h,m,0], out)
```


再进一步，用fragment表示 matmul 结构
```python
# 串行化表达 —— 为了做并行化做铺垫。便于分析哪些for能切，哪些不能
# Q: [B,H,S,D]
# K: [B,H,S,D]
# V: [B,H,S,D]
#
# BM, BN, BK

for b in range(B):  # 可并行
  for h in range(H):  # 可并行
    for m in range(0, S, BM):  # 串行轴

      # Online Softmax state
      row_max = full([BM, 1], -inf)
      row_sum = zeros([BM, 1])
      acc     = zeros([BM, D])

      for n in range(0, S, BN):
        # ----------------------------------
        # 1. QK^T with D-dimension tiling
        # ----------------------------------
        p = zeros([BM, BN])
        pv = zeros([BM, BK])

        for k in range(0, D, BK): # reduce 轴
          qtile = tile_from(Q, [b,h,m,k], [1,1,BM,BK])
          ktile = tile_from(K, [b,h,n,k], [1,1,BN,BK])
          kt = transpose(ktile)
          # p0 = matmul(qtile, kt)  # [BM,BK] @ [BK,BN] = [BM,BN]
          p0 = zeros([BM,BN])
          for mmaM in range(0,BM, mma_m) :
            for mmaN in range(0,BN, mma_n) :
              for mmaK in range(0, BK, mma_k):  # reduce 轴
                fq = frag_from(qtile, mmaM, mmaK)  # [mma_m, mma_k]
                fk = frag_from(kt, mmaK, mmaN)  # [mma_k, mma_n ]
                fv = warp_mma(fq,fk) # [mma_m ,mma_n]
                ptemp = tile_from(p0,[ mmaM, mmaN])  # [mma_m ,mma_n]
                p0temp = add(fv, ptemp)  # [mma_m ,mma_n]
                tile_store_to(p0temp, p0)  # [mma_m ,mma_n] store to [BM,BN]
          p  = add(p, p0)   # [BM,BN]

        # ----------------------------------
        # 2. Scale and causal mask
        # ----------------------------------

        p1 = mul(p, 1.0 / sqrt(D))
        mask = frisk.mask(m, n, BM, BN)  # [BM,BN]
        p2 = add(p1, mask)

        # ----------------------------------
        # 3. Online Softmax
        # ----------------------------------

        block_max = reduce_max(p2, axis=1)  # [BM,1]
        new_max = maximum(row_max, block_max) # [BM,1]

        alpha = exp(row_max - new_max)  # [BM,1]
        p3 = exp(p2 - new_max)   # [BM,BN] - [BM,1]  braodcast-semantic 

        block_sum = reduce_sum(p3, axis=1)
        new_sum = row_sum * alpha + block_sum

        # ----------------------------------
        # 4. Accumulate PV
        # ----------------------------------
        for k in range(0, D, BK):  # reduce 轴
          vtile = tile_from(V, [b,h,n,k], [1,1,BN,BK])  # [BN,BK]

          # pv0 = matmul(p3, vtile)  #  [BM,BN] @ [BN,BK]
          pv0 = zeros([BM,BK])
          for mmaM in range(0,BM, mma_m) :
            for mmaN in range(0,BK, mma_n) :
              for mmaK in range(0, BN, mma_k):  # reduce 轴
                fp3 = frag_from(p3, mmaM, mmaK)  # [mma_m, mma_k]
                fvtile = frag_from(vtile, mmaK, mmaN)  # [mma_k, mma_n ]
                fo = warp_mma(fp3,fvtile) # [mma_m ,mma_n]
                ptemp = tile_from(pv0, mmaM, mmaN)  # [mma_m ,mma_n]
                p0temp = add(fo, ptemp)  # [mma_m ,mma_n]
                tile_store_to(p0temp, pv0, [mmaM,mmaN])  # [mma_m ,mma_n] store to [BM,BN]
          pv  = add(pv, pv0)

        acc = acc * alpha + pv

        row_max = new_max
        row_sum = new_sum

      # ----------------------------------
      # 5. Normalize and store
      # ----------------------------------

      out = acc / row_sum
      tile_store(O, [b,h,m,0], out)
```

改完matmul到frag上的表示后，其他op也能用frag 表示。依据是matmul 参数的传播。
（ buffer-frag 建立联系。 frag = frag_from(buffer) frag也能存回buffer ）
之后，通过Layout分析，可对 frag-buffer-frag 的模式做简化（如果 buffer 的 frag_from 和 frag_to 索引规则一样，就能保持数据在 frag 里）

还要做循环分析 ： 哪些for必须串，哪些能并行(是不是有跨迭代依赖)

```python
# 串行化表达 —— 为了做并行化做铺垫。便于分析哪些for能切，哪些不能
# Q: [B,H,S,D]
# K: [B,H,S,D]
# V: [B,H,S,D]
#
# BM, BN, BK

for b in range(B):  # 可并行 step=1， 可直接分到gpu.blockId 上
  for h in range(H):  # 可并行  step=1， 可直接分到gpu.blockId 上
    for m in range(0, S, BM):  # 可并行

      # Online Softmax state
      row_max = full([BM, 1], -inf)
      row_sum = zeros([BM, 1])
      acc     = zeros([BM, D])

      for n in range(0, S, BN):  # 可并行？
        # ----------------------------------
        # 1. QK^T with D-dimension tiling
        # ----------------------------------
        p = zeros([BM, BN])
        pv = zeros([BM, BK])

        for k in range(0, D, BK): # # 串行
          qtile = tile_from(Q, [b,h,m,k], [1,1,BM,BK])
          ktile = tile_from(K, [b,h,n,k], [1,1,BN,BK])
          kt = transpose(ktile)
          # p0 = matmul(qtile, kt)  # [BM,BK] @ [BK,BN] = [BM,BN]
          p0 = zeros([BM,BN])
          for mmaM in range(0,BM, mma_m) :
            for mmaN in range(0,BN, mma_n) :
              for mmaK in range(0, BK, mma_k):  # # 串行
                fq = frag_from(qtile, mmaM, mmaK)  # [mma_m, mma_k]
                fk = frag_from(kt, mmaK, mmaN)  # [mma_k, mma_n ]
                fv = warp_mma(fq,fk) # [mma_m ,mma_n]
                ptemp = tile_from(p0,[ mmaM, mmaN])  # [mma_m ,mma_n]
                p0temp = add(fv, ptemp)  # [mma_m ,mma_n]
                tile_store_to(p0temp, p0)  # [mma_m ,mma_n] store to [BM,BN]
          p  = add(p, p0)   # [BM,BN]

        # ----------------------------------
        # 2. Scale and causal mask
        # ----------------------------------
        p1 = mul(p, 1.0 / sqrt(D))
        mask = frisk.mask(m, n, BM, BN)  # [BM,BN]
        p2 = add(p1, mask)

        # ----------------------------------
        # 3. Online Softmax
        # ----------------------------------

        block_max = reduce_max(p2, axis=1)  # [BM,1]
        new_max = maximum(row_max, block_max) # [BM,1]

        alpha = exp(row_max - new_max)  # [BM,1]
        p3 = exp(p2 - new_max)   # [BM,BN] - [BM,1]  braodcast-semantic 

        block_sum = reduce_sum(p3, axis=1)
        new_sum = row_sum * alpha + block_sum

        # ----------------------------------
        # 4. Accumulate PV
        # ----------------------------------
        for k in range(0, D, BK):  # # 串行
          vtile = tile_from(V, [b,h,n,k], [1,1,BN,BK])  # [BN,BK]

          # pv0 = matmul(p3, vtile)  #  [BM,BN] @ [BN,BK]
          pv0 = zeros([BM,BK])
          for mmaM in range(0,BM, mma_m) :
            for mmaN in range(0,BK, mma_n) :
              for mmaK in range(0, BN, mma_k):  # # 串行
                fp3 = frag_from(p3, mmaM, mmaK)  # [mma_m, mma_k]
                fvtile = frag_from(vtile, mmaK, mmaN)  # [mma_k, mma_n ]
                fo = warp_mma(fp3,fvtile) # [mma_m ,mma_n]
                ptemp = tile_from(pv0, mmaM, mmaN)  # [mma_m ,mma_n]
                p0temp = add(fo, ptemp)  # [mma_m ,mma_n]
                tile_store_to(p0temp, pv0, [mmaM,mmaN])  # [mma_m ,mma_n] store to [BM,BN]
          pv  = add(pv, pv0)

        acc = acc * alpha + pv

        row_max = new_max
        row_sum = new_sum

      # ----------------------------------
      # 5. Normalize and store
      # ----------------------------------

      out = acc / row_sum
      tile_store(O, [b,h,m,0], out)
```