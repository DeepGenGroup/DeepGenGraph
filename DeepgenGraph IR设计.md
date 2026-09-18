# DeepgenGraph IR设计

## DeepgenGraphIR（Asuka）

```
#loc = loc(unknown)
module {
  func.func @Attn(%arg0: tensor<1x4096x32x128xf16> loc(unknown), %arg1: tensor<1x4096x32x128xf16> loc(unknown), %arg2: tensor<1x4096x32x128xf16> loc(unknown)) -> tensor<1x4096x32x128xf16> {
    %cst = arith.constant dense<1.131250e+01> : tensor<1xf16> loc(#loc)
    %0 = deepgengraph.trilu diagonal = 1, is_upper = true, shape = [4096, 4096], val = 0xFC00 : f16 loc(#loc)
    %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16> loc(#loc)
    %2 = deepgengraph.permute %arg2, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16> loc(#loc)
    %3 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x128x4096xf16> loc(#loc)
    %4 = deepgengraph.dot %1, %3 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc)
    %5 = deepgengraph.div %4, %cst : (tensor<1x32x4096x4096xf16>, tensor<1xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc)
    %6 = deepgengraph.add %5, %0 : (tensor<1x32x4096x4096xf16>, tensor<4096x4096xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc)
    %7 = deepgengraph.convert %6, type = f32 : (tensor<1x32x4096x4096xf16>) -> tensor<1x32x4096x4096xf32> loc(#loc)
    %8 = deepgengraph.exp %7 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32> loc(#loc)
    %9 = deepgengraph.reduce(%8), dim = -1, op =  ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32> loc(#loc)
    %10 = deepgengraph.div %8, %9 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32> loc(#loc)
    %11 = deepgengraph.convert %10, type = f16 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16> loc(#loc)
    %12 = deepgengraph.dot %11, %2 : (tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16> loc(#loc)
    %13 = deepgengraph.permute %12, dims = [0, 2, 1, 3] : (tensor<1x32x4096x128xf16>) -> tensor<1x4096x32x128xf16> loc(#loc)
    return %13 : tensor<1x4096x32x128xf16> loc(#loc)
  } loc(#loc)

  deepgengraph.kernel @Attn_p2(%Q: tensor<1x4096x32x128xf16>, %V: tensor<1x4096x32x128xf16>, %K: tensor<1x4096x32x128xf16>) -> tensor<1x4096x32x128xf16> attributes {parallel_map = [{arg_dims = [0, 0, 0], res_dims = [0], size_per_unit = 1 : i64, unit_num = 1 : i64}, {arg_dims = [1, -1, -1], res_dims = [1], size_per_unit = 64 : i64, unit_num = 64 : i64}, {arg_dims = [2, 2, 2], res_dims = [2], size_per_unit = 1 : i64, unit_num = 32 : i64}]} {
    %pQ = deepgengraph_triton.ptr_of %Q : (tensor<1x4096x32x128xf16>) -> !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>
    %pV = deepgengraph_triton.ptr_of %V : (tensor<1x4096x32x128xf16>) -> !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>
    %pK = deepgengraph_triton.ptr_of %K : (tensor<1x4096x32x128xf16>) -> !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>
    %O = deepgengraph_triton.empty_ptr type = tensor<1x4096x32x128xf16> : <tensor<1x4096x32x128xf16>>
    deepgengraph_triton.device_kernel args = [%pQ, %pV, %pK, %O], grid = [1, 64, 32] {
    ^bb0(%bz: index, %bx: index, %by: index, %argQ: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, %argV: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, %argK: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, %argO: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>):
      %cst = arith.constant dense<0.127531052> : tensor<1xf32>
      %cst_0 = arith.constant 0xFF800000 : f32
      %cst_1 = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c4096 = arith.constant 4096 : index
      %c128 = arith.constant 128 : index
      // %5 = arith.muli %bx, %c64 : index  // bx * 64
      // %6 = arith.muli %5, %c4096 : index  // bx * 64 * 4096
      // %7 = arith.muli %by, %c128 : index  // by * 128
      // %8 = arith.addi %6, %7 : index  // bx * 64 * 4096 + by * 128
      %5 = arith.muli %by, %c128 : index  // by * 128
      %6 = arith.muli %5, %c4096 : index  // by * 128 * 4096
      %7 = arith.muli %bx, %c64 : index  // bx * 64
      %70 = arith.muli %7, %c128 : index  // bx * 64 * 128
      %8 = arith.addi %6, %70 : index  // by * 128 * 4096 + bx * 64 * 128  [bz , by , bx*64 , 128]
      %9 = deepgengraph_triton.block_ptr_of base = %argQ, base_offset = %8, shape = [64, 128], stride = [4096, 1], offset = [0, 0], block_shape = [64, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<64x128xf16>}>  // 逻辑上[64,128] , 行连续
      %10 = deepgengraph_triton.block_load %9 : (!deepgengraph_triton<block_ptr{tensor<64x128xf16>}>) -> tensor<64x128xf16>
      %11 = deepgengraph_triton.block_ptr_of base = %argO, base_offset = %8, shape = [64, 128], stride = [4096, 1], offset = [0, 0], block_shape = [64, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<64x128xf16>}>  // 逻辑上[64,128] , 行连续
      %12 = deepgengraph.convert %cst, type = f16 : (tensor<1xf32>) -> tensor<1xf16>
      %13 = deepgengraph.mul %10, %12 : (tensor<64x128xf16>, tensor<1xf16>) -> tensor<64x128xf16>
      %14 = deepgengraph.zero shape = [64, 128], type = f32 : () -> tensor<64x128xf32>
      %15 = deepgengraph.zero shape = [64, 1], type = f32 : () -> tensor<64x1xf32>
      %16 = arith.addi %5, %c64 : index  // by * 128 + 64
    %17 = deepgengraph_triton.block_ptr_of base = %argK, base_offset = %8, shape = [128, 4096], stride = [1, 4096], offset = [0, 0], block_shape = [128, 32], order = [0, 1] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x32xf16>}>  // base_offset = by * 128 * 4096 + bx * 64 * 128  逻辑上[128,32] , 列连续。global需按照 [1,32,128,4096] 做permute 
      %18 = deepgengraph_triton.block_ptr_of base = %argV, base_offset = %8, shape = [4096, 128], stride = [4096, 1], offset = [0, 0], block_shape = [32, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<32x128xf16>}>  // 逻辑上[32,128] 行连续
      %temp = arith.muli %bx , %c64 : index 
      %loopUb = arith.addi %temp , %c32 : index // BM * bx + BN
      %19:4 = scf.for %arg10 = %c0 to %loopUb step %c32 iter_args(%tempK = %17, %tempV = %18, %arg13 = %14, %arg14 = %15) -> (!deepgengraph_triton<block_ptr{tensor<128x32xf16>}>, !deepgengraph_triton<block_ptr{tensor<32x128xf16>}>, tensor<64x128xf32>, tensor<64x1xf32>) {
        %22 = deepgengraph_triton.block_load %tempK : (!deepgengraph_triton<block_ptr{tensor<128x32xf16>}>) -> tensor<128x32xf16>
        %23 = deepgengraph_triton.block_load %tempV : (!deepgengraph_triton<block_ptr{tensor<32x128xf16>}>) -> tensor<32x128xf16>
        %24 = deepgengraph.precise_dot_op %13, %22, acc = f32 : (tensor<64x128xf16>, tensor<128x32xf16>) -> tensor<64x32xf32>
        %25 = deepgengraph.mask starts = [%7, %arg10], sizes = [64, 32], type = f32 {
        ^bb0(%arg15: index, %arg16: index):
          %34 = arith.addi %arg15, %c1 : index
          %35 = arith.cmpi ule, %34, %arg16 : index
          %36 = scf.if %35 -> (f32) {
            scf.yield %cst_0 : f32
          } else {
            scf.yield %cst_1 : f32
          }
          deepgengraph.mask_yield %36 : f32
        } : (index, index) -> tensor<64x32xf32>
        %26 = deepgengraph.add %24, %25 : (tensor<64x32xf32>, tensor<64x32xf32>) -> tensor<64x32xf32>
        %27 = deepgengraph.exp2 %26 : (tensor<64x32xf32>) -> tensor<64x32xf32>
        %28 = deepgengraph.reduce(%27, init = %arg14), dim = 1, op =  ADD, keep_dim = true : (tensor<64x32xf32>, tensor<64x1xf32>) -> tensor<64x1xf32>
        %29 = deepgengraph.convert %27, type = f16 : (tensor<64x32xf32>) -> tensor<64x32xf16>
        %30 = deepgengraph.precise_dot_op %29, %23, acc = f32 : (tensor<64x32xf16>, tensor<32x128xf16>) -> tensor<64x128xf32>
        %31 = deepgengraph.add %arg13, %30 : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
        %32 = deepgengraph_triton.block_advance %tempK, offsets = [0, 32] : (!deepgengraph_triton<block_ptr{tensor<128x32xf16>}>) -> !deepgengraph_triton<block_ptr{tensor<128x32xf16>}>  // D x BN
        %33 = deepgengraph_triton.block_advance %tempV, offsets = [32, 0] : (!deepgengraph_triton<block_ptr{tensor<32x128xf16>}>) -> !deepgengraph_triton<block_ptr{tensor<32x128xf16>}>  // BN x D
        scf.yield %32, %33, %31, %28 : !deepgengraph_triton<block_ptr{tensor<128x32xf16>}>, !deepgengraph_triton<block_ptr{tensor<32x128xf16>}>, tensor<64x128xf32>, tensor<64x1xf32>
      }
      %20 = deepgengraph.div %19#2, %19#3 : (tensor<64x128xf32>, tensor<64x1xf32>) -> tensor<64x128xf32>
      %21 = deepgengraph.convert %20, type = f16 : (tensor<64x128xf32>) -> tensor<64x128xf16>
      deepgengraph_triton.block_store %11, %21 : (!deepgengraph_triton<block_ptr{tensor<64x128xf16>}>, tensor<64x128xf16>) -> ()
    } : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>) -> ()
    %4 = deepgengraph_triton.tensor_from %O : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>) -> tensor<1x4096x32x128xf16>
    deepgengraph.return %4 : tensor<1x4096x32x128xf16>
  }
} loc(#loc)

```

IR特点：表达算子内部宏观计算逻辑。基于tensor+ ptr 语义的计算逻辑表示。不考虑数据的存储位置。

没有索引的概念，仅通过指针移动来表达从tensor取一个slice的语义

理由：asuka项目本身采用triton作为后端DSL。triton内部使用指针块+指针偏移。因此IR的设计以方便转换到triton code为优先



2. FriskIR

   ```mlir
   // ---------- after createConvertFriskToBasePass ---------
   module {
     func.func @Attn_p2(%Q: memref<1x32x4096x128xf16, 1>, %V: memref<1x32x4096x128xf16, 1>, %K: memref<1x32x128x4096xf16, 1>, %O: memref<1x32x4096x128xf16, 1>) attributes {thread_num = 128 : i32} {
       %block_id_z = gpu.block_id  z {range = 1 : index}
       %block_id_y = gpu.block_id  y {range = 64 : index}
       %block_id_x = gpu.block_id  x {range = 32 : index}
       %thread_id_x = gpu.thread_id  x {range = 128 : index}
       %c64 = arith.constant 64 : index
       %c1 = arith.constant 1 : index
       %cst = arith.constant 0.000000e+00 : f32
       %cst_0 = arith.constant 0xFF800000 : f32
       %cst_1 = arith.constant 0.127531052 : f32
       %0 = arith.muli %block_id_y, %c64 : index  // by * 64
       %qtile = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<64x128xf16, 3>
       %2 = frisk.buffer_view %Q[0, (%block_id_x ) mod 32, (%block_id_y * 64) mod 4096, 0], ranges = [64, 128] : memref<1x32x4096x128xf16, 1> -> memref<64x128xf16, 1>
       %qtile_copied = frisk.copy %2 to %qtile [affine_map<() -> (2)>] : memref<64x128xf16, 1>, memref<64x128xf16, 3> -> memref<64x128xf16, 3>
       %3 = arith.truncf %cst_1 : f32 to f16
       %qtileScale = frisk.mul %qtile_copied, %3 : memref<64x128xf16, 3>, f16 -> memref<64x128xf16, 3>
       %5 = frisk.zero {initIdx = 2 : index} : memref<64x128xf32>
       %6 = frisk.zero {initIdx = 3 : index} : memref<64x1xf32>
       %ktile = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
       %vtile = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
       %cst_2 = arith.constant dense<0.000000e+00> : vector<64x128xf32>
       %cst_3 = arith.constant dense<0.000000e+00> : vector<64x1xf32>
       %9:2 = affine.for %loopK = 0 to affine_map<()[s0] -> (s0 * 64 + 64)>()[%block_id_y] step 32 iter_args(%oAcc = %cst_2, %qkmerAcc = %cst_3) -> (vector<64x128xf32>, vector<64x1xf32>) {
         %12 = frisk.buffer_view %K[0, %block_id_x mod 32,  0, %loopK mod 4096], ranges = [128, 32] : memref<1x32x128x4096xf16, 1> -> memref<128x32xf16, 1>
         %ktile_copied = frisk.copy %12 to %ktile [affine_map<() -> (2)>] : memref<128x32xf16, 1>, memref<128x32xf16, 3> -> memref<128x32xf16, 3>
         %13 = frisk.buffer_view %V[0, %block_id_x mod 32, %loopK mod 4096, 0], ranges = [32, 128] : memref<1x32x4096x128xf16, 1> -> memref<32x128xf16, 1>
         %vtile_copied = frisk.copy %13 to %vtile [affine_map<() -> (2)>] : memref<32x128xf16, 1>, memref<32x128xf16, 3> -> memref<32x128xf16, 3>
         %qk = frisk.gemm(%qtileScale, %ktile_copied) {transA = false, transB = false} : memref<64x128xf16, 3>, memref<128x32xf16, 3> -> memref<64x32xf32>
         %maskTIle = frisk.mask starts = [%0, %loopK], sizes = [64, 32], type = f32 {  // [64*by, loopK] start, window_sz=[64,32]
         ^bb0(%arg7: index, %arg8: index):
           %23 = arith.addi %arg7, %c1 : index  // i+1
           %24 = arith.cmpi ule, %23, %arg8 : index  // i+1 < j
           %25 = scf.if %24 -> (f32) {  // if(i+1 < j)
             scf.yield %cst_0 : f32  // -inf
           } else {
             scf.yield %cst : f32  // 0
           }
           frisk.mask_yield %25 : f32
         } : (index, index) -> memref<64x32xf32>
         // %maskTile_filled = frisk.fill %maskTIle {value = 0.000000e+00 : f32} : memref<64x32xf32> -> memref<64x32xf32>  // dbg
         %qkm = frisk.add %qk, %maskTIle : memref<64x32xf32>, memref<64x32xf32> -> memref<64x32xf32>
         %qkmeLocal = frisk.exp2 %qkm : memref<64x32xf32> -> memref<64x32xf32>
         %qkmeShm = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<64x32xf32, 3>
         %qkme = frisk.copy %qkmeLocal to %qkmeShm [affine_map<() -> (2)>] : memref<64x32xf32>, memref<64x32xf32,3> -> memref<64x32xf32,3>
         // %qkme = frisk.copy %qkm to %qkmeShm [affine_map<() -> (2)>] : memref<64x32xf32>, memref<64x32xf32,3> -> memref<64x32xf32,3>
   
         %qkme_reduced = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<64x1xf32, 3>
         frisk.reduce %qkme, %qkme_reduced {dim = 1 : i64, kind = "add"} : memref<64x32xf32, 3>, memref<64x1xf32, 3>  // non-ssa semantic
         frisk.sync_threads_in_block  // Make reduction leader writes visible to all readers.
         %19 = frisk.add %qkme_reduced, %qkmerAcc : memref<64x1xf32, 3>, vector<64x1xf32> -> memref<64x1xf32, 3>
         %20 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<64x32xf16, 3>
         %pvWeights_copied = frisk.copy %qkme to %20 [affine_map<() -> (2)>] : memref<64x32xf32, 3>, memref<64x32xf16, 3> -> memref<64x32xf16, 3>
         // Debug PV = 32 * half(0.3) * half(0.1) per iteration.
         frisk.sync_threads_in_block  // Finish all shared copy writes before the debug fill.
         // %vtile_filled = frisk.fill %vtile_copied {value = 2.00000e-1 : f16} : memref<32x128xf16, 3> -> memref<32x128xf16, 3>  // dbg
         // %pvWeights_fill = frisk.fill %pvWeights_copied {value = 3.00000e-1 : f16} : memref<64x32xf16, 3> -> memref<64x32xf16, 3>  // dbg
         frisk.sync_threads_in_block  // Make the debug fill visible before PV reads it.
   
         // Accumulate 2 * block_id_y + 1 PV tiles; the debug denominator is 1.
         %tempO = frisk.gemm(%pvWeights_copied, %vtile_copied) {transA = false, transB = false} : memref<64x32xf16, 3>, memref<32x128xf16, 3> -> memref<64x128xf32>
         %22 = frisk.add %oAcc, %tempO : vector<64x128xf32>, memref<64x128xf32> -> memref<64x128xf32>
         // %qkmerAcc_filled = frisk.fill %19 {value = 1.00000e+00 : f32} : memref<64x1xf32,3> -> memref<64x1xf32,3>  // dbg
         %oAcc_next = frisk.copy %22 to %oAcc [affine_map<() -> (2)>] : memref<64x128xf32>, vector<64x128xf32> -> vector<64x128xf32>
         %qkmerAcc_next = frisk.copy %19 to %qkmerAcc [affine_map<() -> (2)>] : memref<64x1xf32, 3>, vector<64x1xf32> -> vector<64x1xf32>
         affine.yield %oAcc_next, %qkmerAcc_next : vector<64x128xf32>, vector<64x1xf32>
       } {local_yield_transformed = true}
       frisk.sync_threads_in_block  // dbg
       %10 = frisk.div %9#0, %9#1 : vector<64x128xf32>, vector<64x1xf32> -> memref<64x128xf32>
       %oShm = frisk.alloc_buffer {scope = "local", alignment = 16} -> memref<64x128xf16>
       %oShm_copied = frisk.copy %10 to %oShm [affine_map<() -> (2)>] : memref<64x128xf32>, memref<64x128xf16> -> memref<64x128xf16>
       frisk.sync_threads_in_block  // dbg
       %O_copied = frisk.copy %oShm_copied to %O at(%block_id_y, %block_id_x) [affine_map<(d0, d1) -> (0, (d1 + (d0 * 8192) floordiv 524288) mod 32, (d0 * 64) mod 4096, 0)>] : memref<64x128xf16>, memref<1x32x4096x128xf16, 1> -> memref<1x32x4096x128xf16, 1>
       return
     }
   }
   
   ```

   IR特点：基于memref+vector 的kernel内部计算逻辑的宏观表示。相比asukaIR（DeepgenGraphIR），其内存层级更加明确（多出了shm和寄存器抽象）。明确使用buffer索引代替指针语义. 尽可能使用SSA表达

   设计理由：

   使用索引代替指针：便于mlir进行loop展开与优化，尽可能使用编译期常数

   引入memref+vector：显式地明确buffer层级。避免统一采用memref带来的spill问题（分配错误），不依赖mlir::Mem2reg Pass 提升Memref到寄存器

   本质为：memref = handle语义(ptr，可寻址)  , vector=reg语义 (不可寻址)

   SSA形势下，分析user-define 链条更加简单，不依赖于op顺序。

   

## IR转化过程设计

1.DeepgenGraphIR简化，降低到FriskIR
主要Pass


2.基于FriskIR的Layout推断，对每个buffer，明确每个thread持有buffer的哪些元素



3.BlockTile切到threadTile



4.lower到 LLVMIR



5.LLVMIR生成kernel文件。配合launcher执行

