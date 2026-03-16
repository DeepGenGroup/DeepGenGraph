// ===== before ParallelForConversion ====
deepgengraph.kernel @Attn_p3(%arg0: tensor<1x4096x32x128xf16>, %arg1: tensor<1x4096x32x128xf16>, %arg2: tensor<1x4096x32x128xf16>) -> (tensor<1x32x4096x1xf32>, tensor<1x4096x32x128xf16>) attributes {parallel_map = [{arg_dims = [0, 0, 0], res_dims = [0, 0], size_per_unit = 1 : i64, unit_num = 1 : i64}, {arg_dims = [1, -1, -1], res_dims = [2, 1], size_per_unit = 32 : i64, unit_num = 128 : i64}, {arg_dims = [2, 2, 2], res_dims = [1, 2], size_per_unit = 1 : i64, unit_num = 32 : i64}]} {
  %0:2 = deepgengraph.parallel_for %arg0, %arg1, %arg2 {
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: tensor<32x128xf16>, %arg7: tensor<4096x128xf16>, %arg8: tensor<4096x128xf16>):
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant dense<0.127531052> : tensor<1xf32>
    %1 = deepgengraph.permute %arg8, dims = [1, 0] : (tensor<4096x128xf16>) -> tensor<128x4096xf16>
    %2 = deepgengraph.convert %cst_1, type = f16 : (tensor<1xf32>) -> tensor<1xf16>
    %3 = deepgengraph.mul %arg6, %2 : (tensor<32x128xf16>, tensor<1xf16>) -> tensor<32x128xf16>
    %4 = deepgengraph.zero shape = [32, 128], type = f32 : () -> tensor<32x128xf32>
    %5 = deepgengraph.zero shape = [32, 1], type = f32 : () -> tensor<32x1xf32>
    %6 = arith.addi %arg4, %c128 : index
    %7:2 = deepgengraph.dynamic_block_for lb = %c0, ub = %6, step = 128, args = [%1, %arg7], dims = [1, 0], init = [%4, %5] {
    ^bb0(%arg9: index, %arg10: tensor<128x128xf16>, %arg11: tensor<128x128xf16>, %arg12: tensor<32x128xf32>, %arg13: tensor<32x1xf32>):
      %10 = deepgengraph.precise_dot_op %3, %arg10, acc = f32 : (tensor<32x128xf16>, tensor<128x128xf16>) -> tensor<32x128xf32>
      %11 = deepgengraph.mask starts = [%arg4, %arg9], sizes = [32, 128], type = f32 {
      ^bb0(%arg14: index, %arg15: index):
        %18 = arith.addi %arg14, %c1 : index
        %19 = arith.cmpi ule, %18, %arg15 : index
        %20 = scf.if %19 -> (f32) {
          scf.yield %cst_0 : f32
        } else {
          scf.yield %cst : f32
        }
        deepgengraph.mask_yield %20 : f32
      } : (index, index) -> tensor<32x128xf32>
      %12 = deepgengraph.add %10, %11 : (tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
      %13 = deepgengraph.exp2 %12 : (tensor<32x128xf32>) -> tensor<32x128xf32>
      %14 = deepgengraph.reduce(%13, init = %arg13), dim = 1, op =  ADD, keep_dim = true : (tensor<32x128xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
      %15 = deepgengraph.convert %13, type = f16 : (tensor<32x128xf32>) -> tensor<32x128xf16>
      %16 = deepgengraph.precise_dot_op %15, %arg11, acc = f32 : (tensor<32x128xf16>, tensor<128x128xf16>) -> tensor<32x128xf32>
      %17 = deepgengraph.add %arg12, %16 : (tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
      deepgengraph.block_yield block_outs = [], iter_outs = [%17, %14] : tensor<32x128xf32>, tensor<32x1xf32>
    } : (index, index, tensor<128x4096xf16>, tensor<4096x128xf16>, tensor<32x128xf32>, tensor<32x1xf32>) -> (tensor<32x128xf32>, tensor<32x1xf32>)
    %8 = deepgengraph.div %7#0, %7#1 : (tensor<32x128xf32>, tensor<32x1xf32>) -> tensor<32x128xf32>
    %9 = deepgengraph.convert %8, type = f16 : (tensor<32x128xf32>) -> tensor<32x128xf16>
    deepgengraph.parallel_yield %7#1, %9 : tensor<32x1xf32>, tensor<32x128xf16>
  } : (tensor<1x4096x32x128xf16>, tensor<1x4096x32x128xf16>, tensor<1x4096x32x128xf16>) -> (tensor<1x32x4096x1xf32>, tensor<1x4096x32x128xf16>)
  deepgengraph.return %0#0, %0#1 : tensor<1x32x4096x1xf32>, tensor<1x4096x32x128xf16>
}

// ===== after ParallelForConversion ====
tl.prim_func @Attn_p3(%arg0: tensor<1x4096x32x128xf16>, %arg1: tensor<1x4096x32x128xf16>, %arg2: tensor<1x4096x32x128xf16>) -> (tensor<1x32x4096x1xf32>, tensor<1x4096x32x128xf16>) attributes {parallel_map = [{arg_dims = [0, 0, 0], res_dims = [0, 0], size_per_unit = 1 : i64, unit_num = 1 : i64}, {arg_dims = [1, -1, -1], res_dims = [2, 1], size_per_unit = 32 : i64, unit_num = 128 : i64}, {arg_dims = [2, 2, 2], res_dims = [1, 2], size_per_unit = 1 : i64, unit_num = 32 : i64}]} {
  // 表达了args 如何参加计算。使用到的 bx, by,bz , 返回 宏观的大参数
  // (tensor<1x4096x32x128xf16>, tensor<1x4096x32x128xf16>, tensor<1x4096x32x128xf16>) -> (tensor<1x32x4096x1xf32>, tensor<1x4096x32x128xf16>)
  // 小tile 形状 ： deepgengraph.parallel_yield %7#1, %9 : tensor<32x1xf32>, tensor<32x128xf16>
  // 宏观return参数和 小tile之间的 形状关系 : 计算需要blocks数目 -> bx,by,bz 数量
  // 
  %0:2 = deepgengraph.parallel_for %arg0, %arg1, %arg2 {
    // arg0, %arg1, %arg2 -> arg6,7,8  
  //     enum Kind {
  //   kInit,    // 未初始化
  //   kBatch,   // 完全可并行（互不依赖）
  //   kReUse,   // 可并行但存在数据复用
  //   kNonPara  // 不可并行（有依赖 / 归约）
  // };

    // tensor<1x4096x32x128xf16> -> tensor<32x128xf16>  arg_dims = [0, 0, 0], res_dims = [0, 0]
  // tensor<1x4096x32x128xf16> -> tensor<4096x128xf16>  arg_dims = [1, -1, -1], res_dims = [2, 1]
  // tensor<1x4096x32x128xf16> -> tensor<4096x128xf16>)  arg_dims = [2, 2, 2], res_dims = [1, 2]
  ^bb0(%arg3: index, %arg4: index, %arg5: index, %arg6: tensor<32x128xf16>, %arg7: tensor<4096x128xf16>, %arg8: tensor<4096x128xf16>):
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant dense<0.127531052> : tensor<1xf32>
    %1 = deepgengraph.permute %arg8, dims = [1, 0] : (tensor<4096x128xf16>) -> tensor<128x4096xf16>
    %2 = deepgengraph.convert %cst_1, type = f16 : (tensor<1xf32>) -> tensor<1xf16>
    %3 = deepgengraph.mul %arg6, %2 : (tensor<32x128xf16>, tensor<1xf16>) -> tensor<32x128xf16>
    %4 = deepgengraph.zero shape = [32, 128], type = f32 : () -> tensor<32x128xf32>
    %5 = deepgengraph.zero shape = [32, 1], type = f32 : () -> tensor<32x1xf32>
    %6 = arith.addi %arg4, %c128 : index
    // dynamic_block_for 表达了 for循环 & 大尺寸参数-> 小尺寸tile 的映射关系
    // lb = %c0, ub = %6, step = 128 —— 产生新的 affine.for 循环
    // args = [%1, %arg7] init = [%4, %5] —— 产生新的 memref。for之前，用 [%4, %5] 填充
    // dims = [1, 0] 迭代维度。对于 %1 而言为 dim1（倒数第二维）， arg7的dim0 （最后一维）
    %7:2 = deepgengraph.dynamic_block_for lb = %c0, ub = %6, step = 128, args = [%1, %arg7], dims = [1, 0], init = [%4, %5] {
    ^bb0(%arg9: index, %arg10: tensor<128x128xf16>, %arg11: tensor<128x128xf16>, %arg12: tensor<32x128xf32>, %arg13: tensor<32x1xf32>):
      %10 = deepgengraph.precise_dot_op %3, %arg10, acc = f32 : (tensor<32x128xf16>, tensor<128x128xf16>) -> tensor<32x128xf32>
      %11 = deepgengraph.mask starts = [%arg4, %arg9], sizes = [32, 128], type = f32 {
      ^bb0(%arg14: index, %arg15: index):
        %18 = arith.addi %arg14, %c1 : index
        %19 = arith.cmpi ule, %18, %arg15 : index
        %20 = scf.if %19 -> (f32) {
          scf.yield %cst_0 : f32
        } else {
          scf.yield %cst : f32
        }
        deepgengraph.mask_yield %20 : f32
      } : (index, index) -> tensor<32x128xf32>
      %12 = deepgengraph.add %10, %11 : (tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
      %13 = deepgengraph.exp2 %12 : (tensor<32x128xf32>) -> tensor<32x128xf32>
      %14 = deepgengraph.reduce(%13, init = %arg13), dim = 1, op =  ADD, keep_dim = true : (tensor<32x128xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>
      %15 = deepgengraph.convert %13, type = f16 : (tensor<32x128xf32>) -> tensor<32x128xf16>
      %16 = deepgengraph.precise_dot_op %15, %arg11, acc = f32 : (tensor<32x128xf16>, tensor<128x128xf16>) -> tensor<32x128xf32>
      %17 = deepgengraph.add %arg12, %16 : (tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>
      deepgengraph.block_yield block_outs = [], iter_outs = [%17, %14] : tensor<32x128xf32>, tensor<32x1xf32>
    } : (index, index, tensor<128x4096xf16>, tensor<4096x128xf16>, tensor<32x128xf32>, tensor<32x1xf32>) -> (tensor<32x128xf32>, tensor<32x1xf32>)
    %8 = deepgengraph.div %7#0, %7#1 : (tensor<32x128xf32>, tensor<32x1xf32>) -> tensor<32x128xf32>
    %9 = deepgengraph.convert %8, type = f16 : (tensor<32x128xf32>) -> tensor<32x128xf16>
    deepgengraph.parallel_yield %7#1, %9 : tensor<32x1xf32>, tensor<32x128xf16>
  } : (tensor<1x4096x32x128xf16>, tensor<1x4096x32x128xf16>, tensor<1x4096x32x128xf16>) -> (tensor<1x32x4096x1xf32>, tensor<1x4096x32x128xf16>)
  deepgengraph.return %0#0, %0#1 : tensor<1x32x4096x1xf32>, tensor<1x4096x32x128xf16>
}

