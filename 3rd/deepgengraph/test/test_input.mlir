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
      %9 = deepgengraph_triton.block_ptr_of base = %argQ, base_offset = %8, shape = [64, 128], stride = [4096, 1], offset = [0, 0], block_shape = [64, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<64x128xf16>}>
      %10 = deepgengraph_triton.block_load %9 : (!deepgengraph_triton<block_ptr{tensor<64x128xf16>}>) -> tensor<64x128xf16>
      %11 = deepgengraph_triton.block_ptr_of base = %argO, base_offset = %8, shape = [64, 128], stride = [4096, 1], offset = [0, 0], block_shape = [64, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<64x128xf16>}>
      %12 = deepgengraph.convert %cst, type = f16 : (tensor<1xf32>) -> tensor<1xf16>
      %13 = deepgengraph.mul %10, %12 : (tensor<64x128xf16>, tensor<1xf16>) -> tensor<64x128xf16>
      %14 = deepgengraph.zero shape = [64, 128], type = f32 : () -> tensor<64x128xf32>
      %15 = deepgengraph.zero shape = [64, 1], type = f32 : () -> tensor<64x1xf32>
      %16 = arith.addi %5, %c64 : index  // by * 128 + 64
      %17 = deepgengraph_triton.block_ptr_of base = %argK, base_offset = %8, shape = [128, 4096], stride = [1, 4096], offset = [0, 0], block_shape = [128, 32], order = [0, 1] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x32xf16>}>  // base_offset = by * 128 * 4096 + bx * 64 * 128
      %18 = deepgengraph_triton.block_ptr_of base = %argV, base_offset = %8, shape = [4096, 128], stride = [4096, 1], offset = [0, 0], block_shape = [32, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<32x128xf16>}>
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
