#loc = loc(unknown)
module {
  func.func @Attn(%arg0: tensor<1x4096x32x128xf16> loc(unknown), %arg1: tensor<1x4096x32x128xf16> loc(unknown), %arg2: tensor<1x4096x32x128xf16> loc(unknown)) -> tensor<1x4096x32x128xf16> {
    %cst = arith.constant dense<1.131250e+01> : tensor<1xf16> loc(#loc) // 根号128约为11.3137，这个对吗
    %0 = deepgengraph.trilu diagonal = 1, is_upper = true, shape = [4096, 4096], val = 0xFC00 : f16 loc(#loc) // 上三角负无穷
    %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16> loc(#loc) // Q
    %2 = deepgengraph.permute %arg2, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16> loc(#loc) // V
    %3 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x128x4096xf16> loc(#loc) // K^T
    %4 = deepgengraph.dot %1, %3 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc) // Q*k^T
    %5 = deepgengraph.div %4, %cst : (tensor<1x32x4096x4096xf16>, tensor<1xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc) // Q*K^T/根号d
    %6 = deepgengraph.add %5, %0 : (tensor<1x32x4096x4096xf16>, tensor<4096x4096xf16>) -> tensor<1x32x4096x4096xf16> loc(#loc) // mask
    %7 = deepgengraph.convert %6, type = f32 : (tensor<1x32x4096x4096xf16>) -> tensor<1x32x4096x4096xf32> loc(#loc)
    %8 = deepgengraph.exp %7 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32> loc(#loc)
    %9 = deepgengraph.reduce(%8), dim = -1, op =  ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32> loc(#loc)
    %10 = deepgengraph.div %8, %9 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32> loc(#loc)
    %11 = deepgengraph.convert %10, type = f16 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16> loc(#loc) // P
    %12 = deepgengraph.dot %11, %2 : (tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16> loc(#loc)
    %13 = deepgengraph.permute %12, dims = [0, 2, 1, 3] : (tensor<1x32x4096x128xf16>) -> tensor<1x4096x32x128xf16> loc(#loc)
    return %13 : tensor<1x4096x32x128xf16> loc(#loc)
  } loc(#loc)

  deepgengraph.kernel @Attn_p2(%Q: tensor<1x4096x32x128xf16>, %K: tensor<1x4096x32x128xf16>, %V: tensor<1x4096x32x128xf16>) -> tensor<1x4096x32x128xf16> attributes {parallel_map = [{arg_dims = [0, 0, 0], res_dims = [0], size_per_unit = 1 : i64, unit_num = 1 : i64}, {arg_dims = [1, -1, -1], res_dims = [1], size_per_unit = 128 : i64, unit_num = 32 : i64}, {arg_dims = [2, 2, 2], res_dims = [2], size_per_unit = 1 : i64, unit_num = 32 : i64}]} {
    %pQ = deepgengraph_triton.ptr_of %Q : (tensor<1x4096x32x128xf16>) -> !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>
    %pV = deepgengraph_triton.ptr_of %V : (tensor<1x4096x32x128xf16>) -> !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>
    %pK = deepgengraph_triton.ptr_of %K : (tensor<1x4096x32x128xf16>) -> !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>
    %O = deepgengraph_triton.empty_ptr type = tensor<1x4096x32x128xf16> : <tensor<1x4096x32x128xf16>>
    deepgengraph_triton.device_kernel args = [%pQ, %pV, %pK, %O], grid = [1, 32, 32] {
    ^bb0(%bz: index, %bx: index, %by: index, %argQ: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, %argV: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, %argK: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, %argO: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>):
      // BK splits QK's reduction dimension and V/O's output columns.
      %scale32 = arith.constant dense<0.127531052> : tensor<1xf32>
      %scale = deepgengraph.convert %scale32, type = f16 : (tensor<1xf32>) -> tensor<1xf16>
      %negInf = arith.constant 0xFF800000 : f32
      %zero = arith.constant 0.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %cBM = arith.constant 128 : index
      %cBN = arith.constant 128 : index
      %cBK = arith.constant 32 : index
      %cD = arith.constant 128 : index
      %cS = arith.constant 4096 : index
      %cSD = arith.constant 524288 : index
      %cHSD = arith.constant 16777216 : index
      %batchBase = arith.muli %bz, %cHSD : index
      %headOffset = arith.muli %by, %cSD : index
      %headBase = arith.addi %batchBase, %headOffset : index
      %qStart = arith.muli %bx, %cBM : index
      %qOffset = arith.muli %qStart, %cD : index
      %qBase = arith.addi %headBase, %qOffset : index
      %keyEnd = arith.addi %qStart, %cBM : index
      // Output D is NOT a reduction axis of PV. Store each BK-column slice.
      affine.for %od = 0 to 128 step 32 {
        %outOffset = arith.addi %qBase, %od : index
        %outPtr = deepgengraph_triton.block_ptr_of base = %argO, base_offset = %outOffset, shape = [128, 128], stride = [4096, 1], offset = [0, 0], block_shape = [128, 32], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x32xf16>}>
        %outZero = deepgengraph.zero shape = [128, 32], type = f32 : () -> tensor<128x32xf32>
        %rowZero = deepgengraph.zero shape = [128, 1], type = f32 : () -> tensor<128x1xf32>
        %attention:2 = scf.for %n = %c0 to %keyEnd step %cBN iter_args(%accO = %outZero, %accL = %rowZero) -> (tensor<128x32xf32>, tensor<128x1xf32>) {
          %scoreZero = deepgengraph.zero shape = [128, 128], type = f32 : () -> tensor<128x128xf32>
          // Reset Q/K and the f32 score accumulator for EVERY key tile.
          %qk = scf.for %dk = %c0 to %cD step %cBK iter_args(%accQK = %scoreZero) -> (tensor<128x128xf32>) {
            %qTileOffset = arith.addi %qBase, %dk : index
            %kDOffset = arith.muli %dk, %cS : index
            %kHeadOffset = arith.addi %headBase, %kDOffset : index
            %kTileOffset = arith.addi %kHeadOffset, %n : index
            %qPtr = deepgengraph_triton.block_ptr_of base = %argQ, base_offset = %qTileOffset, shape = [128, 128], stride = [4096, 1], offset = [0, 0], block_shape = [128, 32], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x32xf16>}>
            %kPtr = deepgengraph_triton.block_ptr_of base = %argK, base_offset = %kTileOffset, shape = [128, 4096], stride = [1, 4096], offset = [0, 0], block_shape = [32, 128], order = [0, 1] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<32x128xf16>}>
            %qTile = deepgengraph_triton.block_load %qPtr : (!deepgengraph_triton<block_ptr{tensor<128x32xf16>}>) -> tensor<128x32xf16>
            %kTile = deepgengraph_triton.block_load %kPtr : (!deepgengraph_triton<block_ptr{tensor<32x128xf16>}>) -> tensor<32x128xf16>
            %qScaled = deepgengraph.mul %qTile, %scale : (tensor<128x32xf16>, tensor<1xf16>) -> tensor<128x32xf16>
            %partialQK = deepgengraph.precise_dot_op %qScaled, %kTile, acc = f32 : (tensor<128x32xf16>, tensor<32x128xf16>) -> tensor<128x128xf32>
            %nextQK = deepgengraph.add %accQK, %partialQK : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
            scf.yield %nextQK : tensor<128x128xf32>
          } {frisk.simplify_loop}
          %mask = deepgengraph.mask starts = [%qStart, %n], sizes = [128, 128], type = f32 {
          ^bb0(%qi: index, %ki: index):
            %nextRow = arith.addi %qi, %c1 : index
            %future = arith.cmpi ule, %nextRow, %ki : index
            %maskValue = scf.if %future -> (f32) {
              scf.yield %negInf : f32
            } else {
              scf.yield %zero : f32
            }
            deepgengraph.mask_yield %maskValue : f32
          } : (index, index) -> tensor<128x128xf32>
          %masked = deepgengraph.add %qk, %mask : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
          %weights = deepgengraph.exp2 %masked : (tensor<128x128xf32>) -> tensor<128x128xf32>
          %nextL = deepgengraph.reduce(%weights, init = %accL), dim = 1, op = ADD, keep_dim = true : (tensor<128x128xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
          %p = deepgengraph.convert %weights, type = f16 : (tensor<128x128xf32>) -> tensor<128x128xf16>
          %vRowOffset = arith.muli %n, %cD : index
          %vHeadOffset = arith.addi %headBase, %vRowOffset : index
          %vTileOffset = arith.addi %vHeadOffset, %od : index
          %vPtr = deepgengraph_triton.block_ptr_of base = %argV, base_offset = %vTileOffset, shape = [4096, 128], stride = [4096, 1], offset = [0, 0], block_shape = [128, 32], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x32xf16>}>
          %vTile = deepgengraph_triton.block_load %vPtr : (!deepgengraph_triton<block_ptr{tensor<128x32xf16>}>) -> tensor<128x32xf16>
          %partialO = deepgengraph.precise_dot_op %p, %vTile, acc = f32 : (tensor<128x128xf16>, tensor<128x32xf16>) -> tensor<128x32xf32>
          %nextO = deepgengraph.add %accO, %partialO : (tensor<128x32xf32>, tensor<128x32xf32>) -> tensor<128x32xf32>
          scf.yield %nextO, %nextL : tensor<128x32xf32>, tensor<128x1xf32>
        }
        %normalized = deepgengraph.div %attention#0, %attention#1 : (tensor<128x32xf32>, tensor<128x1xf32>) -> tensor<128x32xf32>
        %result = deepgengraph.convert %normalized, type = f16 : (tensor<128x32xf32>) -> tensor<128x32xf16>
        deepgengraph_triton.block_store %outPtr, %result : (!deepgengraph_triton<block_ptr{tensor<128x32xf16>}>, tensor<128x32xf16>) -> ()
      }
    } : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>) -> ()
    %4 = deepgengraph_triton.tensor_from %O : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>) -> tensor<1x4096x32x128xf16>
    deepgengraph.return %4 : tensor<1x4096x32x128xf16>
  }
} loc(#loc)
