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

  deepgengraph.kernel @Attn_p2(%arg0: tensor<1x4096x32x128xf16> loc(unknown), %arg1: tensor<1x4096x32x128xf16> loc(unknown), %arg2: tensor<1x4096x32x128xf16> loc(unknown)) -> tensor<1x4096x32x128xf16> attributes {parallel_map = [{arg_dims = [0, 0, 0], res_dims = [0], size_per_unit = 1 : i64, unit_num = 1 : i64}, {arg_dims = [1, -1, -1], res_dims = [1], size_per_unit = 128 : i64, unit_num = 32 : i64}, {arg_dims = [2, 2, 2], res_dims = [2], size_per_unit = 1 : i64, unit_num = 32 : i64}]} {
    %0 = deepgengraph_triton.ptr_of %arg0 : (tensor<1x4096x32x128xf16>) -> !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>> loc(#loc)
    %1 = deepgengraph_triton.ptr_of %arg1 : (tensor<1x4096x32x128xf16>) -> !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>> loc(#loc)
    %2 = deepgengraph_triton.ptr_of %arg2 : (tensor<1x4096x32x128xf16>) -> !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>> loc(#loc)
    %3 = deepgengraph_triton.empty_ptr type = tensor<1x4096x32x128xf16> : <tensor<1x4096x32x128xf16>> loc(#loc)
    deepgengraph_triton.device_kernel args = [%0, %1, %2, %3], grid = [1, 32, 32] {
    ^bb0(%arg3: index loc(unknown), %arg4: index loc(unknown), %arg5: index loc(unknown), %arg6: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>> loc(unknown), %arg7: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>> loc(unknown), %arg8: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>> loc(unknown), %arg9: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>> loc(unknown)):
      %cst = arith.constant dense<0.127531052> : tensor<1xf32> loc(#loc)
      %cst_0 = arith.constant 0xFF800000 : f32 loc(#loc)
      %cst_1 = arith.constant 0.000000e+00 : f32 loc(#loc)
      %c0 = arith.constant 0 : index loc(#loc)
      %c1 = arith.constant 1 : index loc(#loc)
      %c4096 = arith.constant 4096 : index loc(#loc)
      %c128 = arith.constant 128 : index loc(#loc)
      %5 = arith.muli %arg4, %c128 : index loc(#loc)
      %6 = arith.muli %5, %c4096 : index loc(#loc)
      %7 = arith.muli %arg5, %c128 : index loc(#loc)
      %8 = arith.addi %6, %7 : index loc(#loc)
      %9 = deepgengraph_triton.block_ptr_of base = %arg6, base_offset = %8, shape = [128, 128], stride = [4096, 1], offset = [0, 0], block_shape = [128, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}> loc(#loc)
      %10 = deepgengraph_triton.block_load %9 : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>) -> tensor<128x128xf16> loc(#loc)
      %11 = deepgengraph_triton.block_ptr_of base = %arg9, base_offset = %8, shape = [128, 128], stride = [4096, 1], offset = [0, 0], block_shape = [128, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}> loc(#loc)
      %12 = deepgengraph.convert %cst, type = f16 : (tensor<1xf32>) -> tensor<1xf16> loc(#loc)
      %13 = deepgengraph.mul %10, %12 : (tensor<128x128xf16>, tensor<1xf16>) -> tensor<128x128xf16> loc(#loc)
      %14 = deepgengraph.zero shape = [128, 128], type = f32 : () -> tensor<128x128xf32> loc(#loc)
      %15 = deepgengraph.zero shape = [128, 1], type = f32 : () -> tensor<128x1xf32> loc(#loc)
      %16 = arith.addi %5, %c128 : index loc(#loc)
      %17 = deepgengraph_triton.block_ptr_of base = %arg8, base_offset = %7, shape = [128, 4096], stride = [1, 4096], offset = [0, 0], block_shape = [128, 128], order = [0, 1] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}> loc(#loc)
      %18 = deepgengraph_triton.block_ptr_of base = %arg7, base_offset = %7, shape = [4096, 128], stride = [4096, 1], offset = [0, 0], block_shape = [128, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}> loc(#loc)
      %19:4 = scf.for %arg10 = %c0 to %16 step %c128 iter_args(%arg11 = %17, %arg12 = %18, %arg13 = %14, %arg14 = %15) -> (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>, !deepgengraph_triton<block_ptr{tensor<128x128xf16>}>, tensor<128x128xf32>, tensor<128x1xf32>) {
        %22 = deepgengraph_triton.block_load %arg11 : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>) -> tensor<128x128xf16> loc(#loc)
        %23 = deepgengraph_triton.block_load %arg12 : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>) -> tensor<128x128xf16> loc(#loc)
        %24 = deepgengraph.precise_dot_op %13, %22, acc = f32 : (tensor<128x128xf16>, tensor<128x128xf16>) -> tensor<128x128xf32> loc(#loc)
        %25 = deepgengraph.mask starts = [%5, %arg10], sizes = [128, 128], type = f32 {
        ^bb0(%arg15: index loc(unknown), %arg16: index loc(unknown)):
          %34 = arith.addi %arg15, %c1 : index loc(#loc)
          %35 = arith.cmpi ule, %34, %arg16 : index loc(#loc)
          %36 = scf.if %35 -> (f32) {
            scf.yield %cst_0 : f32 loc(#loc)
          } else {
            scf.yield %cst_1 : f32 loc(#loc)
          } loc(#loc)
          deepgengraph.mask_yield %36 : f32 loc(#loc)
        } : (index, index) -> tensor<128x128xf32> loc(#loc)
        %26 = deepgengraph.add %24, %25 : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32> loc(#loc)
        %27 = deepgengraph.exp2 %26 : (tensor<128x128xf32>) -> tensor<128x128xf32> loc(#loc)
        %28 = deepgengraph.reduce(%27, init = %arg14), dim = 1, op =  ADD, keep_dim = true : (tensor<128x128xf32>, tensor<128x1xf32>) -> tensor<128x1xf32> loc(#loc)
        %29 = deepgengraph.convert %27, type = f16 : (tensor<128x128xf32>) -> tensor<128x128xf16> loc(#loc)
        %30 = deepgengraph.precise_dot_op %29, %23, acc = f32 : (tensor<128x128xf16>, tensor<128x128xf16>) -> tensor<128x128xf32> loc(#loc)
        %31 = deepgengraph.add %arg13, %30 : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32> loc(#loc)
        %32 = deepgengraph_triton.block_advance %arg11, offsets = [0, 128] : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}> loc(#loc)
        %33 = deepgengraph_triton.block_advance %arg12, offsets = [128, 0] : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}> loc(#loc)
        scf.yield %32, %33, %31, %28 : !deepgengraph_triton<block_ptr{tensor<128x128xf16>}>, !deepgengraph_triton<block_ptr{tensor<128x128xf16>}>, tensor<128x128xf32>, tensor<128x1xf32> loc(#loc)
      } loc(#loc)
      %20 = deepgengraph.div %19#2, %19#3 : (tensor<128x128xf32>, tensor<128x1xf32>) -> tensor<128x128xf32> loc(#loc)
      %21 = deepgengraph.convert %20, type = f16 : (tensor<128x128xf32>) -> tensor<128x128xf16> loc(#loc)
      deepgengraph_triton.block_store %11, %21 : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>, tensor<128x128xf16>) -> () loc(#loc)
    } : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>) -> () loc(#loc)
    %4 = deepgengraph_triton.tensor_from %3 : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>) -> tensor<1x4096x32x128xf16> loc(#loc)
    deepgengraph.return %4 : tensor<1x4096x32x128xf16> loc(#loc)
  } loc(#loc)

} loc(#loc)

// ---------- after conversion ---------
// module {
//   func.func @Attn(%arg0: tensor<1x4096x32x128xf16>, %arg1: tensor<1x4096x32x128xf16>, %arg2: tensor<1x4096x32x128xf16>) -> tensor<1x4096x32x128xf16> {
//     %cst = arith.constant dense<1.131250e+01> : tensor<1xf16>
//     %0 = deepgengraph.trilu diagonal = 1, is_upper = true, shape = [4096, 4096], val = 0xFC00 : f16
//     %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
//     %2 = deepgengraph.permute %arg2, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
//     %3 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x128x4096xf16>
//     %4 = deepgengraph.dot %1, %3 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf16>
//     %5 = deepgengraph.div %4, %cst : (tensor<1x32x4096x4096xf16>, tensor<1xf16>) -> tensor<1x32x4096x4096xf16>
//     %6 = deepgengraph.add %5, %0 : (tensor<1x32x4096x4096xf16>, tensor<4096x4096xf16>) -> tensor<1x32x4096x4096xf16>
//     %7 = deepgengraph.convert %6, type = f32 : (tensor<1x32x4096x4096xf16>) -> tensor<1x32x4096x4096xf32>
//     %8 = deepgengraph.exp %7 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
//     %9 = deepgengraph.reduce(%8), dim = -1, op =  ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
//     %10 = deepgengraph.div %8, %9 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
//     %11 = deepgengraph.convert %10, type = f16 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16>
//     %12 = deepgengraph.dot %11, %2 : (tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16>
//     %13 = deepgengraph.permute %12, dims = [0, 2, 1, 3] : (tensor<1x32x4096x128xf16>) -> tensor<1x4096x32x128xf16>
//     return %13 : tensor<1x4096x32x128xf16>
//   }
//   func.func @Attn_p2(%arg0: tensor<1x4096x32x128xf16>, %arg1: tensor<1x4096x32x128xf16>, %arg2: tensor<1x4096x32x128xf16>) -> tensor<1x4096x32x128xf16> {
//     %0 = deepgengraph_triton.ptr_of %arg0 : (tensor<1x4096x32x128xf16>) -> !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>
//     %1 = deepgengraph_triton.ptr_of %arg1 : (tensor<1x4096x32x128xf16>) -> !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>
//     %2 = deepgengraph_triton.ptr_of %arg2 : (tensor<1x4096x32x128xf16>) -> !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>
//     %3 = deepgengraph_triton.empty_ptr type = tensor<1x4096x32x128xf16> : <tensor<1x4096x32x128xf16>>
//     tilelang.with_kernel %0, %1, %2, %3 : !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>> [1, 32, 32] {
//     ^bb0(%bz: index, %by: index, %bx: index, %arg3: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, %arg4: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, %arg5: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, %arg6: !deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>):
//       %cst = arith.constant dense<0.127531052> : tensor<1xf32>
//       %cst_0 = arith.constant 0xFF800000 : f32
//       %cst_1 = arith.constant 0.000000e+00 : f32
//       %c0 = arith.constant 0 : index
//       %c1 = arith.constant 1 : index
//       %c4096 = arith.constant 4096 : index
//       %c128 = arith.constant 128 : index
//       %5 = arith.muli %by, %c128 : index
//       %6 = arith.muli %5, %c4096 : index
//       %7 = arith.muli %bx, %c128 : index
//       %8 = arith.addi %6, %7 : index
//       %9 = deepgengraph_triton.block_ptr_of base = %arg3, base_offset = %8, shape = [128, 128], stride = [4096, 1], offset = [0, 0], block_shape = [128, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}>
//       %10 = deepgengraph_triton.block_load %9 : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>) -> tensor<128x128xf16>
//       %11 = deepgengraph_triton.block_ptr_of base = %arg6, base_offset = %8, shape = [128, 128], stride = [4096, 1], offset = [0, 0], block_shape = [128, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}>
//       %12 = deepgengraph.convert %cst, type = f16 : (tensor<1xf32>) -> tensor<1xf16>
//       %13 = deepgengraph.mul %10, %12 : (tensor<128x128xf16>, tensor<1xf16>) -> tensor<128x128xf16>
//       %14 = deepgengraph.zero shape = [128, 128], type = f32 : () -> tensor<128x128xf32>
//       %15 = deepgengraph.zero shape = [128, 1], type = f32 : () -> tensor<128x1xf32>
//       %16 = arith.addi %5, %c128 : index
//       %17 = deepgengraph_triton.block_ptr_of base = %arg5, base_offset = %7, shape = [128, 4096], stride = [1, 4096], offset = [0, 0], block_shape = [128, 128], order = [0, 1] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}>
//       %18 = deepgengraph_triton.block_ptr_of base = %arg4, base_offset = %7, shape = [4096, 128], stride = [4096, 1], offset = [0, 0], block_shape = [128, 128], order = [1, 0] : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>, index) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}>
//       %19:4 = scf.for %arg7 = %c0 to %16 step %c128 iter_args(%arg8 = %17, %arg9 = %18, %arg10 = %14, %arg11 = %15) -> (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>, !deepgengraph_triton<block_ptr{tensor<128x128xf16>}>, tensor<128x128xf32>, tensor<128x1xf32>) {
//         %22 = deepgengraph_triton.block_load %arg8 : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>) -> tensor<128x128xf16>
//         %23 = deepgengraph_triton.block_load %arg9 : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>) -> tensor<128x128xf16>
//         %24 = deepgengraph.precise_dot_op %13, %22, acc = f32 : (tensor<128x128xf16>, tensor<128x128xf16>) -> tensor<128x128xf32>
//         %25 = deepgengraph.mask starts = [%5, %arg7], sizes = [128, 128], type = f32 {
//         ^bb0(%arg12: index, %arg13: index):
//           %34 = arith.addi %arg12, %c1 : index
//           %35 = arith.cmpi ule, %34, %arg13 : index
//           %36 = scf.if %35 -> (f32) {
//             scf.yield %cst_0 : f32
//           } else {
//             scf.yield %cst_1 : f32
//           }
//           deepgengraph.mask_yield %36 : f32
//         } : (index, index) -> tensor<128x128xf32>
//         %26 = deepgengraph.add %24, %25 : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
//         %27 = deepgengraph.exp2 %26 : (tensor<128x128xf32>) -> tensor<128x128xf32>
//         %28 = deepgengraph.reduce(%27, init = %arg11), dim = 1, op =  ADD, keep_dim = true : (tensor<128x128xf32>, tensor<128x1xf32>) -> tensor<128x1xf32>
//         %29 = deepgengraph.convert %27, type = f16 : (tensor<128x128xf32>) -> tensor<128x128xf16>
//         %30 = deepgengraph.precise_dot_op %29, %23, acc = f32 : (tensor<128x128xf16>, tensor<128x128xf16>) -> tensor<128x128xf32>
//         %31 = deepgengraph.add %arg10, %30 : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
//         %32 = deepgengraph_triton.block_advance %arg8, offsets = [0, 128] : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}>
//         %33 = deepgengraph_triton.block_advance %arg9, offsets = [128, 0] : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>) -> !deepgengraph_triton<block_ptr{tensor<128x128xf16>}>
//         scf.yield %32, %33, %31, %28 : !deepgengraph_triton<block_ptr{tensor<128x128xf16>}>, !deepgengraph_triton<block_ptr{tensor<128x128xf16>}>, tensor<128x128xf32>, tensor<128x1xf32>
//       }
//       %20 = deepgengraph.div %19#2, %19#3 : (tensor<128x128xf32>, tensor<128x1xf32>) -> tensor<128x128xf32>
//       %21 = deepgengraph.convert %20, type = f16 : (tensor<128x128xf32>) -> tensor<128x128xf16>
//       deepgengraph_triton.block_store %11, %21 : (!deepgengraph_triton<block_ptr{tensor<128x128xf16>}>, tensor<128x128xf16>) -> ()
//     }
//     %4 = deepgengraph_triton.tensor_from %3 : (!deepgengraph_triton.ptr<tensor<1x4096x32x128xf16>>) -> tensor<1x4096x32x128xf16>
//     return %4 : tensor<1x4096x32x128xf16>
//   }
// }