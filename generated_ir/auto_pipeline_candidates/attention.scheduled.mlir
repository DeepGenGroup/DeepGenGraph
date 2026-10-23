module {
  func.func @Attn(%arg0: tensor<1x4096x32x128xf16>, %arg1: tensor<1x4096x32x128xf16>, %arg2: tensor<1x4096x32x128xf16>) -> tensor<1x4096x32x128xf16> {
    %cst = arith.constant dense<1.131250e+01> : tensor<1xf16>
    %0 = deepgengraph.trilu diagonal = 1, is_upper = true, shape = [4096, 4096], val = 0xFC00 : f16
    %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %2 = deepgengraph.permute %arg2, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16>
    %3 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x128x4096xf16>
    %4 = deepgengraph.dot %1, %3 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf16>
    %5 = deepgengraph.div %4, %cst : (tensor<1x32x4096x4096xf16>, tensor<1xf16>) -> tensor<1x32x4096x4096xf16>
    %6 = deepgengraph.add %5, %0 : (tensor<1x32x4096x4096xf16>, tensor<4096x4096xf16>) -> tensor<1x32x4096x4096xf16>
    %7 = deepgengraph.convert %6, type = f32 : (tensor<1x32x4096x4096xf16>) -> tensor<1x32x4096x4096xf32>
    %8 = deepgengraph.exp %7 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32>
    %9 = deepgengraph.reduce(%8), dim = -1, op = ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32>
    %10 = deepgengraph.div %8, %9 : (tensor<1x32x4096x4096xf32>, tensor<1x32x4096x1xf32>) -> tensor<1x32x4096x4096xf32>
    %11 = deepgengraph.convert %10, type = f16 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf16>
    %12 = deepgengraph.dot %11, %2 : (tensor<1x32x4096x4096xf16>, tensor<1x32x4096x128xf16>) -> tensor<1x32x4096x128xf16>
    %13 = deepgengraph.permute %12, dims = [0, 2, 1, 3] : (tensor<1x32x4096x128xf16>) -> tensor<1x4096x32x128xf16>
    return %13 : tensor<1x4096x32x128xf16>
  }
  func.func @Attn_p2(%arg0: memref<1x32x4096x128xf16, 1>, %arg1: memref<1x32x4096x128xf16, 1>, %arg2: memref<1x32x4096x128xf16, 1>, %arg3: memref<1x32x4096x128xf16, 1>) attributes {frisk.block_warps = array<i64: 4, 1>, frisk.k_layout_bhsd, thread_num = 256 : i32} {
    %transpose = memref.transpose %arg1 (d0, d1, d2, d3) -> (d0, d1, d3, d2) : memref<1x32x4096x128xf16, 1> to memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1>
    %block_id_z = gpu.block_id z {range = 1 : index}
    %block_id_y = gpu.block_id y {range = 32 : index}
    %block_id_x = gpu.block_id x {range = 32 : index}
    %thread_id_x = gpu.thread_id x {range = 256 : index}
    %cst = arith.constant 0.127531052 : f32
    %cst_0 = arith.constant 0xFF800000 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %0 = frisk.cast %cst : f32 -> f16
    %1 = arith.muli %block_id_y, %c128 : index
    %2 = frisk.zero {initIdx = 0 : index} : memref<128x32xf32>
    %3 = frisk.zero {initIdx = 1 : index} : memref<128x1xf32>
    %4 = frisk.zero {initIdx = 2 : index} : memref<128x32xf32>
    %5 = frisk.zero {initIdx = 3 : index} : memref<128x32xf32>
    %6 = frisk.zero {initIdx = 4 : index} : memref<128x32xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_3 = arith.constant dense<0.000000e+00> : vector<128x1xf32>
    %cst_4 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_5 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %cst_6 = arith.constant dense<0.000000e+00> : vector<128x32xf32>
    %7:5 = affine.for %arg4 = 0 to affine_map<()[s0] -> (s0 * 128 + 128)>()[%block_id_y] step 128 iter_args(%arg5 = %cst_2, %arg6 = %cst_3, %arg7 = %cst_4, %arg8 = %cst_5, %arg9 = %cst_6) -> (vector<128x32xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>) {
      %16 = frisk.zero {initIdx = 0 : index} : memref<128x128xf32>
      %cst_7 = arith.constant dense<0.000000e+00> : vector<128x128xf32>
      %c0 = arith.constant 0 : index
      %c128_8 = arith.constant 128 : index
      %17 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %18 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %19 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
      %20 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<32x128xf16, 3>
      %21 = arith.cmpi slt, %c0, %c128_8 : index
      %22 = arith.subi %c128_8, %c0 : index
      %c0_9 = arith.constant 0 : index
      %23 = arith.cmpi ugt, %22, %c0_9 : index
      %24 = arith.andi %21, %23 : i1
      scf.if %24 {
        %48 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %block_id_y * 16384 + %c0) floordiv 524288) mod 32, (%block_id_z * 131072 + %block_id_x * 4096 + %block_id_y * 128 + %c0 floordiv 128) mod 4096, %c0 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        %49 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %c0 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %block_id_x * 128 + %c0 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        frisk.async_copy_gm_to_shm %48 to %17 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
        frisk.async_copy_gm_to_shm %49 to %19 {alignment = 16 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
      }
      %25 = affine.for %arg10 = 0 to 128 step 32 iter_args(%arg11 = %cst_7) -> (vector<128x128xf32>) {
        %48 = arith.subi %arg10, %c0 : index
        %c32 = arith.constant 32 : index
        %49 = arith.divui %48, %c32 : index
        %c2 = arith.constant 2 : index
        %50 = arith.remui %49, %c2 : index
        %c0_10 = arith.constant 0 : index
        %51 = arith.cmpi eq, %50, %c0_10 : index
        %52:2 = scf.if %51 -> (memref<128x32xf16, 3>, memref<32x128xf16, 3>) {
          scf.yield %17, %19 : memref<128x32xf16, 3>, memref<32x128xf16, 3>
        } else {
          scf.yield %18, %20 : memref<128x32xf16, 3>, memref<32x128xf16, 3>
        }
        frisk.wait_buffer_ready %52#0 : memref<128x32xf16, 3>
        frisk.wait_buffer_ready %52#1 : memref<32x128xf16, 3>
        frisk.sync_threads_in_block
        %53 = arith.subi %c128_8, %arg10 : index
        %c32_11 = arith.constant 32 : index
        %54 = arith.cmpi ugt, %53, %c32_11 : index
        scf.if %54 {
          %60 = affine.apply affine_map<(d0) -> (d0 + 32)>(%arg10)
          %c2_12 = arith.constant 2 : index
          %61 = arith.remui %49, %c2_12 : index
          %c1_13 = arith.constant 1 : index
          %62 = arith.addi %61, %c1_13 : index
          %63 = arith.remui %62, %c2_12 : index
          %c0_14 = arith.constant 0 : index
          %64 = arith.cmpi eq, %63, %c0_14 : index
          %65:2 = scf.if %64 -> (memref<128x32xf16, 3>, memref<32x128xf16, 3>) {
            scf.yield %17, %19 : memref<128x32xf16, 3>, memref<32x128xf16, 3>
          } else {
            scf.yield %18, %20 : memref<128x32xf16, 3>, memref<32x128xf16, 3>
          }
          %66 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %block_id_y * 16384 + %60) floordiv 524288) mod 32, (%block_id_z * 131072 + %block_id_x * 4096 + %block_id_y * 128 + %60 floordiv 128) mod 4096, %60 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
          %67 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %60 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %block_id_x * 128 + %60 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
          frisk.async_copy_gm_to_shm %66 to %65#0 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
          frisk.async_copy_gm_to_shm %67 to %65#1 {alignment = 16 : i64} : memref<32x128xf16, 1>, memref<32x128xf16, 3>
        }
        %55 = frisk.buffer_view %arg0[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %block_id_y * 16384 + %arg10) floordiv 524288) mod 32, (%block_id_z * 131072 + %block_id_x * 4096 + %block_id_y * 128 + %arg10 floordiv 128) mod 4096, %arg10 mod 128], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
        %56 = frisk.buffer_view %transpose[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %arg10 * 4096 + %arg4) floordiv 524288) mod 32, (%block_id_z * 4096 + %block_id_x * 128 + %arg10 + %arg4 floordiv 4096) mod 128, %arg4 mod 4096], ranges = [32, 128] : memref<1x32x128x4096xf16, strided<[16777216, 524288, 1, 128]>, 1> -> memref<32x128xf16, 1>
        %57 = frisk.mul %52#0, %0 : memref<128x32xf16, 3>, f16 -> memref<128x32xf16, 3>
        %58 = frisk.gemm(%57, %52#1) {transA = false, transB = false} : memref<128x32xf16, 3>, memref<32x128xf16, 3> -> memref<128x128xf32>
        %59 = frisk.add %arg11, %58 : vector<128x128xf32>, memref<128x128xf32> -> memref<128x128xf32>
        frisk.copy %59 to %arg11 [affine_map<() -> (2)>] : memref<128x128xf32>, vector<128x128xf32>
        affine.yield %arg11 : vector<128x128xf32>
      } {local_yield_transformed = true, pipeline.num_stages = 2 : i64, pipeline.scheduled, pipeline.slots = 2 : i64}
      %26 = frisk.mask starts = [%1, %arg4], sizes = [128, 128], type = f32 {
      ^bb0(%arg10: index, %arg11: index):
        %48 = arith.addi %arg10, %c1 : index
        %49 = arith.cmpi ule, %48, %arg11 : index
        %50 = arith.select %49, %cst_0, %cst_1 : f32
        frisk.mask_yield %50 : f32
      } : (index, index) -> memref<128x128xf32>
      %27 = frisk.add %25, %26 : vector<128x128xf32>, memref<128x128xf32> -> memref<128x128xf32>
      %28 = frisk.exp2 %27 : memref<128x128xf32> -> memref<128x128xf32, 3>
      %29 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x1xf32, 3>
      frisk.reduce %28, %29 {dim = 1 : i64, kind = "add"} : memref<128x128xf32, 3>, memref<128x1xf32, 3>
      %30 = frisk.add %29, %arg6 : memref<128x1xf32, 3>, vector<128x1xf32> -> memref<128x1xf32, 3>
      %31 = frisk.cast %28 : memref<128x128xf32, 3> -> memref<128x128xf16, 3>
      %32 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %33 = frisk.buffer_view %arg2[0, (%block_id_z * 32 + %block_id_x + (%arg4 * 128) floordiv 524288) mod 32, %arg4 mod 4096, 0], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %33 to %32 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %32 : memref<128x32xf16, 3>
      %34 = frisk.gemm(%31, %32) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3> -> memref<128x32xf32>
      %35 = frisk.add %arg5, %34 : vector<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %36 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %37 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %arg4 * 128 + 32) floordiv 524288) mod 32, %arg4 mod 4096, 32], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %37 to %36 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %36 : memref<128x32xf16, 3>
      %38 = frisk.gemm(%31, %36) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3> -> memref<128x32xf32>
      %39 = frisk.add %arg7, %38 : vector<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %40 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %41 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %arg4 * 128 + 64) floordiv 524288) mod 32, %arg4 mod 4096, 64], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %41 to %40 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %40 : memref<128x32xf16, 3>
      %42 = frisk.gemm(%31, %40) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3> -> memref<128x32xf32>
      %43 = frisk.add %arg8, %42 : vector<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      %44 = frisk.alloc_buffer {scope = "shared", alignment = 16} -> memref<128x32xf16, 3>
      %45 = frisk.buffer_view %arg2[0, ((%block_id_z * 16777216 + %block_id_x * 524288 + %arg4 * 128 + 96) floordiv 524288) mod 32, %arg4 mod 4096, 96], ranges = [128, 32] : memref<1x32x4096x128xf16, 1> -> memref<128x32xf16, 1>
      frisk.async_copy_gm_to_shm %45 to %44 {alignment = 16 : i64} : memref<128x32xf16, 1>, memref<128x32xf16, 3>
      frisk.wait_buffer_ready %44 : memref<128x32xf16, 3>
      %46 = frisk.gemm(%31, %44) {transA = false, transB = false} : memref<128x128xf16, 3>, memref<128x32xf16, 3> -> memref<128x32xf32>
      %47 = frisk.add %arg9, %46 : vector<128x32xf32>, memref<128x32xf32> -> memref<128x32xf32>
      frisk.copy %35 to %arg5 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %30 to %arg6 [affine_map<() -> (2)>] : memref<128x1xf32, 3>, vector<128x1xf32>
      frisk.copy %39 to %arg7 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %43 to %arg8 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
      frisk.copy %47 to %arg9 [affine_map<() -> (2)>] : memref<128x32xf32>, vector<128x32xf32>
      affine.yield %arg5, %arg6, %arg7, %arg8, %arg9 : vector<128x32xf32>, vector<128x1xf32>, vector<128x32xf32>, vector<128x32xf32>, vector<128x32xf32>
    } {local_yield_transformed = true}
    %8 = frisk.div %7#0, %7#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %9 = frisk.cast %8 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %9 to %arg3 at (%block_id_y, %block_id_x, %block_id_z) [affine_map<(d0, d1, d2) -> (0, (d2 * 32 + d1 + (d0 * 16384) floordiv 524288) mod 32, (d0 * 128) mod 4096, 0)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    %10 = frisk.div %7#2, %7#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %11 = frisk.cast %10 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %11 to %arg3 at (%block_id_y, %block_id_x, %block_id_z) [affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 32) floordiv 524288) mod 32, (d0 * 128) mod 4096, 32)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    %12 = frisk.div %7#3, %7#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %13 = frisk.cast %12 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %13 to %arg3 at (%block_id_y, %block_id_x, %block_id_z) [affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 64) floordiv 524288) mod 32, (d0 * 128) mod 4096, 64)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    %14 = frisk.div %7#4, %7#1 : vector<128x32xf32>, vector<128x1xf32> -> memref<128x32xf32>
    %15 = frisk.cast %14 : memref<128x32xf32> -> memref<128x32xf16, 3>
    frisk.copy %15 to %arg3 at (%block_id_y, %block_id_x, %block_id_z) [affine_map<(d0, d1, d2) -> (0, ((d2 * 16777216 + d1 * 524288 + d0 * 16384 + 96) floordiv 524288) mod 32, (d0 * 128) mod 4096, 96)>] : memref<128x32xf16, 3>, memref<1x32x4096x128xf16, 1>
    return
  }
}
