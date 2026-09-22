// Input of the `frisk-pipeline-schedule` pass: `test/test_friskBaseDebug.mlir`
// verbatim, with only the FA3 2-stage / 2-buffer scheme attached to the
// `affine.for`.  The ops of the loop body are untouched.
//
//   pipeline.stage = [0,0, 1,1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1, 2,2,2, 1]
//   pipeline.order = [0,0, 1,1, 2, 4,4,4,4,4,4,4,4,4,4,4,4,4, 3,3,3, 5]
//                     K   V   QK  softmax(mask..copy)           PV rowsum
//
// One entry per top level op of the loop body (22 of them); consecutive ops
// sharing a `(stage, order)` pair form one statement.
//
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
    } {local_yield_transformed = true, pipeline.stage = array<i64: 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1>, pipeline.order = array<i64: 0, 0, 1, 1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 5>}
    frisk.sync_threads_in_block  // dbg
    %10 = frisk.div %9#0, %9#1 : vector<64x128xf32>, vector<64x1xf32> -> memref<64x128xf32>
    %oShm = frisk.alloc_buffer {scope = "local", alignment = 16} -> memref<64x128xf16>
    %oShm_copied = frisk.copy %10 to %oShm [affine_map<() -> (2)>] : memref<64x128xf32>, memref<64x128xf16> -> memref<64x128xf16>
    frisk.sync_threads_in_block  // dbg
    %O_copied = frisk.copy %oShm_copied to %O at(%block_id_y, %block_id_x) [affine_map<(d0, d1) -> (0, (d1 + (d0 * 8192) floordiv 524288) mod 32, (d0 * 64) mod 4096, 0)>] : memref<64x128xf16>, memref<1x32x4096x128xf16, 1> -> memref<1x32x4096x128xf16, 1>
    return
  }
}
