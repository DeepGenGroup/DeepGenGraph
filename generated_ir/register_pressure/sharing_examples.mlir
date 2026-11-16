// The original tile, its extracted fragment and the updated tile coexist.
// Untouched elements and the extracted old fragment share their content.
// The overwritten OLD fragment and the independent patch must both survive.
!f = !frisk.fragment<vector<1x4xf32>, {}>
module {
  func.func @static_fragment_sharing(%tile: vector<2x4xf32>, %patch: !f)
      -> (vector<2x4xf32>, vector<2x4xf32>, !f) {
    %c0 = arith.constant 0 : index
    %old = frisk.extract_frag %tile[%c0, %c0] : vector<2x4xf32> -> <vector<1x4xf32>, {}>
    %updated = frisk.insert_frag %patch, %tile[%c0, %c0] : <vector<1x4xf32>, {}> into vector<2x4xf32>
    return %tile, %updated, %old : vector<2x4xf32>, vector<2x4xf32>, !f
  }

  // The old tile dies at insert. Its overwritten content need not survive.
  func.func @static_fragment_replace(%tile: vector<2x4xf32>, %patch: !f)
      -> vector<2x4xf32> {
    %c0 = arith.constant 0 : index
    %updated = frisk.insert_frag %patch, %tile[%c0, %c0] : <vector<1x4xf32>, {}> into vector<2x4xf32>
    return %updated : vector<2x4xf32>
  }

  // Unknown row: keep the conservative tile+patch union, rather than a second
  // whole tile. The extracted result is already covered by the surviving tile.
  func.func @dynamic_fragment_sharing(%tile: vector<2x4xf32>, %patch: !f, %row: index)
      -> (vector<2x4xf32>, vector<2x4xf32>, !f) {
    %c0 = arith.constant 0 : index
    %old = frisk.extract_frag %tile[%row, %c0] : vector<2x4xf32> -> <vector<1x4xf32>, {}>
    %updated = frisk.insert_frag %patch, %tile[%row, %c0] : <vector<1x4xf32>, {}> into vector<2x4xf32>
    return %tile, %updated, %old : vector<2x4xf32>, vector<2x4xf32>, !f
  }

  // No data is loaded: neither memory contents nor memref descriptors enter
  // peak-live. Only the separate descriptor diagnostic is nonzero.
  func.func @memory_only(%g: memref<2048x2048xf16, 1>, %s: memref<64x64xf16, 3>)
      -> (memref<2048x2048xf16, 1>, memref<64x64xf16, 3>) {
    return %g, %s : memref<2048x2048xf16, 1>, memref<64x64xf16, 3>
  }

  // Two loaded f16 SSA values: data=2, packed-data=2, ideal-packed-data=1.
  // The backing global/shared buffers are excluded from all three metrics.
  func.func @two_f16_from_global_and_shm(%g: memref<2048x2048xf16, 1>, %s: memref<64x64xf16, 3>)
      -> (f16, f16) {
    %c0 = arith.constant 0 : index
    %x = memref.load %g[%c0, %c0] : memref<2048x2048xf16, 1>
    %y = memref.load %s[%c0, %c0] : memref<64x64xf16, 3>
    return %x, %y : f16, f16
  }

  // An explicit pointer or dynamic offset is a value, not backing memory.
  // DataLayout determines pointer/index widths; addresses stay separate from
  // data packing. No claim is made about SGPR versus VGPR allocation.
  func.func @explicit_addresses(%p: !llvm.ptr<1>, %i: i64)
      -> (!llvm.ptr<1>, !llvm.ptr<1>) {
    %q = llvm.getelementptr %p[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
    return %p, %q : !llvm.ptr<1>, !llvm.ptr<1>
  }
}
