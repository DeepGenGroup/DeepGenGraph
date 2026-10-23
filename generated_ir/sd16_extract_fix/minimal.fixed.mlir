module {
  func.func @repro(%arg0: vector<2x4xi32>, %arg1: index, %arg2: index) -> vector<4xi32> {
    %0 = vector.extract %arg0[%arg1, 0] : i32 from vector<2x4xi32>
    %1 = vector.extract %arg0[%arg1, 1] : i32 from vector<2x4xi32>
    %2 = vector.extract %arg0[%arg1, 2] : i32 from vector<2x4xi32>
    %3 = vector.extract %arg0[%arg1, 3] : i32 from vector<2x4xi32>
    %4 = vector.from_elements %0, %1, %2, %3 : vector<4xi32>
    return %4 : vector<4xi32>
  }
}

