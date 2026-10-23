func.func @repro(%v: vector<2x4xi32>, %row: index, %other: index) -> vector<4xi32> {
%e0 = vector.extract %v[%row, 0] : i32 from vector<2x4xi32>
%e1 = vector.extract %v[%row, 1] : i32 from vector<2x4xi32>
%e2 = vector.extract %v[%row, 2] : i32 from vector<2x4xi32>
%e3 = vector.extract %v[%row, 3] : i32 from vector<2x4xi32>
%result = vector.from_elements %e0, %e1, %e2, %e3 : vector<4xi32>
return %result : vector<4xi32>
}
