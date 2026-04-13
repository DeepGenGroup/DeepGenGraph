#ifndef FRISK_ATTRIBUTES_TD
#define FRISK_ATTRIBUTES_TD

include "deepgengraph/Dialect/Frisk/IR/FriskDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "deepgengraph/Dialect/Frisk/IR/FriskEnums.td"

def MemorySpaceAttr : EnumAttr<Frisk_Dialect, MemorySpaceEnum, "memory_space"> {
    let cppNamespace = "::mlir::frisk";
}

def GemmWarpPolicyAttr : EnumAttr<Frisk_Dialect, GemmWarpPolicyEnum, "gemm_warp_policy"> {
    let cppNamespace = "::mlir::frisk";
    let summary = "GEMM warp partitioning policy";
}

def LayoutAttr : AttrDef<Frisk_Dialect, "Layout"> {
    let cppNamespace = "::mlir::frisk";
    let summary = "Describes a logical-to-physical mapping for tile layouts";
    let mnemonic = "layout";
    let assemblyFormat = "`<` $inputShape `,` $forwardIndex (`,` $forwardThread^)? (`,` $replicateSize^)? `>`";

    let parameters = (ins
        AttrParameter<"::mlir::DenseI64ArrayAttr", "input shape">:$inputShape,
        AttrParameter<"::mlir::AffineMapAttr", "forward index mapping">:$forwardIndex,
        OptionalParameter<"::mlir::AffineMapAttr", "thread mapping">:$forwardThread,
        OptionalParameter<"::mlir::IntegerAttr", "replication factor">:$replicateSize
    );

}

def MemSliceAttr : AttrDef<Frisk_Dialect, "MemSlice"> {
    let cppNamespace = "::mlir::frisk";
    let summary = [{
    表示切片语义。例子：
    %arg = memref<10xf32>,  如果其 memrange 为 <[i], #map, [5] >  则表示： %arg[#map(i) : #map(i) + 5] : memref<5xf32>
    }];
    let mnemonic = "memslice";

    let parameters = (ins
        AttrParameter<"::mlir::ArrayAttr", "affineMap ijk var">:$dims,
        AttrParameter<"::mlir::AffineMap", "affineMap">:$map,
        AttrParameter<"::mlir::DenseI64ArrayAttr", "ranges on dims">:$ranges
    );
    let assemblyFormat = "`<` $dims `,` $map `,` `ranges``=` $ranges `>`";
}



#endif // FRISK_ATTRIBUTES_TD
