#ifndef FRISK_ATTRIBUTES_TD
#define FRISK_ATTRIBUTES_TD

include "deepgengraph/Dialect/Frisk/IR/FriskDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "deepgengraph/Dialect/Frisk/IR/FriskEnums.td"

def MemorySpaceAttr : EnumAttr<Frisk_Dialect, MemorySpaceEnum, "memory_space"> {
    let cppNamespace = "::mlir::frisk";
}

def DeviceKindAttr : EnumAttr<Frisk_Dialect, DeviceKindEnum, "device"> {
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

def GPUSwizzleAttr : AttrDef<Frisk_Dialect, "GPUSwizzle"> {
  let cppNamespace = "::mlir::frisk";
  let summary = "GPU shared-memory swizzle parameters";
  let mnemonic = "gpu_swizzle";

  let parameters = (ins
    "int32_t":$groupBits,
    "int32_t":$elementShift,
    "int32_t":$baseShift
  );

  let assemblyFormat = "`<` $groupBits `,` $elementShift `,` $baseShift `>`";
}

def GPULayoutAttr : AttrDef<Frisk_Dialect, "GPULayout"> {
    let cppNamespace = "::mlir::frisk";
    let summary = "Describes a GPU logical-to-physical tile layout";
    let mnemonic = "gpu_layout";

    let parameters = (ins
        AttrParameter<"::mlir::DenseI64ArrayAttr", "logical shape">:$logicalShape,
        AttrParameter<"::mlir::DenseI64ArrayAttr", "padded shape">:$paddedShape,
        AttrParameter<"::mlir::DenseI64ArrayAttr", "padding offsets">:$paddingOffsets,
        AttrParameter<"::mlir::frisk::attr::GPUTargetSpace", "target space">:$targetSpace,
        AttrParameter<"::mlir::AffineMapAttr", "base mapping">:$baseMap,
        OptionalParameter<"::mlir::frisk::GPUSwizzleAttr", "swizzle">:$swizzle
    );

    let assemblyFormat = "`<` $logicalShape `,` $paddedShape `,` $paddingOffsets `,` $targetSpace `,` $baseMap (`,` $swizzle^)? `>`";
}

#endif // FRISK_ATTRIBUTES_TD
