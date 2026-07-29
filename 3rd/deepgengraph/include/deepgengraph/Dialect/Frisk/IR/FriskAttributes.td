#ifndef FRISK_ATTRIBUTES_TD
#define FRISK_ATTRIBUTES_TD

include "deepgengraph/Dialect/Frisk/IR/FriskDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
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


// ------------- Layouts -------------
def SharedSwizzleLayoutAttr : AttrDef<Frisk_Dialect, "SharedSwizzleLayout", [
    DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface,
                                ["getAffineMap", "verifyLayout"]>
]> {
  let cppNamespace = "::mlir::frisk";
  let summary = "GPU shared-memory swizzle layout( NV 下有 32B 64B 128B swizzle)";
  let mnemonic = "sm_swizzle";

  let parameters = (ins
    "int32_t":$groupBits,
    "int32_t":$elementShift,
    "int32_t":$baseShift
  );

  let assemblyFormat = "`<` $groupBits `,` $elementShift `,` $baseShift `>`";
}

def DotOperandLayoutAttr : AttrDef<Frisk_Dialect, "DotOperandLayout", [
    DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface,
                                ["getAffineMap", "verifyLayout"]>
]> {
    let cppNamespace = "::mlir::frisk";
    let summary = "tile级别矩阵乘时, 寄存器上 ab fragment 的layout(如果为 nvidia wgmma_ss, 则应选择 shared_swizzle layout, 而非这个)";
    let mnemonic = "dot_operand";

    let parameters = (ins
    "int32_t" : $warp_id,
    "int32_t" : $lane_id,
    "int32_t" : $reg_id,
    ArrayRefParameter<"int32_t"> : $inst_mnk
    );

    let assemblyFormat = "`<` $warp_id `,` $lane_id `,` $reg_id `,` $inst_mnk `>`";
}

def DotAccumLayoutAttr : AttrDef<Frisk_Dialect, "DotAccumLayout", [
    DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface,
                                ["getAffineMap", "verifyLayout"]>
]> {
    let cppNamespace = "::mlir::frisk";
    let summary = "tile级别矩阵乘时, 寄存器上 accumulator 的layout";
    let mnemonic = "dot_acc";

    let parameters = (ins
    "int32_t" : $warp_id,
    "int32_t" : $lane_id,
    "int32_t" : $reg_id,
    ArrayRefParameter<"int32_t"> : $inst_mnk
    );
    let assemblyFormat = "`<` $warp_id `,` $lane_id `,` $reg_id `,` $inst_mnk `>`";
}

def NaiveLayoutAttr : AttrDef<Frisk_Dialect, "NaiveLayout", [
    DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface,
                                ["getAffineMap", "verifyLayout"]>
]> {
    let cppNamespace = "::mlir::frisk";
    let summary = "无要求的普通 layout";
    let mnemonic = "naive_distributed";

    let parameters = (ins
    "int32_t" : $warp_id,
    "int32_t" : $lane_id,
    "int32_t" : $reg_id
    );
    let assemblyFormat = "`<` $warp_id `,` $lane_id `,` $reg_id  `>`";
}

def GPULayoutAttr : AttrDef<Frisk_Dialect, "GPULayout", [
    DeclareAttrInterfaceMethods<MemRefLayoutAttrInterface,
                                ["getAffineMap", "verifyLayout"]>
]> {
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
