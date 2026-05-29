#ifndef FRISK_ENUMS_TD
#define FRISK_ENUMS_TD

include "mlir/IR/EnumAttr.td"

def MemorySpace_Local : I32EnumAttrCase<"Local", 0>;
def MemorySpace_Global  : I32EnumAttrCase<"Global", 1>;
def MemorySpace_Shared  : I32EnumAttrCase<"Shared", 3>;
def MemorySpace_None  : I32EnumAttrCase<"None", 9>;
def MemorySpace_Unknown  : I32EnumAttrCase<"Unknown", 10>;

def MemorySpaceEnum : I32EnumAttr<"MemorySpace",
    "Memory space for buffer allocation",
    [MemorySpace_Local, MemorySpace_Global, MemorySpace_Shared,
    MemorySpace_None, MemorySpace_Unknown]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::frisk::attr";
}

def GemmWarpPolicy_Square : I32EnumAttrCase<"Square", 0>;
def GemmWarpPolicy_FullRow : I32EnumAttrCase<"FullRow", 1>;
def GemmWarpPolicy_FullCol : I32EnumAttrCase<"FullCol", 2>;

def GemmWarpPolicyEnum : I32EnumAttr<"GemmWarpPolicy",
    "Policy for GEMM warp partitioning",
    [GemmWarpPolicy_Square, GemmWarpPolicy_FullRow, GemmWarpPolicy_FullCol]> {
  let cppNamespace = "::mlir::frisk::attr";
}


def Device_CC : I32EnumAttrCase<"CCore", 0>;
def Device_TC  : I32EnumAttrCase<"TCore", 1>;
def Device_TMA  : I32EnumAttrCase<"TMA", 3>;

def DeviceKindEnum : I32EnumAttr<"DevKind",
    "算子运行在什么类型的硬件(tensor/cuda core, tma)",
    [Device_CC,
    Device_TC,
    Device_TMA]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::frisk::attr";
}

#endif // FRISK_ENUMS_TD
