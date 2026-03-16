#ifndef TILELANG_ENUMS
#define TILELANG_ENUMS


include "deepgengraph/Dialect/TL/IR/TilelangDialect.td"
include "mlir/IR/EnumAttr.td" 

def TL_MemSpaceAttr : I32EnumAttr<
    "MemSpace", "",
    [
      I32EnumAttrCase<"GM", 1, "global">,
      I32EnumAttrCase<"L1", 2, "l1">,
      I32EnumAttrCase<"L0A", 3, "l0a">,
      I32EnumAttrCase<"L0B", 4, "l0b">,
      I32EnumAttrCase<"L0C", 5, "l0c">,
      I32EnumAttrCase<"UB", 6, "ub">,
    ]> {
    let cppNamespace = "::mlir::tilelang";
}

def TL_CrossFlagAttr : I32EnumAttr<
    "CrossFlag", "",
    [
      I32EnumAttrCase<"MTE1", 1, "mte1">,
      I32EnumAttrCase<"MTE2", 2, "mte2">,
      I32EnumAttrCase<"MTE3", 3, "mte3">,
      I32EnumAttrCase<"FIX", 4, "fix">,
      I32EnumAttrCase<"PIPE_V", 5, "v">,
      I32EnumAttrCase<"PIPE_M", 6, "m">,
    ]> {
    let cppNamespace = "::mlir::tilelang";
}



#endif // TILELANG_ENUMS