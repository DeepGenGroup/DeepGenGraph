#ifndef THREADIMP_ENUMS
#define THREADIMP_ENUMS


include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.td"
include "mlir/IR/EnumAttr.td" 

def ThreadImp_MemSpaceAttr : I32EnumAttr<
    "MemSpace", "",
    [
      I32EnumAttrCase<"GM", 1, "global">,
      I32EnumAttrCase<"SHM", 2, "shm">,
      I32EnumAttrCase<"L0A", 3, "l0a">,
      I32EnumAttrCase<"L0B", 4, "l0b">,
      I32EnumAttrCase<"L0C", 5, "l0c">,
      I32EnumAttrCase<"UB", 6, "ub">,
    ]> {
    let cppNamespace = "::mlir::threadimp";
}





#endif // 