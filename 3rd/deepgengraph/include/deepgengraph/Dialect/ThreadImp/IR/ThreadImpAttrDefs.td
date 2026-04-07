#ifndef THREADIMP_ENUMS
#define THREADIMP_ENUMS


include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.td"
include "mlir/IR/EnumAttr.td" 

// 定义 ThreadImp 属性的基类
class ThreadImp_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<ThreadImp_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

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

def ThreadImp_OpKindEnumAttr : I32EnumAttr<
    "OpKind", "",
    [
      I32EnumAttrCase<"add", 1, "add">,
      I32EnumAttrCase<"sub", 2, "sub">,
      I32EnumAttrCase<"mul", 3, "mul">,
      I32EnumAttrCase<"div", 4, "div">,
      I32EnumAttrCase<"exp", 5, "exp">,
      I32EnumAttrCase<"mod", 6, "mod">,
    ]> {
    let cppNamespace = "::mlir::threadimp";
}



// 假设你的 Dialect 类名为 ThreadImp_Dialect
def ThreadImp_TensorEncodingAttr : ThreadImp_Attr<"TensorEncoding", "tensor_encoding"> {
  let summary = "Encoding attribute for ThreadImp tensors.";
  let description = [{
    该属性用于 RankedTensorType 的 encoding 字段，
    显式指定 Tensor 所处的硬件存储空间（Global, SHM, L0, UB 等）。
  }];

  // 这里的参数类型使用的是你定义的 EnumAttr 的 C++ 类型名称
  let parameters = (ins 
    "MemSpace":$memSpace
  );

  // 定义打印和解析格式：#threadimp.tensor_encoding<shm>
  let assemblyFormat = "`<` $memSpace `>`";
}


#endif // 