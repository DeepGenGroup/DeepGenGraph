#ifndef THREADIMP_TYPES
#define THREADIMP_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.td"
include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpAttrDefs.td"

// 定义 ThreadImp 类型的基类
class ThreadImp_Type<string name, string typeMnemonic, list<Trait> traits = []> 
    : TypeDef<ThreadImp_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ThreadImp_PtrType : ThreadImp_Type<"Pointer", "ptr"> {
  let summary = "Pointer type";
  let description = [{
    ThreadImp Pointer type representing a pointer to a ranked tensor, but only contains element type info
  }];

  let parameters = (ins 
    "Type":$elementType,
    "threadimp::MemSpace":$memory_space
  );

  // --- Assembly Format ---
  // 语法示例: !threadimp.ptr<tensor<1x2xf32>, [1, 2], [0, 1], 0, GM>
  let assemblyFormat = [{
    `<` $elementType `,` $memory_space `>`
  }];

  // --- Builders ---
  let genStorageClass = 1;
  let builders = [
    // 标准 Builder：按顺序传入所有参数
    TypeBuilderWithInferredContext<(ins 
      "Type":$elementType,
      "threadimp::MemSpace":$memSpace
    ), [{
      // 在这里将原始 C++ 类型包装成 MLIR Attribute
      return $_get(elementType.getContext(), 
                   elementType,
                   memSpace);
    }]>
  ];
}



#endif // TILELANG_TYPES
