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
  let description = [{}];
  let parameters = (ins 
    "RankedTensorType": $pointee_type,
    DefaultValuedParameter<"MemSpace", "threadimp::MemSpace::GM">:$memory_space  
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins 
    "RankedTensorType":$pointee_type, 
    "threadimp::MemSpace":$memory_space
    ), [{
      return $_get(pointee_type.getContext(), pointee_type, memory_space);
    }]>
  ];

  let assemblyFormat = "`<` $pointee_type (`,` struct($memory_space)^)? `>`";
}



#endif // TILELANG_TYPES
