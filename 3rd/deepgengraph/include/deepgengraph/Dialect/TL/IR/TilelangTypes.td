#ifndef TILELANG_TYPES
#define TILELANG_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "deepgengraph/Dialect/TL/IR/TilelangDialect.td"
include "deepgengraph/Dialect/TL/IR/TilelangAttrDefs.td"

// 定义 TileLang 类型的基类
class TL_Type<string name, string typeMnemonic, list<Trait> traits = []> 
    : TypeDef<Tilelang_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}


def TL_PtrType : TL_Type<"Pointer", "ptr"> {
  let summary = "Pointer type";
  let description = [{}];
  let parameters = (ins 
    "RankedTensorType": $pointee_type,
    DefaultValuedParameter<"MemSpace", "tilelang::MemSpace::GM">:$memory_space  
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins 
    "RankedTensorType":$pointee_type, 
    "tilelang::MemSpace":$memory_space
    ), [{
      return $_get(pointee_type.getContext(), pointee_type, memory_space);
    }]>
  ];

  let assemblyFormat = "`<` $pointee_type (`,` struct($memory_space)^)? `>`";
}

def TL_BlockPtrType : TL_Type<"BlockPointer", "block_ptr"> {
  let summary = "Block Pointer type";
  let description = [{}];
  let parameters = (ins 
    "RankedTensorType": $pointee_type,
    DefaultValuedParameter<"MemSpace", "tilelang::MemSpace::GM">:$memory_space  
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins 
    "RankedTensorType":$pointee_type, 
    "tilelang::MemSpace":$memory_space
    ), [{
      return $_get(pointee_type.getContext(), pointee_type, memory_space);
    }]>
  ];

  let assemblyFormat = "`<` $pointee_type (`,` struct($memory_space)^)? `>`";
}



#endif // TILELANG_TYPES
