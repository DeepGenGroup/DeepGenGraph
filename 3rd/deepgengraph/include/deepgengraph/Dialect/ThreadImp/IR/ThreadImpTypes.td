#ifndef TILELANG_TYPES
#define TILELANG_TYPES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.td"
include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpAttrDefs.td"

// // 定义 TileLang 类型的基类
// class TI_Type<string name, string typeMnemonic, list<Trait> traits = []> 
//     : TypeDef<ThreadImp_Dialect, name, traits> {
//   let mnemonic = typeMnemonic;
// }


// def TI_PtrType : TI_Type<"Pointer", "ptr"> {
//   let summary = "Pointer type";
//   let description = [{}];
//   let parameters = (ins 
//     "RankedTensorType": $pointee_type,
//     DefaultValuedParameter<"MemSpace", "threadimp::MemSpace::GM">:$memory_space  
//   );

//   let builders = [
//     TypeBuilderWithInferredContext<(ins 
//     "RankedTensorType":$pointee_type, 
//     "tilelang::MemSpace":$memory_space
//     ), [{
//       return $_get(pointee_type.getContext(), pointee_type, memory_space);
//     }]>
//   ];

//   let assemblyFormat = "`<` $pointee_type (`,` struct($memory_space)^)? `>`";
// }

// def TI_BlockPtrType : TI_Type<"BlockPointer", "block_ptr"> {
//   let summary = "Block Pointer type";
//   let description = [{}];
//   let parameters = (ins 
//     "RankedTensorType": $pointee_type,
//     DefaultValuedParameter<"MemSpace", "threadimp::MemSpace::GM">:$memory_space  
//   );

//   let builders = [
//     TypeBuilderWithInferredContext<(ins 
//     "RankedTensorType":$pointee_type, 
//     "tilelang::MemSpace":$memory_space
//     ), [{
//       return $_get(pointee_type.getContext(), pointee_type, memory_space);
//     }]>
//   ];

//   let assemblyFormat = "`<` $pointee_type (`,` struct($memory_space)^)? `>`";
// }



#endif // TILELANG_TYPES
