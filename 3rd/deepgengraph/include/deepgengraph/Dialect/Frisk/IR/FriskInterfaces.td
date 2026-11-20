#ifndef FRISK_INTERFACES
#define FRISK_INTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Frisk_ExecutionOpInterface : OpInterface<"ExecutionOpInterface"> {
  let cppNamespace = "::mlir::frisk";
  let description = [{
    Execution and scheduling facts independent of memory effects. Unknown ops
    are boundaries. These facts do not authorize speculation, CSE or motion
    across collective, synchronization or asynchronous task dependencies.
  }];
  let methods = [InterfaceMethod<
    "Query the execution contract at this IR abstraction level.",
    "::mlir::frisk::ExecutionSemantics", "getExecutionSemantics", (ins),
    "", [{ return ::mlir::frisk::inferExecutionSemantics($_op); }]
  >];
}

// def LayoutMap : CppType<"mlir::DenseMap<mlir::Value, ::mlir::Attribute>&">;

def LayoutInterface : OpInterface<"LayoutInterface"> {
  let description = [{}];
  let methods = [
    InterfaceMethod<
      /*description=*/[{}],
      /*retTy=*/"mlir::LogicalResult",
      /*methodName=*/"inferLayout",
      /*args=*/(ins "mlir::OpBuilder &": $builder, 
                    "mlir::DenseMap<mlir::Value, ::mlir::Attribute> &":$LayoutMap),
      /*methodBody=*/"",
      /*defaultImplementation=*/[{
        return mlir::success(false);
      }]
    >
  ];
}



#endif // FRISK_INTERFACES
