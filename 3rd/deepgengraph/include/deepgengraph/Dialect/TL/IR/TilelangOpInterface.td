#ifndef TILELANG_OP_INTERFACES
#define TILELANG_OP_INTERFACES

include "mlir/IR/OpBase.td"

def ScopedOpInterface: OpInterface<"ScopedOpInterface"> {
  let description = [{
    op which is not a function, but contains a default region and a default block as body.
    for example : with T.kernel, T.scope("C"), 
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the scoped region",
      /*retTy=*/"::mlir::Region&",
      /*methodName=*/"getScopedBody",
      /*args=*/(ins),
      /*methodBody=*/[{}],
      /*defaultImplementation=*/[{
        return $_op->getRegion(0);
      }]
    >,
    ];

}


#endif 