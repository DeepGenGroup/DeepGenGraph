#ifndef THREADIMP_OPS
#define THREADIMP_OPS

include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpAttrDefs.td"
include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpDialect.td"
include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpTypes.td"
// include "deepgengraph/Dialect/DeepgengraphTriton/IR/DeepgengraphTritonTypes.td"
include "deepgengraph/Dialect/ThreadImp/IR/ThreadImpOpInterface.td"

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike
include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultRank



// Op Base
class ThreadImp_Op<string mnemonic, list<Trait> traits = []> :
    Op<ThreadImp_Dialect, mnemonic, !listconcat(traits, [])> {}

def GetBlockIdOp : ThreadImp_Op<"getBlockId", [Pure]> {
  let summary = "get blockId x y or z";
  let description = [{}];

  let arguments = (ins I32Attr : $dim);  // 0,1,2 is valid
  let results = (outs Index : $bid);
  let assemblyFormat = [{
    `dim` `=` $dim attr-dict `:` type($bid)
  }];
  
}

def GetThreadIdOp : ThreadImp_Op<"getThreadId", [Pure]> {
  let summary = "getThreadId x y or z";
  let description = [{}];

  let arguments = (ins I32Attr : $dim);  // // 0,1,2 is valid
  let results = (outs Index : $bid);
  let assemblyFormat = [{
    `dim` `=` $dim attr-dict `:` type($bid)
  }];
}

// global copy to shm in block level
def BlockCopyG2S: ThreadImp_Op<"blockcpy_global_to_shm", [Pure, InferTypeOpAdaptor]> {
  let summary = "memcpy from g to s";
  let description = [{}];

  let arguments = (ins
    AnyTensor:$dst_tensor,
    ThreadImp_PtrType:$src_pointer,
    Index:$base_offset,
    DenseI64ArrayAttr:$shape,
    DenseI64ArrayAttr:$stride,
    DenseI64ArrayAttr:$offset,
    DenseI64ArrayAttr:$block_shape,
    DenseI64ArrayAttr:$order
  );

  let builders = [
    // default order (n-1, n-2, ..., 0)
    OpBuilder<(ins "Value":$dst_tensor, "Value":$src_pointer, "Value":$base_offset, "ArrayRef<int64_t>":$shape, "ArrayRef<int64_t>":$stride, "ArrayRef<int64_t>":$offset, "ArrayRef<int64_t>":$block_shape), [{
      SmallVector<int64_t> order;
      for (int i = (int)block_shape.size() - 1; i >= 0; --i) {
        order.push_back(i); 
      }
      build($_builder, $_state, dst_tensor, src_pointer, base_offset, shape, stride, offset, block_shape, order);
    }]>
  ];

  let results = (outs AnyTensor : $result);
  let assemblyFormat = "`dst` `=` $dst_tensor `,` `src` `=` $src_pointer `,` `base_offset` `=` $base_offset `,` `shape` `=` $shape `,` `stride` `=` $stride `,` `offset` `=` $offset `,` `block_shape` `=` $block_shape `,` `order` `=` $order `,`  attr-dict `:` functional-type(operands, results)";
}


def PointerAdvanceOp: ThreadImp_Op<"pointer_advance", [SameOperandsAndResultType]> {
  let summary = "pointer advance";
  let description = [{}];
  let arguments = (ins
    ThreadImp_PtrType:$src_pointer,
    DenseI64ArrayAttr:$offsets
  );
  let results = (outs ThreadImp_PtrType : $next_pointer);
  let assemblyFormat = "$src_pointer `,` `offsets` `=` $offsets attr-dict `:` functional-type(operands, results)";
}

def PreciseDotOp: ThreadImp_Op<"precise_dot_op", [Pure, InferTypeOpAdaptor]> {
  let summary = "Dot operation with output precision at least as high as input precision";
  let description = [{
    This operation performs a dot product on two input tensors with the same element type,
    and ensures the output tensor has a precision (element type) that is at least as high as the inputs.
  }];
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    TypeAttr:$acc_type
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat =  "$lhs `,` $rhs `,` `acc` `=` $acc_type attr-dict `:` functional-type(operands, results)";
}


#endif // TILELANG_OPS