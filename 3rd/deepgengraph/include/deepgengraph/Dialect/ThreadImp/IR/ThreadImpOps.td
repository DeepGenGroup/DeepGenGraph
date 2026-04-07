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

def GetBlockIdOp : ThreadImp_Op<"get_bid", [Pure]> {
  let summary = "get blockId x y or z";
  let description = [{}];

  let arguments = (ins I32Attr : $dim);  // 0,1,2 is valid
  let results = (outs Index : $bid);
  let assemblyFormat = [{
    `dim` `=` $dim attr-dict `:` type($bid)
  }];
}

def GetThreadIdOp : ThreadImp_Op<"get_tid", [Pure]> {
  let summary = "getThreadId x y or z";
  let description = [{}];

  let arguments = (ins I32Attr : $dim);  // // 0,1,2 is valid
  let results = (outs Index : $bid);
  let assemblyFormat = [{
    `dim` `=` $dim attr-dict `:` type($bid)
  }];
}

// 对内存的抽象 : tensor 
// ptr 指向内存区。可移动 
// 
def PointerToOp : ThreadImp_Op<"ptr_to", [Pure, InferTypeOpAdaptor]> {
  let summary = "get a pointer from a tensor";
  let description = [{
    从 global or shm tensor 构造一个指针。
  }];

  let arguments = (ins 
    AnyTensor : $srcTensor,
    ThreadImp_MemSpaceAttr : $memspace
    );
  let results = (outs ThreadImp_PtrType : $pointer);
  let assemblyFormat = [{
    `srcTensor` `=` $srcTensor `,` `memspace` `=` $memspace attr-dict `:` functional-type(operands, results)
  }];
}


def PointerToEmptyGlobalOp: ThreadImp_Op<"ptr_to_empty_global", [InferTypeOpAdaptor]> {
  let summary = "临时性构造一个指向global的指针。该global的内存区为空白的";
  let description = [{}];

  let arguments = (ins TypeAttr:$tensor_type);
  let results = (outs ThreadImp_PtrType:$result);
  let assemblyFormat = "`type` `=` $tensor_type attr-dict `:` type($result)";
}



def AllocSharedOp : ThreadImp_Op<"alloc_shared", [Pure]> {
  let summary = "alloc a shm tensor";
  let description = [{}];
  let arguments = (ins
    DenseI64ArrayAttr : $shape,
    OptionalAttr <F32Attr> : $initValue
  );
  let builders = [
    // 自动生成的 builder 通常够用，但如果需要自定义：
    OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType, 
                   CArg<"FloatAttr", "{}">:$initValue), [{
      $_state.addAttribute("shape", $_builder.getDenseI64ArrayAttr(shape));
      if (initValue) $_state.addAttribute("init_value", initValue);
      $_state.addTypes(RankedTensorType::get(shape, elementType));
    }]>
  ];

  let results = (outs AnyTensor : $result);
  // 优化 assemblyFormat，使其支持可选的初值打印
  let assemblyFormat = [{
    $shape (`init``=` $initValue^)? attr-dict `:` type($result)
  }];
}

// global copy to shm in block level
def CopyGlobalToShm: ThreadImp_Op<"copy_global_to_shm", [
  MemoryEffects<[MemWrite]>, MemoryEffects<[MemRead]>
]> {
  let summary = "memcpy from global to shm";
  let description = [{}];
  let arguments = (ins
    Arg<ThreadImp_PtrType, "source", [MemRead]> : $src_pointer,
    Arg<ThreadImp_PtrType, "destination", [MemWrite]> : $dst_pointer,
    DenseI64ArrayAttr : $offset,
    DenseI64ArrayAttr : $block_shape,
    DenseI64ArrayAttr : $src_stride,
    DenseI64ArrayAttr : $src_order
  );
  let results = (outs);
  let assemblyFormat = [{
  `src` `=` $src_pointer `dst``=` $dst_pointer `,``offset` `=` $offset `,``block_shape` `=` $block_shape `,``stride` `=` $src_stride `,``order` `=` $src_order attr-dict `:` functional-type(operands, results)
  }];
}

// global copy to shm in block level
def CopyShmToGlobal: ThreadImp_Op<"copy_shm_to_global", [
  MemoryEffects<[MemWrite]>, MemoryEffects<[MemRead]>
]> {
  let summary = "memcpy from shm to global";
  let description = [{}];
  let arguments = (ins
    Arg<ThreadImp_PtrType, "source", [MemRead]> : $src_pointer,
    Arg<ThreadImp_PtrType, "destination", [MemWrite]> : $dst_pointer,
    DenseI64ArrayAttr : $offset,
    DenseI64ArrayAttr : $block_shape,
    DenseI64ArrayAttr : $dst_stride,
    DenseI64ArrayAttr : $dst_order
  );
  let results = (outs);
  let assemblyFormat = [{
  `src``=` $src_pointer `dst` `=` $dst_pointer `,` `offset` `=` $offset `,``block_shape` `=` $block_shape `,``stride` `=` $dst_stride `,``order` `=` $dst_order attr-dict `:` functional-type(operands, results)
  }];
}

def ReadShmOp : ThreadImp_Op<"read_shm", [Pure]> {
  let summary = "read data from shmptr";
  let description = [{}];
  let arguments = (ins
    ThreadImp_PtrType : $shm_ptr
  );
  let results = (outs AnyTensor : $resultData);
  let assemblyFormat = [{
  `srcShm``=` $shm_ptr attr-dict `:` functional-type(operands, results)
  }];
}

def WriteShmOp : ThreadImp_Op<"write_shm", [Pure]> {
  let summary = "write tensor data to shm through shm_ptr. Returns a new shm_ptr (便于区分ReadWrite顺序 以及数据流跟踪)";
  let description = [{}];
  let arguments = (ins
    ThreadImp_PtrType : $shm_ptr,
    AnyTensor : $data
  );
  let results = (outs ThreadImp_PtrType : $result_ptr);
  let assemblyFormat = [{
  `dst``=` $shm_ptr `,` `data``=` $data  attr-dict `:` functional-type(operands, results)
  }];
}

def PointerAdvanceOp: ThreadImp_Op<"ptr_advance", [SameOperandsAndResultType]> {
  let summary = "pointer advance";
  let description = [{}];
  let arguments = (ins
    ThreadImp_PtrType:$src_pointer,
    DenseI64ArrayAttr:$offsets
  );
  let results = (outs ThreadImp_PtrType : $next_pointer);
  let assemblyFormat = "$src_pointer `,` `offsets` `=` $offsets attr-dict `:` functional-type(operands, results)";
}

def PreciseMatmulOp: ThreadImp_Op<"preciseMatmul", [Pure, InferTypeOpAdaptor]> {
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

def CopyShmToReg : ThreadImp_Op<"copyShmToReg", [Pure]> {
  let summary = "copy shm to reg and return data";
  let description = [{
    This operation performs a dot product on two input tensors with the same element type,
    and ensures the output tensor has a precision (element type) that is at least as high as the inputs.
  }];
  let arguments = (ins
    AnyTensor : $src,
    I32Attr : $threads
  );
  let results = (outs AnyTensor:$dst);
  let assemblyFormat =  [{
    $src `,` `threads` `=` $threads  attr-dict `:` functional-type(operands, results)
  }];
}

def CopyRegToShm : ThreadImp_Op<"copyRegToShm", [Pure]> {
  let summary = "Dot operation with output precision at least as high as input precision";
  let description = [{
    This operation performs a dot product on two input tensors with the same element type,
    and ensures the output tensor has a precision (element type) that is at least as high as the inputs.
  }];
  let arguments = (ins
    AnyTensor : $src,
    I32Attr : $threads
  );
  let results = (outs AnyTensor:$dst);
  let assemblyFormat =  [{
    $src `,` `threads` `=` $threads  attr-dict `:` functional-type(operands, results)
  }];
}

def ThreadElementwiseBinaryOp: ThreadImp_Op<"threadElementwiseBinary", [Pure, InferTypeOpAdaptor]> {
  let summary = "Dot operation with output precision at least as high as input precision";
  let description = [{
    This operation performs a dot product on two input tensors with the same element type,
    and ensures the output tensor has a precision (element type) that is at least as high as the inputs.
  }];
  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    ThreadImp_OpKindEnumAttr : $kind
  );
  let results = (outs AnyTensor:$result);
  let assemblyFormat =  [{
    $lhs `,` $rhs `,` `kind` `=` $kind  attr-dict `:` functional-type(operands, results)
  }];
}


#endif // TILELANG_OPS