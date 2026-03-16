#ifndef TILELANG_OPS
#define TILELANG_OPS

include "deepgengraph/Dialect/TL/IR/TilelangAttrDefs.td"
include "deepgengraph/Dialect/TL/IR/TilelangDialect.td"
include "deepgengraph/Dialect/TL/IR/TilelangOpInterface.td"
include "deepgengraph/Dialect/TL/IR/TilelangTypes.td"

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
class Tilelang_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tilelang_Dialect, mnemonic, !listconcat(traits, [])> {}


class Tilelang_BinaryElementWiseOp<string mnemonic, list<Trait> traits = []> :
  Tilelang_Op<mnemonic,
  !listconcat(traits, [Pure, SameOperandsAndResultElementType])> {

  let arguments = (ins
    AnyMemRef:$lhs,
    AnyMemRef:$rhs
  );

  let results = (outs AnyMemRef:$result);
  let assemblyFormat =  [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
  let hasVerifier = 1;
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs) , [{
  std::vector<Value> vr = {lhs, rhs };
  $_state.addOperands(vr);
  std::vector<Type> retTypes = {mlir::cast<MemRefType>(lhs.getType())} ;
  $_state.addTypes(retTypes);
  }] >
  ];
  let extraClassDeclaration = [{
  }];
}


class Tilelang_UnaryElementWiseOp<string mnemonic, list<Trait> traits = []> :
  Tilelang_Op<mnemonic,
  !listconcat(traits, [Pure, SameOperandsAndResultElementType])> {

  let arguments = (ins
    AnyMemRef:$input
  );

  let results = (outs AnyMemRef:$result);
  let assemblyFormat =  [{
    $input attr-dict `:` type($input) `->` type($result)
  }];
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$input) , [{
  std::vector<Value> vr = {input };
  $_state.addOperands(input);
  std::vector<Type> retTypes = {mlir::cast<MemRefType>(input.getType())} ;
  $_state.addTypes(retTypes);
  }] >
  ];
  let extraClassDeclaration = [{
  }];
}

def TileAddOp: Tilelang_BinaryElementWiseOp<"tile_add", [Commutative]> {  // Commutative : 符合交换律
  let summary = "add";
  let description = [{
  }];
}

def TileMulOp: Tilelang_BinaryElementWiseOp<"tile_mul", [Commutative]> {  // Commutative : 符合交换律
  let summary = "mul";
  let description = [{
  }];
}

def TileSubOp: Tilelang_BinaryElementWiseOp<"tile_sub", []> {  
  let summary = "sub";
  let description = [{
  }];
}

def TileDivOp: Tilelang_BinaryElementWiseOp<"tile_div", []> {  
  let summary = "div";
  let description = [{
  }];
}

def TileMaxOp: Tilelang_BinaryElementWiseOp<"tile_max", []> {  
  let summary = "max";
  let description = [{
  }];
}

def TileMinOp: Tilelang_BinaryElementWiseOp<"tile_min", []> {  
  let summary = "min";
  let description = [{
  }];
}

def TileExpOp: Tilelang_UnaryElementWiseOp<"tile_exp", []> {  
  let summary = "exp";
  let description = [{
  }];
}

def TileLnOp: Tilelang_UnaryElementWiseOp<"tile_ln", []> {  
  let summary = "ln";
  let description = [{
  }];
}

def TileAbsOp: Tilelang_UnaryElementWiseOp<"tile_abs", []> {  
  let summary = "abs";
  let description = [{
  }];
}

def AnnotateAddressOp : Tilelang_Op<"annotate_addr", [
  // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "T.annotate_address ";
  let arguments = (ins
    DictArrayAttr : $addrOffsets
  );
  let assemblyFormat = [{
    $addrOffsets attr-dict
  }];
}

def GemmV0Op : Tilelang_Op<"gemm_v0", [
  // DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "T.gemm_v0(a,b,c, transA=False, transB=false, init=False)";
  let arguments = (ins
    AnyMemRef : $matrixA,
    AnyMemRef : $matrixB,
    AnyMemRef : $matrixC,  // out
    I1Attr : $transA,
    I1Attr : $transB,
    I1Attr : $init
  );
  // let results = (outs
  //   I32 : $ret
  // );
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins 
    "Value":$matrixA, "Value": $matrixB, "Value" : $matrixC, 
    "bool" : $transA, "bool" : $transB, "bool" : $init) >
  ];
  let assemblyFormat = [{
    $matrixA $matrixB $matrixC attr-dict `:` type($matrixA) `*` type($matrixB) `->` type($matrixC)
  }];
  let hasVerifier = 1;
}

def SerialForOp : Tilelang_Op<"serial_for", [
  ScopedOpInterface ,
  // 声明该 Op 包含递归内存副作用
  RecursiveMemoryEffects, 
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>
]> {
  let summary = "for i in T.serial(lb, ub)";
  let arguments = (ins
    AnyTypeOf<[AnyInteger, Index]> : $lb,
    AnyTypeOf<[AnyInteger, Index]> : $ub
  );
  let skipDefaultBuilders = 1;
  let regions = (region SizedRegion<1>:$body);
  let builders = [
    OpBuilder<(ins "Value":$lb, "Value": $ub) >
  ];
  let assemblyFormat = [{
    $lb `,` $ub attr-dict `:` type($lb) `,` type($ub) $body
  }];
}

def SetCrossFlagOp : Tilelang_Op<"set_cross_flag", [
]> {
  let summary = "T.set_cross_flag(mte1, 1)";
  let arguments = (ins
    TL_CrossFlagAttr : $flag,
    I32Attr : $id
  );
  let results = (outs
    I32 : $ret
  );
  let assemblyFormat = [{
    $flag `,` $id `,` attr-dict `:` type($ret)
  }];
}

def WaitCrossFlagOp : Tilelang_Op<"wait_cross_flag", [
]> {
  let summary = "T.wait_cross_flag(1)";
  let arguments = (ins
    I32 : $flagId
  );
  let assemblyFormat = [{
    $flagId `{` attr-dict `}`
  }];
}

def CopyOp : Tilelang_Op<"copy", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "T.copy()";
  let arguments = (ins
    AnyMemRef : $src,
    AffineMapAttr : $src_map,
    AnyMemRef : $dst,
    AffineMapAttr : $dst_map
  );
  let hasVerifier = 1;
  let assemblyFormat = [{
    $src`:`type($src) $src_map `to` $dst`:`type($dst) $dst_map `,` attr-dict 
  }];
}

def AllocOp : Tilelang_Op<"alloc", [ ]> {
  let summary = "T.alloc_ub([block_M // 2, block_N], accum_dtype)";
  let arguments = (ins
    TL_MemSpaceAttr : $memspace,
    I64ArrayAttr : $shape,
    TypeAttr : $elementType
  );
  let results = (outs
    AnyMemRef : $result
  );
  // 添加自定义 builder，方便在 C++ 中直接调用
  let builders = [
    OpBuilder<(ins "MemSpace":$memspace, "ArrayRef<int64_t>":$shape, "Type":$elementType)>
  ];
  // 关键修复：显式指明打印结果类型，并使用 attr-dict 保证通用属性不丢失
  let assemblyFormat = [{
    $memspace $shape `x` $elementType attr-dict `:` type($result)
  }];
}

def YieldOp : Tilelang_Op<"yield", [Terminator]> {
  let summary = "indicates val reused among iterations for T.SerialFor";
  let arguments = (ins Variadic<AnyType>:$srcs);
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];
  let assemblyFormat = "attr-dict ($srcs^ `:` type($srcs))?";
}


def ReturnOp : Tilelang_Op<"return", [Pure, HasParent<"PrimFuncOp">, /*MemRefsNormalizable, */ReturnLike, Terminator]> {
  let summary = "Terminator for prim_func op. Indicate the vals to be returned";
  let description = [{
    return
  }];

  let arguments = (ins Variadic<AnyType>:$srcs);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let assemblyFormat = "attr-dict ($srcs^ `:` type($srcs))?";
  // let hasVerifier = 1;
}

def WithKernelOp : Tilelang_Op<"with_kernel", [
    ScopedOpInterface ,
    // 声明该 Op 包含递归内存副作用（通常 Kernel 会读写内存）
    RecursiveMemoryEffects, 
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
    NoTerminator
  ]> {
  
  let summary = "with T.Kernel(block_num, is_npu=True) as ()";
  let description = [{
    with T.Kernel(block_num, is_npu=True) as (cid, vid) : is_npu=True
    with T.Kernel(block_num, is_npu=True) as (bx, by, bz) : is_npu=False
  }];

  let arguments = (ins
    Variadic<AnyType>:$args,
    DenseI64ArrayAttr:$grid
  );

  // 定义 Region
  // SizedRegion<1> 表示该 Region 至少包含一个 Block
  let regions = (region SizedRegion<1>:$body);

  let builders = [
    OpBuilder<(ins "ValueRange":$operands, "ArrayRef<int64_t>":$grid), [{
      $_state.addOperands(operands);
      $_state.getOrAddProperties<Properties>().grid = $_builder.getDenseI64ArrayAttr(grid);
      auto region = $_state.addRegion();
      auto block = new Block();
      region->push_back(block);
      for (size_t i = 0; i < grid.size(); ++i) {
        region->front().addArgument($_builder.getIndexType(), $_state.location); 
      }
      for (auto v: operands) {
        region->front().addArgument(v.getType(), v.getLoc());
      }
    }]>
  ];

  let skipDefaultBuilders = 1;

  // 定义 MLIR 文本格式
  // cid 和 vid 将作为 entry_block 的参数显示
  let assemblyFormat = [{
    $args `:` type($args)  $grid attr-dict-with-keyword $body
  }];

  let extraClassDeclaration = [{
    mlir::Value getBodyArgs(uint32_t idx){
      return getScopedBody().getArgument(idx);
    }
  }];
}

def WithScopeOp : Tilelang_Op<"with_scope", [
    ScopedOpInterface ,
    // 声明该 Op 包含递归内存副作用（通常 Kernel 会读写内存）
    RecursiveMemoryEffects, 
    NoTerminator
  ]> {
  
  let summary = "with T.Scope('C')";
  let description = [{
  }];

  // 定义 Region
  // SizedRegion<1> 表示该 Region 至少包含一个 Block
  let regions = (region SizedRegion<1>:$body);
  let builders = [
    OpBuilder<(ins "bool":$is_cube)>
  ];
  // 定义 MLIR 文本格式
  // cid 和 vid 将作为 entry_block 的参数显示
  let assemblyFormat = [{
    attr-dict-with-keyword $body
  }];
}


def PrimFuncOp: Tilelang_Op<"prim_func", [FunctionOpInterface, IsolatedFromAbove]> {
  let summary = "@T.prim_func";
  let description = [{
  用于声明一个 tilelang 中的 prim_func
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       UnitAttr:$no_inline);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];  
  let extraClassDeclaration = [{
    static PrimFuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static PrimFuncOp create(Location location, StringRef name, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static PrimFuncOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // OpAsmOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Allow the dialect prefix to be omitted.
    static StringRef getDefaultDialect() { return "tilelang"; }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let hasCustomAssemblyFormat = 1;

}



def ParallelForOp: Tilelang_Op<"parallel_for", [Pure, IsolatedFromAbove, HasParent<"WithKernelOp">, SingleBlockImplicitTerminator<"tilelang::ParallelYieldOp">]> {
  let summary = "";
  let description = [{}];
  let arguments = (ins Variadic<AnyTensor>:$inputs);
  let results = (outs Variadic<AnyTensor>:$outputs);
  let regions = (region SizedRegion<1>:$region);
  let extraClassDeclaration = [{
    Value getIterArgInEntry(int i) {
      auto entry = &getRegion().front();
      return entry->getArgument(i);
    }
  }];
  let assemblyFormat = "$inputs $region attr-dict `:` functional-type(operands, results)";
}

def ParallelYieldOp: Tilelang_Op<"parallel_yield", [Pure, ReturnLike, Terminator, HasParent<"ParallelForOp">]> {
  let summary = "";
  let description = [{}];
  let arguments = (ins Variadic<AnyTensor>:$srcs);
  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];
  let assemblyFormat = "attr-dict $srcs `:` type($srcs)";
}

#endif // TILELANG_OPS