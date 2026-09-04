#ifndef FRISK_OPS
#define FRISK_OPS

include "deepgengraph/Dialect/Frisk/IR/FriskDialect.td"
include "deepgengraph/Dialect/Frisk/IR/FriskEnums.td"
include "deepgengraph/Dialect/Frisk/IR/FriskAttributes.td"
include "deepgengraph/Dialect/Frisk/IR/FriskTypes.td"
// include "Dialect/Frisk/IR/FriskInterfaces.td"

include "mlir/IR/OpBase.td"
// include "mlir/IR/AttrTypeBase.td"
// include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td" // SymbolUserOpInterface
include "mlir/Interfaces/CallInterfaces.td" // CallOpInterface
include "mlir/IR/OpAsmInterface.td" // OpAsmOpInterface
include "mlir/Interfaces/FunctionInterfaces.td" // FunctionOpInterface
include "mlir/Interfaces/SideEffectInterfaces.td" // Pure
include "mlir/Interfaces/ControlFlowInterfaces.td" // ReturnLike

include "mlir/Interfaces/InferTypeOpInterface.td" // SameOperandsAndResultRank


// Op Base
class Frisk_Op<string mnemonic, list<Trait> traits = []> : 
  Op<Frisk_Dialect, mnemonic, traits>;

def ImplicitFriskTerminator : SingleBlockImplicitTerminator<"EndOp">;

// def MemRefOrVector : AnyTypeOf<[AnyMemRef], "memref or vector type">;
// def MemRefOrVector : AnyTypeOf<[AnyMemRef, AnyVectorOfNonZeroRank], "memref or vector type">;

// KernelOp
def Frisk_KernelOp : Frisk_Op<"kernel", [
  ImplicitFriskTerminator, CallableOpInterface, FunctionOpInterface,
  IsolatedFromAbove
]> {
    let summary = "Kernel operation";
  let description = [{
    A simple kernel operation that represents a compute kernel.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<StrAttr>:$sym_visibility,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );

  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type)>
  ];

  let extraClassDeclaration = [{
    ::mlir::Region *getCallableRegion() { return &getRegion(); }

    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    bool isDeclaration() { return false; }

    unsigned getNumArguments() {
      return getBody(0)->getNumArguments();
    }

    BlockArgument getArgument(unsigned idx) {
      return getBody(0)->getArgument(idx);
    }

    Block* addEntryBlock() {
      auto& region = getRegion();
      if (!region.empty()) {
        llvm_unreachable("entry block already exists");
      }
      auto* entry = new Block;
      region.push_back(entry);
      // block add args
      mlir::Type funcType = getFunctionType();
      if (auto funcTy = mlir::dyn_cast<FunctionType>(funcType)) {
        for (auto type : funcTy.getInputs()) {
          entry->addArgument(type, getLoc());
        }
      }
      mlir::OpBuilder builder(entry, entry->begin());
      builder.create<EndOp>(getLoc());
      return entry;
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// parallelOp
def Frisk_ParallelOp : Frisk_Op<"parallel", [
  HasParent<"KernelOp">,
  ImplicitFriskTerminator, AffineScope, 
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>
]> {
  let summary = "parallel opeartion";
  let description = [{
    parallel
  }];

  let arguments = (ins
    I64Attr:$threads,
    DenseI64ArrayAttr:$ranges
  );

  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "ArrayRef<int64_t>":$ranges, "int64_t":$thread_num)>
  ];

  let extraClassDeclaration = [{
    std::vector<int64_t> getGrid() {
      std::vector<int64_t> ranges;
      for (auto range : getRangesAttr().asArrayRef()) {
        ranges.push_back(range);
      }
      return ranges;
    }

    unsigned getGridDims() { 
      return getRangesAttr().size(); 
    }

    int64_t getThreadNum() {
      return getThreads();
    }

    MutableArrayRef<BlockArgument> getIVs() {
      return getBody()->getArguments();
    }

    Block* addEntryBlock() {
      auto& region = getRegion();
      if (!region.empty()) {
        llvm_unreachable("entry block already exists");
      }
      auto* entry = new Block;
      region.push_back(entry);
      for (unsigned i=0; i<getGridDims(); i++) {
        entry->addArgument(IndexType::get(getContext()), getLoc());
      }
      mlir::OpBuilder builder(entry, entry->begin());
      builder.create<EndOp>(getLoc());
      return entry;
    }

    ::mlir::LogicalResult inferLayout(::mlir::OpBuilder &builder,
                                      llvm::DenseMap<::mlir::Value, ::mlir::Attribute> &layoutMap);
  }];

  let hasVerifier = 0;
  let hasCustomAssemblyFormat = 1;
}

// blockOp
def Frisk_BlockOp : Frisk_Op<"block", [RecursiveMemoryEffects, ImplicitFriskTerminator, AffineScope]> {
  let summary = "block opeartion";
  let description = [{
    block
  }];

  let arguments = (ins
    DenseI64ArrayAttr:$ranges
  );

  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "ArrayRef<int64_t>":$ranges, 
               CArg<"function_ref<void(ValueRange)>", "nullptr">:$bodyBuilder)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn = function_ref<void(ValueRange)>;

    std::vector<int64_t> getBlockRanges() {
      std::vector<int64_t> ranges;
      for (auto range : getRangesAttr().asArrayRef()) {
        ranges.push_back(range);
      }
      return ranges;
    }

    int64_t getIterationNum() {
      int64_t iter_num = 1;
      for (int64_t range : getBlockRanges()) {
        iter_num *= range;
      }
      return iter_num;
    }

    int64_t getBlockDim() {
      return getRangesAttr().size();
    }

    MutableArrayRef<BlockArgument> getIVs() {
      return getBody()->getArguments();
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// forOp
def Frisk_ForOp : Frisk_Op<"for", [RecursiveMemoryEffects, ImplicitFriskTerminator]> {
  let summary = "for opeartion";
  let description = [{
    for
  }];

  let arguments = (ins I64Attr:$lower, I64Attr:$upper, I64Attr:$step);

  let regions = (region SizedRegion<1>:$region);

  let builders = [OpBuilder<(ins "int64_t":$lowerBound, 
                  "int64_t":$upperBound,
              CArg<"int64_t", "1">:$step, 
              CArg<"function_ref<void(Value)>", "nullptr">:$bodyBuilder)>];
  
  let extraClassDeclaration = [{
    using BodyBuilderFn = function_ref<void(Value)>;

    BlockArgument getInductionVar() { return getBody()->getArgument(0); }
    Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front();
    }

    // /// Returns information about the lower bound as a single object.
    // int64_t getLowerBound() { return getProperties().lower; }

    // /// Returns information about the upper bound as a single object.
    // int64_t getUpperBound() { return getProperties().upper; }

    // /// Returns loop step.
    // int64_t getStep() { return getProperties().step; }
  }];

  let hasVerifier = 0;
  let hasCustomAssemblyFormat = 1;
}

// EndOp
def Frisk_EndOp : Frisk_Op<"end", [Terminator, ReturnLike]> {
  let summary = "Invisible kernel termination operation";
  let description = [{
    An invisible terminator operation for frisk kernels that dose not appear in printed IR.
  }];

  let arguments = (ins);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, {}); }]>
  ];

  let hasVerifier = 0;
  let assemblyFormat = "attr-dict";
}

// gemmOp
def Frisk_GemmOp: Frisk_Op<"gemm", [MemoryEffects<[MemWrite]>, MemoryEffects<[MemRead]>]> {
  let summary = "General matrix multiplication operation";
  let description = [{}];
  let arguments = (ins
    AnyMemRef:$A,  // shm
    AnyMemRef:$B,
    AnyMemRef:$C,

    BoolAttr:$transA,
    BoolAttr:$transB 
  );

  let results = (outs);

  let extraClassDeclaration = [{
    Value getMatrixA() { return getA(); }
    Value getMatrixB() { return getB(); }
    Value getMatrixC() { return getC(); }
    ValueRange getAllMatrix() {
      return ValueRange(getOperation()->getOperands());
    }
    ::mlir::LogicalResult inferLayout(::mlir::OpBuilder &builder,
                                      llvm::DenseMap<::mlir::Value, ::mlir::Attribute> &layoutMap);
  }];


  let hasVerifier = 1;
  let assemblyFormat = " `(` $A `,` $B `)` `to` $C attr-dict `:` type($A) `,` type($B) `,` type($C)";
}

// ifOp
def Frisk_IfOp : Frisk_Op<"if", [RecursiveMemoryEffects, ImplicitFriskTerminator]> {
  let summary = "if branch operation";
  let description = [{
    if-else branch operation with IntegerSet condition.
  }];

  let arguments = (ins IntegerSetAttr:$condition, Variadic<Index>:$operands);
  let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

  let extraClassDeclaration = [{
    bool hasElse() { return !getElseRegion().empty(); }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// WarpGroupOp
def Frisk_WarpGroupOp : Frisk_Op<"warp_group", [RecursiveMemoryEffects, ImplicitFriskTerminator]> {
  let summary = "warpgroup operation";
  let description = [{
    warp group operation.
  }];

  let arguments = (ins I64Attr:$warpGroupNum);
  let regions = (region SizedRegion<1>:$region);

  let builders = [
    OpBuilder<(ins "uint64_t":$warpGroupNum)>
  ];
  
  let extraClassDeclaration = [{

  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// gemmwaitOp
def Frisk_GemmWaitOp : Frisk_Op<"gemm_wait", [MemoryEffects<[MemRead]>]> {
  let summary = "wait for gemm write completion";
  let description = [{
    Wait until the specified memref buffer is written back after gemm commit.
  }];

  let arguments = (ins Arg<AnyMemRef, "", [MemRead]>:$buffer);

  let extraClassDeclaration = [{
    Value getWaitBuffer() { return getBuffer(); }
  }];

  let assemblyFormat = "$buffer attr-dict `:` type($buffer)";
}

// bufferViewOp
def Frisk_BufferViewOp : Frisk_Op<"buffer_view", [MemoryEffects<[MemRead]>]> {
  let summary = "create a memref view from source buffer and slice";
  let description = [{
    Create a view memref from a source memref using dynamic indices, an affine
    map and per-dimension ranges.
    The result memref keeps source element type and memory space, and uses
    ranges as result shape.
  }];

  let arguments = (ins
    AnyMemRef:$source,
    Variadic<Index>:$indices,
    AffineMapAttr:$indexMap,
    DenseI64ArrayAttr:$ranges
  );

  let results = (outs
    Res<AnyMemRef>:$view
  );

  let builders = [
    OpBuilder<(ins "Value":$source, "ValueRange":$indices, "AffineMap":$indexMap, "ArrayRef<int64_t>":$ranges)>,
    OpBuilder<(ins "Value":$source, "ValueRange":$indices, "AffineMapAttr":$indexMap, "DenseI64ArrayAttr":$ranges)>
  ];

  let extraClassDeclaration = [{
    MemRefType getSourceType() { return cast<MemRefType>(getSource().getType()); }
    MemRefType getViewType() { return cast<MemRefType>(getView().getType()); }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}
// alloc buffer
def Frisk_AllocBufferOp: Frisk_Op<"alloc_buffer", [MemoryEffects<[MemAlloc]>]> {
  let summary = "allocate buffer operation";
  let description = [{ 
    Allocates a buffer with the specified shape, element type and optional attributes.
  }];

  let arguments = (ins 
    DenseI64ArrayAttr:$shape,
    TypeAttr:$elementType,
    DefaultValuedAttr<I64Attr, "0">:$alignment,
    DefaultValuedAttr<I64Attr, "0">:$memorySpace
  );

  let results = (outs 
    Res<AnyMemRef>:$result
  );

  let builders = [
    OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType)>,
    OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType, "int64_t":$alignment)>,
    OpBuilder<(ins "ArrayRef<int64_t>":$shape, "Type":$elementType, "int64_t":$alignment, "int64_t":$memorySpace)>
  ];

  let extraClassDeclaration = [{
    MemRefType getMemRefType() { 
      return cast<MemRefType>(getResult().getType()); 
    }
    
    // 根据 memorySpace 获取 scope
    StringRef getScope() {
      switch (getMemorySpace()) {
        case 0: return "local";
        case 1: return "global"; 
        case 3: return "shared";
        case 5: return "local";
        default: return "unknown";
      }
    }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

// copy
def Frisk_CopyOp : Frisk_Op<"copy", [
    MemoryEffects<[MemRead, MemWrite]>,
    DeclareOpInterfaceMethods<OpAsmOpInterface>
]> {
  let summary = "copy data with affine-based destination offset";
  let description = [{
    The `copy` operation performs a memory copy from a source memref to a destination memref.
    The destination address is calculated using an affine map applied to the provided 
    map_operands.
    
    Example:
      frisk.copy %src to %dst at (%i, %j) [affine_map<(d0, d1) -> (0, 0, d0, d1)>] 
        : memref<64x32xf32>, memref<1x32x1024x128xf32>
  }];

  let arguments = (ins 
    Arg<AnyMemRef>:$src,
    Arg<AnyMemRef>:$dst,
    Variadic<Index>:$map_operands,
    AffineMapAttr:$offset_map
  );

  let builders = [
    // 构造器 1: 默认构造器，自动生成全零偏移映射
    OpBuilder<(ins "Value":$src, "Value":$dst), [{
      auto dstType = cast<MemRefType>(dst.getType());
      auto offsetMap = AffineMap::getConstantMap(dstType.getRank(), $_builder.getContext());
      build($_builder, $_state, src, dst, ValueRange{}, offsetMap);
    }]>
    // ,
    // // 构造器 2: 全参数构造器
    // OpBuilder<(ins "Value":$src, "Value":$dst, "ValueRange":$map_operands, "AffineMap":$offset_map), [{
    //   $_state.addOperands({src, dst});
    //   $_state.addOperands(map_operands);
    //   $_state.addAttribute("offset_map", AffineMapAttr::get(offset_map));
    // }]>
  ];

  let assemblyFormat = [{
    $src `to` $dst (`at` `(` $map_operands^ `)`)? `[` $offset_map `]` attr-dict 
    `:` type($src) `,` type($dst)
  }];

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Value getSrcMemRef() { return getOperand(0); }
    Value getDstMemRef() { return getOperand(1); }
  }];
}


// convert layout op
def Frisk_ConvertLayoutOp: Frisk_Op<"convert_layout", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "layout conversion";
  let description = [{ 
    layout convert
  }];

  let arguments = (ins 
    Arg<AnyMemRef, "", [MemWrite, MemRead]>:$memref,
    AnyAttr:$value
  );

  let extraClassDeclaration = [{
    MemRefType getMemType() { return cast<MemRefType>(getMemref().getType()); }
  }];

  // let hasVerifier = 1;
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

// 
def Frisk_FillOp: Frisk_Op<"fill", [MemoryEffects<[MemWrite]>]> {
  let summary = "fill";
  let description = [{ 
    fill
  }];

  let arguments = (ins 
    Arg<AnyMemRef, "", [MemWrite]>:$memref,
    AnyAttr:$value
  );

  let extraClassDeclaration = [{
    MemRefType getMemType() { return cast<MemRefType>(getMemref().getType()); }
  }];

  let hasVerifier = 1;
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$memref attr-dict `:` type($memref)";
}

// reduceOp
def Frisk_ReduceOp: Frisk_Op<"reduce", [MemoryEffects<[MemWrite]>, MemoryEffects<[MemRead]>]> {
  let summary = "reduce";
  let description = [{ 
    reduce
  }];

  let arguments = (ins 
    Arg<AnyMemRef, "", [MemRead]>:$src,  // read
    Arg<AnyMemRef, "", [MemWrite]>:$dst,  // write
    StrAttr:$kind,
    I64Attr:$dim
  );

  let extraClassDeclaration = [{
    MemRefType getSrcType() { return cast<MemRefType>(getSrc().getType()); }
    MemRefType getDstType() { return cast<MemRefType>(getDst().getType()); }
    // Value getSrcValue() { return getSrc(); }
    // Value getDstValue() { return getDst(); }

    ::mlir::LogicalResult inferLayout(::mlir::OpBuilder &builder,
                                      llvm::DenseMap<::mlir::Value, ::mlir::Attribute> &layoutMap);
  }];

  let hasVerifier = 1;
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

// init_barrier
def Frisk_InitBarrierOp: Frisk_Op<"init_barrier", [Pure]> {
  let summary = "Initialize barriers";
  let description = [{
    Create a list of barrier handles.
    The `num` attribute specifies how many barriers to create.
    The op returns `num` results, each of which is a `frisk.barrier` handle.
  }];

  let arguments = (ins
    I64Attr:$num
  );

  let results = (outs
    Variadic<Frisk_BarrierType>:$barriers
  );

  let builders = [
    OpBuilder<(ins "int64_t":$num)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "attr-dict `:` type($barriers)";
}

// barrier_arrive
def Frisk_BarrierArrive: Frisk_Op<"barrier_arrive", []> {
  let summary = "Arrive at a barrier";
  let description = [{
    Signal arrival at a synchronization barrier.
  }];

  let arguments = (ins
    Frisk_BarrierType:$barrier
  );

  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";
}

// barrier_wait
def Frisk_BarrierWait: Frisk_Op<"barrier_wait", []> {
  let summary = "Wait on a barrier";
  let description = [{
    Wait on a synchronization barrier with an index-typed phase value.
  }];

  let arguments = (ins
    Frisk_BarrierType:$barrier,
    Index:$phase
  );

  let assemblyFormat = "$barrier `,` $phase attr-dict `:` type($barrier) `,` type($phase)";
}

// barrier_arrive_expect_tx
def Frisk_BarrierArriveExpectTx: Frisk_Op<"barrier_arrive_expect_tx", []> {
  let summary = "Arrive at a barrier with expected transfer size";
  let description = [{
    Signal arrival at a synchronization barrier and provide expected transfer
    size in bytes.
  }];

  let arguments = (ins
    Frisk_BarrierType:$barrier,
    I64Attr:$expected_tx_bytes
  );

  let assemblyFormat = "$barrier attr-dict `:` type($barrier)";
}

// async copy
def Frisk_AsyncCopy : Frisk_Op<"async_copy", []> {
  let summary = "Async copy buffer using TMA with affine coordinates";
  let description = [{
    使用 TMA (Tensor Memory Accelerator) 异步拷贝数据。
    该操作通过 AffineMap 及其操作数（MapOperands）计算出在全局内存中的坐标，
    并将数据异步搬运到共享内存（dst_smem）中。传输完成后触发指定的 Barrier。

    示例:
    ```mlir
    // 对应伪代码中的 coords=[0, %by*128]
    #map = affine_map<(d0) -> (0, d0 * 128)>
    
    frisk.async_copy %smem, %tma_desc [%by]
      map: #map, barrier: %barrier
      : memref<1x128xf32, 3>, type(%tma_desc), type(%barrier)
    ```
  }];

  let arguments = (ins
    Arg<AnyMemRef, "Destination shared memory buffer", [MemWrite]>:$dst_smem,
    Frisk_TmaDescType:$tma_desc,
    Variadic<Index>:$map_operands,     // 计算坐标所需的输入变量（如伪代码中的 %by）
    AffineMapAttr:$coordinate_map,    // 用于映射出多维坐标的 AffineMap 属性
    Frisk_BarrierType:$barrier
  );

  // 语法格式化，使其接近标准的 Affine 风格
  let assemblyFormat = [{
    $dst_smem `,` $tma_desc `[` $map_operands `]` 
    `map` `:` $coordinate_map `,` `barrier` `:` $barrier
    attr-dict `:` type($dst_smem) `,` type($tma_desc) `,` type($barrier)
  }];

  // 可以在此处添加验证器（Verifier）来确保 map 的输出维度符合 TMA 的要求
  let hasVerifier = 1;
}

// get global tma descriptor
def Frisk_GetGlobalDescOp : Frisk_Op<"get_global_desc", [Pure]> {
  let summary = "Create a TMA descriptor from a global memref";
  let description = [{
    该操作通过绑定一个全局内存的 `memref` 来创建并初始化一个 TMA 描述符。
    输入的 `memref` 代表全局内存的完整张量结构，`box_size` 指定了未来通过该描述符
    进行 TMA 异步搬运时的基本块（Tile）大小。

    示例:
    ```mlir
    // 将一个 2048x2048 的全局内存绑定为 TMA 描述符，指定每次搬运 128x128 的块
    %tma_desc = frisk.get_global_desc %global_mem
      box_size = [128, 128]
      : (memref<2048x2048xf16>) -> !frisk.tma_desc<memref<2048x2048xf16>>
    ```
  }];

  let arguments = (ins
    Arg<AnyMemRef, "Global memory buffer to be mapped">:$global_mem,
    DenseI64ArrayAttr:$box_size // 硬件 TMA 需要知道高维切片每个维度的基本大小 (Tile Size)
  );

  let results = (outs
    Frisk_TmaDescType:$tma_desc
  );

  // 语法格式化
  let assemblyFormat = [{
    $global_mem `box_size` `=` $box_size attr-dict `:` `(` type($global_mem) `)` `->` type($tma_desc)
  }];

  // 验证器：确保输出的 TmaDescType 内部携带的类型与输入的 global_mem 类型完全一致
  let hasVerifier = 1;
}

// wgMMA (Warpgroup Matrix Multiply Accumulate)
def Frisk_WgMmaAsyncSSOp : Frisk_Op<"wgmma_ss_async", [AttrSizedOperandSegments]> {
  let summary = "Warpgroup-level asynchronous matrix multiply-accumulate";
  let description = [{
    执行 Warpgroup 级别的异步矩阵乘法累加操作 (D = A * B + C)。
    该操作由整个 Warpgroup (128 threads) 协同完成。
    AB 必须位于 shared , Acc 必须位于 local

    smAMap / smBMap 是 AffineMap，描述如何从 smAMapOperands / smBMapOperands
    （动态 SSA 值，如 tile 索引、warpgroup id 等）计算出 smA / smB 在
    shared memory 中的线性 byte offset，供 wgmma.fence / wgmma.async 使用。

    典型用法：
      smAMap = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
      smAMapOperands = [%row, %col, %stride]
  }];

  let arguments = (ins
    // ── Matrix buffers ──────────────────────────────────────────────
    Arg<AnyMemRef, "Matrix A in shared memory", [MemRead]>:$smA,
    Arg<AnyMemRef, "Matrix B in shared memory", [MemRead]>:$smB,
    Arg<AnyMemRef, "Accumulator C in local (register) memory", [MemRead, MemWrite]>:$localAcc,

    // ── Offset affine maps for smA ───────────────────────────────────
    // AffineMapAttr 描述 smA 的 offset 计算公式
    AffineMapAttr:$smAMap,
    // 对应 smAMap 中 dim / symbol 的动态 SSA 操作数
    Variadic<Index>:$smAMapOperands,

    // ── Offset affine maps for smB ───────────────────────────────────
    AffineMapAttr:$smBMap,
    Variadic<Index>:$smBMapOperands,

    // ── Shape descriptor ─────────────────────────────────────────────
    DenseI64ArrayAttr:$mnk   // [M, N, K]
  );

  let results = (outs
  );

  let assemblyFormat = [{
    $smA `[` $smAMap
      ( `(` $smAMapOperands^ `)` )?
    `]`
    $smB `[` $smBMap
      ( `(` $smBMapOperands^ `)` )?
    `]`
    $localAcc
    `mnk` `=` $mnk
    attr-dict
    `:` type($smA) `,` type($smB) `,` type($localAcc)
  }];
  // let hasVerifier = 1;
}

// wgMMA (Warpgroup Matrix Multiply Accumulate)
def Frisk_WgMmaAsyncLSOp : Frisk_Op<"wgmma_ls_async", []> {
  let summary = "Warpgroup-level asynchronous matrix multiply-accumulate";
  let description = [{
    执行 Warpgroup 级别的异步矩阵乘法累加操作 (D = A * B + C)。
    该操作由整个 Warpgroup (128 threads) 协同完成。
    B 必须位于 shared , A & Acc 必须位于 local

    smBMap 是 AffineMap，描述如何从 smBMapOperands
    （动态 SSA 值，如 tile 索引、warpgroup id 等）计算出  smB 在
    shared memory 中的线性 byte offset，供 wgmma.fence / wgmma.async 使用。

    典型用法：
      smBMap = affine_map<(d0, d1)[s0] -> (d0 * s0 + d1)>
      smBMapOperands = [%row, %col, %stride]
  }];

  let arguments = (ins
    // ── Matrix buffers ──────────────────────────────────────────────
    Arg<AnyMemRef, "matrix A data in lcoal memory", [MemRead]>:$localA,
    Arg<AnyMemRef, "Matrix B in shared memory", [MemRead]>:$smB,
    Arg<AnyMemRef, "Accumulator C in local (register) memory", [MemRead, MemWrite]>:$localAcc,

    // ── Offset affine maps for smB ───────────────────────────────────
    AffineMapAttr:$smBMap,
    Variadic<Index>:$smBMapOperands,

    // ── Shape descriptor ─────────────────────────────────────────────
    DenseI64ArrayAttr:$mnk   // [M, N, K]
  );

  let results = (outs);

  let assemblyFormat = [{
    $localA 
    $smB `[` $smBMap
      ( `(` $smBMapOperands^ `)` )?
    `]`
    $localAcc
    `mnk` `=` $mnk
    attr-dict
    `:` type($localA) `,` type($smB) `,` type($localAcc)
  }];
  // let hasVerifier = 1;
}

// 在instSelect阶段，我们以确保参数正确性。因此这里不需要特意强调acc和operator的type以及尺寸。一切由lowerinfo记录
def Frisk_WarpMmaOp : Frisk_Op<"warp_mma", [
    // 声明该 Op 具有内存副作用 (MemRead 和 MemWrite)
    MemoryEffects<[MemRead, MemWrite]>
]> {
  let summary = "MMAC 16x16x16 FP16 到 FP32 的矩阵乘加指令 (累加器原地读写)";
  let description = [{
    进行 C = A * B + C 的矩阵乘加计算，计算结果直接写回累加器 C 中。
    注：参数类型约束为 MemRef，但具体元素类型(ElementType)与尺寸(Shape)校验由 instSelect 阶段及 lowerinfo 处理。

    A: 只读 MemRef
    B: 只读 MemRef
    C: 读写 MemRef
  }];

  // 定义输入参数：A, B, C
  // 使用 AnyMemRef 强调内存引用语义，但不限制其元素类型与 Shape
  let arguments = (ins 
    Arg<AnyMemRef, "只读输入 A", [MemRead]>:$a,
    Arg<AnyMemRef, "只读输入 B", [MemRead]>:$b,
    Arg<AnyMemRef, "读写累加器 C", [MemRead, MemWrite]>:$c
  );

  // 原地修改累加器，无 SSA 返回值
  let results = (outs);

  // 指定自定义汇编格式
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) `,` type($c)";
}

def Frisk_SyncThreadsInBlockOp : Frisk_Op<"sync_threads_in_block", [
    MemoryEffects<[MemRead, MemWrite]>
]> {
  let summary = "Synchronize all threads in a block";
  let description = [{
    Synchronize all threads within the current block.  This operation can
    optionally carry a shared-memory buffer operand that records the memref
    whose accesses must be waited on.
  }];

  let arguments = (ins Optional<AnyMemRef>:$buffer);
  let results = (outs);

  let builders = [
    OpBuilder<(ins), [{
      build($_builder, $_state, ::mlir::Value{});
    }]>
  ];

  let assemblyFormat = "attr-dict ($buffer^ `:` type($buffer))?";
  let hasVerifier = 1;
}



#endif // FRISK_OPS
