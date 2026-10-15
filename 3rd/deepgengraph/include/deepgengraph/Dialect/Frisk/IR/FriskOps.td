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

// Fresh logical values; see FriskTraits.h for the storage/aliasing contract.
def Frisk_ComputedTile : NativeOpTrait<"ComputedTile"> {
  let cppNamespace = "::mlir::frisk";
}

// KernelOp
def Frisk_KernelOp : Frisk_Op<"kernel", [
  ImplicitFriskTerminator, CallableOpInterface, SymbolName, SymbolVisibility, FunctionOpInterface,
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

  let skipDefaultBuilders = 1;
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
def Frisk_GemmOp: Frisk_Op<"gemm", [Frisk_ComputedTile, MemoryEffects<[MemWrite]>, MemoryEffects<[MemRead]>]> {
  let summary = "General matrix multiplication operation";
  let description = [{}];
  let arguments = (ins
    Frisk_ScalarOrMemRefOrVector:$A,  // shm
    Frisk_ScalarOrMemRefOrVector:$B,

    BoolAttr:$transA,
    BoolAttr:$transB
  );

  let results = (outs
    Frisk_ScalarOrMemRefOrVector:$C
  );

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
  let assemblyFormat = " `(` $A `,` $B `)` attr-dict `:` type($A) `,` type($B) `->` type($C)";
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
def Frisk_BufferViewOp : Frisk_Op<"buffer_view", [Pure]> {
  let summary = "describe a logical buffer slice with an affine origin";
  let description = [{
    Describe a logical slice without reading memory or constructing a physical
    memref descriptor. The affine map computes the origin in the source buffer;
    slice coordinates are added to its trailing dimensions. Leading dimensions
    select a single element. Nested views compose these index mappings.
    The result keeps the source element type and memory space, with ranges as
    its logical shape. Lowering folds the mapping into accesses to the original
    buffer, whose layout determines the physical strides.
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

    For a memref destination, an optional result aliases the complete destination
    buffer, including when only a slice is copied. It is not a snapshot of the
    source registers. Physical shared-memory stores may use the source tile's
    distribution; subsequent readers choose their own distribution.

    Example:
      frisk.copy %src to %dst at (%i, %j) [affine_map<(d0, d1) -> (0, 0, d0, d1)>]
        : memref<64x32xf32>, memref<1x32x1024x128xf32>
  }];

  let arguments = (ins
    Arg<Frisk_MemRefOrVector>:$src,
    Arg<Frisk_MemRefOrVector>:$dst,
    Variadic<Index>:$map_operands,
    AffineMapAttr:$offset_map
  );

  let results = (outs
    Variadic<Frisk_MemRefOrVector>:$result
  );

  let builders = [
    // 构造器 1: 默认构造器，自动生成全零偏移映射
    OpBuilder<(ins "Value":$src, "Value":$dst), [{
      auto dstType = cast<ShapedType>(dst.getType());
      auto offsetMap = AffineMap::getConstantMap(dstType.getRank(), $_builder.getContext());
      $_state.addOperands({src, dst});
      $_state.addAttribute("offset_map", AffineMapAttr::get(offsetMap));
    }]>,
    // 构造器 2: 旧代码路径默认不产生 SSA result
    OpBuilder<(ins "Value":$src, "Value":$dst, "ValueRange":$map_operands, "AffineMap":$offset_map), [{
      $_state.addOperands({src, dst});
      $_state.addOperands(map_operands);
      $_state.addAttribute("offset_map", AffineMapAttr::get(offset_map));
    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    Value getSrcMemRef() { return getOperand(0); }
    Value getDstMemRef() { return getOperand(1); }
    bool hasValueResult() { return getNumResults() != 0; }
    Value getValueResult() { return getOperation()->getResult(0); }
  }];
}

// copy
def Frisk_AsyncCopyGmToShmOp : Frisk_Op<"async_copy_gm_to_shm", [
    MemoryEffects<[MemRead, MemWrite]>
]> {
  let summary = "Issue a block-cooperative global-to-shared tile copy";
  let description = [{
    Copies equal-shaped, equal-element-type memref tiles from global (1) to
    shared (3) memory. Views express source and destination slices. All threads
    of the block execute this op with the same descriptors. Issuing the copy
    does NOT make the destination ready. A wait_buffer_ready on the destination
    must precede consumers, reuse, or release of either buffer. The source must
    remain unchanged until completion. There is no implicit type conversion.

    This is a semantic operation: lowering may use asynchronous DMA or a
    synchronous copy when disabled by the user or unsupported by the layout.
    A 4-byte-aligned source and destination permit the gfx936 dword DMA path;
    the alignment attribute is a caller guarantee on the tile origins.
    Full-width plans carry frisk.dma_bytes selected from HWSpecification and
    optionally frisk.copy_threads (whole waves). All block threads still
    execute completion and publication. Unsupported explicit widths fail
    instead of silently selecting a synchronous or narrower copy.

    After physical packing, the destination may be a flat tile of equal element
    count. In that form, frisk.source_map is an affine map from one physical
    destination element index to logical source coordinates. The verifier
    requires a bounded bijection; synchronous fallback uses the same map.
    Target lowering may stage a copy when source and destination packet orders
    differ, preserving the named destination's physical layout and readiness.
  }];
  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst,
      DefaultValuedAttr<I64Attr, "4">:$alignment);
  let assemblyFormat = "$src `to` $dst attr-dict `:` type($src) `,` type($dst)";
  let hasVerifier = 1;
}

def Frisk_WaitBufferReadyOp : Frisk_Op<"wait_buffer_ready", [
    MemoryEffects<[MemRead, MemWrite]>
]> {
  let summary = "Make a shared tile ready for block-wide consumption";
  let description = [{
    All threads in the block participate with the same shared buffer descriptor.
    On return, prior copies into this buffer are complete and visible to the
    block. Backends may wait for additional buffers. Async gfx936 lowering uses
    wait.asyncmark 0 followed by a block barrier. Synchronous lowering only
    needs a block barrier, which subsequent dependency analysis may remove.
    This does not prove that consumers have finished: buffer reuse after reads
    may still need sync_threads_in_block. Keep this op with its consumer tile
    when software pipelining, including the rotated buffer slot.
  }];
  let arguments = (ins AnyMemRef:$buffer);
  let assemblyFormat = "$buffer attr-dict `:` type($buffer)";
  let hasVerifier = 1;
}

def Frisk_ReadMatrixFromShmOp : Frisk_Op<"read_matrix_from_shm", [
    MemoryEffects<[MemRead]>
]> {
  let summary = "Collective HCU matrix read from shared memory into registers";
  let description = [{
    Wraps the fixed-shape HCUDsReadMatrixIntrinsic instructions listed in
    IntrinsicsHCU.log. (m, n, element_bit_width, transpose, alt) selects an
    exact instruction; alt=1 denotes the _alt suffix, not _alt1. The dimensions
    and element bit width describe the instruction, not the per-lane result.

    src[indices] supplies the calling lane's LDS pointer using the memref's
    actual offset and strides. offset is the intrinsic's i32 immediate byte
    offset. The source must be byte-addressable shared storage (space 3);
    packed b4 data may use an i8 backing memref. The returned fixed 1-D vector
    is explicitly typed, matching the intrinsic's overloaded anyvector ABI.
    No implicit cast, scalar-load expansion or lane permutation is performed.
    The source storage type need not match the instruction or result type.

    This is a convergent collective read, not speculatable. Callers must supply
    the instruction's required active lanes, address layout and ready LDS data.
    It neither waits for pending DMA nor inserts a block barrier. The intrinsic
    list alone does not specify lane ownership or a bank-conflict-free layout.

    Supported variants:
      m32x64_b4; m32x32_b8; m32x16_b16; m32x16_b16_alt;
      m32x8_b32; m32x32_b8_alt2; m64x16_b8_alt4; m16x16_b32;
      mt16x16_b32; mt16x32_b16_alt2.
    The separate matrix_format/trans_format/padbyte intrinsics have different
    control operands and are not represented by this fixed-shape operation.

    Example:
      %r = frisk.read_matrix_from_shm %sm[%i] {
        m = 32 : i64, n = 16 : i64, element_bit_width = 16 : i64
      } : memref<4096xi8, 3> -> vector<4xi32>
  }];
  let arguments = (ins AnyMemRef:$src, Variadic<Index>:$indices,
      I64Attr:$m, I64Attr:$n, I64Attr:$element_bit_width,
      DefaultValuedAttr<BoolAttr, "false">:$transpose,
      DefaultValuedAttr<I64Attr, "0">:$alt,
      DefaultValuedAttr<I32Attr, "0">:$offset);
  let results = (outs AnyVectorOfNonZeroRank:$result);
  let assemblyFormat = [{
    $src `[` $indices `]` attr-dict `:` type($src) `->` type($result)
  }];
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    // Base LLVM intrinsic name, without overloaded result/pointer suffixes.
    // Empty for an unsupported combination (also rejected by verify()).
    ::llvm::StringRef getIntrinsicName();
  }];
}

def Frisk_CopyToRegOp : Frisk_Op<"copy_to_reg", [
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
    Variadic<Index>:$map_operands,
    AffineMapAttr:$offset_map
  );

  let results = (outs
    AnyVectorOfNonZeroRank:$result
  );

  let builders = [
    // 构造器: 传入返回类型 (vector) 与输入 memref
    OpBuilder<(ins "Type":$resultType, "Value":$src), [{
      auto srcType = cast<MemRefType>(src.getType());
      auto offsetMap = AffineMap::getConstantMap(srcType.getRank(), $_builder.getContext());
      build($_builder, $_state, resultType, src, ValueRange{}, offsetMap);
    }]>
  ];


  let assemblyFormat = [{
    $src (`at` `(` $map_operands^ `)`)? `[` $offset_map `]` attr-dict
    `:` type($src) `->` type($result)
  }];

  // let hasVerifier = 1;

  let extraClassDeclaration = [{
    Value getSrcMemRef() { return getOperand(0); }
  }];
}



// 2. 定义通用的二元算子基类模板
class Frisk_ArithmeticOp<string mnemonic, list<Trait> traits = []> :
    Frisk_Op<mnemonic, !listconcat(traits, [Frisk_ComputedTile, MemoryEffects<[MemRead]>])> {

  let summary = "Element-wise " # mnemonic # " operation on memrefs or scalars";
  let description = "Performs element-wise " # mnemonic # " operation on memrefs or scalar values (f16/f32).";

  let arguments = (ins
    Frisk_ScalarOrMemRefOrVector:$lhs,
    Frisk_ScalarOrMemRefOrVector:$rhs
  );

  let results = (outs
    Frisk_MemRefOrVector:$result
  );

  let extraClassDeclaration = [{
    ::mlir::Type getLhsType() { return getLhs().getType(); }
    ::mlir::Type getRhsType() { return getRhs().getType(); }
    ::mlir::Type getResultType() { return getResult().getType(); }

    bool isLhsScalar() { return ::llvm::isa<::mlir::FloatType>(getLhsType()); }
    bool isRhsScalar() { return ::llvm::isa<::mlir::FloatType>(getRhsType()); }
  }];

  // let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

class Frisk_UnaryArithmeticOp<string mnemonic, list<Trait> traits = []> :
    Frisk_Op<mnemonic, !listconcat(traits, [Frisk_ComputedTile, MemoryEffects<[MemRead]>])> {

  let summary = "Element-wise " # mnemonic # " operation on memrefs or scalars";
  let description = "Performs element-wise " # mnemonic # " operation on memrefs or scalar values (f16/f32).";

  let arguments = (ins
    Frisk_ScalarOrMemRefOrVector:$operand
  );

  let results = (outs
    Frisk_MemRefOrVector:$result
  );

  let extraClassDeclaration = [{
    ::mlir::Type getOperandType() { return getOperand().getType(); }
    ::mlir::Type getResultType() { return getResult().getType(); }

    bool isOperandScalar() { return ::llvm::isa<::mlir::FloatType>(getOperandType()); }
  }];

  // let hasVerifier = 1;
  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
}

// 3. 实例化具体的算子
def Frisk_AddOp : Frisk_ArithmeticOp<"add", [Commutative]>; // 加法支持交换律 Trait
def Frisk_SubOp : Frisk_ArithmeticOp<"sub">;
def Frisk_MulOp : Frisk_ArithmeticOp<"mul", [Commutative]>; // 乘法支持交换律 Trait
def Frisk_DivOp : Frisk_ArithmeticOp<"div">;
def Frisk_Exp2Op : Frisk_UnaryArithmeticOp<"exp2">;

// Element-wise data type conversion.
def Frisk_CastOp : Frisk_Op<"cast", [
    Frisk_ComputedTile, MemoryEffects<[MemRead]>, SameOperandsAndResultShape
]> {
  let summary = "Convert the data type of a scalar, memref or vector";
  let description = [{
    Converts each input element to the element type of the result, preserving
    the shape. Scalar inputs produce scalar results. This is a numeric
    conversion, not a reinterpretation of the input bits.
  }];

  let arguments = (ins
    AnyTypeOf<[AnyMemRef, AnyVectorOfNonZeroRank, AnyFloat, AnyInteger, Index]>:$operand
  );
  let results = (outs
    AnyTypeOf<[AnyMemRef, AnyVectorOfNonZeroRank, AnyFloat, AnyInteger, Index]>:$result
  );

  let assemblyFormat = "$operand attr-dict `:` type($operand) `->` type($result)";
}


// convert layout op
def Frisk_ConvertLayoutOp: Frisk_Op<"convert_layout", [MemoryEffects<[MemWrite, MemRead]>]> {
  let summary = "layout conversion";
  let description = [{
    layout convert
  }];

  let arguments = (ins
    Arg<Frisk_MemRefOrVector, "", [MemWrite, MemRead]>:$memref,
    AnyAttr:$value
  );

  let results = (outs
    Variadic<Frisk_MemRefOrVector>:$result
  );

  let extraClassDeclaration = [{
    ShapedType getMemType() { return cast<ShapedType>(getMemref().getType()); }
  }];

  let assemblyFormat = [{
    $memref  attr-dict `:` type($memref) `->` type($result)
  }];
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

  let results = (outs
    Variadic<Frisk_MemRefOrVector>:$result
  );

  let extraClassDeclaration = [{
    MemRefType getMemType() { return cast<MemRefType>(getMemref().getType()); }
    bool hasValueResult() { return getNumResults() != 0; }
    Value getValueResult() { return getOperation()->getResult(0); }
  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
}

//
def Frisk_ZeroOp: Frisk_Op<"zero", [Frisk_ComputedTile, Pure]> {
  let summary = "zero";
  let description = [{
    zero
  }];

  let arguments = (ins
  );
  let results = (outs
    AnyMemRef:$result
  );
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = " attr-dict `:`  type($result)";
}


def MaskYieldOp: Frisk_Op<"mask_yield", [Pure, ReturnLike, Terminator, HasParent<"frisk::MaskOp">]> {
  let summary = "";
  let description = [{}];
  let arguments = (ins Variadic<AnyType> :$srcs);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, ValueRange{});
  }]>];

  let assemblyFormat = "attr-dict $srcs `:` type($srcs)";
}

def MaskOp: Frisk_Op<"mask", [Frisk_ComputedTile, Pure, SingleBlockImplicitTerminator<"frisk::MaskYieldOp">, InferTypeOpAdaptor]> {
  let summary = "";
  let description = [{}];

  let arguments = (ins
    Variadic<Index>:$starts,
    DenseI64ArrayAttr:$sizes,
    TypeAttr:$element_type
  );

  let results = (outs AnyMemRef:$result);

  let regions = (region SizedRegion<1>:$region);

  // let builders = [
  // ];

  let extraClassDeclaration = [{
    Value getStart(int i) {
      return getOperation()->getOperand(i);
    }
    Value getIterArgInEntry(int i) {
      auto entry = &getRegion().front();
      return entry->getArgument(i);
    }
  }];

  let assemblyFormat = "`starts` `=` `[` $starts `]` `,` `sizes` `=` $sizes `,` `type` `=` $element_type $region attr-dict `:` functional-type(operands, results)";

  // let hasVerifier = 1;
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
def Frisk_WarpMmaRROp : Frisk_Op<"warp_mma_rr", [
    // Register MMA has no memory effects; do not mark it speculatable (collective).
    MemoryEffects<[]>
]> {
  let summary = "Register-fragment matrix multiply-accumulate";
  let description = [{
    Computes a new SSA register fragment C = A * B + C. A, B, and C are
    vectors; the instruction shape and layout are selected by lowerinfo.
    This collective instruction has no memory effects and is not speculatable.

    The enclosing module may set `frisk.use_hcu_mmac = true` to lower the
    supported, unmodified DCU MMAC templates to the DTK HCU intrinsic ABI.
    Other instructions retain inline assembly. The default remains the
    hardware-tested inline assembly path until the intrinsic path is validated
    with the deployment DTK and GPU. Setting the attribute to false explicitly
    selects that fallback, including its MMAC hazard padding.
  }];

  let arguments = (ins
    AnyVectorOfNonZeroRank:$a,
    AnyVectorOfNonZeroRank:$b,
    AnyVectorOfNonZeroRank:$c
  );

  let results = (outs
    AnyVectorOfNonZeroRank:$result
  );

  // 指定自定义汇编格式
  let assemblyFormat = "$a `,` $b `,` $c attr-dict `:` type($a) `,` type($b) `,` type($c) `->` type($result)";
}

def Frisk_SyncThreadsInBlockOp : Frisk_Op<"sync_threads_in_block", [
    MemoryEffects<[MemRead, MemWrite]>
]> {
  let summary = "Synchronize all threads in a block";
  let description = [{
    Synchronize all threads within the current block.
  }];

  let arguments = (ins);
  let results = (outs);

  let assemblyFormat = "attr-dict";
}

def Frisk_ToThreadTileOp: Frisk_Op<"to_threadTile", [Pure]> {
  let summary = "to_threadTile";
  let description = [{
    临时转换op。表达从 block-tile -> thread-tile 后期会和 Frisk_FromThreadTileOp 配对消除
  }];

  let arguments = (ins
    Frisk_MemRefOrVector : $blockTile
  );
  let results = (outs
    Frisk_MemRefOrVector:$threadTile
  );
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    $blockTile attr-dict `:` type($blockTile) `->` type($threadTile)
  }];
}

def Frisk_FromThreadTileOp: Frisk_Op<"from_threadTile", [Pure]> {
  let summary = "from_threadTile";
  let description = [{
    临时转换op。表达从 thread-tile -> block-tile. 后期会和 Frisk_ToThreadTileOp 配对消除
  }];

  let arguments = (ins
    Frisk_MemRefOrVector : $threadTile
  );
  let results = (outs
    Frisk_MemRefOrVector : $blockTile
  );
  // let hasCustomAssemblyFormat = 1;
  let assemblyFormat = [{
    $threadTile attr-dict `:` type($threadTile) `->` type($blockTile)
  }];
}


// A scheduling unit executes exactly once. Keep its scalar implementation
// encapsulated until fragment fusion/pipelining has finished. It deliberately
// has no inlining canonicalizer and is not speculatable across control flow.
def Frisk_FragmentOp : Frisk_Op<"fragment", [SingleBlock, NoRegionArguments,
    RecursiveMemoryEffects]> {
  let summary = "One fragment load, computation, gather or store";
  let arguments = (ins StrAttr:$kind);
  let results = (outs Variadic<AnyType>:$values);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "$kind attr-dict-with-keyword (`->` `(` type($values)^ `)`)? $body";
  let hasVerifier = 1;
}

def Frisk_FragmentYieldOp : Frisk_Op<"fragment_yield", [Pure, Terminator,
    HasParent<"FragmentOp">]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

#endif // FRISK_OPS
