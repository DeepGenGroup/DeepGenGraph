#include <map>
#include <functional>
#include <string>

#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskEnums.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
// #include "deepgengraph/Analysis/LowerInfo.h"

namespace mlir::frisk {
#define GEN_PASS_DEF_CONVERTFRISKTOBASE
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

using friskMs = frisk::attr::MemorySpace;

// struct KernelOpConversion : public OpConversionPattern<KernelOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(KernelOp kernelOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
//     FunctionType funcType = mlir::dyn_cast<FunctionType>(kernelOp.getFunctionType());
//     ArrayRef<Type> inputTypes = funcType.getInputs();
//     // new func
//     func::FuncOp funcOp = rewriter.create<func::FuncOp>(kernelOp.getLoc(), kernelOp.getSymName(), funcType);
//     if (auto threadNumAttr = kernelOp->getAttr("thread_num")) {
//       funcOp->setAttr("thread_num", threadNumAttr);
//     }
//     auto& region = funcOp->getRegion(0);
//     region.emplaceBlock();
//     auto& body = funcOp.front();
//     SmallVector<Location> locs(inputTypes.size(), kernelOp.getLoc());
//     body.addArguments(inputTypes, locs);

//     auto& oldBlock = kernelOp->getRegion(0).front();
//     auto& newBlock = funcOp->getRegion(0).front();
//     // replace all uses with
//     for (unsigned i=0; i<oldBlock.getNumArguments(); ++i) {
//         Value oldArg = oldBlock.getArgument(i);
//         Value newArg = newBlock.getArgument(i);
//         rewriter.replaceAllUsesWith(oldArg,newArg);
//     }
//     // move operation from origin kernelOp
//     newBlock.getOperations().splice(newBlock.getOperations().begin(), oldBlock.getOperations());
//     // llvm::outs() <<  << "\n";
//     rewriter.eraseOp(&(newBlock.back()));
//     // add returnop
//     rewriter.setInsertionPointToEnd(&body);
//     rewriter.create<func::ReturnOp>(funcOp.getLoc());
//     // remove origin kernelOp
//     rewriter.eraseOp(kernelOp);
//     return success();
//   }
// };

struct KernelOpConversion : public OpConversionPattern<KernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(KernelOp kernelOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // 1. 获取函数签名类型
    FunctionType funcType = mlir::dyn_cast<FunctionType>(kernelOp.getFunctionType());
    if (!funcType) return failure();

    // 2. 使用 rewriter 创建新的 func::FuncOp
    func::FuncOp funcOp = rewriter.create<func::FuncOp>(kernelOp.getLoc(), kernelOp.getSymName(), funcType);

    // 3. 转移所需的属性
    if (auto threadNumAttr = kernelOp->getAttr("thread_num")) {
      funcOp->setAttr("thread_num", threadNumAttr);
    }

    // 4. 将原 kernelOp 的 Region 移动（内联）到新的 funcOp 中
    // 这一步由 rewriter 接管，会自动保留原有的 BlockArguments 及其所有 Use 关系，
    // 替代了原本手动 create block -> add argument -> replaceAllUsesWith -> splice operations 的繁琐步骤。
    rewriter.inlineRegionBefore(kernelOp.getRegion(), funcOp.getRegion(), funcOp.getRegion().end());

    // 5. 替换原有的 Terminator
    // 获取刚刚移动过来的 Block 及其最后一个操作（原 terminator）
    Block &body = funcOp.getRegion().front();
    Operation *terminator = body.getTerminator();
    
    // 将插入点设置在原 terminator 处，并使用 rewriter 将其替换为 func::ReturnOp
    rewriter.setInsertionPoint(terminator);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(terminator);

    // 6. 使用 rewriter 安全删除原 kernelOp
    rewriter.eraseOp(kernelOp);

    return success();
  }
};

// struct ParallelOpConversion : public OpConversionPattern<ParallelOp> {
//   using OpConversionPattern::OpConversionPattern;

//   LogicalResult matchAndRewrite(ParallelOp parallelOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
//     constexpr gpu::Dimension dims[] = {gpu::Dimension::z, gpu::Dimension::y, gpu::Dimension::x};
//     SmallVector<Value, 4> bids;
//     // create gpu blockidx
//     auto grid = parallelOp.getGrid();
//     rewriter.setInsertionPoint(parallelOp);
//     for (unsigned i=0; i<grid.size(); i++) {
//       auto bidOp = rewriter.create<gpu::BlockIdOp>(parallelOp.getLoc(), dims[i]);
//       bidOp->setAttr("range", rewriter.getI32IntegerAttr(grid[i]));
//       bids.push_back(bidOp);
//     }
//     // create gpu threadIdx
//     auto tidOp = rewriter.create<gpu::ThreadIdOp>(parallelOp.getLoc(), gpu::Dimension::x);
//     tidOp->setAttr("range", rewriter.getI32IntegerAttr(parallelOp.getThreadNum()));
//     // set kernelOp
//     Operation *op = parallelOp->getParentOp();
//     rewriter.modifyOpInPlace(op, [&](){
//       op->setAttr("thread_num", rewriter.getI32IntegerAttr(parallelOp.getThreadNum()));
//     });
//     // collect
//     auto& block = parallelOp->getRegion(0).front();
//     SmallVector<Operation*> opsToMove;
//     for (auto &op : block.getOperations()) {
//       if (!op.hasTrait<OpTrait::IsTerminator>()) {
//         opsToMove.push_back(&op);
//       }
//     }
//     // move
//     Operation *pos = parallelOp.getOperation();
//     for (Operation *op : opsToMove) {
//       // op->moveAfter(pos);
//       rewriter.moveOpAfter(op, pos);
//       pos = op;
//     }
//     // replace uses
//     for (unsigned i=0; i<block.getNumArguments(); ++i) {
//       Value oldArg = block.getArgument(i);
//       rewriter.replaceAllUsesWith(oldArg, bids[i]);
//     }
//     rewriter.eraseOp(parallelOp);
//     return success();
//   }
// };

struct ParallelOpConversion : public OpConversionPattern<ParallelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ParallelOp parallelOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = parallelOp.getLoc();
    constexpr gpu::Dimension dims[] = {gpu::Dimension::z, gpu::Dimension::y, gpu::Dimension::x};

    // 1. 生成 gpu blockIdx
    auto grid = parallelOp.getGrid();
    SmallVector<Value, 4> bids;
    for (unsigned i = 0; i < grid.size(); i++) {
      auto bidOp = rewriter.create<gpu::BlockIdOp>(loc, dims[i]);
      // 对于新创建的 Op，直接 setAttr 是安全的，因为它们还未对其他 Pass 可见
      bidOp->setAttr("range", rewriter.getIndexAttr(grid[i]));
      bids.push_back(bidOp);
    }

    // 2. 生成 gpu threadIdx
    auto tidOp = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    tidOp->setAttr("range", rewriter.getIndexAttr(parallelOp.getThreadNum()));

    // 3. 规范化修改 Parent Op（原代码这里用得很标准）
    Operation *parentOp = parallelOp->getParentOp();
    rewriter.modifyOpInPlace(parentOp, [&]() {
      parentOp->setAttr("thread_num", rewriter.getI32IntegerAttr(parallelOp.getThreadNum()));
    });

    // 4. 提取要处理的 Block
    Block &block = parallelOp.getRegion().front();

    // 准备 Block Arguments 的替代值
    SmallVector<Value, 4> replArgs;
    for (unsigned i = 0; i < block.getNumArguments(); ++i) {
      replArgs.push_back(bids[i]);
    }

    // 5. 规范化的 Inline 操作
    // 步骤 A: 首先删除原 block 内的 terminator，防止它被错误地移入 Parent Block 导致 IR 结构破坏
    rewriter.eraseOp(block.getTerminator());

    // 步骤 B: 使用 rewriter 的内联方法，这会自动移动所有剩余的 Ops 并替换原 Block arguments
    // 该方法受 Dialect Conversion 框架的完全监控并支持回滚
    rewriter.inlineBlockBefore(&block, parallelOp, replArgs);

    // 6. 删除原 Op
    rewriter.eraseOp(parallelOp);

    return success();
  }
};

struct ForOpConversion : public OpConversionPattern<ForOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(ForOp dforOp, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    uint64_t lb = dforOp.getLower();
    uint64_t ub = dforOp.getUpper();
    uint64_t step = dforOp.getStep();
    Value div = dforOp.getInductionVar();
    Value aiv;
    auto aforOp = rewriter.create<affine::AffineForOp>(dforOp.getLoc(), lb, ub, step, mlir::ValueRange({}), 
      [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
        aiv = iv;
      });
    // move
    aforOp.getBody()->getOperations().splice(aforOp.getBody()->getOperations().begin(), dforOp.getBody()->getOperations());
    rewriter.eraseOp(&(aforOp.getBody()->back()));
    rewriter.setInsertionPointToEnd(aforOp.getBody());
    rewriter.create<affine::AffineYieldOp>(aforOp.getLoc());
    // replace
    div.replaceAllUsesWith(aiv);
    rewriter.eraseOp(dforOp);
    return success();
  }
};


class AffineForOpConversion : public OpConversionPattern<affine::AffineForOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(affine::AffineForOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const override {
    // 分析affineFor。 如果内部存在 frisk.copy 到 memref local 的情况，替换为 vector，并用yield做 loop-carried 处理
    if(op->hasAttr("local_yield_transformed")){
      return failure();
    }

    auto isLocalMemRef = [](Value value) -> bool {
      auto type = dyn_cast<MemRefType>(value.getType());
      return type && type.getMemorySpaceAsInt() == int(friskMs::Local);
    };

    auto canPromoteInLoop = [&](Value value) -> bool {
      if (!isLocalMemRef(value)) {
        return false;
      }
      Operation *defOp = value.getDefiningOp();
      return !defOp || !op->isAncestor(defOp);
    };

    SmallVector<Value> localBuffers;
    auto addLocalBuffer = [&](Value value) {
      if (!canPromoteInLoop(value)) {
        return;
      }
      if (!llvm::is_contained(localBuffers, value)) {
        localBuffers.push_back(value);
      }
    };

    op->walk([&](frisk::CopyOp copy) {
      addLocalBuffer(copy.getSrcMemRef());
      addLocalBuffer(copy.getDstMemRef());
    });
    op->walk([&](frisk::FillOp fill) {
      addLocalBuffer(fill.getMemref());
    });
    op->walk([&](affine::AffineLoadOp load) {
      addLocalBuffer(load.getMemref());
    });
    op->walk([&](affine::AffineStoreOp store) {
      addLocalBuffer(store.getMemref());
    });
    op->walk([&](frisk::WarpMmaRROp wmma) {
      addLocalBuffer(wmma.getA());
      addLocalBuffer(wmma.getB());
      addLocalBuffer(wmma.getC());
    });

    if (localBuffers.empty()) {
      rewriter.modifyOpInPlace(op, [&]() {
        op->setAttr("local_yield_transformed", rewriter.getBoolAttr(true));
      });
      return success();
    }

    SmallVector<VectorType> vectorTypes;
    SmallVector<Value> vectorInits;
    vectorTypes.reserve(localBuffers.size());
    vectorInits.reserve(localBuffers.size());
    rewriter.setInsertionPoint(op);
    for (Value buffer : localBuffers) {
      auto memrefType = dyn_cast<MemRefType>(buffer.getType());
      if (!memrefType || !memrefType.hasStaticShape()) {
        return failure();
      }
      auto vectorType =
          VectorType::get(memrefType.getShape(), memrefType.getElementType());
      Attribute zeroAttr = rewriter.getZeroAttr(memrefType.getElementType());
      if (!zeroAttr) {
        return failure();
      }
      auto denseAttr = DenseElementsAttr::get(vectorType, zeroAttr);
      auto init =
          rewriter.create<arith::ConstantOp>(op.getLoc(), vectorType, denseAttr);
      vectorTypes.push_back(vectorType);
      vectorInits.push_back(init.getResult());
    }

    auto copyNonStructuralForAttrs = [&](affine::AffineForOp from,
                                         affine::AffineForOp to) {
      for (NamedAttribute attr : from->getAttrs()) {
        StringRef name = attr.getName().getValue();
        if (name == "lowerBoundMap" || name == "upperBoundMap" ||
            name == "operandSegmentSizes" || name == "step") {
          continue;
        }
        to->setAttr(attr.getName(), attr.getValue());
      }
    };

    SmallVector<Value> newInits(adaptor.getInits().begin(), adaptor.getInits().end());
    newInits.append(vectorInits.begin(), vectorInits.end());

    auto newForOp = rewriter.create<affine::AffineForOp>(
        op.getLoc(), adaptor.getLowerBoundOperands(), op.getLowerBoundMap(),
        adaptor.getUpperBoundOperands(), op.getUpperBoundMap(),
        op.getStepAsInt(), newInits);
    copyNonStructuralForAttrs(op, newForOp);
    newForOp->setAttr("local_yield_transformed", rewriter.getBoolAttr(true));

    auto getBufferIndex = [&](Value value) -> std::optional<unsigned> {
      for (auto [idx, buffer] : llvm::enumerate(localBuffers)) {
        if (buffer == value) {
          return idx;
        }
      }
      return std::nullopt;
    };

    auto makeSplatAttr = [&](Attribute attr, Type elementType) -> Attribute {
      if (auto typedAttr = dyn_cast<TypedAttr>(attr)) {
        if (typedAttr.getType() == elementType) {
          return attr;
        }
      }
      if (auto floatAttr = dyn_cast<FloatAttr>(attr)) {
        if (auto floatType = dyn_cast<FloatType>(elementType)) {
          return rewriter.getFloatAttr(floatType, floatAttr.getValue());
        }
      }
      if (auto integerAttr = dyn_cast<IntegerAttr>(attr)) {
        if (auto integerType = dyn_cast<IntegerType>(elementType)) {
          return rewriter.getIntegerAttr(integerType, integerAttr.getValue());
        }
        if (isa<IndexType>(elementType)) {
          return rewriter.getIndexAttr(integerAttr.getInt());
        }
      }
      return Attribute();
    };

    std::function<LogicalResult(Block *, Block *, IRMapping &, SmallVector<Value> &)>
        cloneBody = [&](Block *oldBlock, Block *newBlock, IRMapping &mapper,
                        SmallVector<Value> &currentVectors) -> LogicalResult {
      auto remapValue = [&](Value value) -> Value {
        if (auto idx = getBufferIndex(value)) {
          return currentVectors[*idx];
        }
        return mapper.lookupOrDefault(value);
      };

      auto setCurrentVector = [&](Value buffer, Value value) {
        if (auto idx = getBufferIndex(buffer)) {
          currentVectors[*idx] = value;
          mapper.map(buffer, value);
        }
      };

      auto remapValueRange = [&](ValueRange values) {
        SmallVector<Value> remapped;
        remapped.reserve(values.size());
        for (Value value : values) {
          remapped.push_back(remapValue(value));
        }
        return remapped;
      };

      auto buildAffineAccessIndices = [&](Location loc, AffineMap map,
                                          ValueRange operands) {
        SmallVector<Value> mappedOperands = remapValueRange(operands);
        SmallVector<Value> indices;
        indices.reserve(map.getNumResults());
        for (AffineExpr expr : map.getResults()) {
          auto resultMap = AffineMap::get(map.getNumDims(), map.getNumSymbols(),
                                          expr, rewriter.getContext());
          indices.push_back(
              rewriter.create<affine::AffineApplyOp>(loc, resultMap,
                                                     mappedOperands));
        }
        return indices;
      };

      // rewriter.eraseOp(newBlock->getTerminator());
      rewriter.setInsertionPointToEnd(newBlock);
      for (Operation &childOp : oldBlock->without_terminator()) {
        if (auto nestedFor = dyn_cast<affine::AffineForOp>(childOp)) {
          SmallVector<Value> nestedInits =
              remapValueRange(nestedFor.getInits());
          nestedInits.append(currentVectors.begin(), currentVectors.end());
          SmallVector<Value> lowerBoundOperands =
              remapValueRange(nestedFor.getLowerBoundOperands());
          SmallVector<Value> upperBoundOperands =
              remapValueRange(nestedFor.getUpperBoundOperands());
          auto newNestedFor = rewriter.create<affine::AffineForOp>(
              nestedFor.getLoc(), lowerBoundOperands,
              nestedFor.getLowerBoundMap(), upperBoundOperands,
              nestedFor.getUpperBoundMap(), nestedFor.getStepAsInt(),
              nestedInits);
          copyNonStructuralForAttrs(nestedFor, newNestedFor);
          newNestedFor->setAttr("local_yield_transformed",
                                rewriter.getBoolAttr(true));

          IRMapping nestedMapper;
          nestedMapper.map(nestedFor.getBody()->getArgument(0),
                           newNestedFor.getBody()->getArgument(0));
          for (auto [oldArg, newArg] :
               llvm::zip(nestedFor.getRegionIterArgs(),
                         newNestedFor.getRegionIterArgs().take_front(
                             nestedFor.getNumIterOperands()))) {
            nestedMapper.map(oldArg, newArg);
          }

          SmallVector<Value> nestedCurrentVectors;
          nestedCurrentVectors.reserve(localBuffers.size());
          unsigned vectorArgStart = 1 + nestedFor.getNumIterOperands();
          for (unsigned i = 0; i < localBuffers.size(); ++i) {
            Value arg = newNestedFor.getBody()->getArgument(vectorArgStart + i);
            nestedCurrentVectors.push_back(arg);
            nestedMapper.map(localBuffers[i], arg);
          }

          if (failed(cloneBody(nestedFor.getBody(), newNestedFor.getBody(),
                               nestedMapper, nestedCurrentVectors))) {
            return failure();
          }

          for (auto [oldResult, newResult] :
               llvm::zip(nestedFor.getResults().take_front(
                             nestedFor.getNumResults()),
                         newNestedFor.getResults().take_front(
                             nestedFor.getNumResults()))) {
            mapper.map(oldResult, newResult);
          }
          for (unsigned i = 0; i < localBuffers.size(); ++i) {
            setCurrentVector(localBuffers[i],
                             newNestedFor->getResult(nestedFor.getNumResults() + i));
          }
          rewriter.setInsertionPointAfter(newNestedFor);
          continue;
        }

        if (auto loadOp = dyn_cast<affine::AffineLoadOp>(childOp)) {
          if (auto idx = getBufferIndex(loadOp.getMemref())) {
            SmallVector<Value> indices = buildAffineAccessIndices(
                loadOp.getLoc(), loadOp.getAffineMap(), loadOp.getMapOperands());
            SmallVector<int64_t> staticPosition(indices.size(),
                                                ShapedType::kDynamic);
            auto positionAttr = rewriter.getDenseI64ArrayAttr(staticPosition);
            auto extractOp = rewriter.create<vector::ExtractOp>(
                loadOp.getLoc(), loadOp.getType(), currentVectors[*idx],
                indices, positionAttr);
            mapper.map(loadOp.getResult(), extractOp.getResult());
            continue;
          }
        }

        if (auto storeOp = dyn_cast<affine::AffineStoreOp>(childOp)) {
          if (auto idx = getBufferIndex(storeOp.getMemref())) {
            SmallVector<Value> indices =
                buildAffineAccessIndices(storeOp.getLoc(), storeOp.getAffineMap(),
                                         storeOp.getMapOperands());
            SmallVector<OpFoldResult> position;
            position.reserve(indices.size());
            for (Value index : indices) {
              position.push_back(index);
            }
            auto insertOp = rewriter.create<vector::InsertOp>(
                storeOp.getLoc(), remapValue(storeOp.getValueToStore()),
                currentVectors[*idx], position);
            setCurrentVector(storeOp.getMemref(), insertOp.getResult());
            continue;
          }
        }

        if (auto fillOp = dyn_cast<frisk::FillOp>(childOp)) {
          if (auto idx = getBufferIndex(fillOp.getMemref())) {
            Attribute splat =
                makeSplatAttr(fillOp.getValueAttr(), vectorTypes[*idx].getElementType());
            if (!splat) {
              return failure();
            }
            auto denseAttr = DenseElementsAttr::get(vectorTypes[*idx], splat);
            auto constantOp = rewriter.create<arith::ConstantOp>(
                fillOp.getLoc(), vectorTypes[*idx], denseAttr);
            setCurrentVector(fillOp.getMemref(), constantOp.getResult());
            continue;
          }
        }

        auto *cloned = rewriter.clone(childOp, mapper);
        for (auto [oldResult, newResult] :
             llvm::zip(childOp.getResults(), cloned->getResults())) {
          if (!mapper.lookupOrNull(oldResult)) {
            mapper.map(oldResult, newResult);
          }
        }
        if (auto oldWmma = dyn_cast<frisk::WarpMmaRROp>(childOp)) {
          if (auto newWmma = dyn_cast<frisk::WarpMmaRROp>(cloned)) {
            if (oldWmma->getNumResults() == 1 && newWmma->getNumResults() == 1) {
              setCurrentVector(oldWmma.getC(), newWmma.getResult());
            }
          }
        }
      }

      auto oldYield = cast<affine::AffineYieldOp>(oldBlock->getTerminator());
      SmallVector<Value> yieldValues = remapValueRange(oldYield.getOperands());
      yieldValues.append(currentVectors.begin(), currentVectors.end());
      rewriter.setInsertionPointToEnd(newBlock);
      rewriter.create<affine::AffineYieldOp>(oldYield.getLoc(), yieldValues);
      return success();
    };

    IRMapping mapper;
    mapper.map(op.getBody()->getArgument(0), newForOp.getBody()->getArgument(0));
    for (auto [oldArg, newArg] :
         llvm::zip(op.getRegionIterArgs(),
                   newForOp.getRegionIterArgs().take_front(op.getNumIterOperands()))) {
      mapper.map(oldArg, newArg);
    }

    SmallVector<Value> currentVectors;
    currentVectors.reserve(localBuffers.size());
    unsigned vectorArgStart = 1 + op.getNumIterOperands();
    for (unsigned i = 0; i < localBuffers.size(); ++i) {
      Value arg = newForOp.getBody()->getArgument(vectorArgStart + i);
      currentVectors.push_back(arg);
      mapper.map(localBuffers[i], arg);
    }

    if (failed(cloneBody(op.getBody(), newForOp.getBody(), mapper,
                         currentVectors))) {
      return failure();
    }

    SmallVector<Value> replacementResults(
        newForOp.getResults().take_front(op.getNumResults()).begin(),
        newForOp.getResults().take_front(op.getNumResults()).end());
    rewriter.replaceOp(op, replacementResults);

    for (auto [idx, buffer] : llvm::enumerate(localBuffers)) {
      Value result = newForOp->getResult(op.getNumResults() + idx);
      buffer.replaceUsesWithIf(result, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return user->getBlock() == newForOp->getBlock() &&
               newForOp->isBeforeInBlock(user);
      });
      Operation *defOp = buffer.getDefiningOp();
      if (defOp && defOp->use_empty()) {
        rewriter.eraseOp(defOp);
      }
    }

    return success();
  }
};



class ConvertFriskToBase : public impl::ConvertFriskToBaseBase<ConvertFriskToBase> {
public:
  
  void runOnOperation(){
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    ConversionTarget target(*context);

    // clang-format off
    target.addLegalDialect<
      frisk::FriskDialect,
      arith::ArithDialect,
      affine::AffineDialect,
      math::MathDialect,
      func::FuncDialect,
      memref::MemRefDialect,
      scf::SCFDialect,
      gpu::GPUDialect,
      vector::VectorDialect>();

    target.addIllegalOp<KernelOp>();
    target.addIllegalOp<ParallelOp>();
    target.addIllegalOp<ForOp>();
    target.addDynamicallyLegalOp<affine::AffineForOp>([](affine::AffineForOp op) {
      return op->hasAttr("local_yield_transformed");
    });
    RewritePatternSet patterns(context);
    patterns.add<KernelOpConversion>(context);
    patterns.add<ParallelOpConversion>(context);
    patterns.add<ForOpConversion>(context);
    patterns.add<AffineForOpConversion>(context);
    if (failed(applyPartialConversion(mod, target, std::move(patterns)))){
      return signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createConvertFriskToBasePass() {
  return std::make_unique<ConvertFriskToBase>();
}

}
