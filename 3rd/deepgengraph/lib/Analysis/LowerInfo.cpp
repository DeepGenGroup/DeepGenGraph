#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Analysis/HardwareSpecification.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace mlir::frisk {

template<typename T>
void show_vector(llvm::SmallVector<T, 2> vec, const std::string& name) {
  llvm::outs() << "[" << name << "]: {";
  for (size_t i=0; i<vec.size(); ++i) {
    llvm::outs() << vec[i];
    if (i != vec.size()-1) {
      llvm::outs() << ", ";
    }
  }
  llvm::outs() << "}\n";
}

#define LLVM_OUT_MSG(msg)  llvm::outs() << msg << "\n";llvm::outs().flush()

LowerInfo::LowerInfo(int w) : warp_threads(w){ }

llvm::SmallVector<Operation*, 5>
LowerInfoAnalysis::collectNeedInferOps(mlir::Operation *kernelOp) {
  auto _kernelOp = mlir::dyn_cast<func::FuncOp>(kernelOp);
  if (!_kernelOp) {
    assert(false);
  }

  llvm::SmallVector<Operation*, 5> need_infer_ops{};
  _kernelOp.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (isa<CopyOp, BlockOp, GemmOp, ReduceOp>(op)) {
      need_infer_ops.push_back(op);
    }
  });
  return need_infer_ops;
}

std::pair<int, int> LowerInfoAnalysis::squareFactor(int n) {
  int a = static_cast<int>(std::sqrt(n));
  while (a >= 1) {
    if (n % a == 0) {
      int b = n / a;
      return {b, a};
    }
    --a;
  }
  return {n, 1};
}

uint64_t LowerInfoAnalysis::getRegionThreadNum(Operation *op) {
  uint64_t thread_num = 0;
  Operation *parentOp = op->getParentOp();
  while (parentOp != nullptr) {
    if (isa<KernelOp>(parentOp)) {
      if (auto intElem = dyn_cast_or_null<IntegerAttr>(parentOp->getAttr("thread_num"))) {
        thread_num = intElem.getValue().getZExtValue();
        break;
      }
    } else if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {
      if (auto intElem = dyn_cast_or_null<IntegerAttr>(funcOp->getAttr("thread_num"))) {
        thread_num = intElem.getValue().getZExtValue();
        break;
      }
    } else if (auto warpGroupOp = dyn_cast<WarpGroupOp>(parentOp)) {
      thread_num = warpGroupOp.getWarpGroupNum() * 128;
      break;
    }
    parentOp = parentOp->getParentOp();
  }
  return thread_num;
}

LowerInfo &LowerInfoAnalysis::setBufferInfo(DenseMap<Value, LowerInfo> &buf_info_maps,
                                            Value buffer, const LowerInfo &info) {
  auto it = buf_info_maps.find(buffer);
  if (it == buf_info_maps.end()) {
    it = buf_info_maps.insert(std::make_pair(buffer, info)).first;
  } else {
    it->second = info;
  }
  return it->second;
}

LowerInfoAnalysis::GemmProblem LowerInfoAnalysis::getGemmProblem(GemmOp gemmOp) {
  GemmProblem problem;
  problem.A = gemmOp.getMatrixA();
  problem.B = gemmOp.getMatrixB();
  problem.C = gemmOp.getMatrixC();
  problem.aType = gemmOp.getA().getType();
  problem.bType = gemmOp.getB().getType();
  problem.cType = gemmOp.getC().getType();
  auto shapeC = problem.cType.getShape();
  auto shapeA = problem.aType.getShape();
  problem.inElemBitWidth = problem.aType.getElementTypeBitWidth();
  problem.bm = shapeC[0];
  problem.bn = shapeC[1];
  problem.bk = gemmOp.getTransA() ? shapeA[1] : shapeA[0];
  return problem;
}

MMAInstInfo* LowerInfoAnalysis::selectGemmInst(LowerInfoAnalysis::GemmProblem problem, HWSpecification* hw){
  MMAInstInfo* ret = nullptr;
  int max_m = 0, max_n = 0, max_k = 0;
  auto _GetFriskDTypeFromBuffer = [](MemRefType ty){
    auto ety = ty.getElementType();
    if(ety.isF16()){
      return FriskDType::f16;
    }
    if(ety.isF32()){
      return FriskDType::f32;
    }
  };
  auto aTy = _GetFriskDTypeFromBuffer(problem.aType);
  auto bTy = _GetFriskDTypeFromBuffer(problem.bType);
  auto cTy = _GetFriskDTypeFromBuffer(problem.cType);
  for(auto& inst : hw->gemmInfo.validInsts){
    // 如果 inst 的 mnk在其中最大，且mnk小于 bm bn bk, 且dtype符合 problem的elementType, 则选择该inst
    if(inst.desc_c.elementType == cTy && inst.desc_a.elementType == aTy && inst.desc_b.elementType == bTy){
      if(inst.m <= problem.bm && inst.n <= problem.bn && inst.k <= problem.bk 
        && inst.m * inst.n > max_m * max_n
      ){
        max_m = inst.m;
        max_n = inst.n;
        ret = &inst;
      }
    }
  }
  return ret;
}

bool LowerInfoAnalysis::getDirectGemmBlockLayout(
  uint64_t thread_num,
  std::array<int64_t, 2> &block_layout,
  HWSpecification* hw
) {
  if(hw->getKind() == HW_KIND_NVIDIA){
    int warpgroup_num = thread_num / 128;
    auto [y, x] = squareFactor(warpgroup_num);
    if (warpgroup_num <= 0 || y <= 0 || x <= 0) {
      LLVM_OUT_MSG("warpgroup layout err");
      return false;
    }
    block_layout = {y * 4, x};
    return true;
  }
  else if(hw->getKind() == HW_KIND_DCU){
    int warp_num = thread_num / hw->getWarpsize();
    auto [y, x] = squareFactor(warp_num);
    if (warp_num <= 0 || y <= 0 || x <= 0) {
      LLVM_OUT_MSG("warp_num layout err");
      return false;
    }
    block_layout = {y , x};
    return true;
  }
}

LowerInfo LowerInfoAnalysis::makeDirectGemmCInfo(OpBuilder b, const GemmProblem &problem,
                                                 MMAInstInfo *mma, uint64_t thread_num,
                                                 HWSpecification *hw,
                                                 std::array<int64_t, 2> block_layout
                                                ) {
  LowerInfo info{hw->getWarpsize()};
  if(hw->getKind() == HW_KIND_NVIDIA){
    info.buffer = problem.C;
    info.thread_bound = thread_num;
    info.thread_widths = {1, 32 / static_cast<int64_t>(problem.inElemBitWidth)};
    info.warp_layout = {8, 4};
    info.block_layout = block_layout;
    info.warp_widths = info.getWarpWidths(info.thread_widths, info.warp_layout);
    info.warp_repeat = {2, mma->n / info.warp_widths[1]};
    info.block_widths = info.getBlockWidths(info.warp_widths, info.warp_repeat, info.block_layout);
    info.block_repeat = {problem.bm / info.block_widths[0], problem.bn / info.block_widths[1]};
    info.warp_indices = info.getWarpIndices(b, info.block_layout);
    info.lane_indices = info.getThreadIndices(b, info.warp_layout);
  }
  else if(hw->getKind() == HW_KIND_DCU){
    info.buffer = problem.C;
    info.thread_bound = thread_num;
    info.thread_widths = mma->desc_c.thread_creg;
    info.warp_layout = mma->desc_c.warp_layout;
    info.block_layout = block_layout;
    info.warp_widths = PointwiseDot(mma->desc_c.warp_layout, mma->desc_c.thread_creg);
    info.warp_repeat = mma->desc_c.warp_repeat;
    info.block_widths = info.getBlockWidths(info.warp_widths, info.warp_repeat, info.block_layout);
    info.block_repeat = {problem.bm / info.block_widths[0], problem.bn / info.block_widths[1]};
    info.warp_indices = info.getWarpIndices(b, info.block_layout);
    info.lane_indices = info.getThreadIndices(b, info.warp_layout);
  }
  return info;
}

LowerInfo LowerInfoAnalysis::makeRelyGemmCInfo(OpBuilder b, const GemmProblem &problem,
                                               MMAInstInfo *mma, uint64_t thread_num,
                                               HWSpecification *hw,
                                               const LowerInfo &source_info,
                                               bool source_is_a) {
  LowerInfo info{hw->getWarpsize()};
  info.buffer = problem.C;
  info.thread_bound = thread_num;
  info.warp_layout = source_info.warp_layout;
  info.block_layout = source_info.block_layout;
  if (source_is_a) {
    info.thread_widths = {source_info.thread_widths[0],
                          32 / static_cast<int64_t>(problem.inElemBitWidth)};
    info.warp_widths = info.getWarpWidths(info.thread_widths, info.warp_layout);
    info.warp_repeat = {source_info.warp_repeat[0], mma->n / info.warp_widths[1]};
    info.block_widths = info.getBlockWidths(info.warp_widths, info.warp_repeat, info.block_layout);
    info.block_repeat = {source_info.block_repeat[0], problem.bn / info.block_widths[1]};
  } else {
    info.thread_widths = {1, source_info.thread_widths[1]};
    info.warp_widths = info.getWarpWidths(info.thread_widths, info.warp_layout);
    info.warp_repeat = {2, source_info.warp_repeat[1]};
    info.block_widths = info.getBlockWidths(info.warp_widths, info.warp_repeat, info.block_layout);
    info.block_repeat = {problem.bm / info.block_widths[0], source_info.block_repeat[1]};
  }
  info.warp_indices = info.getWarpIndices(b, info.block_layout);
  info.lane_indices = info.getThreadIndices(b, info.warp_layout);
  return info;
}

void LowerInfoAnalysis::applyDirectGemmAInfo(LowerInfo &info, const GemmProblem &problem,
                                             MMAInstInfo *mma, AffineExpr zero, HWSpecification* hw) {
  info.buffer = problem.A;
  if(hw->getKind() == HW_KIND_NVIDIA){
    info.thread_widths[1] = 32 / static_cast<int64_t>(problem.inElemBitWidth);
    info.warp_widths[1] = 0;
    info.warp_repeat[1] = mma->k / info.warp_layout[1] / info.thread_widths[1];
    info.block_widths[1] = mma->k;
    info.block_repeat[1] = problem.bk / mma->k;
    info.warp_indices[1] = zero;
    info.lane_indices[1] = zero;
  }
  else if(hw->getKind() == HW_KIND_DCU){
    info.thread_widths = mma->desc_a.thread_creg ;
    info.warp_widths = mma->desc_a.get_warp_widths();
    info.warp_repeat = mma->desc_a.warp_repeat;
    info.block_widths[1] = mma->k;
    info.block_repeat[1] = problem.bk / mma->k;
    info.warp_indices[1] = zero;
    info.lane_indices[1] = zero;
  }
}

void LowerInfoAnalysis::applyRelyGemmAInfo(LowerInfo &info, const GemmProblem &problem,
                                           MMAInstInfo *mma, AffineExpr zero) {
  info.buffer = problem.A;

  info.thread_widths[1] = 0;
  info.warp_widths[1] = 0;
  info.warp_repeat[1] = 0;
  info.block_widths[1] = mma->k;
  info.block_repeat[1] = problem.bk / mma->k;
  info.warp_indices[1] = zero;
  info.lane_indices[1] = zero;


}

void LowerInfoAnalysis::applyGemmBInfo(LowerInfo &info, const GemmProblem &problem,
                                       MMAInstInfo *mma, AffineExpr zero, HWSpecification* hw) {
  info.buffer = problem.B;
  if(hw->getKind() == HW_KIND_NVIDIA){
    info.thread_widths[0] = 0;
    info.warp_widths[0] = 0;
    info.warp_repeat[0] = 0;
    info.block_widths[0] = mma->k;
    info.block_repeat[0] = problem.bk / mma->k;
    info.warp_indices[0] = zero;
    info.lane_indices[0] = zero;
  }
  else if(hw->getKind() == HW_KIND_DCU) {
    // info.thread_widths[0] = 0;
    // info.warp_widths[0] = 0;
    // info.warp_repeat[0] = 0;
    info.thread_widths = mma->desc_b.thread_creg;
    info.warp_widths = mma->desc_b.get_warp_widths();
    info.warp_repeat = mma->desc_b.warp_repeat;
    info.block_widths[0] = mma->k;
    info.block_repeat[0] = problem.bk / mma->k;
    info.warp_indices[0] = zero;
    info.lane_indices[0] = zero;
  }
}

bool LowerInfoAnalysis::inferCopyOp(Operation *op, DenseMap<Value, LowerInfo> &buf_info_maps) {
  if (auto copyOp = dyn_cast<CopyOp>(op)) {
    Value dst = copyOp.getDstMemRef();
    Value src = copyOp.getSrcMemRef();
    if (auto dstInfo = buf_info_maps.find(dst); dstInfo != buf_info_maps.end()) {
      LowerInfo &srcInfo = setBufferInfo(buf_info_maps, src, dstInfo->second);
      srcInfo.buffer = src;
      return true;
    } else if (auto srcInfo = buf_info_maps.find(src); srcInfo != buf_info_maps.end()) {
      LowerInfo &dstInfo = setBufferInfo(buf_info_maps, dst, srcInfo->second);
      dstInfo.buffer = dst;
      return true;
    }
    LLVM_OUT_MSG("---- inferError 0");
    return false;
  }
  return false;
}

bool LowerInfoAnalysis::inferBlockOp(Operation *op, DenseMap<Value, LowerInfo> &buf_info_maps) {
  auto blockOp = dyn_cast<BlockOp>(op);
  if (!blockOp) {
    return false;
  }

  Value store_buf;
  llvm::SmallVector<Value, 3> load_bufs;
  auto uppers = blockOp.getBlockRanges();
  blockOp.walk<mlir::WalkOrder::PreOrder>([&](Operation *nestedOp) {
    if (auto loadOp = dyn_cast<affine::AffineLoadOp>(nestedOp)) {
      load_bufs.push_back(loadOp.getMemRef());
    } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(nestedOp)) {
      store_buf = storeOp.getMemref();
    }
  });

  auto get_main_info_func = [&](Value buf) -> bool {
    MemRefType lty = dyn_cast<MemRefType>(buf.getType());
    if (lty.getRank() == uppers.size()) {
      auto shape = lty.getShape();
      bool is_shape_equl = true;
      for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] != uppers[i]) {
          is_shape_equl = false;
          break;
        }
      }
      if (is_shape_equl) return true;
    }
    LLVM_OUT_MSG("---- inferError 1");
    return false;
  };

  LowerInfo *info = nullptr;
  for (const auto &lbuf : load_bufs) {
    if (buf_info_maps.count(lbuf) && get_main_info_func(lbuf)) {
      info = &buf_info_maps.find(lbuf)->second;
      break;
    }
  }
  if (info == nullptr && buf_info_maps.count(store_buf) && get_main_info_func(store_buf)) {
    info = &buf_info_maps.find(store_buf)->second;
  }
  if (info == nullptr) {
    LLVM_OUT_MSG("---- inferError 3");
    return false;
  }

  load_bufs.push_back(store_buf);
  LowerInfo sourceInfo = *info;
  for (const Value &buf : load_bufs) {
    if (!buf_info_maps.count(buf)) {
      LowerInfo &bufInfo = setBufferInfo(buf_info_maps, buf, sourceInfo);
      bufInfo.buffer = buf;
    }
  }
  return true;
}

bool LowerInfoAnalysis::checkGemmProblem(LowerInfoAnalysis::GemmProblem p, HWSpecification* hw){
  // TODO : 根据 hw属性检查GEMM 是否符合硬件要求
  return true;
}

bool LowerInfoAnalysis::inferGemmOp(Operation *op, DenseMap<Value, LowerInfo> &buf_info_maps,
                                    HWSpecification *hw) {
  auto gemmOp = dyn_cast<GemmOp>(op);
  if (!gemmOp) {
    return false;
  }

  OpBuilder b(op);
  uint64_t thread_num = getRegionThreadNum(op);
  GemmProblem problem = getGemmProblem(gemmOp);
  if (!checkGemmProblem(problem, hw)) {
    LLVM_OUT_MSG("gemmProblem check failed!");
    return false;
  }
  
  std::array<int64_t, 2> block_layout;  // block 内的warp 分布[x,y]
  if (!getDirectGemmBlockLayout(thread_num, block_layout, hw)) {
    return false;
  }
  if(!checkGemmProblem(problem, hw)){
    assert(false);
  }
  // assert(problem.bm % 64 == 0 && "BM must great more than MMA_M");
  // assert(problem.bn % 8 == 0 && "BN must great more than min MMA_N");
  // assert(problem.inElemBitWidth * problem.bk % 32 == 0 && "BK must great more than MMA_K");
  // assert((problem.cType.getMemorySpaceAsInt() == 0 || problem.cType.getMemorySpaceAsInt() == 5) &&
  //        "C must be local buffer.");
  // MmaShape mma = getMmaShape(problem, hw);
  MMAInstInfo* instInfo = selectGemmInst(problem, hw);

  LowerInfo ic = makeDirectGemmCInfo(b, problem, instInfo, thread_num, hw, block_layout);
  setBufferInfo(buf_info_maps, problem.C, ic);

  auto zero = b.getAffineConstantExpr(0);
  LowerInfo &aInfo = setBufferInfo(buf_info_maps, problem.A, LowerInfo(ic));
  applyDirectGemmAInfo(aInfo, problem, instInfo, zero, hw);

  LowerInfo &bInfo = setBufferInfo(buf_info_maps, problem.B, LowerInfo(ic));
  applyGemmBInfo(bInfo, problem, instInfo, zero,hw);
  return true;
}

bool LowerInfoAnalysis::inferRelyGemmOp(Operation *op, DenseMap<Value, LowerInfo> &buf_info_maps,
                                        HWSpecification *hw) {
  auto gemmOp = dyn_cast<GemmOp>(op);
  if (!gemmOp) {
    return false;
  }

  OpBuilder b(op);
  uint64_t thread_num = getRegionThreadNum(op);
  GemmProblem problem = getGemmProblem(gemmOp);
  if(!checkGemmProblem(problem, hw)){
    assert(false);
  }
  // assert(problem.bm % 64 == 0 && "BM must great more than MMA_M");
  // assert(problem.bn % 8 == 0 && "BN must great more than min MMA_N");
  // assert(problem.inElemBitWidth * problem.bk % (32*8) == 0 && "BK must great more than MMA_K");
  // assert((problem.cType.getMemorySpaceAsInt() == 0 || problem.cType.getMemorySpaceAsInt() == 5) &&
  //        "C must be local buffer.");
  // assert((problem.bType.getMemorySpaceAsInt() == 3) && "B must be shared buffer.");
  // MmaShape mma = getMmaShape(problem, hw);
  auto mma = selectGemmInst(problem, hw);

  LowerInfo ic{hw->getWarpsize()};
  if (buf_info_maps.count(problem.A) || buf_info_maps.count(problem.B)) {
    bool source_is_a = buf_info_maps.count(problem.A);
    LowerInfo info = source_is_a ? buf_info_maps.find(problem.A)->second
                                 : buf_info_maps.find(problem.B)->second;
    assert(thread_num == info.thread_bound && "Thread Number is not equl");
    ic = makeRelyGemmCInfo(b, problem, mma, thread_num, hw, info, source_is_a);
    setBufferInfo(buf_info_maps, problem.C, ic);
  } else {
    ic = buf_info_maps.find(problem.C)->second;
  }

  auto zero = b.getAffineConstantExpr(0);
  if (buf_info_maps.count(problem.A)) {
    LowerInfo &bInfo = setBufferInfo(buf_info_maps, problem.B, ic);
    applyGemmBInfo(bInfo, problem, mma, zero,hw);
  }
  if (buf_info_maps.count(problem.B)) {
    LowerInfo &aInfo = setBufferInfo(buf_info_maps, problem.A, ic);
    applyRelyGemmAInfo(aInfo, problem, mma, zero);
  }
  return true;
}

bool LowerInfoAnalysis::inferReduceOp(Operation *op, DenseMap<Value, LowerInfo> &buf_info_maps) {
  auto reduceOp = dyn_cast<ReduceOp>(op);
  if (!reduceOp) {
    return false;
  }

  Value dst = reduceOp.getDst();
  Value src = reduceOp.getSrc();
  uint64_t dim = reduceOp.getDim();
  if (!buf_info_maps.count(src)) {
    LLVM_OUT_MSG("---- inferError 4");
    return false;
  }
  LowerInfo &dstInfo = setBufferInfo(buf_info_maps, dst, buf_info_maps.find(src)->second);
  dstInfo.buffer = dst;

  auto zero = OpBuilder(op).getAffineConstantExpr(0);
  auto erase_affine_dim_func = [&](std::array<AffineExpr, 2> &vec) -> bool {
    if (dim >= vec.size()) return false;
    for (size_t i = dim; i + 1 < vec.size(); ++i) {
      vec[i] = vec[i + 1];
    }
    vec.back() = zero;
    return true;
  };
  auto erase_i64_dim_func = [&](std::array<int64_t, 2> &vec) -> bool {
    if (dim >= vec.size()) return false;
    for (size_t i = dim; i + 1 < vec.size(); ++i) {
      vec[i] = vec[i + 1];
    }
    vec.back() = 1;
    return true;
  };

  if (!erase_affine_dim_func(dstInfo.warp_indices) ||
      !erase_affine_dim_func(dstInfo.lane_indices) ||
      !erase_i64_dim_func(dstInfo.warp_layout) ||
      !erase_i64_dim_func(dstInfo.block_layout) ||
      !erase_i64_dim_func(dstInfo.warp_repeat) ||
      !erase_i64_dim_func(dstInfo.block_repeat) ||
      !erase_i64_dim_func(dstInfo.thread_widths) ||
      !erase_i64_dim_func(dstInfo.warp_widths) ||
      !erase_i64_dim_func(dstInfo.block_widths)) {
    LLVM_OUT_MSG("---- inferError 5");
    return false;
  }
  return true;
}

bool LowerInfoAnalysis::inferDirectOp(Operation *op, DenseMap<Value, LowerInfo> &buf_info_maps,
                                      HWSpecification *hw) {
  if (auto gemmOp = dyn_cast<GemmOp>(op)) {
    (void)gemmOp;
    return inferGemmOp(op, buf_info_maps, hw);
  }
  return false;
}

bool LowerInfoAnalysis::inferRelyOp(Operation *op, DenseMap<Value, LowerInfo> &buf_info_maps,
                                    HWSpecification *hw) {
  llvm::SmallVector<Value, 8> memrefsToCheck;
  for (const auto &opd : op->getOperands()) {
    if (isa<MemRefType>(opd.getType())) {
      memrefsToCheck.push_back(opd);
    }
  }
  if (auto blockOp = dyn_cast<BlockOp>(op)) {
    blockOp.walk<mlir::WalkOrder::PreOrder>([&](Operation *nestedOp) {
      if (auto loadOp = dyn_cast<affine::AffineLoadOp>(nestedOp)) {
        memrefsToCheck.push_back(loadOp.getMemRef());
      } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(nestedOp)) {
        memrefsToCheck.push_back(storeOp.getMemref());
      }
    });
  }

  bool all_in = true;
  size_t count = 0;
  for (const Value &memref : memrefsToCheck) {
    if (!buf_info_maps.count(memref)) {
      all_in = false;
      count++;
    }
  }
  if (all_in) {
    LLVM_OUT_MSG("buf已全部推断");
    return true;
  }
  if (!all_in && count == memrefsToCheck.size()) {
    LLVM_OUT_MSG("无已推断buf, 需要gemmOp做锚点");
    return false;
  }

  if (inferCopyOp(op, buf_info_maps)) {
    return true;
  }
  if (inferBlockOp(op, buf_info_maps)) {
    return true;
  }
  if (inferRelyGemmOp(op, buf_info_maps, hw)) {
    return true;
  }
  if (inferReduceOp(op, buf_info_maps)) {
    return true;
  }

  assert(false && "the operation can not be recognized.");
  LLVM_OUT_MSG("---- inferError 6");
  return false;
}

/**
 * @brief lowinfo 推断
 目的：以kernel中的首个gemm为出发点，向两侧推断线程应该持有的寄存器buffer形状。尽可能减少算子之间 local->shm 的写回
 block中，block_layout 和 warp_layout 按照 H100 gemm tensorcore 的计算规则固定
 * 
 * @param kernelOp 
 * @param hwKind : dcu,nvidia 
 * @return DenseMap<Value, LowerInfo> 
 */
DenseMap<Value, LowerInfo> LowerInfoAnalysis::run(mlir::Operation* kernelOp, const std::string& hwKind ,const std::string& version){
  auto hw = GetHWSpecification(hwKind, version, kernelOp->getContext());
  DenseMap<Value, LowerInfo> buf_info_maps{};
  SmallVector<Operation*, 5> need_infer_ops = collectNeedInferOps(kernelOp);

  bool exsit_dircet_infer_op = false;
  for (size_t i=0; i<need_infer_ops.size(); ++i) {
    if (inferDirectOp(need_infer_ops[i], buf_info_maps, hw)) {
      exsit_dircet_infer_op = true;
      need_infer_ops.erase(need_infer_ops.begin() + i);
      break;
    }
  }
  if (!exsit_dircet_infer_op) {
    assert(false && "LowerInfo infer filed.");
  }

  while (!need_infer_ops.empty()) {
    bool progress = false;
    SmallVector<Operation*, 5> pendingOps;
    pendingOps.reserve(need_infer_ops.size());
    for (Operation *op : need_infer_ops) {
      if (inferRelyOp(op, buf_info_maps, hw)) {
        progress = true;
      } else {
        pendingOps.push_back(op);
      }
    }
    if (!progress) {
      llvm::errs() << "[LowerInfo] infer failed: unresolved ops remain (" << pendingOps.size() << ")\n";
      for (Operation *op : pendingOps) {
        llvm::errs() << "[E] unresolved op: " << *op << "\n";
      }
      assert(false && "LowerInfo infer failed.");
    }
    need_infer_ops.swap(pendingOps);
  }
  return buf_info_maps;
}

}
