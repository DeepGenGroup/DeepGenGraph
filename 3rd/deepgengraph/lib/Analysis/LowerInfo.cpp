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
#include "llvm/ADT/STLExtras.h"
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

static std::array<int64_t, 2> getThreadOwnDataSize(const LowerInfo &info,
                                                   bool includeBaseRepeat) {
  auto [tw0, tw1] = info.get_thread_widths();
  auto [wr0, wr1] = info.get_warp_repeat();
  if (!includeBaseRepeat) {
    return {tw0 * wr0, tw1 * wr1};
  }
  auto [br0, br1] = info.get_block_repeat();
  return {tw0 * wr0 * br0, tw1 * wr1 * br1};
}

LowerInfo *LowerInfoMap::getLowerInfo(const mlir::Value &buffer,
                                      mlir::Operation *op) {
  auto it = infoMap.find(std::make_pair(buffer, op));
  if (it == infoMap.end()) {
    return nullptr;
  }
  return &it->second;
}

LowerInfo *LowerInfoMap::addLowerInfo(mlir::Operation *op, LowerInfo info) {
  assert(info.buffer != nullptr);
  auto key = std::make_pair(info.buffer, op);
  auto it = infoMap.find(key);
  if (it == infoMap.end()) {
    it = infoMap.insert(std::make_pair(key, std::move(info))).first;
  } else {
    it->second = std::move(info);
  }
  it->second.buffer = key.first;
  return &it->second;
}

void LowerInfoMap::getOpsOrder(mlir::Operation* rootNode){
  unsigned idx = 1;
  rootNode->walk<WalkOrder::PreOrder>([&](mlir::Operation* subOp){
    opOrder.try_emplace(subOp, idx++);
  });
}


LowerInfo *LowerInfoMap::getNearestInferedInfo(const mlir::Value &buffer,
                                               mlir::Operation *currOp,
                                               bool isBefore) {
  if (currOp == nullptr) {
    return nullptr;
  }
  if(opOrder.empty()){
    assert(false && "must run LowerInfoMap::getOpsOrder() first");
    return nullptr;
  }
  auto currOrderIt = opOrder.find(currOp);
  if (currOrderIt == opOrder.end()) {
    assert(false && "currOp must be recorded in LowerInfoMap::getOpsOrder()");
    return nullptr;
  }
  unsigned currOrder = currOrderIt->second;
  unsigned bestOrder = 0;
  LowerInfo *nearestInfo = nullptr;
  for (auto &entry : infoMap) {
    auto& buf = entry.first.first;
    auto op = entry.first.second;
    auto& info = entry.second;
    auto opOrderIt = opOrder.find(op);
    if (opOrderIt == opOrder.end()) {
      continue;
    }
    unsigned order = opOrderIt->second;
    if (buffer != buf) {
      continue;
    }
    if (isBefore && order <= currOrder &&
        (nearestInfo == nullptr || order > bestOrder)) {
      nearestInfo = &info;
      bestOrder = order;
    } else if (!isBefore && order >= currOrder &&
               (nearestInfo == nullptr || order < bestOrder)) {
      nearestInfo = &info;
      bestOrder = order;
    }
  }
  return nearestInfo;
}

static LowerInfo *getNearestInferedInfoEither(LowerInfoMap &infoMap,
                                              const mlir::Value &buffer,
                                              mlir::Operation *currOp) {
  if (auto *info = infoMap.getNearestInferedInfo(buffer, currOp, true)) {
    return info;
  }
  return infoMap.getNearestInferedInfo(buffer, currOp, false);
}

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

// 根据gemm问题，选择最大的 wmma指令计算（先不考虑memspace，后续有Pass处理）
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
  info.mmaInst = mma;
  info.block_layout = block_layout;
  info.block_layout_order = {0, 1};
  if(hw->getKind() == HW_KIND_NVIDIA){
    info.buffer = problem.C;
    info.thread_bound = thread_num;
    info.base_layout.thread_creg = {1, 32 / static_cast<int64_t>(problem.inElemBitWidth)};
    info.base_layout.warp_layout = {8, 4};
    info.base_layout.warp_repeat = {2, mma->n / info.get_warp_widths()[1]};
    info.block_repeat = {problem.bm / info.get_block_widths()[0],
                         problem.bn / info.get_block_widths()[1]};
  }
  else if(hw->getKind() == HW_KIND_DCU){
    info.buffer = problem.C;
    info.thread_bound = thread_num;
    info.base_layout = mma->desc_c;
    info.block_repeat = {problem.bm / info.get_block_widths()[0],
                         problem.bn / info.get_block_widths()[1]};
  }
  // 对于gemmC，单个线程持有的数据应考虑wmmaInst在block-level buffer上滑动的情况。每次计算得到一个inst区域的C
  info.thread_own_data_size = getThreadOwnDataSize(info, true);
  return info;
}

LowerInfo LowerInfoAnalysis::makeRelyGemmCInfo(OpBuilder b, const GemmProblem &problem,
                                               MMAInstInfo *mma, uint64_t thread_num,
                                               HWSpecification *hw,
                                               const LowerInfo &source_info,
                                               bool source_is_a) {
  LowerInfo info{hw->getWarpsize()};
  info.mmaInst = mma;
  info.buffer = problem.C;
  info.thread_bound = thread_num;
  std::array<int64_t, 2> block_layout;
  if (!getDirectGemmBlockLayout(thread_num, block_layout, hw)) {
    block_layout = source_info.get_block_layout();
  }
  info.block_layout = block_layout;
  info.block_layout_order = source_info.block_layout_order;
  if (hw->getKind() == HW_KIND_DCU) {
    info.base_layout = mma->desc_c;
  } else {
    info.base_layout.thread_creg = {1, 32 / static_cast<int64_t>(problem.inElemBitWidth)};
    info.base_layout.warp_layout = source_info.get_warp_layout();
    info.base_layout.warp_repeat = {2, mma->n / info.get_warp_widths()[1]};
  }
  info.block_repeat = {problem.bm / info.get_block_widths()[0],
                       problem.bn / info.get_block_widths()[1]};
  // 对于gemmC，单个线程持有的数据应考虑wmmaInst在block-level buffer上滑动的情况。每次计算得到一个inst区域的C
  info.thread_own_data_size = getThreadOwnDataSize(info, true);

  return info;
}

void LowerInfoAnalysis::applyDirectGemmAInfo(LowerInfo &info, const GemmProblem &problem,
                                             MMAInstInfo *mma, AffineExpr zero, HWSpecification* hw) {
  info.buffer = problem.A;
  info.mmaInst = mma;
  if(hw->getKind() == HW_KIND_NVIDIA){
    auto blockLayout = info.get_block_layout();
    info.base_layout.thread_creg[1] = 32 / static_cast<int64_t>(problem.inElemBitWidth);
    info.base_layout.warp_repeat[1] =
        mma->k / info.get_warp_layout()[1] / info.get_thread_widths()[1];
    info.block_layout = {blockLayout[0], 1};
    info.block_repeat = {problem.bm / info.get_block_widths()[0],
                         problem.bk / info.get_block_widths()[1]};
  }
  else if(hw->getKind() == HW_KIND_DCU){
    auto blockLayout = info.get_block_layout();
    info.base_layout = mma->desc_a;
    info.block_layout = {blockLayout[0], 1};
    info.block_repeat = {problem.bm / info.get_block_widths()[0],
                         problem.bk / info.get_block_widths()[1]};
  }
  // 对于gemm A/B ，wmmaInst在block-level buffer上滑动时，每次清空AB即可。故线程仅需要持有单个wmmaInst所需区域的AB
  info.thread_own_data_size = getThreadOwnDataSize(info, false);
}

void LowerInfoAnalysis::applyRelyGemmAInfo(LowerInfo &info, const GemmProblem &problem,
                                           MMAInstInfo *mma, AffineExpr zero) {
  info.buffer = problem.A;
  auto blockLayout = info.get_block_layout();
  info.base_layout = mma->desc_a;
  info.block_layout = {blockLayout[0], 1};
  info.block_repeat = {problem.bm / info.get_block_widths()[0],
                       problem.bk / info.get_block_widths()[1]};
  info.thread_own_data_size = getThreadOwnDataSize(info, false);
}

void LowerInfoAnalysis::applyGemmBInfo(LowerInfo &info, const GemmProblem &problem,
                                       MMAInstInfo *mma, AffineExpr zero, HWSpecification* hw) {
  info.buffer = problem.B;
  info.mmaInst = mma;
  if(hw->getKind() == HW_KIND_NVIDIA){
    auto blockLayout = info.get_block_layout();
    info.base_layout.thread_creg[0] = 1;
    info.base_layout.warp_repeat[0] = mma->k / info.get_warp_widths()[0];
    info.block_layout = {1, blockLayout[1]};
    info.block_repeat = {problem.bk / info.get_block_widths()[0],
                         problem.bn / info.get_block_widths()[1]};
  }
  else if(hw->getKind() == HW_KIND_DCU) {
    auto blockLayout = info.get_block_layout();
    info.base_layout = mma->desc_b;
    info.block_layout = {1, blockLayout[1]};
    info.block_repeat = {problem.bk / info.get_block_widths()[0],
                         problem.bn / info.get_block_widths()[1]};
  }
  // 对于gemm A/B ，wmmaInst在block-level buffer上滑动时，每次清空AB即可。故线程仅需要持有计算单个wmmaInst所需的A/B （最少持有）
  info.thread_own_data_size = getThreadOwnDataSize(info, false);
}

bool LowerInfoAnalysis::inferCopyOp(Operation *op, LowerInfoMap &buf_info_maps) {
  // copyop : 根据 src dst的一方推定 另一方
  if (auto copyOp = dyn_cast<CopyOp>(op)) {
    Value dst = copyOp.getDstMemRef();
    Value src = copyOp.getSrcMemRef();
    auto dstInfo = getNearestInferedInfoEither(buf_info_maps, dst, op);
    auto srcInfo = getNearestInferedInfoEither(buf_info_maps, src, op);
    LowerInfo *sourceInfo = dstInfo != nullptr ? dstInfo : srcInfo;

    auto isLowerInfoOKForCalculate = [](Value buffer, LowerInfo& info){
      auto memShape = mlir::cast<MemRefType>(buffer.getType()).getShape();
      int64_t required_sz = 1;
      int64_t info_thead_own_sz = 1;
      for(auto dim : memShape){
        required_sz *= dim;
      }
      for(auto dim : info.thread_own_data_size){
        info_thead_own_sz *= dim;
      }
      // 比较完成计算所需的 thread 数据量和 LowerInfo中指定的 thread 持有数据量
      if(required_sz / LowerInfoAnalysis::block_threads > info_thead_own_sz){
        // info指定的线程数据量偏少。需要进一步repeat才能满足 op 计算需要
        info.get_warp_repeat();
      }
      else{
        // 数据量充足
        return true;
      }

    };

    if (sourceInfo != nullptr) {
      LowerInfo source = *sourceInfo;



      if (buf_info_maps.getLowerInfo(src, op) == nullptr) {
        LowerInfo i = source;
        i.buffer = src;
        buf_info_maps.addLowerInfo(op, i);
      }
      if (buf_info_maps.getLowerInfo(dst, op) == nullptr) {
        LowerInfo i = source;
        i.buffer = dst;
        buf_info_maps.addLowerInfo(op, i);
      }
      return true;
    }
    LLVM_OUT_MSG("---- inferCopyOp error");
    return false;
  }
  return false;
}

bool LowerInfoAnalysis::inferBlockOp(Operation *op, LowerInfoMap &buf_info_maps) {
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
    info = getNearestInferedInfoEither(buf_info_maps, lbuf, op);
    if (info != nullptr && get_main_info_func(lbuf)) {
      break;
    }
  }
  // loadbufs 均没推断
  auto storeLowerInfo = getNearestInferedInfoEither(buf_info_maps, store_buf, op);
  if (info == nullptr && storeLowerInfo != nullptr && get_main_info_func(store_buf)) {
    info = storeLowerInfo;
  }
  if (info == nullptr) {
    LLVM_OUT_MSG("---- blockOp::loadbufs storebufs 均没推定");
    return false;
  }
  // 根据 load storebuf 的一方推定另一方
  LowerInfo sourceInfo = *info;
  // 补充： checkCompatibility(sourceInfo) 检查适配性
  // blockOp : 描述pointWise的op。特征为：outBuffer上，每个点的数据互相独立，不可替代。
  //  storeOp buffer：输出。 每个线程需完整持有结果的一部分。数据规模必须等于 m=总规模/线程数
  //  loadOp buffer：输入。 每个线程持有的数据量不必等于 m. 可以为 m/(I*J),  配合 寄存器复用+for IJ 循环完成 store buffer的覆盖
  //  取决于buffer是否已有 Lowerinfo信息
  auto checkCompatibleAndInfer = [](Operation* op, LowerInfo& info, Value targetBuffer, bool isStoreBuffer, LowerInfoMap& infoMap){
    auto memType = mlir::cast<MemRefType>(targetBuffer.getType());
    auto shape = memType.getShape();
    auto [infoThreadData0, infoThreadData1] = info.get_thread_own_data_size();

    auto [wl0, wl1] = info.get_warp_layout();  // warp_threads
    auto [bl0, bl1] = info.get_block_layout();  // block_warps
    auto safeDiv = [](int64_t lhs, int64_t rhs) -> int64_t {
      return rhs == 0 ? 0 : lhs / rhs;
    };
    int64_t required_sz0 = safeDiv(shape[0], wl0 * bl0);  // 分到每个线程的计算量：0轴
    int64_t required_sz1 = safeDiv(shape[1], wl1 * bl1);  // 分到每个线程的计算量：1轴
    int64_t required_all = safeDiv(shape[0] * shape[1], wl0 * bl0 * wl1 * bl1);  // 线程计算量总数
    int64_t infoThreadData_all = infoThreadData0 * infoThreadData1;
    if(isStoreBuffer){  // 推定的为输出buffer
      if(required_sz0 !=0 && required_sz1 != 0
          && required_sz0 == infoThreadData0
          && required_sz1 == infoThreadData1
      ){
        // 完全一致, 可直接推定
        llvm::outs() << "完全一致!\n"; llvm::outs().flush();
        LowerInfo targetInfo = info; targetInfo.buffer = targetBuffer;
        infoMap.addLowerInfo(op, targetInfo);
      }
      else if(required_all == infoThreadData_all){
        // 数据量一致，布局不同。可按照 info指定的访问模式 覆盖buffer
        llvm::outs() << "数据量一致，布局不同!\n"; llvm::outs().flush();
        assert(false);  // 理论上，不应出现这种状况
      }
      else if(required_all > infoThreadData_all){
        // 数据量不同。info指定的访问模式无法完全覆盖buffer的计算任务，需要增大每个thread的数据持有量 （用 info 的 thread_own_buffer做下平铺）
        int64_t k0 = safeDiv(required_sz0, infoThreadData0);
        int64_t k1 = safeDiv(required_sz1, infoThreadData1);
        llvm::outs() << "需增加info.thread_own 持有量. (k0,k1) = " << k0 <<"," << k1 << " | br " <<
          info.block_repeat[0] << "," << info.block_repeat[1] << "\n";llvm::outs().flush();

        LowerInfo targetInfo = info; targetInfo.buffer = targetBuffer;
        // 扩大寄存器数目， 缩减 block_repeat 次数
        info.thread_own_data_size = {required_sz0, required_sz1};
        info.block_repeat[0] /= k0;
        info.block_repeat[1] /= k1;
        infoMap.addLowerInfo(op, targetInfo);
      }
      else{
        int64_t k0 = safeDiv(infoThreadData0, required_sz0);
        int64_t k1 = safeDiv(infoThreadData1, required_sz1);
        llvm::outs() << "需降低 info.thread_own 持有量. (k0,k1)= " << k0<<","<<k1 << "\n";llvm::outs().flush();
        LowerInfo targetInfo = info; targetInfo.buffer = targetBuffer;
        infoMap.addLowerInfo(op, targetInfo);
      }
    }
    else{  // 推定输入buffer
      if(required_sz0 !=0 && required_sz1 != 0
          && required_sz0 == infoThreadData0
          && required_sz1 == infoThreadData1
      ){
        // 完全一致, 可直接推定
        llvm::outs() << "完全一致!\n"; llvm::outs().flush();
        LowerInfo targetInfo = info; targetInfo.buffer = targetBuffer;
        infoMap.addLowerInfo(op, targetInfo);
      }
      else if(required_all == infoThreadData_all){
        // 数据量一致，布局不同。可按照 info指定的访问模式 覆盖buffer
        llvm::outs() << "数据量一致，布局不同!\n"; llvm::outs().flush();
        assert(false);  // 理论上，不应出现这种状况
      }
      else if(required_all > infoThreadData_all){
        // 数据量不同。info指定的访问模式无法完全覆盖buffer的计算任务，
        // 可用 block_repeat for循环覆盖整个buffer，单个线程持有数据不变。后续IR 生成时需要协调inBuffer  outBuffer的循环体结构(block_repeat 不相同)
        /*
          for(brA0,brA1){
            subA = getThreadElements(A[brA0,brA1]); 
            smallTile = someOp(subA);
            store(smallTile, outCReg[brA0, brA1 对应的小分区]);  // 多次迭代，将小tile 分次放入 outC的 reg
          }
          对blockOp，inBuffer和outBuffer一般形状相同。 生成for循环时，只需要按照 block_repeat 大的
          之后 outBuffer 遵照输入buffer进行访问即可
        */
        // 或者提高单个线程持有元素个数，减少 block_repeat 次数
        int64_t k0 = safeDiv(required_sz0, infoThreadData0);
        int64_t k1 = safeDiv(required_sz1, infoThreadData1);
        llvm::outs() << "需增加info.thread_own 持有量. (k0,k1) = " << k0 <<"," << k1 << " | br " <<
          info.block_repeat[0] << "," << info.block_repeat[1] << "\n";llvm::outs().flush();
        LowerInfo targetInfo = info; targetInfo.buffer = targetBuffer;
        infoMap.addLowerInfo(op, targetInfo);
      }
      else{
        int64_t k0 = safeDiv(infoThreadData0, required_sz0);
        int64_t k1 = safeDiv(infoThreadData1, required_sz1);
        llvm::outs() << "info.thread_own 持有量超过需求 ,info: " << infoThreadData0  << "," << infoThreadData1 << " | require " << required_sz0 << ","<< required_sz1 << "\n";llvm::outs().flush();
        LowerInfo targetInfo = info; targetInfo.buffer = targetBuffer;
        infoMap.addLowerInfo(op, targetInfo);
      }
    }
  };

  for (const Value &buf : load_bufs) {
    auto info = buf_info_maps.getLowerInfo(buf, op);
    if (info == nullptr) {
      auto bufInfo = sourceInfo;
      checkCompatibleAndInfer(op, sourceInfo, buf, false, buf_info_maps); 
    }
  }
  auto store_info = buf_info_maps.getLowerInfo(store_buf, op);
  if (store_info == nullptr) {
    checkCompatibleAndInfer(op, sourceInfo, store_buf, true, buf_info_maps);
  }
  return true;
}

bool LowerInfoAnalysis::checkGemmProblem(LowerInfoAnalysis::GemmProblem p, HWSpecification* hw){
  // TODO : 根据 hw属性检查GEMM 是否符合硬件要求
  return true;
}

bool LowerInfoAnalysis::inferGemmOp(Operation *op, LowerInfoMap &buf_info_maps,
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
  llvm::outs() <<"mma_inst - " << instInfo->name.c_str() << "\n"; llvm::outs().flush();

  LowerInfo ic = makeDirectGemmCInfo(b, problem, instInfo, thread_num, hw, block_layout);
  ic.mmaInst = instInfo;
  buf_info_maps.addLowerInfo(op, ic);

  auto zero = b.getAffineConstantExpr(0);
  LowerInfo aInfo = ic;
  LowerInfo bInfo = ic;
  applyDirectGemmAInfo(aInfo, problem, instInfo, zero, hw);
  buf_info_maps.addLowerInfo(op, aInfo);
  applyGemmBInfo(bInfo, problem, instInfo, zero,hw);
  buf_info_maps.addLowerInfo(op, bInfo);
  return true;
}

bool LowerInfoAnalysis::inferRelyGemmOp(Operation *op, LowerInfoMap &buf_info_maps,
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

  auto zero = b.getAffineConstantExpr(0);
  // 若A或B已经推断：根据A或B的info得到C的info；否则C的info从 map里找
  auto infoA = getNearestInferedInfoEither(buf_info_maps, problem.A, op);
  auto infoB = getNearestInferedInfoEither(buf_info_maps, problem.B, op);
  if(infoA != nullptr){
    // A 已经推断, A->C
    LowerInfo sourceA = *infoA;
    if (buf_info_maps.getLowerInfo(problem.A, op) == nullptr) {
      sourceA.buffer = problem.A;
      buf_info_maps.addLowerInfo(op, sourceA);
    }
    auto ic = makeRelyGemmCInfo(b, problem, mma, thread_num, hw, sourceA, true);
    buf_info_maps.addLowerInfo(op, ic);
    // 检查B是否已推断
    if(infoB == nullptr){
      // C->B
      ic.buffer = problem.B;
      infoB = buf_info_maps.addLowerInfo(op, ic);
      applyGemmBInfo(*infoB, problem, mma, zero, hw);
    } else if (buf_info_maps.getLowerInfo(problem.B, op) == nullptr) {
      LowerInfo bInfo = *infoB;
      bInfo.buffer = problem.B;
      buf_info_maps.addLowerInfo(op, bInfo);
    }
    return true;
  }
  else{
    // A没被推断。检查B是否已推断
    if(infoB != nullptr){
      // B->C
      LowerInfo sourceB = *infoB;
      if (buf_info_maps.getLowerInfo(problem.B, op) == nullptr) {
        sourceB.buffer = problem.B;
        buf_info_maps.addLowerInfo(op, sourceB);
      }
      auto ic = makeRelyGemmCInfo(b, problem, mma, thread_num, hw, sourceB, false);
      buf_info_maps.addLowerInfo(op, ic);
      // C->A
      ic.buffer = problem.A;
      infoA = buf_info_maps.addLowerInfo(op, ic);
      applyRelyGemmAInfo(*infoA, problem, mma, zero);
      return true;
    }
  }
  // AB 均无推断
  return false;
}

// reduce : 根据src推定dst的layout
bool LowerInfoAnalysis::inferReduceOp(Operation *op, LowerInfoMap &buf_info_maps) {
  auto reduceOp = dyn_cast<ReduceOp>(op);
  if (!reduceOp) {
    return false;
  }


  Value dst = reduceOp.getDst();
  Value src = reduceOp.getSrc();
  uint64_t dim = reduceOp.getDim();
  auto srcInfo = buf_info_maps.getNearestInferedInfo(src, op);
  if (srcInfo == nullptr) {
    LLVM_OUT_MSG("---- inferError 4");
    return false;
  }
  // 根据src 推断 dst
  LowerInfo dstInfo = *srcInfo;
  dstInfo.buffer = dst;
  auto _dstInfo = buf_info_maps.addLowerInfo(op, dstInfo);

  auto erase_i64_dim_func = [&](std::array<int64_t, 2> &vec) -> bool {
    if (dim >= vec.size()) return false;
    for (size_t i = dim; i + 1 < vec.size(); ++i) {
      vec[i] = vec[i + 1];
    }
    vec.back() = 1;
    return true;
  };

  if (!erase_i64_dim_func(_dstInfo->base_layout.warp_layout) ||
      !erase_i64_dim_func(_dstInfo->base_layout.warp_layout_order) ||
      !erase_i64_dim_func(_dstInfo->base_layout.thread_creg) ||
      !erase_i64_dim_func(_dstInfo->base_layout.thread_creg_order) ||
      !erase_i64_dim_func(_dstInfo->base_layout.warp_repeat) ||
      !erase_i64_dim_func(_dstInfo->base_layout.warp_repeat_order) ||
      !erase_i64_dim_func(_dstInfo->block_layout) ||
      !erase_i64_dim_func(_dstInfo->block_layout_order) ||
      !erase_i64_dim_func(_dstInfo->block_repeat) ||
      !erase_i64_dim_func(_dstInfo->thread_own_data_size)) {
    LLVM_OUT_MSG("---- inferError 5");
    return false;
  }
  return true;
}

bool LowerInfoAnalysis::inferDirectOp(Operation *op, LowerInfoMap &buf_info_maps,
                                      HWSpecification *hw) {
  if (auto gemmOp = dyn_cast<GemmOp>(op)) {
    (void)gemmOp;
    return inferGemmOp(op, buf_info_maps, hw);
  }
  return false;
}

bool LowerInfoAnalysis::inferRelyOp(Operation *op, LowerInfoMap &buf_info_maps, HWSpecification *hw) {
  // 提取op的所有memref 参数
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
  // 检查是否已经推断过
  bool all_in = true;
  bool has_anchor = false;
  size_t count = 0;
  for (const Value &memref : memrefsToCheck) {
    // 检查op的buffer是否已被推定过
    auto info = buf_info_maps.getLowerInfo(memref, op);
    // 检查buffer是否在op前后已经存在可传播的锚点
    auto lastInfo = getNearestInferedInfoEither(buf_info_maps, memref, op);
    if (info == nullptr) {
      all_in = false;
      count++;
    }
    if(lastInfo != nullptr){
      has_anchor = true;
    }
  }
  if (all_in) {
    LLVM_OUT_MSG("buf已全部推断");
    return true;
  }
  if (!all_in && count == memrefsToCheck.size() && !has_anchor) {
    LLVM_OUT_MSG("无已推断buf, 需要gemmOp做锚点");
    return false;
  }

  if (inferCopyOp(op, buf_info_maps)) {
    return true;
  }
  if (inferBlockOp(op, buf_info_maps)) {
    return true;
  }
  // notes: 修改infer逻辑后 这里待商榷
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

LowerInfoMap LowerInfoAnalysis::buf_info_maps {};
int LowerInfoAnalysis::block_threads = 0;
/**
 * @brief lowinfo 推断
 目的：以kernel中的首个gemm为出发点，向两侧推断线程应该持有的寄存器buffer形状。尽可能减少算子之间 local->shm 的写回
 block中，block_layout 和 warp_layout 按照 H100 gemm tensorcore 的计算规则固定
 * 
 * @param kernelOp 
 * @param hwKind : dcu,nvidia 
 * @return DenseMap<Value, LowerInfo> 
 */
LowerInfoMap* LowerInfoAnalysis::run(mlir::Operation* kernelOp, const std::string& hwKind ,const std::string& version){
  auto hw = GetHWSpecification(hwKind, version, kernelOp->getContext());

  buf_info_maps = LowerInfoMap{};
  buf_info_maps.getOpsOrder(kernelOp);

  // 获取kernelOp的 thread_num 属性, 用于判断 lowerInfo中 thread_own_data_sz 是否能满足op计算要求
  LowerInfoAnalysis::block_threads = kernelOp->getAttrOfType<IntegerAttr>("thread_num").getInt();
  llvm::outs() << "LowerInfoAnalysis::block_threads = " << LowerInfoAnalysis::block_threads << "\n"; llvm::outs().flush();

  SmallVector<Operation*, 5> need_infer_ops = collectNeedInferOps(kernelOp);

  auto old_size = need_infer_ops.size();
  // need_infer_ops 中的gemm进行直接推定 
  need_infer_ops.erase(llvm::remove_if(need_infer_ops, [&](mlir::Operation* op){
    return inferDirectOp(op, buf_info_maps, hw);
  }), need_infer_ops.end());
  
  
  if (old_size == need_infer_ops.size()) {
    assert(false && "LowerInfo infer failed. No direct infer anchor op");
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
  return &buf_info_maps;
}

}
