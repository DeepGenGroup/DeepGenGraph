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
#include "mlir/IR/PatternMatch.h"
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
  auto [iwu0, iwu1] = info.warpInstUnroll;
  if (!includeBaseRepeat) {
    return {tw0 * wr0 * iwu0, tw1 * wr1 * iwu1};
  }
  auto [br0, br1] = info.get_block_repeat();
  return {tw0 * wr0 *iwu0 * br0, tw1 * wr1 *iwu1* br1};
}

LowerInfo *LowerInfoMap::getLowerInfo(const mlir::Value &buffer,
                                      mlir::Operation *op) {
  auto it = infoMap.find(std::make_pair(buffer, op));
  if (it != infoMap.end()) {
    return &it->second;
  }
  auto candidateIt = m_candidates.find(buffer);
  if (candidateIt == m_candidates.end()) {
    return nullptr;
  }
  for (auto &candidate : candidateIt->second) {
    if (candidate.op == op) {
      return &candidate;
    }
  }
  return nullptr;
}

static bool isSameLinearLayout(const LinearLayout2DDesc &lhs,
                               const LinearLayout2DDesc &rhs) {
  return lhs.memspace == rhs.memspace &&
         lhs.elementType == rhs.elementType &&
         lhs.warp_layout == rhs.warp_layout &&
         lhs.warp_layout_order == rhs.warp_layout_order &&
         lhs.thread_creg == rhs.thread_creg &&
         lhs.thread_creg_order == rhs.thread_creg_order &&
         lhs.warp_repeat == rhs.warp_repeat &&
         lhs.warp_repeat_order == rhs.warp_repeat_order &&
         lhs.wg_layout == rhs.wg_layout &&
         lhs.wg_layout_order == rhs.wg_layout_order;
}

// 比较两个info间，除了op buffer 外是否相同
static bool isSameLayout(const LowerInfo &lhs, const LowerInfo &rhs) {
  // 是否需要考虑 ignoreDim的情况？
  if(lhs.warp_threads == rhs.warp_threads &&
        isSameLinearLayout(lhs.base_layout, rhs.base_layout) &&
        lhs.block_layout == rhs.block_layout &&
        lhs.block_layout_order == rhs.block_layout_order &&
        lhs.thread_own_data_size == rhs.thread_own_data_size){
    auto br_wu0_x = lhs.block_repeat[0] * lhs.warpInstUnroll[0];
    auto br_wu0_y = lhs.block_repeat[1] * lhs.warpInstUnroll[1];
    auto br_wu1_x = rhs.block_repeat[0] * rhs.warpInstUnroll[0];
    auto br_wu1_y = rhs.block_repeat[1] * rhs.warpInstUnroll[1];
    if(br_wu0_x == br_wu1_x && br_wu0_y == br_wu1_y){
      return true;
    }
  }
  return false;
}

// 比较info的 tod_sz. 如果参数中 ignoreDim有效，则忽略对应维度（此时 ignoreDim 必为 0 或 1）
static int getMaxThreadOwnDatasz(const LowerInfo& lhs, coordXY_t& tod_sz, int ignoreDim){
  auto x = lhs.thread_own_data_size[0] * lhs.thread_own_data_size[1];
  auto y = tod_sz[0] * tod_sz[1];
  if(ignoreDim < 0){
    if(y < x){
      tod_sz = lhs.thread_own_data_size;
    }
  }
  else{
    x = lhs.thread_own_data_size[1-ignoreDim];
    y = tod_sz[1-ignoreDim];
    if(y < x){
      tod_sz[1-ignoreDim] = lhs.thread_own_data_size[1-ignoreDim];
    }
  }
}

static bool isSameCandidate(const LowerInfo &lhs, const LowerInfo &rhs) {
  return lhs.op == rhs.op 
    && lhs.buffer == rhs.buffer 
    && lhs.pos == rhs.pos 
    && isSameLayout(lhs, rhs);
}

// 判断Layout 冲突：op buffer 相同，但数据分布规则不同。视为冲突(不区分 in out， 因为是同一个buffer)
static bool isLowerInfoConflict(const LowerInfo &lhs, const LowerInfo &rhs) {
  // return lhs.op == rhs.op && lhs.buffer == rhs.buffer && !isSameLayout(lhs, rhs);
  return lhs.buffer == rhs.buffer && !isSameLayout(lhs, rhs);
}

void LowerInfoMap::addLowerInfo(mlir::Operation *op, LowerInfo info, bool isConflict) {
  assert(info.buffer != nullptr);
  info.op = op;
  llvm::outs() << "[debug] addLayout : " << op->getName().getStringRef() << " - "<< info.buffer;

  auto &candidates = m_candidates[info.buffer];
  if (!llvm::any_of(candidates, [&](const LowerInfo &candidate) {
        return isSameCandidate(candidate, info);
      })) {
    candidates.push_back(std::move(info));
  }
  llvm::outs() << "\n";
  return;
}

static void recalcLayout(LowerInfo& info, const coordXY_t& new_thread_own_data_sz, unsigned pos){
  // 调整 info.thread_own_data 后，线程持有数据增加
  // 单次指令计算区域仍不变，为 warp_layout * warp_repeat * thread_creg -> inst在buffer上平铺次数没变
  // 线程持有数据多了 -> block_repeat 少了，但每次平铺需要额外 unroll k0*k1 次 inst操作
  // 本质是将 k0*k1 次的inst 所用数据都放进 thread_own_data 里。

  if(pos >= unsigned(LowerInfo::BufPos::Out)){
    // buffer有作为 out参数的时候 ： 需要按最大数据量
    auto k0 = new_thread_own_data_sz[0] / info.thread_own_data_size[0];
    auto k1 = new_thread_own_data_sz[1] / info.thread_own_data_size[1];
    info.thread_own_data_size = new_thread_own_data_sz;
    info.block_repeat[0] /= k0;  // 以 thread_own_data_size 为单位进行的 buffer 平铺次数减少
    info.block_repeat[1] /= k1;
    info.warpInstUnroll = {k0,k1};  // 单次平铺内，需额外进行 {k0,k1} inst 展开以算满 new_thread_own_data_sz
  }
  else{
    // buffer 仅作为 in 参数被读取 : 无需修改LowerInfo。
    llvm::outs() << "[recalcLayout] buffer 仅作为输入。不需修改Layout \n";
  }
}

void LowerInfoMap::conflictResolve() {
  infoMap.clear();
  
  for (auto &entry : m_candidates) {
    Value buffer = entry.getFirst();
    auto &bufferInfoCandidates = entry.getSecond();
    if (bufferInfoCandidates.empty() || bufferInfoCandidates.size() == 1) {
      continue;
    }
    // candidate 按照opOrder 排序
    llvm::sort(bufferInfoCandidates, [&](const LowerInfo &a, const LowerInfo &b) {
      int orderA = opOrder.lookup(a.op);
      int orderB = opOrder.lookup(b.op);
      return orderA < orderB; // 升序排列
    });
    // 判断是否存在冲突，以及BaseLayout 是否不同
    const LowerInfo &selectedInfo = bufferInfoCandidates.front();
    bool hasConflict = llvm::any_of(bufferInfoCandidates, [&](const LowerInfo &candidate) {
      return isLowerInfoConflict(selectedInfo, candidate);
    });
    bool isBaselayoutMismatch = llvm::any_of(bufferInfoCandidates, [&](const LowerInfo &candidate) {
      return !isSameLinearLayout(selectedInfo.base_layout, candidate.base_layout);
    });
    
    if(isBaselayoutMismatch){
      // baselayout 不同。无法协商 —— 需插入 convertLayoutOp。LowerInfo 需注明 needConvertFrom = srcLowerInfo
      LowerInfo* lastInfo = nullptr;
      for(int i=0;i<bufferInfoCandidates.size();++i){
        if(lastInfo == nullptr){
          bufferInfoCandidates[i].convertFrom = nullptr;
          lastInfo = &bufferInfoCandidates[i];
          continue;
        }
        if(isLowerInfoConflict(*lastInfo, bufferInfoCandidates[i])){
          bufferInfoCandidates[i].convertFrom = lastInfo;
          lastInfo = &bufferInfoCandidates[i];
        }
      }
    }
    else{
      if(hasConflict){
        // 看下buffer是否作为op的输入/输出参数。
        unsigned bufferPos = 0;
        for(auto info : bufferInfoCandidates){
          bufferPos |= unsigned(info.pos);
        }
        // 最大尺寸的 thread_own_data_sz 
        coordXY_t max_tod_sz = {0,0};
        for(int i=0;i<bufferInfoCandidates.size();++i){
          int ignoreDim = -1;
          if(mlir::isa<frisk::ReduceOp>(bufferInfoCandidates[i].op)){
            // 当buffer作为 reduceOp的 out buffer时，可忽略 reducedim 的layout数值
            if(bufferInfoCandidates[i].pos >= LowerInfo::BufPos::Out){
              auto realType = mlir::dyn_cast<frisk::ReduceOp>(bufferInfoCandidates[i].op);
              ignoreDim = realType.getDim();
            }
          }
          getMaxThreadOwnDatasz(bufferInfoCandidates[i], max_tod_sz, ignoreDim);
        }
        assert(max_tod_sz[0] > 0);
        assert(max_tod_sz[1] > 0);
        // 根据 max_tod_sz, 调整 buffer对应的所有op下的LowerInfo
        for(auto &info : bufferInfoCandidates){
          if(info.op != nullptr){
            recalcLayout(info, max_tod_sz, bufferPos);
            auto key = std::make_pair(buffer, info.op);
            infoMap.try_emplace(key, info);
          }
        }
      }
    }
    // post check : 检查buffer下所有的info须保持一致
    LowerInfo* prev = nullptr;
    for(int i=0;i<bufferInfoCandidates.size();++i){
      if(prev== nullptr){
        prev = &bufferInfoCandidates[i]; 
        continue;
      }
      assert(prev->buffer == bufferInfoCandidates[i].buffer);
      if(isLowerInfoConflict(*prev, bufferInfoCandidates[i])){
        if(bufferInfoCandidates[i].convertFrom == nullptr){
          bufferInfoCandidates[i].show("crashed");
          prev->show("anchor");
          llvm::outs().flush(); assert(false);
        }
        prev = &bufferInfoCandidates[i];
        continue;
      }
      if(bufferInfoCandidates[i].convertFrom == nullptr){
        if(isLowerInfoConflict(*prev, bufferInfoCandidates[i])){
          bufferInfoCandidates[i].show("crashed");
          prev->show("anchor");
          llvm::outs().flush(); assert(false);
        }
      }
    }
  }
  // 结果整理
  for(auto& ent : m_candidates){
    auto buffer = ent.getFirst();
    auto &lowerInfos = ent.getSecond();
    for(auto& info : lowerInfos){
      // infoMap.insert(const std::pair<std::pair<mlir::Value, mlir::Operation *>, mlir::frisk::LowerInfo> &KV)
      infoMap.insert({{info.buffer, info.op}, info});
    }
  }
  // m_candidates.clear();
}

const SmallVector<Operation*>& LowerInfoMap::getOpsOrder(mlir::Operation* rootNode){
  if(opOrder.empty()){
    opOrderVec.push_back(nullptr);
    unsigned idx = 1;
    rootNode->walk<WalkOrder::PreOrder>([&](mlir::Operation* subOp){
      opOrder.try_emplace(subOp, idx++);
      opOrderVec.push_back(subOp);
    });
  }
  return opOrderVec;
}

void LowerInfoMap::print(){
  for(auto[k,lowInfo] : infoMap){
    auto [value,operation] = k;
    lowInfo.show( operation->getName().getStringRef().data() );
  }
}

// 获取距离 currOp 前向/后向 的已经推定的 LowerInfo
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
  auto updateNearest = [&](Value buf, Operation *op, LowerInfo &info) {
    auto opOrderIt = opOrder.find(op);
    if (opOrderIt == opOrder.end()) {
      return;
    }
    unsigned order = opOrderIt->second;
    if (buffer != buf) {
      return;
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
  };

  for (auto &entry : infoMap) {
    updateNearest(entry.first.first, entry.first.second, entry.second);
  }
  for (auto &entry : m_candidates) {
    Value buf = entry.getFirst();
    for (auto &candidate : entry.getSecond()) {
      updateNearest(buf, candidate.op, candidate);
    }
  }
  return nearestInfo;
}

static LowerInfo *getNearestInferedInfoEither(LowerInfoMap &infoMap,
                                              const mlir::Value &buffer,
                                              mlir::Operation *currOp,
                                              bool preferBefore = true) {
  if (auto *info = infoMap.getNearestInferedInfo(buffer, currOp, preferBefore)) {
    return info;
  }
  return infoMap.getNearestInferedInfo(buffer, currOp, !preferBefore);
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
                         problem.bn / info.get_block_widths()[1]};  // 这里的 block_repeat 可能不准。因为 block_layout 可根据conflict动态调整
  }
  // 对于gemmC，单个线程持有的数据应考虑wmmaInst在block-level buffer上滑动的情况。每次计算得到一个inst区域的C
  info.thread_own_data_size = getThreadOwnDataSize(info, true);
  info.pos = LowerInfo::BufPos::Out;
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
  info.pos = LowerInfo::BufPos::Out;
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
  info.pos = LowerInfo::BufPos::In;
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
  info.pos = LowerInfo::BufPos::In;
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
  info.pos = LowerInfo::BufPos::In;
}

bool LowerInfoAnalysis::inferCopyOp(Operation *op, LowerInfoMap &buf_info_maps,
                                    bool preferBefore) {
  // copyop : 根据 src dst的一方推定 另一方
  if (auto copyOp = dyn_cast<CopyOp>(op)) {
    Value dst = copyOp.getDstMemRef();
    Value src = copyOp.getSrcMemRef();
    auto dstInfo = getNearestInferedInfoEither(buf_info_maps, dst, op, preferBefore);
    auto srcInfo = getNearestInferedInfoEither(buf_info_maps, src, op, preferBefore);
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
      LowerInfo srcCandidate = source;
      srcCandidate.buffer = src;
      srcCandidate.pos = LowerInfo::BufPos::In;
      buf_info_maps.addLowerInfo(op, srcCandidate);

      LowerInfo dstCandidate = source;
      dstCandidate.buffer = dst;
      dstCandidate.pos = LowerInfo::BufPos::Out;
      buf_info_maps.addLowerInfo(op, dstCandidate);
      return true;
    }
    LLVM_OUT_MSG("---- inferCopyOp error");
    return false;
  }
  return false;
}

bool LowerInfoAnalysis::inferBlockOp(Operation *op, LowerInfoMap &buf_info_maps,
                                     bool preferBefore) {
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
    info = getNearestInferedInfoEither(buf_info_maps, lbuf, op, preferBefore);
    if (info != nullptr && get_main_info_func(lbuf)) {
      break;
    }
  }
  // loadbufs 均没推断
  auto storeLowerInfo = getNearestInferedInfoEither(buf_info_maps, store_buf, op, preferBefore);
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
    auto [infoThreadData0, infoThreadData1] = info.get_thread_own_data_size();  // base

    auto [wl0, wl1] = info.get_warp_layout();  // warp_threads
    auto [bl0, bl1] = info.get_block_layout();  // block_warps
    auto safeDiv = [](int64_t lhs, int64_t rhs) -> int64_t {
      return rhs == 0 ? 0 : lhs / rhs;
    };
    
    int64_t required_sz0 = safeDiv(shape[0], wl0 * bl0);  // 分到每个线程的计算量：0轴
    int64_t required_sz1 = safeDiv(shape[1], wl1 * bl1);  // 分到每个线程的计算量：1轴
    int64_t required_all = safeDiv(shape[0] * shape[1], wl0 * bl0 * wl1 * bl1);  // 线程计算量总数
  
    assert(required_all > 0);
    if(required_sz0 == 0){ required_sz0 = 1;}  // 至少计算一个元素
    if(required_sz1 == 0){ required_sz1 = 1;}

    int64_t infoThreadData_all = infoThreadData0 * infoThreadData1;
    if(isStoreBuffer){  // 推定的为输出buffer
      if(required_sz0 !=0 && required_sz1 != 0
          && required_sz0 == infoThreadData0
          && required_sz1 == infoThreadData1
      ){
        // 完全一致, 可直接推定
        llvm::outs() << "完全一致!\n"; llvm::outs().flush();
        LowerInfo targetInfo = info; targetInfo.buffer = targetBuffer;
        targetInfo.pos = LowerInfo::BufPos::Out;
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
        targetInfo.thread_own_data_size = {required_sz0,required_sz1};
        targetInfo.pos = LowerInfo::BufPos::Out;
        infoMap.addLowerInfo(op, targetInfo, true);  // isCOnflict=true 表示该buffer的Lowerinfo需要协商一致。放在conflict里以待后续检查
      }
      else{
        int64_t k0 = safeDiv(infoThreadData0, required_sz0);
        int64_t k1 = safeDiv(infoThreadData1, required_sz1);
        llvm::outs() << "需降低 info.thread_own 持有量. (k0,k1)= " << k0<<","<<k1 << "\n";llvm::outs().flush();
        LowerInfo targetInfo = info; targetInfo.buffer = targetBuffer;
        targetInfo.thread_own_data_size = {required_sz0, required_sz1};
        targetInfo.pos = LowerInfo::BufPos::Out;
        infoMap.addLowerInfo(op, targetInfo, true);
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
        targetInfo.pos = LowerInfo::BufPos::In;
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
        targetInfo.thread_own_data_size = {required_sz0, required_sz1};
        targetInfo.pos = LowerInfo::BufPos::In;
        infoMap.addLowerInfo(op, targetInfo, true);
      }
      else{
        int64_t k0 = safeDiv(infoThreadData0, required_sz0);
        int64_t k1 = safeDiv(infoThreadData1, required_sz1);
        llvm::outs() << "info.thread_own 持有量超过需求 ,info: " << infoThreadData0  << "," << infoThreadData1 << " | require " << required_sz0 << ","<< required_sz1 << "\n";llvm::outs().flush();
        LowerInfo targetInfo = info; targetInfo.buffer = targetBuffer;
        targetInfo.thread_own_data_size = {required_sz0, required_sz1};
        targetInfo.pos = LowerInfo::BufPos::In;
        infoMap.addLowerInfo(op, targetInfo, true);
      }
    }
  };

  for (const Value &buf : load_bufs) {
    LowerInfo candidateInfo = sourceInfo;
    checkCompatibleAndInfer(op, candidateInfo, buf, false, buf_info_maps); 
  }
  LowerInfo candidateInfo = sourceInfo;
  checkCompatibleAndInfer(op, candidateInfo, store_buf, true, buf_info_maps);
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
                                        HWSpecification *hw, bool preferBefore) {
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
  auto infoA = getNearestInferedInfoEither(buf_info_maps, problem.A, op, preferBefore);
  auto infoB = getNearestInferedInfoEither(buf_info_maps, problem.B, op, preferBefore);
  if(infoA != nullptr){
    // A 已经推断, A->C
    LowerInfo sourceA = *infoA;
    sourceA.buffer = problem.A;
    buf_info_maps.addLowerInfo(op, sourceA);
    auto ic = makeRelyGemmCInfo(b, problem, mma, thread_num, hw, sourceA, true);
    buf_info_maps.addLowerInfo(op, ic);
    // C->B
    LowerInfo bInfo = ic;
    applyGemmBInfo(bInfo, problem, mma, zero, hw);
    buf_info_maps.addLowerInfo(op, bInfo);
    return true;
  }
  else{
    // A没被推断。检查B是否已推断
    if(infoB != nullptr){
      // B->C
      LowerInfo sourceB = *infoB;
      sourceB.buffer = problem.B;
      buf_info_maps.addLowerInfo(op, sourceB);
      auto ic = makeRelyGemmCInfo(b, problem, mma, thread_num, hw, sourceB, false);
      buf_info_maps.addLowerInfo(op, ic);
      // C->A
      LowerInfo aInfo = ic;
      applyRelyGemmAInfo(aInfo, problem, mma, zero);
      buf_info_maps.addLowerInfo(op, aInfo);
      return true;
    }
  }
  // AB 均无推断
  return false;
}

// reduce : 根据src推定dst的layout
// 将 src切分到block threads上。每个线程持有几个元素，先做局部reduce
// 之后通过shuffle 做warp内的跨线程通信 （对new & old值做运算 (sum, max 等)）
bool LowerInfoAnalysis::inferReduceOp(Operation *op, LowerInfoMap &buf_info_maps,
                                      bool preferBefore) {
  auto reduceOp = dyn_cast<ReduceOp>(op);
  if (!reduceOp) {
    return false;
  }

  Value dst = reduceOp.getDst();
  Value src = reduceOp.getSrc();
  uint64_t dim = reduceOp.getDim();
  auto srcInfo = getNearestInferedInfoEither(buf_info_maps, src, op, preferBefore);
  if (srcInfo == nullptr) {
    LLVM_OUT_MSG("---- inferError 4");
    return false;
  }
  // 检查srcInfo
  auto srcType = mlir::cast<MemRefType>(src.getType());
  auto dstType = mlir::cast<MemRefType>(dst.getType());
  assert(srcType.getMemorySpaceAsInt() == int(friskMs::Shared));
  assert(dstType.getMemorySpaceAsInt() == int(friskMs::Shared));
  auto shape = srcType.getShape();
  auto required_src = shape[0] * shape[1] / LowerInfoAnalysis::block_threads;
  
  auto required_sz0 = shape[0] / (srcInfo->get_warp_layout()[0] * srcInfo->get_block_layout()[0]);
  auto required_sz1 = shape[1] / (srcInfo->get_warp_layout()[1] * srcInfo->get_block_layout()[1]);

  // 根据src 推断 dst
  LowerInfo _dstInfo = *srcInfo;
  _dstInfo.pos = LowerInfo::BufPos::Out;
  _dstInfo.buffer = dst;
  _dstInfo.ignoreDim = dim;
  _dstInfo.thread_own_data_size[0] = required_sz0;
  _dstInfo.thread_own_data_size[1] = required_sz1;
  _dstInfo.thread_own_data_size[dim] = 1; 

  buf_info_maps.addLowerInfo(op, _dstInfo);
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

bool LowerInfoAnalysis::inferRelyOp(Operation *op, LowerInfoMap &buf_info_maps,
                                    HWSpecification *hw, bool collectConflict,
                                    bool preferBefore) {
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
    auto lastInfo = getNearestInferedInfoEither(buf_info_maps, memref, op, preferBefore);
    if (info == nullptr) {
      all_in = false;
      count++;
    }
    if(lastInfo != nullptr){
      has_anchor = true;
    }
  }
  if (memrefsToCheck.empty()) {
    return true;
  }
  if (all_in && !collectConflict) {
    LLVM_OUT_MSG("buf已全部推断");
    return true;
  }
  if (!all_in && count == memrefsToCheck.size() && !has_anchor) {
    LLVM_OUT_MSG("无已推断buf, 需要gemmOp做锚点");
    return false;
  }

  if (inferCopyOp(op, buf_info_maps, preferBefore)) {
    return true;
  }
  if (inferBlockOp(op, buf_info_maps, preferBefore)) {
    return true;
  }
  // notes: 修改infer逻辑后 这里待商榷
  if (inferRelyGemmOp(op, buf_info_maps, hw, preferBefore)) {
    return true;
  }
  if (inferReduceOp(op, buf_info_maps, preferBefore)) {
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

// TODO : 建立 LowerInfo之间的双向链表，明确推断链条。当后面的LowerInfo发生冲突，需要修改，可直接传播到同链条的所有 info
// 
LowerInfoMap* LowerInfoAnalysis::run(mlir::Operation* kernelOp, const std::string& hwKind ,const std::string& version){
  auto hw = GetHWSpecification(hwKind, version, kernelOp->getContext());

  buf_info_maps = LowerInfoMap{};
  const auto& opOrderVec = buf_info_maps.getOpsOrder(kernelOp);
  
  // 获取kernelOp的 thread_num 属性, 用于判断 lowerInfo中 thread_own_data_sz 是否能满足op计算要求
  LowerInfoAnalysis::block_threads = kernelOp->getAttrOfType<IntegerAttr>("thread_num").getInt();
  llvm::outs() << "LowerInfoAnalysis::block_threads = " << LowerInfoAnalysis::block_threads << "\n"; llvm::outs().flush();

  SmallVector<Operation*, 5> need_infer_ops = collectNeedInferOps(kernelOp);

  auto old_size = need_infer_ops.size();
  // need_infer_ops 中的gemm进行直接推定 
  need_infer_ops.erase(llvm::remove_if(need_infer_ops, [&](mlir::Operation* op){
    return inferDirectOp(op, buf_info_maps, hw);
  }), need_infer_ops.end());
  
  SmallVector<int> gemmOpIds {};
  for(int i=0;i<opOrderVec.size();++i){
    auto op = opOrderVec[i];
    if(op != nullptr && mlir::isa<frisk::GemmOp>(op)){
      gemmOpIds.push_back(i);
    }
  }
  
  if (old_size == need_infer_ops.size()) {
    assert(false && "LowerInfo infer failed. No direct infer anchor op");
  }
  
  // 以 gemmOpIds 为锚点，前向/后向推断 LowerInfo。
  // 根据 op 性质从已知 buffer 推定未知 buffer；已存在的不同候选由 addLowerInfo
  // 放入 m_candidates，最后通过 conflictResolve() 统一处理。

  auto isInferTargetOp = [](Operation *op) {
    return op != nullptr && isa<CopyOp, BlockOp, GemmOp, ReduceOp>(op);
  };
  auto tryInferAt = [&](int opId, bool collectConflict, bool preferBefore) -> bool {
    if (opId <= 0 || opId >= static_cast<int>(opOrderVec.size())) {
      return false;
    }
    Operation *op = opOrderVec[opId];
    if (!isInferTargetOp(op)) {
      return false;
    }
    auto inferSuccess = inferRelyOp(op, buf_info_maps, hw, collectConflict, preferBefore);
    if(inferSuccess){
      auto it =llvm::find_if(need_infer_ops, [&](Operation* _op){
        return _op != nullptr && _op == op;
      });
      if(it != need_infer_ops.end()){
        need_infer_ops.erase(it);
      }
    }
    return inferSuccess;
  };

  // gemm 作为 anchor，分别向前、向后传播 Layout。
  int lastOpId = static_cast<int>(opOrderVec.size()) - 1;
  for (int anchorIdx = 0; anchorIdx < static_cast<int>(gemmOpIds.size()); ++anchorIdx) {
    int gemmId = gemmOpIds[anchorIdx];
    int leftBound = anchorIdx == 0 ? 1 : gemmOpIds[anchorIdx - 1] + 1;
    int rightBound = anchorIdx + 1 == static_cast<int>(gemmOpIds.size())
                         ? lastOpId
                         : gemmOpIds[anchorIdx + 1] - 1;

    for (int i = gemmId - 1; i >= leftBound; --i) {
      tryInferAt(i, /*collectConflict=*/true, /*preferBefore=*/false);
    }
    for (int i = gemmId + 1; i <= rightBound; ++i) {
      tryInferAt(i, /*collectConflict=*/true, /*preferBefore=*/true);
    }
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
  // 冲突解决
  buf_info_maps.conflictResolve();

  return &buf_info_maps;
}

}
