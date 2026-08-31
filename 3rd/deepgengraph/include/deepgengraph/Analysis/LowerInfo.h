#ifndef FRISK_ANALYSIS_INFERLOWERINFO_H
#define FRISK_ANALYSIS_INFERLOWERINFO_H

#include "deepgengraph/Analysis/HardwareSpecification.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <cstdint>
#include <utility>
#include <vector>
#include "deepgengraph/Dialect/Frisk/Utils/Utils.h"
#include "mlir/Support/LLVM.h"

struct HWSpecification;

namespace mlir::frisk {

#define TID  "threadIdx"

static const char* WARP_LABELS[] = {"iv_warpX", "iv_warpY"};
static const char* THREAD_LABELS[] = {"iv_threadX", "iv_threadY"};
static const char* BLOCK_LABELS[] = {"iv_blockX", "iv_blockY"};

class LowerInfoAnalysis ;
/**
 * @brief LowerInfo
 * 其表示了 在block层面的一块buffer，在降低到线程层面后，线程如何从 block-level的buffer里 根据自己的tid 去RW 该buffer里的数据（即索引[x,y]）
[x,y] 可通过不同级别的 loop_iv （block_repeat, warp_repeat, thread_width） 配合 wid laneid 算出来
 */
class LowerInfo {
  friend LowerInfoAnalysis;
public:
  enum class BufPos : int {
    In = 1,  // buffer作为op输入参数  0b01
    Out = 2  // buffer作为op输出参数  0b10
  };
  Value buffer = nullptr;
  mlir::Operation* op = nullptr;
  int warp_threads;
  BufPos pos = LowerInfo::BufPos::In;
  coordXY_t warpInstUnroll = {1,1};
  int ignoreDim = -1;
  LowerInfo* convertFrom = nullptr;  // 表示该Layout使用前，需要添加 LayoutConvertOp，从 convertFrom Layout转换到到自己（即：reg->shm->reg）

public:
  explicit LowerInfo(int _warp_threads);
  MMAInstInfo*  mmaInst = nullptr;

/**
  * 字段说明
//  $![截图](/data2/xsl/DeepGenGraph/image_comments/LowerInfo.png)

上述布局示例中，warp_inst 对应单个warp级别指令（如wmma）构成的基础计算区域，称为 base_layout. 该部分的布局完全由硬件决定。一般手册中已经固定了访问模式
- base_layout中: thread_creg+order, warp_layout+order, warp_repeat+order 唯一确定一个base_Layout 布局
    其中： thread_creg = thread 计算的连续区域大小 
          order = 行列优先顺序.[0,1] 表示先迭代0轴，再迭代1轴，即列优先，反之为行优先
          warp_layout = warp中线程排布形状。NVIDIA下，单个warp含32线程,如[4,8]; AMD/DCU 下=64. 如[16,4]
          warp_repeat = 单个warp中所有线程计算的连续区域。对应 【图1】中的 warp_inst
- singeIter表示单次循环计算的区域。其中，可能含有多个 warp_inst 指令。单个循环内 warp_inst 的排列称为 warpInstUnroll
- 
*/

  LinearLayout2DDesc base_layout;  
  
  std::array<int64_t, 2> block_layout = {1, 1};  // block内的warp布局，plan1=[2,1], plan2=[1,2]。用户自行决定
  std::array<int64_t, 2> block_layout_order = {0, 1};  // block内warp布局的行列优先顺序 上例中为[1,0] 行优先（列优先也可）
  std::array<int64_t, 2> block_repeat = {1, 1};  // 为了覆盖buffer，warp_inst 需要迭代的次数。上例中为 i=0,1,2,3 布局为 [2,2] or [4,1]. 其中行列优先顺序无所谓，不影响结果

  std::array<int64_t, 2>  thread_own_data_size;  // thread级别IR表达上，每个线程应当持有的（最少）buffer元素量，才能完成op的计算
  
  coordXY_t get_warp_layout() const {
    return base_layout.warp_layout;
  }
  coordXY_t get_block_layout() const {
    return block_layout;
  }
  coordXY_t get_block_layout_order() const {
    return block_layout_order;
  }
  coordXY_t get_warp_repeat() const {
    return base_layout.warp_repeat;
  }
  coordXY_t get_block_repeat() const {
    return block_repeat;
  }
  // 单个inst中，每个线程处理的连续元素数
  coordXY_t get_thread_widths() const {
    return base_layout.thread_creg;
  } 
  // kernel中，每个线程持有多少buffer的数据
  coordXY_t get_thread_own_data_size() const {
    return thread_own_data_size;
  }
  // kernel中，buffer下 每个线程的总计算数据量
  std::array<int64_t, 2> get_thread_total_widths() const {
    std::array<int64_t, 2> ret;
    for(int i=0;i<2;++i){
      ret[i] = (base_layout.thread_creg[i] * base_layout.warp_repeat[i] * block_repeat[i]);
    }
    if (ignoreDim >= 0 && static_cast<size_t>(ignoreDim) < ret.size()) {
      ret[ignoreDim] = 1;
    }
    return ret;
  }
  // 单个warp计算的连续区域
  coordXY_t get_warp_widths() const {
    return base_layout.get_warp_widths();
  }
  // 单个warp级别指令计算的区域
  coordXY_t get_warpInst_widths() const {
    return get_warp_widths() * get_warp_repeat();
  }
  coordXY_t get_block_widths() const {
    return get_warpInst_widths() * get_block_layout() * warpInstUnroll;
  }

  const auto& getOperandLabels() const {
    return mapOperandsLabel;
  }
  const auto& getIterVarLabels() const {
    return iterVarLabels;
  }
  const auto& getItervarUbs() const {
    return ivUpperBounds;
  }

  void show(const char* label = nullptr) {
    auto printI64Vec = [&](const char *name, const std::array<int64_t, 2> &vec) {
      llvm::outs() << name << ": [";
      for (size_t i = 0; i < vec.size(); ++i) {
        llvm::outs() << vec[i];
        if (i + 1 < vec.size()) llvm::outs() << ", ";
      }
      llvm::outs() << "]\n";
    };
    auto printExprVec = [&](const char *name, const std::array<AffineExpr, 2> &vec) {
      llvm::outs() << name << ": [";
      for (size_t i = 0; i < vec.size(); ++i) {
        vec[i].print(llvm::outs());
        auto& raw = vec[i];
        // llvm::outs() << " simplified:[ " << mlir::simplifyAffineExpr(raw, dimCount, 0) << "] ";
        if (i + 1 < vec.size()) llvm::outs() << ", ";

      }
      llvm::outs() << "]\n";
    };
    const char* _label = " ";
    if(label != nullptr){
      _label = label;
    }
    llvm::outs() << "=== LowerInfo "<< _label <<" ===\n";
    llvm::outs() << "buffer: ";
    if (buffer) {
      buffer.print(llvm::outs());
    } else {
      llvm::outs() << "<null>";
    }
    llvm::outs() << "\n";
    llvm::outs() << "buffer_memory: ";
    if (!buffer || !isa<MemRefType>(buffer.getType())) {
      llvm::outs() << "<non-memref>\n";
    } else {
      int64_t memorySpace = cast<MemRefType>(buffer.getType()).getMemorySpaceAsInt();
      if (memorySpace == 3) {
        llvm::outs() << "shared(memory_space=3)\n";
      } else if (memorySpace == 0 || memorySpace == 5) {
        llvm::outs() << "register/local(memory_space=" << memorySpace << ")\n";
      } else {
        llvm::outs() << "unknown(memory_space=" << memorySpace << ")\n";
      }
      // auto affineMapIndices = getAffineMap();
      // printExprVec("getAffineMap()", affineMapIndices);
    }
    if(op != nullptr){
      llvm::outs() << "op: " << op->getName().getStringRef() << "\n";  
    }
    else{
      llvm::outs() << "op: null\n";  
    }
    llvm::outs() << "thread_bound: " << thread_bound << "\n";
    
    printI64Vec("creg", base_layout.thread_creg) ;  // consistent reg
    printI64Vec("creg_order", base_layout.thread_creg_order) ;
    printI64Vec("warp_layout", get_warp_layout());
    printI64Vec("warp_layout_order", base_layout.warp_layout_order);
    printI64Vec("block_layout", get_block_layout());
    printI64Vec("block_layout_order", block_layout_order);
    printI64Vec("warp_repeat", get_warp_repeat());
    printI64Vec("thread_widths", get_thread_widths());
    printI64Vec("warp_widths", get_warp_widths());
    printI64Vec("block_widths", get_block_widths());
    printI64Vec("block_repeat", get_block_repeat());
    printI64Vec("warpInstUnroll", warpInstUnroll);
    printI64Vec("thread_own_data", get_thread_own_data_size());
    llvm::outs() << "ignoreDim = " << ignoreDim << "\n";
    llvm::outs() << "pos = " << int(pos) << "\n";
    auto cvf = convertFrom == nullptr? "NULL" : "notNull";
    llvm::outs() << "convertFrom =" << cvf << "\n";
    llvm::outs() << "=================\n";
  }

  mlir::AffineMap getAffineMap() {
    // 根据上述信息，生成不同层面的索引
    // 强制重新计算
    mapOperandsLabel.clear();
    iterVarLabels.clear();
    ivUpperBounds.clear();
    mapOperandsLabel.push_back(TID);
    OpBuilder b{buffer.getContext()};
    unsigned int pos = 0;
    // iterVar 顺序 ： tid br0 br1  instUnroll0 instUnroll1 warp_repeat_flat  creg_flat 
    auto tid = b.getAffineDimExpr(pos++);  // 根据warp_layout & order, 分解为tx ty
    auto i_br0 = b.getAffineDimExpr(pos++);  // block_repeat 无order限制
    auto i_br1 = b.getAffineDimExpr(pos++);
    
    auto i_iu0 = b.getAffineDimExpr(pos++);  // inst unroll 没order限制
    auto i_iu1 = b.getAffineDimExpr(pos++);
    
    auto i_wr_flatten = b.getAffineDimExpr(pos++);  // warp_repeat 有order限制。需传入flattenId，然后分解
    auto i_creg_flatten = b.getAffineDimExpr(pos++);  // thread_creg 有order限制。需传入flattenId，然后分解
    
    std::array<AffineExpr, 2> indices{0,0};
    
    // 分解为xy分量
    auto[t0,t1] = UnflattenIndexToXY(tid, base_layout.warp_layout_order, base_layout.warp_layout);
    auto[i_wr0, i_wr1] = UnflattenIndexToXY(i_wr_flatten, base_layout.warp_repeat_order, base_layout.warp_repeat);
    auto[i_reg0, i_reg1] = UnflattenIndexToXY(i_creg_flatten, base_layout.thread_creg_order, base_layout.thread_creg);
    
    indices[0] = i_br0 * get_block_widths()[0] + i_iu0 * get_warpInst_widths()[0] + i_wr0 * get_warp_widths()[0] + t0 * get_thread_widths()[0] + i_reg0;
    indices[1] = i_br1 * get_block_widths()[1] + i_iu1 * get_warpInst_widths()[1] + i_wr1 * get_warp_widths()[1] + t1 * get_thread_widths()[1] + i_reg1;

    auto affine_map = mlir::AffineMap::get(pos, 0, indices, buffer.getContext());
  
    return affine_map;
  }

  std::vector<affine::AffineForOp> getForLoops(mlir::OpBuilder& b, mlir::Location loc, std::vector<mlir::Value>& iterVars){
    int ub_wr = flat_size(get_warp_repeat());
    int ub_reg = flat_size(get_thread_widths());
    auto br = get_block_repeat();

    std::vector<int> ubs = {(int)br[0], (int)br[1],
      (int)warpInstUnroll[0], (int)warpInstUnroll[1], 
    ub_wr, ub_reg
    };
    return createNestedAffineFor(b, loc, ubs, iterVars);
  }

private:

  int64_t thread_bound;
  std::vector<const char*> mapOperandsLabel;  // mapOperands 的标签
  std::vector<const char*> iterVarLabels;  // for 循环的标签
  std::vector<int> ivUpperBounds;  // 迭代变量的上界

};

class LowerInfoMap {
public:
  using LowerInfoMapTy = DenseMap<std::pair<Value, Operation*> , LowerInfo>;
  // 进行op顺序分析
  const SmallVector<Operation*>& getOpsOrder(mlir::Operation* rootNode);
  // 查询 <buffer，op> 对应的LowerInfo
  LowerInfo* getLowerInfo(const mlir::Value& buffer, mlir::Operation* op);
  // 添加 lowerinfo（info中已经含有buffer）
  void addLowerInfo(mlir::Operation* op, LowerInfo info, bool isConflict=false);
  void conflictResolve();
  // 根据buffer查找infoMap，找到其中距离currOp最近的之前/之后的Op的 LowerInfo
  LowerInfo* getNearestInferedInfo(const mlir::Value& buffer, mlir::Operation* currOp, bool isBefore = true);
  auto begin() { return infoMap.begin(); }
  auto end() { return infoMap.end(); }
  void print();
private:
  LowerInfoMapTy infoMap;  // 存放解决完冲突后的 LowerInfo 信息
  DenseMap<mlir::Value, SmallVector<LowerInfo, 4>> m_candidates;  // 先存放所有LowerInfo

  DenseMap<Operation*, unsigned> opOrder;  // 存放 op 顺序
  SmallVector<Operation*> opOrderVec;
};

class LowerInfoAnalysis {
public:
  static LowerInfoMap* run(mlir::Operation* kernelOp,
                                        const std::string& hwKind = HW_KIND_DCU,
                                        const std::string& version = HW_VERSION_DCU_BW1000);                                        
  struct GemmProblem {
    Value A;
    Value B;
    Value C;
    MemRefType aType;
    MemRefType bType;
    MemRefType cType;
    unsigned inElemBitWidth;
    int64_t bm;
    int64_t bn;
    int64_t bk;
  };
  static GemmProblem getGemmProblem(GemmOp gemmOp);
  static MMAInstInfo* selectGemmInst(GemmProblem problem, HWSpecification* hw);

private:
  static LowerInfoMap buf_info_maps;
  static int block_threads ;
  static llvm::SmallVector<Operation*, 5> collectNeedInferOps(mlir::Operation *kernelOp);
  static std::pair<int, int> squareFactor(int n);
  static uint64_t getRegionThreadNum(Operation *op);
  static bool checkGemmProblem(GemmProblem p, HWSpecification* hw);
  static bool getDirectGemmBlockLayout(uint64_t thread_num,
                                       std::array<int64_t, 2> &block_layout, HWSpecification* hw);
  static LowerInfo makeDirectGemmCInfo(OpBuilder b, const GemmProblem &problem,
                                       MMAInstInfo *mma, uint64_t thread_num,
                                       HWSpecification *hw,
                                       std::array<int64_t, 2> block_layout);
  static LowerInfo makeRelyGemmCInfo(OpBuilder b, const GemmProblem &problem,
                                     MMAInstInfo *mma, uint64_t thread_num,
                                     HWSpecification *hw, const LowerInfo &source_info,
                                     bool source_is_a);
  static void applyDirectGemmAInfo(LowerInfo &info, const GemmProblem &problem,
                                   MMAInstInfo *mma, AffineExpr zero, HWSpecification* hw);
  static void applyRelyGemmAInfo(LowerInfo &info, const GemmProblem &problem,
                                 MMAInstInfo *mma, AffineExpr zero );
  static void applyGemmBInfo(LowerInfo &info, const GemmProblem &problem,
                             MMAInstInfo *mma, AffineExpr zero, HWSpecification* hw);
  static bool inferDirectOp(Operation *op, LowerInfoMap& infoMap ,HWSpecification *hw);
  static bool inferRelyOp(Operation *op, LowerInfoMap& infoMap, HWSpecification *hw,
                          bool collectConflict = false, bool preferBefore = true);
  static bool inferCopyOp(Operation *op, LowerInfoMap &buf_info_maps,
                          bool preferBefore = true);
  static bool inferBlockOp(Operation *op, LowerInfoMap &buf_info_maps,
                           bool preferBefore = true);
  static bool inferGemmOp(Operation *op, LowerInfoMap &buf_info_maps,
                          HWSpecification *hw);
  static bool inferRelyGemmOp(Operation *op, LowerInfoMap &buf_info_maps,
                              HWSpecification *hw, bool preferBefore = true);
  static bool inferReduceOp(Operation *op, LowerInfoMap &buf_info_maps,
                            bool preferBefore = true);

  // void getTest() {
  //   llvm::outs() << "[D]need_infer_ops size: " << need_infer_ops.size() << "\n";
  //   llvm::outs() << "[D]buf_info_maps size: " << buf_info_maps.size() << "\n";
  // }
  // void showAllInfo() {
  //   llvm::outs() << "[D]show all lower info, count: " << buf_info_maps.size() << "\n";
  //   for ( auto &it : buf_info_maps) {
  //     llvm::outs() << "[D]buffer key: ";
  //     it.first.print(llvm::outs());
  //     llvm::outs() << "\n";
  //     it.second.show();
  //   }
  // }

  // LowerInfo getInfo(const Value& buffer){
  //   // return buf_info_maps.at(buffer);
  // }

};

}

#endif
