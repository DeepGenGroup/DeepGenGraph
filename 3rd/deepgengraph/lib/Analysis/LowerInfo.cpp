#include "deepgengraph/Analysis/LowerInfo.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <string>
#include <system_error>
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


void LowerInfoAnalysis::run() {
  _kernelOp.walk<mlir::WalkOrder::PreOrder>([&](Operation* op) {
    if (isa<CopyOp, BlockOp, GemmOp, ReduceOp>(op)) {
      need_infer_ops.push_back(op);
    }
  });

  auto square_func = [](int n) -> std::pair<int, int> {
    int a = static_cast<int>(std::sqrt(n));
    while (a >= 1) {
      if (n % a == 0) {
        int b = n / a;
        return {b, a};
      }
      --a;
    }
    return {n, 1};
  };

  auto get_region_thread_num_func = [&](Operation *op) -> uint64_t {
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
  };

  auto dircet_infer_func = [&](Operation *op) ->bool {
    // get region thread num
    OpBuilder b(op);
    uint64_t thread_num = get_region_thread_num_func(op);
    if (auto gemmOp = dyn_cast<GemmOp>(op)) {
      Value A = gemmOp.getMatrixA();
      Value B = gemmOp.getMatrixB();
      Value C = gemmOp.getMatrixC();
      MemRefType _a = gemmOp.getA().getType();
      MemRefType _b = gemmOp.getB().getType();
      MemRefType _c = gemmOp.getC().getType();
      // ab有一个不是shared memroy
      if (_a.getMemorySpaceAsInt() != 3 || _b.getMemorySpaceAsInt() != 3) {
        return false;
      }
      // warpgroup 数量和布局
      int warpgroup_num = thread_num / 128;
      auto [y, x] = square_func(warpgroup_num);  // warpgroup layout
      if (warpgroup_num <= 0 || y <= 0 || x <= 0) {  
        return false;
      }
      auto shapeC = _c.getShape();
      auto shapeA = _a.getShape();
      unsigned in_elem_width = _a.getElementTypeBitWidth();
      int64_t bm_ = shapeC[0];
      int64_t bn_ = shapeC[1];
      int64_t bk_ = gemmOp.getTransA() ? shapeA[1] : shapeA[0];
      assert(bm_ % 64 == 0 && "BM must great more than MMA_M");
      assert(bn_ % 8 == 0 && "BN must great more than min MMA_N");
      assert(in_elem_width * bk_ % 32 == 0 && "BK must great more than MMA_K");
      // must memory
      assert((_c.getMemorySpaceAsInt() == 0 || _c.getMemorySpaceAsInt() == 5) &&
             "C must be local buffer.");
      // 任意精度的计算规模
      int mma_n = bn_ >= 256 ? 256 : bn_;
      int mma_m = 64;
      int mma_k = 32 / in_elem_width;
      // C LowerInfo
      LowerInfo ic;
      ic.buffer = C;
      ic.thread_bound = thread_num;
      ic.thread_widths = {1, 32 / static_cast<int64_t>(in_elem_width)};
      ic.warp_layout = {8, 4};
      ic.block_layout = {y * 4, x};
      ic.warp_widths = LowerInfo::getWarpWidths(ic.thread_widths, ic.warp_layout);
      ic.warp_repeat = {2, mma_n / ic.warp_widths[1]};
      ic.block_widths = LowerInfo::getBlockWidths(ic.warp_widths, ic.warp_repeat, ic.block_layout);
      ic.block_repeat = {bm_ / ic.block_widths[0], bn_ / ic.block_widths[1]};
      ic.warp_indices = LowerInfo::getWarpIndices(b, ic.block_layout);
      ic.thread_indices = LowerInfo::getThreadIndices(b, ic.warp_layout);
      buf_info_maps[C] = ic;
      // A lowerInfo no tran
      auto zore = b.getAffineConstantExpr(0);
      buf_info_maps[A] = ic;
      buf_info_maps[A].buffer = A;
      buf_info_maps[A].thread_widths[1] = 0;
      buf_info_maps[A].warp_widths[1] = 0;
      buf_info_maps[A].warp_repeat[1] = 0;
      buf_info_maps[A].block_widths[1] = mma_k;
      buf_info_maps[A].block_repeat[1] = bk_ / mma_k;
      buf_info_maps[A].warp_indices[1] = zore;
      buf_info_maps[A].thread_indices[1] = zore;
      // B lowerInfo no tran
      buf_info_maps[B] = ic;
      buf_info_maps[B].buffer = B;
      buf_info_maps[B].thread_widths[0] = 0;
      buf_info_maps[B].warp_widths[0] = 0;
      buf_info_maps[B].warp_repeat[0] = 0;
      buf_info_maps[B].block_widths[0] = mma_k;
      buf_info_maps[B].block_repeat[0] = bk_ / mma_k;
      buf_info_maps[B].warp_indices[0] = zore;
      buf_info_maps[B].thread_indices[0] = zore;
      return true;
    }
    // others operation
    return false;
  };

  auto rely_infer_func = [&](Operation *op) -> bool {
    OpBuilder b(op);
    uint64_t thread_num = get_region_thread_num_func(op);
    // 对所有 op 统一做“是否已知全部 memref”判断。
    // BlockOp 需要额外检查其内部 affine load/store 的 memref。
    llvm::SmallVector<Value, 8> memrefsToCheck;
    for (const auto &opd : op->getOperands()) {
      if (isa<MemRefType>(opd.getType())) memrefsToCheck.push_back(opd);
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
        all_in = false; count++;
      }
    }
    if (all_in) return true;  // buf已全部推断
    if (!all_in && count == memrefsToCheck.size()) return false;        // 无已推断buf
    // 进入推断
    if (auto copyOp = dyn_cast<CopyOp>(op)) {  // copyOp
      Value dst = copyOp.getDstMemRef();
      Value src = copyOp.getSrcMemRef();
      if (buf_info_maps.count(dst)) {
        buf_info_maps[src] = buf_info_maps[dst];
        buf_info_maps[src].buffer = src;
        return true;
      } else if (buf_info_maps.count(src)) {
        buf_info_maps[dst] = buf_info_maps[src];
        buf_info_maps[dst].buffer = dst;
        return true;
      }
      return false;
    } else if (auto blockOp = dyn_cast<BlockOp>(op)) {  // blockOp
      Value store_buf;
      llvm::SmallVector<Value, 3> load_bufs;
      int64_t dim = blockOp.getBlockDim();
      auto uppers = blockOp.getBlockRanges();
      // 收集输入输出buffer
      blockOp.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
        if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
          load_bufs.push_back(loadOp.getMemRef());
        } else if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
          store_buf = storeOp.getMemref();
        }
      });
      // 在buffer中找到能与upper对齐的且拥有info的buffer
      // 然后把该buffer的info作为传播info，传播到operation中的其他buffer上去
      auto get_main_info_func = [&](Value buf) -> bool {
        MemRefType lty = dyn_cast<MemRefType>(buf.getType());
        if (lty.getRank() == uppers.size()) {  // dim equl
          auto shape = lty.getShape();
          bool is_shape_equl = true;
          for (size_t i=0; i<shape.size(); i++) {  // shape equl
            if (shape[i] != uppers[i]) {
              is_shape_equl = false;
              break;
            }
          }
          if (is_shape_equl) return true;
        }
        return false;
      };

      LowerInfo *info = nullptr;
      for (const auto &lbuf: load_bufs) {  // 输入 buffer
        if (buf_info_maps.count(lbuf) && get_main_info_func(lbuf)) {
          info = &buf_info_maps[lbuf];
          break;
        }
      }
      if (info == nullptr && buf_info_maps.count(store_buf) && get_main_info_func(store_buf)) {  // 输出 buffer
        info = &buf_info_maps[store_buf];
      }
      // 判断是否info是否有值
      if (info == nullptr) return false;

      load_bufs.push_back(store_buf);
      for (const Value& buf: load_bufs) {
        if (!buf_info_maps.count(buf)) {
          buf_info_maps[buf] = *info;
          buf_info_maps[buf].buffer = buf;
          
        }
      }
      return true;

    } else if (auto gemmOp = dyn_cast<GemmOp>(op)) {  // blockOp
      Value A = gemmOp.getMatrixA();
      Value B = gemmOp.getMatrixB();
      Value C = gemmOp.getMatrixC();
      MemRefType _a = gemmOp.getA().getType();
      MemRefType _b = gemmOp.getB().getType();
      MemRefType _c = gemmOp.getC().getType();
      // datas
      auto shapeC = _c.getShape();
      auto shapeA = _a.getShape();
      unsigned in_elem_width = _a.getElementTypeBitWidth();
      int64_t bm_ = shapeC[0];
      int64_t bn_ = shapeC[1];
      int64_t bk_ = gemmOp.getTransA() ? shapeA[1] : shapeA[0];
      assert(bm_ % 64 == 0 && "BM must great more than MMA_M");
      assert(bn_ % 8 == 0 && "BN must great more than min MMA_N");
      assert(in_elem_width * bk_ % 32 == 0 && "BK must great more than MMA_K");
      // must memory
      assert((_c.getMemorySpaceAsInt() == 0 || _c.getMemorySpaceAsInt() == 5) &&
             "C must be local buffer.");
      assert((_b.getMemorySpaceAsInt() == 3) && "B must be shared buffer.");
      // 任意精度的计算规模 
      int mma_n = bn_ >= 256 ? 256 : bn_;
      int mma_m = 64;
      int mma_k = 32 / in_elem_width;

      LowerInfo ic;
      if (buf_info_maps.count(A) || buf_info_maps.count(B)) {
        // 传播源
        LowerInfo info = buf_info_maps.count(A) ? buf_info_maps[A] : buf_info_maps[B];
        assert(thread_num == info.thread_bound && "Thread Number is not equl");
        // C lowerInfo
        ic.buffer = C;
        ic.thread_bound = thread_num;
        ic.warp_layout = info.warp_layout;
        ic.block_layout = info.block_layout;
        if (buf_info_maps.count(A)) {
          ic.thread_widths = {info.thread_widths[0], 32 / static_cast<int64_t>(in_elem_width)};
          ic.warp_widths = LowerInfo::getWarpWidths(ic.thread_widths, ic.warp_layout);
          ic.warp_repeat = {info.warp_repeat[0], mma_n / ic.warp_widths[1]};
          ic.block_widths = LowerInfo::getBlockWidths(ic.warp_widths, ic.warp_repeat, ic.block_layout);
          ic.block_repeat = {info.block_repeat[0], bn_ / ic.block_widths[1]};
        } else {
          ic.thread_widths = {1, info.thread_widths[1]};
          ic.warp_widths = LowerInfo::getWarpWidths(ic.thread_widths, ic.warp_layout);
          ic.warp_repeat = {2, info.warp_repeat[1]};
          ic.block_widths = LowerInfo::getBlockWidths(ic.warp_widths, ic.warp_repeat, ic.block_layout);
          ic.block_repeat = {bm_ / ic.block_widths[0], info.block_repeat[1]};
        }
        ic.warp_indices = LowerInfo::getWarpIndices(b, ic.block_layout);
        ic.thread_indices = LowerInfo::getThreadIndices(b, ic.warp_layout);
        buf_info_maps[C] = ic;
      } else {  // C exsit
        ic = buf_info_maps[C];
      }
      // AB lowerinfo
      auto zore = b.getAffineConstantExpr(0);
      if (buf_info_maps.count(A)) {
        buf_info_maps[B] = ic;
        buf_info_maps[B].buffer = B;
        buf_info_maps[B].thread_widths[0] = 0;
        buf_info_maps[B].warp_widths[0] = 0;
        buf_info_maps[B].warp_repeat[0] = 0;
        buf_info_maps[B].block_widths[0] = mma_k;
        buf_info_maps[B].block_repeat[0] = bk_ / mma_k;
        buf_info_maps[B].warp_indices[0] = zore;
        buf_info_maps[B].thread_indices[0] = zore;
      }
      if (buf_info_maps.count(B)) {
        buf_info_maps[A] = ic;
        buf_info_maps[A].buffer = A;
        buf_info_maps[A].thread_widths[1] = 0;
        buf_info_maps[A].warp_widths[1] = 0;
        buf_info_maps[A].warp_repeat[1] = 0;
        buf_info_maps[A].block_widths[1] = mma_k;
        buf_info_maps[A].block_repeat[1] = bk_ / mma_k;
        buf_info_maps[A].warp_indices[1] = zore;
        buf_info_maps[A].thread_indices[1] = zore;
      }
      return true;
    } else if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
      Value dst = reduceOp.getDst();
      Value src = reduceOp.getSrc();
      uint64_t dim = reduceOp.getDim();
      if (!buf_info_maps.count(src)) {  // unexsit 
        return false;
      }
      buf_info_maps[dst] = buf_info_maps[src];
      LowerInfo &dstInfo = buf_info_maps[dst];
      dstInfo.buffer = dst;

      auto erase_dim_func = [&](auto &vec) -> bool {
        if (dim >= vec.size()) return false;
        vec.erase(vec.begin() + static_cast<int64_t>(dim));
        return true;
      };

      // 沿 dim 归约后，dst 仅保留非归约维度对应的 lower 信息。
      if (!erase_dim_func(dstInfo.warp_indices) ||
          !erase_dim_func(dstInfo.thread_indices) ||
          !erase_dim_func(dstInfo.warp_layout) ||
          !erase_dim_func(dstInfo.block_layout) ||
          !erase_dim_func(dstInfo.warp_repeat) ||
          !erase_dim_func(dstInfo.block_repeat) ||
          !erase_dim_func(dstInfo.thread_widths) ||
          !erase_dim_func(dstInfo.warp_widths) ||
          !erase_dim_func(dstInfo.block_widths)) {
        return false;
      }
      return true;
    } else {
      assert(false && "the operation can not be recognized.");
      return false;
    }
  };

  // 直接推断
  // llvm::outs() << "[DDD]need_infer_ops or size: " << need_infer_ops.size() << "\n";
  bool exsit_dircet_infer_op = false;
  for (size_t i=0; i<need_infer_ops.size(); ++i) {
    if (dircet_infer_func(need_infer_ops[i])) {
      exsit_dircet_infer_op = true;
      // llvm::outs() << "[DDDD]" << *need_infer_ops[i] << "\n";
      need_infer_ops.erase(need_infer_ops.begin() + i);
      break;
    }
  }
  // llvm::outs() << "[DDD]need_infer_ops after size: " << need_infer_ops.size() << "\n";
  if (!exsit_dircet_infer_op) {
    assert(false && "LowerInfo infer filed.");
  }

  // 依赖推断：每轮尽可能消解可推断的 op，直到稳定。
  while (!need_infer_ops.empty()) {
    bool progress = false;
    SmallVector<Operation*, 5> pendingOps;
    pendingOps.reserve(need_infer_ops.size());
    for (Operation *op : need_infer_ops) {
      if (rely_infer_func(op)) {
        progress = true;
      } else {
        pendingOps.push_back(op);
      }
    }
    if (!progress) {
      llvm::errs() << "[LowerInfo] infer failed: unresolved ops remain (" << pendingOps.size() << ")\n";
      for (Operation *op : pendingOps) {
        llvm::errs() << "  unresolved op: " << *op << "\n";
      }
      assert(false && "LowerInfo infer failed.");
    }
    need_infer_ops.swap(pendingOps);
  }
}

}
