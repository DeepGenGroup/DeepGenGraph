// Conservative range proofs for triangular mask coordinates. Unknown inputs,
// overflow, or unsupported regions keep the original index/FP implementation.
#ifndef DEEPGENGRAPH_CAUSAL_MASK_LOWERING_H
#define DEEPGENGRAPH_CAUSAL_MASK_LOWERING_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <cstdint>
#include <optional>

namespace mlir::frisk::causal_mask {
struct Range { int64_t lo, hi; };
using Bounds = std::optional<Range>;

inline Bounds combine(Bounds a, Bounds b, char op) {
  if (!a || !b) return {};
  int64_t lo, hi;
  if (op == '+') {
    if (llvm::AddOverflow(a->lo,b->lo,lo) || llvm::AddOverflow(a->hi,b->hi,hi)) return {};
  } else if (op == '-') {
    if (llvm::SubOverflow(a->lo,b->hi,lo) || llvm::SubOverflow(a->hi,b->lo,hi)) return {};
  } else {
    int64_t p[4];
    if (llvm::MulOverflow(a->lo,b->lo,p[0]) || llvm::MulOverflow(a->lo,b->hi,p[1]) ||
        llvm::MulOverflow(a->hi,b->lo,p[2]) || llvm::MulOverflow(a->hi,b->hi,p[3])) return {};
    lo = *std::min_element(p,p+4); hi = *std::max_element(p,p+4);
  }
  return Range{lo,hi};
}

inline Bounds affineRange(AffineExpr e, ArrayRef<Bounds> inputs, unsigned dims) {
  if (auto c = dyn_cast<AffineConstantExpr>(e)) return Range{c.getValue(),c.getValue()};
  if (auto d = dyn_cast<AffineDimExpr>(e)) return inputs[d.getPosition()];
  if (auto s = dyn_cast<AffineSymbolExpr>(e)) return inputs[dims+s.getPosition()];
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin) return {};
  auto a = affineRange(bin.getLHS(),inputs,dims), b = affineRange(bin.getRHS(),inputs,dims);
  if (!a || !b) return {};
  if (bin.getKind() == AffineExprKind::Add) return combine(a,b,'+');
  if (bin.getKind() == AffineExprKind::Mul) return combine(a,b,'*');
  if (b->lo != b->hi || b->lo <= 0) return {};
  if (bin.getKind() == AffineExprKind::FloorDiv)
    return Range{llvm::divideFloorSigned(a->lo,b->lo),llvm::divideFloorSigned(a->hi,b->lo)};
  if (bin.getKind() == AffineExprKind::Mod) {
    if (llvm::divideFloorSigned(a->lo,b->lo) == llvm::divideFloorSigned(a->hi,b->lo))
      return Range{llvm::mod(a->lo,b->lo),llvm::mod(a->hi,b->lo)};
    return Range{0,b->lo-1};
  }
  return {};
}

inline Bounds indexRange(Value v, unsigned depth = 0);
inline Bounds mapRange(AffineMap map, ValueRange operands, bool upper, unsigned depth) {
  SmallVector<Bounds> inputs;
  for (Value v : operands) inputs.push_back(indexRange(v,depth+1));
  Bounds result;
  for (AffineExpr e : map.getResults()) {
    auto r = affineRange(e,inputs,map.getNumDims());
    // Layout maps often spell modulo as x - (x floordiv c) * c.
    // Enumerate small one-dimensional domains to retain that correlation;
    // every point is evaluated with overflow checks, never by sampling.
    int64_t width;
    if (inputs.size()==1 && inputs[0] &&
        !llvm::SubOverflow(inputs[0]->hi,inputs[0]->lo,width) && width>=0 && width<256) {
      Bounds exact;
      for (int64_t offset=0; offset<=width; ++offset) {
        int64_t point=inputs[0]->lo+offset;
        Bounds singleton=Range{point,point};
        auto value=affineRange(e,ArrayRef<Bounds>(singleton),map.getNumDims());
        if (!value) { exact.reset(); break; }
        if (!exact) exact=value;
        else exact=Range{std::min(exact->lo,value->lo),std::max(exact->hi,value->hi)};
      }
      if (exact) r=exact;
    }
    if (!r) return {};
    if (!result) result = r;
    else if (upper) result = Range{std::min(result->lo,r->lo),std::min(result->hi,r->hi)};
    else result = Range{std::max(result->lo,r->lo),std::max(result->hi,r->hi)};
  }
  return result;
}

inline Bounds indexRange(Value v, unsigned depth) {
  if (depth > 48 || !v.getType().isIndex()) return {};
  APInt constant;
  if (matchPattern(v,m_ConstantInt(&constant))) return Range{constant.getSExtValue(),constant.getSExtValue()};
  if (auto id = v.getDefiningOp<gpu::BlockIdOp>()) {
    if (auto range = id->getAttrOfType<IntegerAttr>("range"); range && range.getInt()>0)
      return Range{0,range.getInt()-1};
  }
  if (auto id = v.getDefiningOp<gpu::ThreadIdOp>()) {
    if (auto range = id->getAttrOfType<IntegerAttr>("range"); range && range.getInt()>0)
      return Range{0,range.getInt()-1};
    if (id.getDimension() == gpu::Dimension::x)
      if (auto f = id->getParentOfType<func::FuncOp>())
        if (auto n = f->getAttrOfType<IntegerAttr>("thread_num"); n && n.getInt()>0)
          return Range{0,n.getInt()-1};
  }
  if (auto loop = affine::getForInductionVarOwner(v)) {
    auto lo = mapRange(loop.getLowerBoundMap(),loop.getLowerBoundOperands(),false,depth);
    auto hi = mapRange(loop.getUpperBoundMap(),loop.getUpperBoundOperands(),true,depth);
    if (lo && hi && hi->hi > lo->lo) return Range{lo->lo,hi->hi-1};
    return {};
  }
  if (auto arg = dyn_cast<BlockArgument>(v))
    if (auto loop = dyn_cast_or_null<scf::ForOp>(arg.getOwner()->getParentOp());
        loop && v == loop.getInductionVar()) {
      auto lo = indexRange(loop.getLowerBound(),depth+1), hi = indexRange(loop.getUpperBound(),depth+1);
      auto step = indexRange(loop.getStep(),depth+1);
      int64_t next;
      if (lo && hi && step && step->lo>0 && hi->hi>lo->lo &&
          !llvm::AddOverflow(hi->hi-1,step->hi,next)) return Range{lo->lo,hi->hi-1};
      return {};
    }
  if (auto apply = v.getDefiningOp<affine::AffineApplyOp>())
    return mapRange(apply.getAffineMap(),apply.getMapOperands(),false,depth);
  auto *op = v.getDefiningOp();
  if (!op || op->getNumOperands()!=2) return {};
  auto a=indexRange(op->getOperand(0),depth+1), b=indexRange(op->getOperand(1),depth+1);
  if (isa<arith::AddIOp>(op)) return combine(a,b,'+');
  if (isa<arith::SubIOp>(op)) return combine(a,b,'-');
  if (isa<arith::MulIOp>(op)) return combine(a,b,'*');
  if (!a || !b || a->lo<0 || b->lo<=0) return {};
  if (isa<arith::DivUIOp,arith::DivSIOp>(op)) return Range{a->lo/b->hi,a->hi/b->lo};
  if (isa<arith::RemUIOp,arith::RemSIOp>(op)) return Range{0,b->hi-1};
  return {};
}

struct Triangle { int64_t diagonal; Value masked, clear; };
inline std::optional<Triangle> match(Block *body) {
  if (body->getNumArguments()!=2 || body->getTerminator()->getNumOperands()!=1) return {};
  // No side effects or nontrivial control flow may be dropped by specialization.
  if (!llvm::all_of(body->without_terminator(),[](Operation &op) {
        return isa<arith::ConstantOp,arith::AddIOp,arith::CmpIOp,arith::SelectOp>(op);
      })) return {};
  auto select=body->getTerminator()->getOperand(0).getDefiningOp<arith::SelectOp>();
  auto cmp=select ? select.getCondition().getDefiningOp<arith::CmpIOp>() : arith::CmpIOp{};
  auto add=cmp ? cmp.getLhs().getDefiningOp<arith::AddIOp>() : arith::AddIOp{};
  APInt diagonal;
  if (!cmp || cmp.getPredicate()!=arith::CmpIPredicate::ule || !add ||
      add.getLhs()!=body->getArgument(0) || cmp.getRhs()!=body->getArgument(1) ||
      !matchPattern(add.getRhs(),m_ConstantInt(&diagonal)) || diagonal.isNegative()) return {};
  return Triangle{diagonal.getSExtValue(),select.getTrueValue(),select.getFalseValue()};
}

inline bool isNegativeInfinityAndZero(const Triangle &t) {
  auto masked=t.masked.getDefiningOp<arith::ConstantOp>();
  auto clear=t.clear.getDefiningOp<arith::ConstantOp>();
  auto m=masked ? dyn_cast<FloatAttr>(masked.getValue()) : FloatAttr{};
  auto c=clear ? dyn_cast<FloatAttr>(clear.getValue()) : FloatAttr{};
  return m && c && m.getValue().isInfinity() && m.getValue().isNegative() &&
         c.getValue().isZero() && !c.getValue().isNegative();
}
} // namespace mlir::frisk::causal_mask
#endif
