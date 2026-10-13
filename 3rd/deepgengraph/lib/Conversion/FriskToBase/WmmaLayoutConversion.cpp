#include "WmmaLayoutConversion.h"
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::frisk {
#define GEN_PASS_DEF_PLANWMMALAYOUTCONVERSIONS
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

void selectGemmAInputLayoutMethod(func::FuncOp kernel, StringRef method) {
  SmallVector<Value> worklist;
  kernel.walk([&](GemmOp gemm) {
    if (gemm->hasAttr("tiled"))
      return;
    gemm->removeAttr("frisk.b_permute");
    gemm->removeAttr("frisk.wmma_layout_applied");
    gemm->removeAttr("frisk.a_input_layout_method");
    worklist.push_back(gemm.getA());
  });
  DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second || !isa<ShapedType>(value.getType()))
      continue;
    // Legacy tensor carriers are updated by whole-tile copy, then yielded.
    // Follow the value written to them, not unrelated users or scalar indices.
    for (Operation *user : value.getUsers()) {
      auto copy = dyn_cast<CopyOp>(user);
      if (!copy || copy.getDst() != value)
        continue;
      auto map = copy.getOffsetMap();
      if (map.getNumInputs() == 0 && map.getNumResults() == 1 &&
          map.getResult(0) == getAffineConstantExpr(
              cast<ShapedType>(value.getType()).getRank(), kernel.getContext()))
        worklist.push_back(copy.getSrc());
    }
    if (auto result = dyn_cast<OpResult>(value)) {
      Operation *producer = result.getOwner();
      unsigned index = result.getResultNumber();
      if (auto gemm = dyn_cast<GemmOp>(producer)) {
        gemm->setAttr("frisk.a_input_layout_method", StringAttr::get(kernel.getContext(), method));
        if (method == "b-permute")
          gemm->setAttr("frisk.b_permute", UnitAttr::get(kernel.getContext()));
        // Do not traverse through a GEMM into its A/B operands.
        continue;
      }
      if (auto loop = dyn_cast<affine::AffineForOp>(producer)) {
        worklist.push_back(loop.getBody()->getTerminator()->getOperand(index));
        worklist.push_back(loop.getInits()[index]);
      } else if (auto loop = dyn_cast<scf::ForOp>(producer)) {
        worklist.push_back(loop.getBody()->getTerminator()->getOperand(index));
        worklist.push_back(loop.getInitArgs()[index]);
      } else if (auto branch = dyn_cast<scf::IfOp>(producer)) {
        worklist.push_back(branch.thenYield()->getOperand(index));
        worklist.push_back(branch.elseYield()->getOperand(index));
      } else if (producer->hasTrait<frisk::ComputedTile>()) {
        llvm::append_range(worklist, producer->getOperands());
      } else if (auto copy = dyn_cast<CopyOp>(producer)) {
        worklist.push_back(copy.getSrc());
      }
    } else if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getArgNumber() == 0)
        continue;
      unsigned index = arg.getArgNumber() - 1;
      if (auto loop = dyn_cast<affine::AffineForOp>(arg.getOwner()->getParentOp())) {
        worklist.push_back(loop.getInits()[index]);
        worklist.push_back(loop.getBody()->getTerminator()->getOperand(index));
      } else if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
        worklist.push_back(loop.getInitArgs()[index]);
        worklist.push_back(loop.getBody()->getTerminator()->getOperand(index));
      }
    }
  }
}

static DictionaryAttr encodeLayout(const LowerInfo &info, OpBuilder &b) {
  NamedAttrList attrs;
  auto pair = [&](StringRef name, coordXY_t value) {
    attrs.set(name, b.getDenseI64ArrayAttr(value));
  };
  pair("thread_widths", info.get_thread_widths());
  pair("warp_repeat", info.get_warp_repeat());
  pair("warp_layout", info.get_warp_layout());
  pair("warp_layout_order", info.base_layout.warp_layout_order);
  pair("thread_creg_order", info.base_layout.thread_creg_order);
  pair("warp_repeat_order", info.base_layout.warp_repeat_order);
  pair("warp_inst_unroll", info.warpInstUnroll);
  pair("block_repeat", info.block_repeat);
  pair("block_layout", info.block_layout);
  pair("block_layout_order", info.block_layout_order);
  attrs.set("warp_threads", b.getI64IntegerAttr(info.warp_threads));
  attrs.set("ignore_dim", b.getI64IntegerAttr(info.ignoreDim));
  if (info.warpIdDivisor != 1)
    attrs.set("warp_id_divisor", b.getI64IntegerAttr(info.warpIdDivisor));
  if (info.logicalTransform)
    attrs.set("logical_transform", AffineMapAttr::get(info.logicalTransform));
  return attrs.getDictionary(b.getContext());
}

void annotateWmmaLayoutConversion(ConvertLayoutOp op, const LowerInfo &from,
                                  const LowerInfo &to) {
  OpBuilder b(op);
  op->setAttr("frisk.source_layout", encodeLayout(from, b));
  op->setAttr("frisk.target_layout", encodeLayout(to, b));
  if (from.op)
    if (auto method = from.op->getAttrOfType<StringAttr>("frisk.a_input_layout_method"))
      op->setAttr("frisk.a_input_layout_method", method);
  // mmaInst also propagates through elementwise operations. Compare the
  // distribution with desc_c, so A/B operands do not acquire the C role.
  if (!from.mmaInst)
    return;
  const auto &c = from.mmaInst->desc_c;
  if (from.get_thread_widths() != c.thread_creg ||
      from.get_warp_repeat() != c.warp_repeat ||
      from.get_warp_layout() != c.warp_layout ||
      from.base_layout.warp_layout_order != c.warp_layout_order)
    return;
  op->setAttr("frisk.layout_role", b.getStringAttr("wmmaC"));
  NamedAttrList inst;
  inst.set("name", b.getStringAttr(from.mmaInst->asm_str));
  inst.set("shape", b.getDenseI64ArrayAttr(
      {from.mmaInst->m, from.mmaInst->n, from.mmaInst->k}));
  op->setAttr("frisk.wmma_instruction", inst.getDictionary(b.getContext()));
}

DictionaryAttr getWmmaShufflePlan(ConvertLayoutOp op) {
  auto role = op->getAttrOfType<StringAttr>("frisk.layout_role");
  auto inst = op->getAttrOfType<DictionaryAttr>("frisk.wmma_instruction");
  auto src = op->getAttrOfType<DictionaryAttr>("frisk.source_layout");
  auto dst = op->getAttrOfType<DictionaryAttr>("frisk.target_layout");
  auto type = op.getMemType();
  if (!role || role.getValue() != "wmmaC" || !inst || !src || !dst ||
      type.getRank() != 2 || !type.hasStaticShape() ||
      !type.getElementType().isF32() || op->getNumResults() != 1 ||
      op->getResult(0).getType() != type)
    return {};
  // The XOR transpose only proves the untransformed hardware C mapping.
  if (src.get("logical_transform") || dst.get("logical_transform"))
    return {};
  if (src.get("warp_id_divisor") != dst.get("warp_id_divisor"))
    return {};
  auto integer = [](DictionaryAttr attrs, StringRef name, int64_t value) {
    auto attr = attrs.getAs<IntegerAttr>(name);
    return attr && attr.getInt() == value;
  };
  auto pair = [](DictionaryAttr attrs, StringRef name, coordXY_t value) {
    auto attr = attrs.getAs<DenseI64ArrayAttr>(name);
    return attr && attr.asArrayRef() == ArrayRef<int64_t>(value);
  };
  auto lanes = src.getAs<DenseI64ArrayAttr>("warp_layout");
  auto shape = inst.getAs<DenseI64ArrayAttr>("shape");
  if (!lanes || lanes.size() != 2 || !shape || shape.size() != 3 ||
      !inst.getAs<StringAttr>("name"))
    return {};
  int64_t rows = lanes[0], group = lanes[1];
  if (rows <= 0 || group <= 1 || rows > 64 || group > 64 ||
      !llvm::isPowerOf2_64(rows) || !llvm::isPowerOf2_64(group) ||
      rows * group != 64 || shape[0] != rows ||
      shape[1] != group * group || shape[2] <= 0 ||
      !integer(src, "warp_threads", 64) || !integer(dst, "warp_threads", 64) ||
      !integer(src, "ignore_dim", -1) || !integer(dst, "ignore_dim", -1) ||
      !pair(src, "warp_layout_order", {0, 1}) ||
      !pair(src, "thread_widths", {1, 1}) ||
      !pair(src, "warp_repeat", {1, group}) ||
      !pair(dst, "thread_widths", {1, group}) ||
      !pair(dst, "warp_repeat", {1, 1}))
    return {};
  // Outer coordinates must agree: a lane-only permutation cannot move data
  // between waves, instruction tiles, or repetitions. Orders on singleton
  // register axes are immaterial, but must still be valid permutations.
  for (auto layout : {src, dst})
    for (StringRef name : {"thread_creg_order", "warp_repeat_order"})
      if (!pair(layout, name, {0, 1}) && !pair(layout, name, {1, 0}))
        return {};
  for (StringRef name : {"warp_layout", "warp_layout_order",
                         "warp_inst_unroll", "block_repeat", "block_layout",
                         "block_layout_order"}) {
    auto a = src.getAs<DenseI64ArrayAttr>(name);
    auto b = dst.getAs<DenseI64ArrayAttr>(name);
    if (!a || a.size() != 2 || a != b)
      return {};
  }
  if (!pair(src, "block_layout_order", {0, 1}) &&
      !pair(src, "block_layout_order", {1, 0}))
    return {};
  for (unsigned dim = 0; dim < 2; ++dim) {
    int64_t extent = shape[dim];
    for (StringRef name : {"warp_inst_unroll", "block_repeat", "block_layout"}) {
      int64_t count = src.getAs<DenseI64ArrayAttr>(name)[dim];
      if (count <= 0 || extent > type.getDimSize(dim) / count)
        return {};
      extent *= count;
    }
    if (extent != type.getDimSize(dim))
      return {};
  }
  OpBuilder b(op);
  SmallVector<int64_t> masks;
  for (int64_t bit = 1; bit < group; bit *= 2)
    masks.push_back(rows * bit);
  NamedAttrList plan;
  plan.set("algorithm", b.getStringAttr("xor_transpose"));
  plan.set("width", b.getI64IntegerAttr(64));
  plan.set("registers", b.getI64IntegerAttr(group));
  plan.set("xor_masks", b.getDenseI64ArrayAttr(masks));
  return plan.getDictionary(b.getContext());
}

void planWmmaLayoutConversions(func::FuncOp kernel, bool contiguousOutputs) {
  if (contiguousOutputs)
    kernel.walk([&](GemmOp op) {
      if (!op->hasAttr("tiled"))
        op->setAttr("frisk.contiguous_c", UnitAttr::get(op.getContext()));
    });
  kernel.walk([&](ConvertLayoutOp op) {
    op->removeAttr("frisk.shuffle_plan");
    if (auto method = op->getAttrOfType<StringAttr>("frisk.a_input_layout_method");
        method && method.getValue() == "shared")
      return;
    if (auto plan = getWmmaShufflePlan(op))
      op->setAttr("frisk.shuffle_plan", plan);
  });
}

namespace {
class PlanWmmaLayoutConversions
    : public impl::PlanWmmaLayoutConversionsBase<PlanWmmaLayoutConversions> {
  void runOnOperation() override {
    planWmmaLayoutConversions(getOperation(), contiguousOutputs);
  }
};
} // namespace
std::unique_ptr<Pass> createPlanWmmaLayoutConversionsPass() {
  return std::make_unique<PlanWmmaLayoutConversions>();
}
} // namespace mlir::frisk
