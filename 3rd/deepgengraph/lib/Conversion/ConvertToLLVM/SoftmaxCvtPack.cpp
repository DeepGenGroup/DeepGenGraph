// Select approximate softmax conversion in ThreadIR, while producer/consumer
// types are explicit. Target lowering only materializes this verified contract.
#include "deepgengraph/Conversion/ConvertToLLVM/SoftmaxCvtPack.h"
#include "ThreadScheduleUtils.h"
#include "mlir/Dialect/Math/IR/Math.h"

namespace mlir::frisk {
namespace {
using namespace thread_schedule;
constexpr StringLiteral Marker = "frisk.softmax_cvt_pack";

bool match(arith::TruncFOp cast) {
  auto *ctx = cast.getContext();
  if (cast.getIn().getType() != VectorType::get({4}, Float32Type::get(ctx)) ||
      cast.getType() != VectorType::get({4}, Float16Type::get(ctx)) ||
      cast.getResult().use_empty() || cast.getRoundingmode())
    return false;
  // The f32 exp2 may also feed the denominator. Only change this narrowing
  // edge, never its producer or the original FP32 reduction DAG.
  if (!cast.getIn().getDefiningOp<math::Exp2Op>()) return false;
  for (OpOperand &use : cast.getResult().getUses())
    if (!mma(use.getOwner()) || use.getOperandNumber() != 0)
      return false;
  return true;
}

class ThreadSoftmaxCvtPackPass
    : public PassWrapper<ThreadSoftmaxCvtPackPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ThreadSoftmaxCvtPackPass)
  ThreadSoftmaxCvtPackPass() = default;
  ThreadSoftmaxCvtPackPass(const ThreadSoftmaxCvtPackPass &other) : PassWrapper(other) {}
  ThreadSoftmaxCvtPackPass(bool enabled, bool native) {
    mode = enabled ? "rtz" : "off";
    nativeExp2 = native;
  }
  Option<std::string> mode{*this, "mode", llvm::cl::init("off")};
  Option<bool> nativeExp2{*this, "native-exp2", llvm::cl::init(false)};
  StringRef getArgument() const final { return "frisk-thread-softmax-cvt-pack"; }
  StringRef getDescription() const final {
    return "Select RTZ softmax/PV register conversion in ThreadIR";
  }
  void getDependentDialects(DialectRegistry &r) const final {
    r.insert<FriskDialect, arith::ArithDialect, math::MathDialect>();
  }
  void runOnOperation() final {
    if (mode != "off" && mode != "rtz") {
      getOperation().emitError("use mode=off|rtz (rtz changes rounding)");
      return signalPassFailure();
    }
    if (mode == "off" || !nativeExp2) return;
    bool conflict = false;
    getOperation().walk([&](func::FuncOp f) {
      if (auto old = f->getAttrOfType<StringAttr>(Marker))
        if (old.getValue() != "rtz") {
          f.emitError("conflicting softmax cvt pack mode"); conflict = true;
        }
    });
    if (conflict) return signalPassFailure();
    getOperation().walk([&](func::FuncOp f) {
      // thread_num identifies a kernel in this pipeline, before GPU attributes
      // and the LLVM AMDGPU_KERNEL calling convention are materialized.
      auto threads = f->getAttrOfType<IntegerAttr>("thread_num");
      if (f.isExternal() || !threads || threads.getInt() <= 0 || protectedFunction(f)) return;
      SmallVector<arith::TruncFOp> casts;
      f.walk([&](arith::TruncFOp cast) { if (match(cast)) casts.push_back(cast); });
      for (auto cast : casts) {
        OpBuilder b(cast);
        auto packed = b.create<ConvertPackOp>(cast.getLoc(), cast.getType(),
                                             cast.getIn(), b.getStringAttr("rtz"));
        cast.getResult().replaceAllUsesWith(packed.getResult());
        cast.erase();
      }
      if (casts.empty()) return;
      if (!f->hasAttr(Marker)) markFunction(f, Marker, "rtz");
      llvm::errs() << "Thread softmax cvt pack: fragments=" << casts.size()
                   << ", packed-pairs=" << 2 * casts.size()
                   << ", rounding=rtz, function=" << f.getName() << '\n';
    });
  }
};
} // namespace

bool parseSoftmaxCvtPackOption(llvm::StringRef arg, bool &enabled, bool &consumed) {
  consumed = arg.starts_with("--softmax-cvt-pack");
  if (!consumed) return true;
  if (!arg.consume_front("--softmax-cvt-pack=")) return false;
  if (arg != "off" && arg != "rtz") return false;
  enabled = arg == "rtz";
  return true;
}

std::unique_ptr<Pass> createThreadSoftmaxCvtPackPass(bool enabled, bool nativeExp2) {
  return std::make_unique<ThreadSoftmaxCvtPackPass>(enabled, nativeExp2);
}
} // namespace mlir::frisk
