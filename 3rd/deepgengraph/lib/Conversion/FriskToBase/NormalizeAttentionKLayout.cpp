// ===========================================================================
// Pass: NormalizeAttentionKLayout（normalize-attention-k-layout）   优化汇总.py 第 3 项，block 阶段
//
// 作用：
//   把 Attn_p2 的 K 输入从 BHDS 改成 BHSD（连续布局），内部用 memref.transpose 这种零拷贝
//   view 还原 QK tile 需要的逻辑视图。这样 K 在 GM 里按连续地址读取，后面的 DMA 更规整。
//
// 会改变外部 ABI：调用方必须按 BHSD 提供 K 的存储。
//
// 硬编码前提（只服务这个例子，不是通用 Pass）：
//   - 函数名是 Attn_p2，4 个 memref 参数，无返回值。
//   - 参数 0 = Q，参数 3 = O，K 是 1/2 中 trailing shape 相对 Q 翻转的那个。
//   - S == D 时形状无法区分 K 和 V，必须用 key-arg 显式指定。
//
// 产物：
//   frisk.k_layout_bhsd（函数属性）    标记已处理，重复运行时直接返回。
//   frisk.shared_role（参数属性）      q / k / v，后面的 swizzle 和 LDS 规划靠它识别角色。
// ===========================================================================
#include "deepgengraph/Conversion/FriskToBase/Passes.h"
#include "deepgengraph/Dialect/Frisk/IR/FriskDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::frisk {
#define GEN_PASS_DEF_NORMALIZEATTENTIONKLAYOUT
#include "deepgengraph/Conversion/FriskToBase/Passes.h.inc"

namespace {
class NormalizeAttentionKLayout
    : public impl::NormalizeAttentionKLayoutBase<NormalizeAttentionKLayout> {
  void runOnOperation() override {
    auto module = getOperation();
    auto func = module.lookupSymbol<func::FuncOp>("Attn_p2");
    if (!func || func->hasAttr("frisk.k_layout_bhsd"))
      return;
    auto reject = [&](const Twine &message) {
      func.emitError(message);
      signalPassFailure();
    };
    if (func.isDeclaration() || func.getNumArguments() != 4 ||
        func.getNumResults() != 0) {
      reject("K layout normalization expects the four-argument Attn_p2 kernel");
      return;
    }
    auto qType = dyn_cast<MemRefType>(func.getArgument(0).getType());
    if (!qType || qType.getRank() != 4 || !qType.hasStaticShape()) {
      reject("K layout normalization expects a static rank-four Q");
      return;
    }
    // bhds：Q 是 BHSD，交换最后两维得到 K 在 BHDS 布局下应有的形状，用来识别 K。
    SmallVector<int64_t> bhds(qType.getShape());
    std::swap(bhds[2], bhds[3]);
    // index：K 的参数下标。-1 表示按形状自动识别。
    int index = keyArg;
    if (index == -1) {
      if (bhds[2] == bhds[3]) {
        reject("S == D is ambiguous; specify key-arg=1 or key-arg=2 for BHDS K");
        return;
      }
      for (int i : {1, 2}) {
        auto type = dyn_cast<MemRefType>(func.getArgument(i).getType());
        if (type && type.getShape() == ArrayRef<int64_t>(bhds)) {
          if (index != -1) {
            reject("multiple BHDS candidates; specify key-arg explicitly");
            return;
          }
          index = i;
        }
      }
      if (index == -1) {
        // Already BHSD is a no-op only for the exact supported signature.
        if (func.getArgument(1).getType() == qType &&
            func.getArgument(2).getType() == qType &&
            func.getArgument(3).getType() == qType)
          return;
        reject("cannot identify BHDS K in the Attn_p2 signature");
        return;
      }
    }
    if (index != 1 && index != 2) {
      reject("key-arg must be 1 or 2");
      return;
    }
    // 下面先校验完整签名，再改写，避免改到一半才失败。
    auto key = func.getArgument(index);
    auto oldType = dyn_cast<MemRefType>(key.getType());
    if (!oldType || oldType.getShape() != ArrayRef<int64_t>(bhds) ||
        !oldType.getLayout().isIdentity() ||
        oldType.getElementType() != qType.getElementType() ||
        oldType.getMemorySpace() != qType.getMemorySpace() ||
        func.getArgument(3 - index).getType() != qType ||
        func.getArgument(3).getType() != qType) {
      reject("expected contiguous BHDS K and matching BHSD Q/V/O");
      return;
    }
    // This pass changes an externally launched kernel ABI, not call sites.
    if (!SymbolTable::symbolKnownUseEmpty(func, module)) {
      reject("cannot change Attn_p2 ABI while symbol uses exist in the module");
      return;
    }
    for (Operation *user : key.getUsers()) {
      if (!isa<frisk::BufferViewOp>(user)) {
        reject("K layout normalization currently supports buffer_view users only");
        return;
      }
    }
    // 改写：直接改参数类型和函数签名，然后插入 transpose 把新布局还原成旧的逻辑视图，
    // 原来的 buffer_view 使用者全部改用这个 transpose 结果。
    auto newType = MemRefType::get(qType.getShape(), oldType.getElementType(),
                                  AffineMap(), oldType.getMemorySpace());
    SmallVector<Type> inputs(func.getFunctionType().getInputs());
    inputs[index] = newType;
    key.setType(newType);
    func.setType(FunctionType::get(&getContext(), inputs,
                                  func.getFunctionType().getResults()));
    OpBuilder builder(&func.front(), func.front().begin());
    auto permutation = AffineMap::getPermutationMap(
        ArrayRef<unsigned>{0, 1, 3, 2}, &getContext());
    // Only a strided view: there is no allocation or full-tensor transpose.
    auto transposed = builder.create<memref::TransposeOp>(
        func.getLoc(), key, AffineMapAttr::get(permutation));
    key.replaceAllUsesExcept(transposed.getResult(), transposed);
    // Preserve semantic role provenance through later physical-layout planning.
    // This helper already validated the example ABI; generic passes never
    // guess K versus V from argument order or identical BHSD tensor shapes.
    func.setArgAttr(0, "frisk.shared_role", builder.getStringAttr("q"));
    func.setArgAttr(index, "frisk.shared_role", builder.getStringAttr("k"));
    func.setArgAttr(3 - index, "frisk.shared_role", builder.getStringAttr("v"));
    func->setAttr("frisk.k_layout_bhsd", builder.getUnitAttr());
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createNormalizeAttentionKLayoutPass() {
  return std::make_unique<NormalizeAttentionKLayout>();
}
} // namespace mlir::frisk
