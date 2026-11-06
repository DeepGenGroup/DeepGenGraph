// ===========================================================================
// Pass: lazySoftmaxBase（LazySoftmaxBasePass）  LLVM 阶段，--lazy-softmax-base=<T>
//
// 作用：
//   safe online softmax 每轮都把 S 减到新的 running max M、并把 O、L 乘上 alpha 换基准。
//   每个元素一次减法、每个累加器一次乘法，是 MMAC 之外最大的一块 VALU 开销。
//   但 O/L 的比值与缩放无关，基准不必每轮都推进到真实最大值。
//
//   before: 每轮  M' = max(peak, M);  alpha = 2^(M-M');  S -= M';  O *= alpha;  L *= alpha
//   after:  状态改成 N = -B（B 是当前基准）。QK 的累加器初值用 N，算出来直接就是 S' = S - B，
//           减法消失。只有当本 wave 某行的 S' 峰值超过 T 时，才进慢路径：
//             d = 首轮 ? peak : max(peak, 0);  S' -= d;  O *= 2^-d;  L *= 2^-d;  N -= d
//
// 为什么等价：
//   慢路径里 peak 是 S' = S - B 的行最大值，即 peak_orig - B。于是
//     d = max(peak_orig - B, 0)，  B + d = max(peak_orig, B) = 原来的 M'，
//     S' - d = S - (B+d) = S - M'，  2^-d = 2^(B - (B+d)) = 2^(M - M')。
//   所以慢路径与原递推逐条对应，B 取代 M。对“跨 lane 行最大值在该行各 lane 内一致”的依赖
//   和原来完全相同，没有引入新前提：原来的 alpha 也是按 lane 算出来再乘到 O、L 上的。
//   快路径只是跳过一次重缩放，此时 S' <= T，所以 W = 2^S' <= 2^T，fp16 的 P 不会溢出
//   （T 要留在 15 以内）；基准不超过真实最大值，所以 P 只会比原来大，下溢不会变差。
//
//   首轮必须强制进慢路径：N 从 0 开始而不是 -inf，如果这一轮的分数都远小于 0（例如全是 -100），
//   快路径会让 P 全部下溢成 0、L 为 0，最后输出 0/0。强制首轮把 B 置成真实行最大值，
//   和原来 M 从 -inf 开始、第一轮 M' = peak 的效果一致。
//   首轮的 alpha 取 1 而不是 2^-d：此时 O、L 都是 0，而 2^-d 可能是 inf，0 * inf 会得到 NaN。
//
// 分支按 wave 判定（ballot），所以不需要跨 wave 同步，两条路径的 barrier 数相同；
// 慢路径只有纯 ALU，没有访存、同步和 MMAC，匹配时会逐条检查。
// 某个 lane 不需要重缩放时 d = max(peak,0) = 0，缩放退化成恒等，所以被 ballot 带进慢路径
// 的 lane 不会被改坏。
//
// 只处理 non-causal：causal 的 -inf mask 没有建模，匹配不到就整个函数放弃。
//
// 必须在依赖 workgroup-uniform 控制流的 Pass 之后运行：它引入的分支只是 wave uniform，
// 会让 analyzeWave64Kernel 之后拒绝这个函数，所以要排在 prefetchNextK 和
// elideBackedgeBarrier 之后。纯寄存器的 SoftmaxCvtPack 可以接在它之后。
//
// 代码结构：
//   resolveField / resolveAggregateField  穿透 insert/extract/shuffle 找字段来源
//   Row / Plan        匹配结果
//   matchRow          一行的 M' / base / alpha 形状
//   plan              函数和循环级前置检查、收集所有要删改的指令
//   apply             执行改写
// ===========================================================================
#include "deepgengraph/Conversion/ConvertToLLVM/LazySoftmaxBase.h"
#include "LateIRPatterns.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <optional>

using namespace llvm;
namespace mlir::frisk {
namespace {
using detail::analyzeWave64Kernel;
using detail::isPureInst;
using detail::isTypedRegisterMMACall;
using Inst = Instruction;

constexpr StringLiteral Marker = "frisk.lazy_softmax_base";
constexpr unsigned MaxRows = 8;

// 一个字段引用：base 这个聚合的第 index 个字段里，被 elements 位掩码选中的元素。
struct FieldRef {
  Value *base = nullptr;
  unsigned index = 0;
  uint64_t elements = 0;
};

unsigned elementCount(Type *type) {
  auto *vector = dyn_cast<FixedVectorType>(type);
  return vector ? vector->getNumElements() : 1;
}
uint64_t fullMask(Type *type) {
  unsigned count = elementCount(type);
  return count >= 64 ? ~0ull : (1ull << count) - 1;
}

// 把 extractvalue [-> shufflevector | extractelement] 这条链解析成字段引用。
// 只接受从单个来源向量里挑常量下标的 shuffle，不做多级组合。
std::optional<FieldRef> resolveField(Value *v) {
  uint64_t mask = 0;
  bool haveMask = false;
  for (unsigned budget = 0; budget < 8; ++budget) {
    if (auto *shuffle = dyn_cast<ShuffleVectorInst>(v)) {
      if (haveMask) return std::nullopt;
      auto *source = dyn_cast<FixedVectorType>(shuffle->getOperand(0)->getType());
      if (!source) return std::nullopt;
      // 第二个操作数必须完全没被选中，否则这不是“取某字段的一段”。
      for (int index : shuffle->getShuffleMask()) {
        if (index < 0 || unsigned(index) >= source->getNumElements()) return std::nullopt;
        mask |= 1ull << unsigned(index);
      }
      haveMask = true;
      v = shuffle->getOperand(0);
      continue;
    }
    if (auto *extract = dyn_cast<ExtractElementInst>(v)) {
      if (haveMask) return std::nullopt;
      auto *index = dyn_cast<ConstantInt>(extract->getIndexOperand());
      if (!index || index->getValue().getActiveBits() > 6) return std::nullopt;
      mask = 1ull << index->getZExtValue();
      haveMask = true;
      v = extract->getVectorOperand();
      continue;
    }
    if (auto *extract = dyn_cast<ExtractValueInst>(v)) {
      if (extract->getNumIndices() != 1) return std::nullopt;
      return FieldRef{extract->getAggregateOperand(), extract->getIndices()[0],
                      haveMask ? mask : fullMask(extract->getType())};
    }
    return std::nullopt;
  }
  return std::nullopt;
}

// 沿 insertvalue 链找出聚合值里第 index 个字段真正的来源。
Value *resolveAggregateField(Value *aggregate, unsigned index) {
  for (unsigned budget = 0; budget < 64; ++budget) {
    auto *insert = dyn_cast<InsertValueInst>(aggregate);
    if (!insert || insert->getNumIndices() != 1) return nullptr;
    if (insert->getIndices()[0] == index) return insert->getInsertedValueOperand();
    aggregate = insert->getAggregateOperand();
  }
  return nullptr;
}

// 广播成宽度 width 的向量；width 为 1 时保持 <1 x float> 或标量。
Value *broadcast(IRBuilder<> &builder, Value *scalar, unsigned width, const Twine &name) {
  Type *type = scalar->getType();
  if (auto *vector = dyn_cast<FixedVectorType>(type)) {
    if (vector->getNumElements() != 1) return nullptr;
    if (width == 1) return scalar;
    return builder.CreateShuffleVector(scalar, SmallVector<int>(width, 0), name);
  }
  Value *lane = builder.CreateInsertElement(
      PoisonValue::get(FixedVectorType::get(type, 1)), scalar, uint64_t(0), name + ".lane");
  if (width == 1) return lane;
  return builder.CreateShuffleVector(lane, SmallVector<int>(width, 0), name);
}

// maxnum 的操作数可能是 <1 x float> 包起来的标量，取出标量形式。
Value *scalarOf(Value *v) {
  if (!v->getType()->isFloatTy()) {
    auto *insert = dyn_cast<InsertElementInst>(v);
    if (!insert || !isa<PoisonValue, UndefValue>(insert->getOperand(0))) return nullptr;
    auto *index = dyn_cast<ConstantInt>(insert->getOperand(2));
    if (!index || !index->isZero()) return nullptr;
    v = insert->getOperand(1);
  }
  return v->getType()->isFloatTy() ? v : nullptr;
}

// 全 0 掩码的广播。不能用 isZeroEltSplat()：它要求不改变向量长度，而 <1 x float>
// 广播成 <4 x float> 正好改变长度。
bool isZeroSplat(ShuffleVectorInst *shuffle) {
  if (!isa<PoisonValue, UndefValue>(shuffle->getOperand(1))) return false;
  for (int index : shuffle->getShuffleMask())
    if (index != 0) return false;
  return !shuffle->getShuffleMask().empty();
}

bool isMaxnum(Inst *i) {
  auto *call = dyn_cast<IntrinsicInst>(i);
  return call && call->getIntrinsicID() == Intrinsic::maxnum && call->arg_size() == 2;
}

// softmax 的一行。index 是 M 聚合里的字段下标。
struct Row {
  unsigned index = 0;
  Inst *oldMaxExtract = nullptr;  // extractvalue M, index
  Value *peak = nullptr;         // 这一行 S' 的跨 lane 最大值（标量）
  Inst *peakInsert = nullptr;    // peak 的 <1 x float> 包装，改写后变成死代码
  Inst *newMax = nullptr;        // maxnum(peak, oldMax)
  Inst *base = nullptr;          // newMax 广播成分数宽度
  Inst *diff = nullptr;          // fsub oldMax, newMax
  Inst *diffElement = nullptr;   // extractelement diff, 0
  CallInst *alphaScalar = nullptr;
  Inst *alphaInsert = nullptr;
  Inst *alphaVector = nullptr;
  SmallVector<Inst *> shifts;    // fsub <N x float> score, base
  SmallVector<Inst *> scales;    // fmul state, alpha
};

struct Plan {
  Loop *loop = nullptr;
  BasicBlock *header = nullptr, *latch = nullptr, *preheader = nullptr, *split = nullptr;
  PHINode *maxState = nullptr, *iv = nullptr;
  Inst *splitAt = nullptr;
  Inst *firstSeed = nullptr;
  SmallVector<Row, 2> rows;
  MapVector<PHINode *, SmallVector<uint64_t>> scaled;  // 被 alpha 缩放的状态 PHI -> 每字段覆盖
  MapVector<Inst *, unsigned> seeds;                   // 零初值 MMAC -> 行号
  SmallVector<Inst *> dead;
};

// 匹配第 index 行的 M' / base / alpha 形状。所有匹配到的指令都登记到 dead。
bool matchRow(Plan &plan, unsigned index, Value *latchValue, std::string &why) {
  auto fail = [&](StringRef reason) {
    why = ("row " + Twine(index) + ": " + reason).str();
    return false;
  };
  Row row;
  row.index = index;
  // oldMax：循环内对 M 这个字段唯一的 extractvalue。
  for (User *user : plan.maxState->users()) {
    auto *extract = dyn_cast<ExtractValueInst>(user);
    if (!extract || !plan.loop->contains(extract->getParent())) continue;
    if (extract->getNumIndices() != 1 || extract->getIndices()[0] != index) continue;
    if (row.oldMaxExtract) return fail("several extracts of the running max field");
    row.oldMaxExtract = extract;
  }
  if (!row.oldMaxExtract) return fail("missing running-max extract");

  // newMax：回边值里这个字段的来源，必须是 maxnum(peak, oldMax)。
  auto *newMax = dyn_cast_or_null<Inst>(resolveAggregateField(latchValue, index));
  if (!newMax || !isMaxnum(newMax)) return fail("field is not a maxnum of the running max");
  auto *call = cast<CallInst>(newMax);
  unsigned oldOperand = call->getArgOperand(1) == row.oldMaxExtract  ? 1
                        : call->getArgOperand(0) == row.oldMaxExtract ? 0
                                                                      : 2;
  if (oldOperand == 2) return fail("maxnum does not take the running max");
  row.newMax = newMax;
  Value *given = call->getArgOperand(1 - oldOperand);
  row.peak = scalarOf(given);
  if (!row.peak) return fail("row peak is not a scalar float");
  if (given != row.peak) row.peakInsert = cast<Inst>(given);
  if (newMax->getType() != row.oldMaxExtract->getType()) return fail("maxnum changes the field type");

  // newMax 的用户只允许三种：分数基准广播、alpha 的差、以及构造回边值的 insertvalue。
  for (User *user : newMax->users()) {
    auto *inst = dyn_cast<Inst>(user);
    if (!inst) return fail("unknown use of the new maximum");
    if (auto *shuffle = dyn_cast<ShuffleVectorInst>(inst)) {
      if (!isZeroSplat(shuffle)) return fail("new maximum is not broadcast");
      if (row.base) return fail("new maximum is broadcast twice");
      row.base = shuffle;
      continue;
    }
    if (inst->getOpcode() == Inst::FSub && inst->getOperand(0) == row.oldMaxExtract &&
        inst->getOperand(1) == newMax) {
      if (row.diff) return fail("several alpha exponents");
      row.diff = inst;
      continue;
    }
    if (isa<InsertValueInst>(inst)) continue;  // 回边聚合，改写后变成死代码
    return fail("unknown use of the new maximum");
  }
  if (!row.base) return fail("missing score base broadcast");
  if (!row.diff) return fail("missing alpha exponent");

  // alpha = exp2(extractelement(diff, 0))，再广播。
  for (User *user : row.diff->users()) {
    auto *extract = dyn_cast<ExtractElementInst>(user);
    auto *constant = extract ? dyn_cast<ConstantInt>(extract->getIndexOperand()) : nullptr;
    if (!extract || !constant || !constant->isZero() || row.diffElement)
      return fail("unknown use of the alpha exponent");
    row.diffElement = extract;
  }
  if (!row.diffElement) return fail("missing alpha exponent element");
  for (User *user : row.diffElement->users()) {
    auto *exp = dyn_cast<CallInst>(user);
    if (!exp || !exp->getCalledFunction() || exp->arg_size() != 1 ||
        !exp->getCalledFunction()->getName().contains("exp2") || row.alphaScalar)
      return fail("alpha is not a single exp2");
    row.alphaScalar = exp;
  }
  if (!row.alphaScalar) return fail("missing alpha exp2");

  for (User *user : row.alphaScalar->users()) {
    auto *inst = dyn_cast<Inst>(user);
    if (!inst) return fail("unknown use of alpha");
    if (auto *insert = dyn_cast<InsertElementInst>(inst)) {
      if (row.alphaInsert) return fail("alpha is broadcast twice");
      row.alphaInsert = insert;
      continue;
    }
    if (inst->getOpcode() == Inst::FMul) {
      row.scales.push_back(inst);
      continue;
    }
    return fail("unknown use of alpha");
  }
  if (row.alphaInsert) {
    for (User *user : row.alphaInsert->users()) {
      auto *shuffle = dyn_cast<ShuffleVectorInst>(user);
      if (!shuffle || !isZeroSplat(shuffle) || row.alphaVector)
        return fail("unknown use of the alpha lane vector");
      row.alphaVector = shuffle;
    }
    if (!row.alphaVector) return fail("missing alpha broadcast");
    for (User *user : row.alphaVector->users()) {
      auto *inst = dyn_cast<Inst>(user);
      if (!inst || inst->getOpcode() != Inst::FMul) return fail("unknown use of the alpha vector");
      row.scales.push_back(inst);
    }
  }
  if (row.scales.empty()) return fail("alpha scales nothing");

  // base 只许用于分数的减法，且必须是减数。
  for (User *user : row.base->users()) {
    auto *inst = dyn_cast<Inst>(user);
    if (!inst || inst->getOpcode() != Inst::FSub || inst->getOperand(1) != row.base)
      return fail("unknown use of the score base");
    row.shifts.push_back(inst);
  }
  if (row.shifts.empty()) return fail("score base shifts nothing");

  for (Inst *inst : {row.oldMaxExtract, row.peakInsert, row.newMax, row.base, row.diff,
                     row.diffElement, cast<Inst>(row.alphaScalar), row.alphaInsert,
                     row.alphaVector})
    if (inst) plan.dead.push_back(inst);
  plan.dead.append(row.shifts);
  plan.dead.append(row.scales);
  plan.rows.push_back(row);
  return true;
}

bool plan(Function &function, LoopInfo &loops, DominatorTree &dt, Plan &plan, std::string &why) {
  auto fail = [&](StringRef reason) {
    why = reason.str();
    return false;
  };
  if (std::distance(loops.begin(), loops.end()) != 1) return fail("requires one top-level loop");
  plan.loop = *loops.begin();
  plan.header = plan.loop->getHeader();
  plan.latch = plan.loop->getLoopLatch();
  plan.preheader = plan.loop->getLoopPreheader();
  if (!plan.latch || !plan.preheader || plan.loop->getNumBlocks() < 2)
    return fail("requires a preheader and a single latch");
  for (BasicBlock *block : plan.loop->blocks())
    if (block != plan.header && isa<PHINode>(&block->front()))
      return fail("requires PHIs only in the loop header");

  // running max 的状态 PHI：聚合类型，入口值每个字段都是 -inf。
  for (PHINode &phi : plan.header->phis()) {
    auto *type = dyn_cast<ArrayType>(phi.getType());
    if (!type || type->getNumElements() > MaxRows) continue;
    Type *field = type->getElementType();
    if (!field->isFloatTy() && !(isa<FixedVectorType>(field) && elementCount(field) == 1 &&
                                 field->getScalarType()->isFloatTy()))
      continue;
    auto *initial = dyn_cast<Constant>(phi.getIncomingValueForBlock(plan.preheader));
    if (!initial) continue;
    bool negativeInfinity = true;
    for (unsigned r = 0; r < type->getNumElements() && negativeInfinity; ++r) {
      Constant *element = initial->getAggregateElement(r);
      for (unsigned e = 0; element && e < elementCount(field); ++e) {
        auto *scalar = dyn_cast_or_null<ConstantFP>(
            isa<FixedVectorType>(field) ? element->getAggregateElement(e) : element);
        negativeInfinity &= scalar && scalar->getValue().isNegInfinity();
      }
      negativeInfinity &= element != nullptr;
    }
    if (!negativeInfinity) continue;
    if (plan.maxState) return fail("several running-max candidates");
    plan.maxState = &phi;
  }
  if (!plan.maxState) return fail("missing a running-max state PHI initialized to -inf");
  if (plan.maxState->getNumIncomingValues() != 2) return fail("running max has several entries");

  // 零起点、步长 1 的归纳变量，用来判断首轮。
  for (PHINode &phi : plan.header->phis()) {
    if (!phi.getType()->isIntegerTy() || phi.getNumIncomingValues() != 2) continue;
    auto *initial = dyn_cast<ConstantInt>(phi.getIncomingValueForBlock(plan.preheader));
    auto *next = dyn_cast<BinaryOperator>(phi.getIncomingValueForBlock(plan.latch));
    if (!initial || !initial->isZero() || !next || next->getOpcode() != Inst::Add) continue;
    auto *step = dyn_cast<ConstantInt>(next->getOperand(1));
    if (next->getOperand(0) != &phi || !step || !step->isOne()) continue;
    if (plan.iv) return fail("several zero-based unit-stride induction variables");
    plan.iv = &phi;
  }
  if (!plan.iv) return fail("missing a zero-based unit-stride induction variable");

  Value *latchValue = plan.maxState->getIncomingValueForBlock(plan.latch);
  unsigned fields = cast<ArrayType>(plan.maxState->getType())->getNumElements();
  if (!fields) return fail("the running-max state has no fields");
  for (unsigned r = 0; r < fields; ++r)
    if (!matchRow(plan, r, latchValue, why)) return false;
  // 回边聚合的整条 insertvalue 链在状态改接到 join PHI 之后全部变成死代码。
  for (Value *chain = latchValue; auto *insert = dyn_cast<InsertValueInst>(chain);
       chain = insert->getAggregateOperand()) {
    if (!plan.loop->contains(insert->getParent())) return fail("the state is rebuilt outside the loop");
    plan.dead.push_back(insert);
  }

  // 每个被 alpha 缩放的状态字段：必须整体被同一行的 alpha 覆盖一次。
  for (Row &row : plan.rows)
    for (Inst *scale : row.scales) {
      Value *alpha = row.alphaVector && scale->getOperand(1) == row.alphaVector ? row.alphaVector
                     : scale->getOperand(1) == row.alphaScalar ? cast<Value>(row.alphaScalar)
                                                               : nullptr;
      if (!alpha) return fail("alpha is not the second factor of a scale");
      auto field = resolveField(scale->getOperand(0));
      if (!field) return fail("cannot resolve the scaled state field");
      auto *state = dyn_cast<PHINode>(field->base);
      if (!state || state->getParent() != plan.header)
        return fail("the scaled value is not a loop state field");
      if (field->index != row.index) return fail("alpha scales another row's field");
      auto *type = dyn_cast<ArrayType>(state->getType());
      if (!type || type->getNumElements() != fields) return fail("unexpected scaled state shape");
      auto &covered = plan.scaled[state];
      covered.resize(fields);
      if (covered[field->index] & field->elements) return fail("the state field is scaled twice");
      covered[field->index] |= field->elements;
    }
  for (auto &entry : plan.scaled) {
    Type *field = cast<ArrayType>(entry.first->getType())->getElementType();
    for (uint64_t covered : entry.second)
      if (covered != fullMask(field)) return fail("a scaled state field is only partly scaled");
  }
  if (plan.scaled.empty()) return fail("no scaled state");

  // 切点的位置由两头夹出来：
  //   必须在它之前算好 —— 各行最大值、以及要移位的分数本身（慢路径要用）。
  //   必须在它之后出现 —— 分数减法、状态缩放、以及任何对被缩放状态的读取
  //                      （这样它们才能看到 join 处汇合后的值）。
  // 换基准的其余计算（旧最大值、M'、基准广播、alpha）改写后都变成死代码，位置不受限制。
  plan.split = plan.rows.front().shifts.front()->getParent();
  if (!plan.loop->contains(plan.split)) return fail("the rescale is outside the loop");
  for (Inst *inst : plan.dead)
    if (!plan.loop->contains(inst->getParent())) return fail("the rescale escapes the loop");

  SmallVector<Inst *> mustFollow;
  for (Row &row : plan.rows) {
    for (Inst *shift : row.shifts) {
      if (shift->getParent() != plan.split)
        return fail("the score shifts are spread over several blocks");
      mustFollow.push_back(shift);
    }
    mustFollow.append(row.scales);
  }
  for (auto &entry : plan.scaled)
    for (Use &use : entry.first->uses()) {
      auto *user = cast<Inst>(use.getUser());
      if (plan.loop->contains(user->getParent())) mustFollow.push_back(user);
    }
  for (Inst *inst : mustFollow)
    if (inst->getParent() == plan.split && (!plan.splitAt || inst->comesBefore(plan.splitAt)))
      plan.splitAt = inst;
  if (!plan.splitAt) return fail("nothing to move past the rescale");

  auto beforeSplit = [&](Value *v) {
    auto *inst = dyn_cast<Inst>(v);
    if (!inst) return true;
    return inst->getParent() == plan.split ? inst->comesBefore(plan.splitAt)
                                           : dt.dominates(inst->getParent(), plan.split);
  };
  auto afterSplit = [&](Inst *inst) {
    return inst->getParent() == plan.split ? !inst->comesBefore(plan.splitAt)
                                           : dt.dominates(plan.split, inst->getParent());
  };
  for (Inst *inst : mustFollow)
    if (!afterSplit(inst)) return fail("the rescaled state is used before the rescale");
  for (Row &row : plan.rows) {
    if (!beforeSplit(row.peak)) return fail("the row peak is not available at the branch");
    for (Inst *shift : row.shifts)
      if (!beforeSplit(shift->getOperand(0)))
        return fail("a shifted score is not available at the branch");
  }

  // 零初值的 QK 累加链：链尾必须正好被一条分数减法消费，由它定出行号。
  DenseMap<Inst *, Inst *> shiftOf;
  for (Row &row : plan.rows)
    for (Inst *shift : row.shifts) {
      auto *score = dyn_cast<Inst>(shift->getOperand(0));
      if (!score) return fail("the shifted score is not an instruction");
      if (!shiftOf.insert({score, shift}).second) return fail("a score is shifted twice");
    }
  DenseMap<Inst *, unsigned> rowOfShift;
  for (Row &row : plan.rows)
    for (Inst *shift : row.shifts) rowOfShift[shift] = row.index;

  for (BasicBlock *block : plan.loop->blocks())
    for (Inst &inst : *block) {
      if (!isTypedRegisterMMACall(&inst)) continue;
      auto *seed = dyn_cast<Constant>(cast<CallInst>(&inst)->getArgOperand(2));
      if (!seed || !seed->isNullValue()) continue;
      Inst *end = &inst;
      for (unsigned budget = 0; budget < 64; ++budget) {
        Inst *next = nullptr;
        for (User *user : end->users()) {
          auto *candidate = dyn_cast<Inst>(user);
          if (candidate && isTypedRegisterMMACall(candidate) &&
              cast<CallInst>(candidate)->getArgOperand(2) == end)
            next = next ? nullptr : candidate;
        }
        if (!next) break;
        end = next;
      }
      auto found = shiftOf.find(end);
      if (found == shiftOf.end()) return fail("a zero-seeded MMAC chain feeds no score shift");
      if (!inst.comesBefore(plan.splitAt) || inst.getParent() != plan.split)
        return fail("a zero-seeded MMAC is not before the rescale");
      plan.seeds[&inst] = rowOfShift[found->second];
      if (!plan.firstSeed || inst.comesBefore(plan.firstSeed)) plan.firstSeed = &inst;
    }
  unsigned shifts = 0;
  for (Row &row : plan.rows) shifts += row.shifts.size();
  if (plan.seeds.size() != shifts)
    return fail("the zero-seeded MMAC chains do not match the shifted scores");
  return true;
}

void apply(Plan &p, float threshold) {
  LLVMContext &ctx = p.split->getContext();
  Module *module = p.split->getModule();
  Type *floatType = Type::getFloatTy(ctx);
  DebugLoc where = p.splitAt->getDebugLoc();

  // 1. QK 的累加器初值改成 N 的该行字段。
  {
    IRBuilder<> builder(p.firstSeed);
    builder.SetCurrentDebugLocation(p.firstSeed->getDebugLoc());
    unsigned width = elementCount(cast<CallInst>(p.firstSeed)->getArgOperand(2)->getType());
    SmallVector<Value *> seed(p.rows.size());
    for (Row &row : p.rows) {
      Value *field = builder.CreateExtractValue(p.maxState, row.index, "lazy.base");
      seed[row.index] = broadcast(builder, field, width, "lazy.seed");
    }
    for (auto &entry : p.seeds)
      cast<CallInst>(entry.first)->setArgOperand(2, seed[entry.second]);
  }
  // N 从 0 开始（基准 B = 0），不再是 -inf。
  p.maxState->setIncomingValueForBlock(p.preheader,
                                       Constant::getNullValue(p.maxState->getType()));

  // 2. 切块，并在前半块末尾建立 wave 级判定。
  // LoopInfo 在切块后就过期了，join 和 slow 都不在它的块集合里，所以先记下来再补上。
  SmallPtrSet<BasicBlock *, 8> inLoop(p.loop->block_begin(), p.loop->block_end());
  BasicBlock *join = p.split->splitBasicBlock(p.splitAt, "lazy.join");
  inLoop.insert(join);
  BasicBlock *slow = BasicBlock::Create(ctx, "lazy.slow", p.split->getParent(), join);
  Value *first = nullptr, *take = nullptr;
  {
    IRBuilder<> builder(p.split->getTerminator());
    builder.SetCurrentDebugLocation(where);
    Value *peak = p.rows.front().peak;
    for (unsigned r = 1; r < p.rows.size(); ++r)
      peak = builder.CreateBinaryIntrinsic(Intrinsic::maxnum, peak, p.rows[r].peak, nullptr,
                                           "lazy.peak");
    Value *over = builder.CreateFCmpOGT(peak, ConstantFP::get(floatType, threshold), "lazy.over");
    // ballot：本 wave 有任何 lane 需要重缩放就整个 wave 进慢路径。
    Value *mask = builder.CreateCall(
        Intrinsic::getOrInsertDeclaration(module, Intrinsic::amdgcn_ballot, {builder.getInt64Ty()}),
        {over}, "lazy.mask");
    Value *any = builder.CreateIsNotNull(mask, "lazy.any");
    first = builder.CreateICmpEQ(p.iv, Constant::getNullValue(p.iv->getType()), "lazy.first");
    take = builder.CreateOr(first, any, "lazy.take");
    Inst *previous = p.split->getTerminator();
    builder.CreateCondBr(take, slow, join);
    previous->eraseFromParent();
  }

  // 3. 慢路径：纯 ALU 的重缩放。
  SmallVector<Value *> shifted, scaledState(p.scaled.size()), nextBase;
  {
    IRBuilder<> builder(slow);
    builder.SetCurrentDebugLocation(where);
    SmallVector<Value *> delta(p.rows.size()), alpha(p.rows.size());
    for (Row &row : p.rows) {
      // d = 首轮 ? peak : max(peak, 0)，和原来 M' = max(peak, M) 对应。
      Value *positive = builder.CreateBinaryIntrinsic(
          Intrinsic::maxnum, row.peak, ConstantFP::get(floatType, 0.0), nullptr, "lazy.positive");
      delta[row.index] = builder.CreateSelect(first, row.peak, positive, "lazy.delta");
      Value *negated = builder.CreateFSub(ConstantFP::get(floatType, 0.0), delta[row.index],
                                          "lazy.negated");
      auto *exp = cast<CallInst>(builder.CreateCall(row.alphaScalar->getCalledFunction(),
                                                    {negated}, "lazy.exp"));
      exp->setAttributes(row.alphaScalar->getAttributes());
      exp->copyFastMathFlags(row.alphaScalar);
      // 首轮 O、L 都是 0，而 2^-d 可能是 inf，0 * inf 会得到 NaN。
      alpha[row.index] = builder.CreateSelect(first, ConstantFP::get(floatType, 1.0), exp,
                                              "lazy.alpha");
    }
    for (Row &row : p.rows)
      for (Inst *shift : row.shifts) {
        Value *step = broadcast(builder, delta[row.index], elementCount(shift->getType()),
                                "lazy.delta.v");
        auto *result = cast<Inst>(builder.CreateFSub(shift->getOperand(0), step, "lazy.shifted"));
        result->copyFastMathFlags(shift);
        shifted.push_back(result);
      }
    unsigned slot = 0;
    for (auto &entry : p.scaled) {
      PHINode *state = entry.first;
      Type *field = cast<ArrayType>(state->getType())->getElementType();
      Value *rebuilt = PoisonValue::get(state->getType());
      for (Row &row : p.rows) {
        Value *old = builder.CreateExtractValue(state, row.index, "lazy.state");
        Value *factor = broadcast(builder, alpha[row.index], elementCount(field), "lazy.alpha.v");
        Value *scaled = builder.CreateFMul(old, factor, "lazy.scaled");
        rebuilt = builder.CreateInsertValue(rebuilt, scaled, row.index, "lazy.state.next");
      }
      scaledState[slot++] = rebuilt;
    }
    Value *rebuilt = PoisonValue::get(p.maxState->getType());
    Type *field = cast<ArrayType>(p.maxState->getType())->getElementType();
    for (Row &row : p.rows) {
      Value *old = builder.CreateExtractValue(p.maxState, row.index, "lazy.base.old");
      Value *step = broadcast(builder, delta[row.index], elementCount(field), "lazy.delta.b");
      Value *next = builder.CreateFSub(old, step, "lazy.base.next");
      rebuilt = builder.CreateInsertValue(rebuilt, next, row.index, "lazy.base.state");
    }
    nextBase.push_back(rebuilt);
    builder.CreateBr(join);
  }

  // 4. join 处汇合，并把快路径上的原值接到 PHI。
  {
    IRBuilder<> builder(&*join->getFirstInsertionPt());
    builder.SetCurrentDebugLocation(where);
    unsigned index = 0;
    for (Row &row : p.rows)
      for (Inst *shift : row.shifts) {
        PHINode *phi = builder.CreatePHI(shift->getType(), 2, "lazy.score");
        phi->addIncoming(shift->getOperand(0), p.split);
        phi->addIncoming(shifted[index++], slow);
        shift->replaceAllUsesWith(phi);
      }
    unsigned slot = 0;
    for (auto &entry : p.scaled) {
      PHINode *state = entry.first;
      PHINode *phi = builder.CreatePHI(state->getType(), 2, "lazy.state");
      phi->addIncoming(state, p.split);
      phi->addIncoming(scaledState[slot++], slow);
      state->replaceUsesWithIf(phi, [&](Use &use) {
        auto *user = cast<Inst>(use.getUser());
        return user != phi && user->getParent() != slow && inLoop.contains(user->getParent());
      });
    }
    PHINode *base = builder.CreatePHI(p.maxState->getType(), 2, "lazy.base.join");
    base->addIncoming(p.maxState, p.split);
    base->addIncoming(nextBase.front(), slow);
    p.maxState->setIncomingValueForBlock(p.latch, base);
  }
  // 缩放移进了慢路径，快路径上的原 fmul 直接透传它的被乘数。
  for (Row &row : p.rows)
    for (Inst *scale : row.scales) scale->replaceAllUsesWith(scale->getOperand(0));

  // 5. 清理。顺序未知，所以反复扫到不再有进展。
  SmallVector<Inst *> pending;
  SmallPtrSet<Inst *, 32> queued;
  for (Inst *inst : p.dead)
    if (queued.insert(inst).second) pending.push_back(inst);
  for (bool progress = true; progress;) {
    progress = false;
    SmallVector<Inst *> again;
    for (Inst *inst : pending) {
      if (!inst->use_empty()) {
        again.push_back(inst);
        continue;
      }
      inst->eraseFromParent();
      progress = true;
    }
    pending = std::move(again);
  }
  assert(pending.empty() && "lazy softmax base left dead instructions behind");
}
} // namespace

bool parseLazySoftmaxBaseOption(StringRef arg, bool &enabled, float &threshold, bool &consumed) {
  consumed = arg.starts_with("--lazy-softmax-base");
  if (!consumed) return true;
  if (!arg.consume_front("--lazy-softmax-base=")) return false;
  if (arg == "off" || arg == "0") {
    enabled = false;
    return true;
  }
  double value = 0;
  if (arg.getAsDouble(value) || !std::isfinite(value) || value <= 0 || value > 15) return false;
  enabled = true;
  threshold = float(value);
  return true;
}

PreservedAnalyses LazySoftmaxBasePass::run(Function &function, FunctionAnalysisManager &) {
  auto skip = [&](StringRef why) {
    errs() << "Lazy softmax base: skipped, function=" << function.getName() << ": " << why << "\n";
    return PreservedAnalyses::all();
  };
  if (function.isDeclaration() || function.hasFnAttribute(Marker)) return PreservedAnalyses::all();
  if (function.getCallingConv() != CallingConv::AMDGPU_KERNEL ||
      function.hasFnAttribute(Attribute::OptimizeNone) ||
      function.hasFnAttribute(Attribute::StrictFP))
    return skip("requires an optimizable AMDGPU kernel");
  std::string why;
  // 这里只用它的 wave64 / 固定 workgroup / uniform 控制流前置检查。本 Pass 不碰 LDS，
  // 但引入的分支之后不再 workgroup uniform，后续 Pass 不能依赖这项分析。
  if (!analyzeWave64Kernel(function, why)) return skip(why);
  DominatorTree dt(function);
  LoopInfo loops(dt);
  Plan matched;
  if (!plan(function, loops, dt, matched, why)) return skip(why);
  apply(matched, threshold);
  function.addFnAttr(Marker, "v1");
  errs() << "Lazy softmax base: rows=" << matched.rows.size()
         << ", seeded-accumulators=" << matched.seeds.size()
         << ", scaled-state=" << matched.scaled.size() << ", threshold=" << threshold
         << ", function=" << function.getName() << "\n";
  return PreservedAnalyses::none();
}

void lazySoftmaxBase(Module &module, bool enabled, float threshold) {
  if (!enabled) return;
  FunctionAnalysisManager analyses;
  LazySoftmaxBasePass pass(threshold);
  for (Function &function : module) pass.run(function, analyses);
}
} // namespace mlir::frisk
