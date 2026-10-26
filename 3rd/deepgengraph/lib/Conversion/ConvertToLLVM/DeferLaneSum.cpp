#include "deepgengraph/Conversion/ConvertToLLVM/DeferLaneSum.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include <map>
#include <set>
#include <vector>

using namespace llvm;
namespace mlir::frisk {
namespace {
using Path = std::vector<unsigned>;
struct Slice {
  Value *value;
  Path path;
  bool operator<(const Slice &b) const {
    return value != b.value ? std::less<Value *>{}(value, b.value) : path < b.path;
  }
  bool operator==(const Slice &b) const { return value == b.value && path == b.path; }
};
bool prefix(ArrayRef<unsigned> a, ArrayRef<unsigned> b) {
  return a.size() <= b.size() && std::equal(a.begin(), a.end(), b.begin());
}
// Resolve a field through aggregate/vector construction without changing IR.
Slice resolve(Slice s) {
  for (unsigned budget = 0; budget < 256; ++budget) {
    if (auto *e = dyn_cast<ExtractValueInst>(s.value)) {
      s.path.insert(s.path.begin(), e->idx_begin(), e->idx_end());
      s.value = e->getAggregateOperand();
    } else if (auto *e = dyn_cast<ExtractElementInst>(s.value)) {
      auto *index = dyn_cast<ConstantInt>(e->getIndexOperand());
      if (!index || index->getValue().getActiveBits() > 32) break;
      s.path.insert(s.path.begin(), index->getZExtValue());
      s.value = e->getVectorOperand();
    } else if (auto *i = dyn_cast<InsertValueInst>(s.value)) {
      ArrayRef<unsigned> indices = i->getIndices();
      if (prefix(indices, s.path)) {
        s.path.erase(s.path.begin(), s.path.begin() + indices.size());
        s.value = i->getInsertedValueOperand();
      } else if (!prefix(s.path, indices)) s.value = i->getAggregateOperand();
      else break;
    } else if (auto *i = dyn_cast<InsertElementInst>(s.value)) {
      auto *index = dyn_cast<ConstantInt>(i->getOperand(2));
      if (s.path.empty() || !index) break;
      if (index->equalsInt(s.path.front())) {
        s.path.erase(s.path.begin()); s.value = i->getOperand(1);
      } else s.value = i->getOperand(0);
    } else if (auto *c = dyn_cast<Constant>(s.value)) {
      if (s.path.empty()) break;
      auto *part = c->getAggregateElement(s.path.front());
      if (!part) break;
      s.path.erase(s.path.begin()); s.value = part;
    } else break;
  }
  return s;
}
bool enumerate(Type *type, Path path, std::vector<Path> &out) {
  if (out.size() >= 128 || path.size() > 8) return false;
  if (type->isFloatTy() || type->isIntegerTy(32)) { out.push_back(path); return true; }
  uint64_t count = 0;
  if (auto *s = dyn_cast<StructType>(type)) count = s->getNumElements();
  else if (auto *a = dyn_cast<ArrayType>(type)) count = a->getNumElements();
  else if (auto *v = dyn_cast<FixedVectorType>(type)) count = v->getNumElements();
  else return false;
  if (count > 128) return false;
  for (unsigned i = 0; i < count; ++i) {
    Type *element = isa<StructType>(type) ? cast<StructType>(type)->getElementType(i)
        : isa<ArrayType>(type) ? cast<ArrayType>(type)->getElementType()
                              : cast<FixedVectorType>(type)->getElementType();
    auto next = path; next.push_back(i);
    if (!enumerate(element, next, out)) return false;
  }
  return true;
}
StringRef callee(Value *v) {
  auto *c = dyn_cast<CallInst>(v);
  return c && c->getCalledFunction() ? c->getCalledFunction()->getName() : StringRef();
}
struct Reduction {
  Value *leaf = nullptr;
  unsigned mask = 0;
  SmallVector<unsigned> bits; // outermost to innermost
  SmallPtrSet<Instruction *, 32> nodes;
};

class Proof {
  Function &function;
  const DataLayout &layout;
  unsigned wave, threads;
  bool linearWorkitems;
  std::map<Value *, unsigned> masks;
  std::map<std::pair<Value *, unsigned>, bool> fixedUniformity;

  Constant *evaluate(Value *v, unsigned tid, unsigned &budget) {
    if (!budget--) return nullptr;
    if (auto *c = dyn_cast<ConstantInt>(v)) return c;
    auto *i = dyn_cast<Instruction>(v);
    if (!i || !i->getType()->isIntegerTy()) return nullptr;
    if (auto *call = dyn_cast<CallInst>(i)) {
      StringRef name = callee(i);
      if (name == "llvm.amdgcn.workitem.id.x" && linearWorkitems)
        return ConstantInt::get(i->getType(), tid);
      if (name != "llvm.amdgcn.mbcnt.lo" && name != "llvm.amdgcn.mbcnt.hi") return nullptr;
      auto *mask = dyn_cast<ConstantInt>(call->getArgOperand(0));
      auto *base = dyn_cast_or_null<ConstantInt>(evaluate(call->getArgOperand(1), tid, budget));
      if (!mask || !mask->isMinusOne() || !base) return nullptr;
      unsigned lane = tid % wave;
      unsigned count = name.ends_with(".lo") ? std::min(lane, 32u) : (lane > 32 ? lane - 32 : 0);
      return ConstantInt::get(i->getType(), base->getZExtValue() + count);
    }
    if (!isa<BinaryOperator, CastInst, ICmpInst, SelectInst>(i)) return nullptr;
    SmallVector<Constant *> operands;
    for (Value *operand : i->operands()) {
      auto *c = evaluate(operand, tid, budget);
      if (!c) return nullptr;
      operands.push_back(c);
    }
    auto *folded = ConstantFoldInstOperands(i, operands, layout);
    return dyn_cast_or_null<ConstantInt>(folded);
  }
public:
  Proof(Function &f, unsigned wave, unsigned threads, bool linear)
      : function(f), layout(f.getDataLayout()), wave(wave), threads(threads), linearWorkitems(linear) {}
  void clearCaches() { masks.clear(); fixedUniformity.clear(); }
  bool fixedUniform(Value *v, unsigned group) {
    if (!v->getType()->isIntegerTy()) return false;
    auto key = std::make_pair(v, group);
    auto found = fixedUniformity.find(key);
    if (found != fixedUniformity.end()) return found->second;
    SmallVector<Constant *> values;
    for (unsigned tid = 0; tid < threads; ++tid) {
      unsigned budget = 512;
      auto *value = evaluate(v, tid, budget);
      if (!value) return fixedUniformity[key] = false;
      values.push_back(value);
    }
    for (unsigned tid = 0; tid < threads; ++tid)
      for (unsigned bit = 1; bit < wave; bit *= 2)
        if ((group & bit) && values[tid] != values[tid ^ bit])
          return fixedUniformity[key] = false;
    return fixedUniformity[key] = true;
  }
  unsigned xorMask(Value *address) {
    auto found = masks.find(address);
    if (found != masks.end()) return found->second;
    unsigned mask = 0;
    for (unsigned tid = 0; tid < threads; ++tid) {
      unsigned budget = 512;
      auto *c = dyn_cast_or_null<ConstantInt>(evaluate(address, tid, budget));
      if (!c || c->getValue().getActiveBits() > 32) { mask = 0; break; }
      uint64_t bytes = c->getZExtValue();
      unsigned m = (bytes / 4) ^ (tid % wave);
      if (bytes % 4 || bytes >= wave * 4 || !isPowerOf2_32(m) || (tid && m != mask)) {
        mask = 0; break;
      }
      mask = m;
    }
    return masks[address] = mask;
  }
  // A stage combines v with a bijective XOR permutation of exactly v.
  bool stage(Value *v, unsigned kind, Value *&previous, unsigned &bit,
             SmallPtrSetImpl<Instruction *> &nodes) {
    auto *i = dyn_cast<Instruction>(v);
    if (!i) return false;
    Value *a = nullptr, *b = nullptr;
    if (kind == Instruction::Add || kind == Instruction::FAdd) {
      if (i->getOpcode() != kind) return false;
      a = i->getOperand(0); b = i->getOperand(1);
    } else {
      auto *c = dyn_cast<IntrinsicInst>(i);
      if (!c || c->getIntrinsicID() != kind) return false;
      a = c->getArgOperand(0); b = c->getArgOperand(1);
    }
    for (unsigned side = 0; side < 2; ++side) {
      Value *plain = side ? b : a, *shuffled = side ? a : b;
      SmallVector<Instruction *> casts;
      if (plain->getType()->isFloatTy()) {
        auto *cast = dyn_cast<BitCastInst>(shuffled);
        if (!cast || !cast->getSrcTy()->isIntegerTy(32)) continue;
        casts.push_back(cast); shuffled = cast->getOperand(0);
      }
      auto *call = dyn_cast<IntrinsicInst>(shuffled);
      if (!call || call->getIntrinsicID() != Intrinsic::amdgcn_ds_bpermute) continue;
      Value *input = call->getArgOperand(1);
      if (plain->getType()->isFloatTy()) {
        auto *cast = dyn_cast<BitCastInst>(input);
        if (!cast || cast->getOperand(0) != plain) continue;
        casts.push_back(cast); input = cast->getOperand(0);
      }
      if (input != plain || !(bit = xorMask(call->getArgOperand(0)))) continue;
      previous = plain;
      nodes.insert(i); nodes.insert(call);
      nodes.insert(casts.begin(), casts.end());
      return true;
    }
    return false;
  }
  Reduction reduction(Value *v, unsigned kind) {
    Reduction r;
    for (unsigned n = 0; n < 6; ++n) {
      Value *previous; unsigned bit;
      if (!stage(v, kind, previous, bit, r.nodes)) break;
      if (r.mask & bit) { r.mask = 0; return r; }
      r.mask |= bit; r.bits.push_back(bit); v = previous;
    }
    r.leaf = v;
    return r;
  }
  bool uniform(Slice s, unsigned group, bool collectives,
               std::set<Slice> &active, unsigned &budget) {
    if (!budget--) return false;
    s = resolve(s);
    if (auto *c = dyn_cast<Constant>(s.value))
      return !isa<UndefValue, PoisonValue>(c) && !c->containsUndefOrPoisonElement() && s.path.empty();
    if (isa<Argument>(s.value)) return true; // kernel scalar/pointer ABI
    auto *i = dyn_cast<Instruction>(s.value);
    if (!i) return false;
    if (s.path.empty() && fixedUniform(i, group)) return true;
    if (auto *p = dyn_cast<PHINode>(i)) {
      // Inductive proof: every initial and backedge input must be uniform.
      // Only PHI cycles are accepted, and no successful proof is cached across
      // queries (a failed cyclic proof must not leave optimistic cache entries).
      if (!active.insert(s).second) return true;
      bool ok = true;
      for (Value *input : p->incoming_values())
        if (!uniform({input, s.path}, group, collectives, active, budget)) { ok = false; break; }
      active.erase(s);
      return ok;
    }
    if (collectives && s.path.empty()) {
      unsigned kind = i->getOpcode();
      if (auto *c = dyn_cast<IntrinsicInst>(i)) kind = c->getIntrinsicID();
      if (kind == Instruction::Add || kind == Instruction::FAdd ||
          kind == Intrinsic::maxnum || kind == Intrinsic::minnum ||
          kind == Intrinsic::smax || kind == Intrinsic::smin ||
          kind == Intrinsic::umax || kind == Intrinsic::umin) {
        auto r = reduction(i, kind);
        if (r.mask && (r.mask & group) == group) return true;
      }
    }
    if (auto *c = dyn_cast<CallInst>(i)) {
      StringRef name = callee(i);
      if (name.starts_with("llvm.amdgcn.workgroup.id.")) return true;
      // A deliberately small set of deterministic lane-wise functions.
      auto *intrinsic = dyn_cast<IntrinsicInst>(c);
      if (!intrinsic) return false;
      switch (intrinsic->getIntrinsicID()) {
      case Intrinsic::exp2: case Intrinsic::amdgcn_exp2:
      case Intrinsic::fabs: case Intrinsic::maxnum: case Intrinsic::minnum:
      case Intrinsic::smax: case Intrinsic::smin: case Intrinsic::umax: case Intrinsic::umin:
      case Intrinsic::sqrt: case Intrinsic::log2: break;
      default: return false;
      }
      for (Value *a : c->args())
        if (!uniform({a, a->getType()->isVectorTy() ? s.path : Path{}}, group, collectives, active, budget)) return false;
      return true;
    }
    if (!isa<BinaryOperator, CastInst, CmpInst, SelectInst, GetElementPtrInst>(i)) return false;
    for (Value *operand : i->operands())
      if (!uniform({operand, operand->getType()->isVectorTy() ? s.path : Path{}}, group, collectives, active, budget)) return false;
    return true;
  }
  bool uniform(Value *v, unsigned group, bool collectives) {
    std::set<Slice> active; unsigned budget = 8192;
    return uniform({v, {}}, group, collectives, active, budget);
  }
};

struct Plan {
  PHINode *phi;
  Path path;
  BinaryOperator *update, *scale = nullptr;
  unsigned reducedOperand;
  unsigned stateReads = 0;
  Reduction sum;
  SmallVector<Use *> outside;
  BasicBlock *exit;
};

bool stateUses(Value *v, Path remaining, Loop *loop, Plan &plan,
               DominatorTree &dt, std::set<Slice> &seen) {
  if (!seen.insert({v, remaining}).second) return true;
  bool scalar = remaining.empty() && (v->getType()->isFloatTy() || v->getType()->isIntegerTy(32));
  auto *vec = dyn_cast<FixedVectorType>(v->getType());
  bool one = vec && vec->getNumElements() == 1 && remaining == Path{0};
  for (Use &use : v->uses()) {
    auto *i = dyn_cast<Instruction>(use.getUser());
    if (!i) return false;
    if (!loop->contains(i) && (scalar || one)) {
      // LCSSA forwarding PHIs: replace their consumers, never an incoming use
      // with a value defined after the PHIs in the exit block.
      if (auto *p = dyn_cast<PHINode>(i)) {
        if (p->getParent() != plan.exit || p->getNumIncomingValues() != 1 ||
            !stateUses(p, remaining, loop, plan, dt, seen)) return false;
      } else {
        if (!dt.dominates(plan.exit, i->getParent())) return false;
        plan.outside.push_back(&use);
      }
      continue;
    }
    if (auto *e = dyn_cast<ExtractValueInst>(i)) {
      auto idx = e->getIndices();
      if (prefix(idx, remaining)) {
        Path next(remaining.begin() + idx.size(), remaining.end());
        if (!stateUses(e, next, loop, plan, dt, seen)) return false;
      } else if (prefix(remaining, idx)) return false;
      continue; // a disjoint field
    }
    if (auto *e = dyn_cast<ExtractElementInst>(i)) {
      auto *index = dyn_cast<ConstantInt>(e->getIndexOperand());
      if (!index || remaining.empty()) return false;
      if (index->equalsInt(remaining.front())) {
        Path next(remaining.begin() + 1, remaining.end());
        if (!stateUses(e, next, loop, plan, dt, seen)) return false;
      }
      continue;
    }
    if (!scalar || i != (plan.scale ? plan.scale : plan.update)) return false;
    ++plan.stateReads;
  }
  return true;
}
bool updateUses(Value *v, PHINode *phi, SmallPtrSetImpl<Value *> &seen) {
  if (!seen.insert(v).second) return true;
  for (User *u : v->users()) {
    if (u == phi) continue;
    if (!isa<InsertValueInst, InsertElementInst>(u) || !updateUses(u, phi, seen)) return false;
  }
  return true;
}

bool analyze(PHINode *phi, const Path &path, Loop *loop, Proof &proof,
             DominatorTree &dt, bool approximate, Plan &plan) {
  auto *pre = loop->getLoopPreheader(), *latch = loop->getLoopLatch();
  Slice init = resolve({phi->getIncomingValueForBlock(pre), path});
  auto *zero = dyn_cast<Constant>(init.value);
  if (!init.path.empty() || !zero || !zero->isNullValue()) return false;
  Slice back = resolve({phi->getIncomingValueForBlock(latch), path});
  auto *update = dyn_cast<BinaryOperator>(back.value);
  if (!back.path.empty() || !update || update->getParent() != latch ||
      (update->getOpcode() != Instruction::Add && update->getOpcode() != Instruction::FAdd) ||
      (!approximate && update->getType()->isFloatingPointTy())) return false;
  Slice state{phi, path};
  for (unsigned side = 0; side < 2; ++side) {
    Value *prior = update->getOperand(1 - side);
    BinaryOperator *scale = nullptr;
    if (!(resolve({prior, {}}) == state)) {
      scale = dyn_cast<BinaryOperator>(prior);
      unsigned mul = update->getOpcode() == Instruction::Add ? Instruction::Mul : Instruction::FMul;
      if (!scale || scale->getOpcode() != mul || scale->getParent() != latch || !scale->hasOneUse()) continue;
      if (!(resolve({scale->getOperand(0), {}}) == state) &&
          !(resolve({scale->getOperand(1), {}}) == state)) continue;
    }
    auto r = proof.reduction(update->getOperand(side), update->getOpcode());
    if (!r.mask) continue;
    if (scale) {
      Value *alpha = resolve({scale->getOperand(0), {}}) == state ? scale->getOperand(1) : scale->getOperand(0);
      if (!proof.uniform(alpha, r.mask, true)) continue;
    }
    bool privateTree = true;
    for (auto *node : r.nodes) {
      if (node->getParent() != latch) privateTree = false;
      for (User *u : node->users())
        if (u != update && !r.nodes.contains(dyn_cast<Instruction>(u))) privateTree = false;
    }
    if (!privateTree) continue;
    plan.phi = phi; plan.path = path; plan.update = update; plan.scale = scale;
    plan.reducedOperand = side; plan.sum = std::move(r); plan.exit = loop->getExitBlock();
    std::set<Slice> seen;
    if (!stateUses(phi, path, loop, plan, dt, seen) || plan.outside.empty() || plan.stateReads != 1) return false;
    SmallPtrSet<Value *, 16> updates;
    if (!updateUses(update, phi, updates)) return false;
    // Inserting this update into a second field would create an untracked state.
    std::vector<Path> fields;
    if (!enumerate(phi->getType(), {}, fields)) return false;
    for (auto &field : fields)
      if (field != path && resolve({phi->getIncomingValueForBlock(latch), field}).value == update) return false;
    return true;
  }
  return false;
}

Value *extract(IRBuilder<> &builder, Value *value, const Path &path) {
  for (unsigned index : path) {
    if (value->getType()->isVectorTy()) value = builder.CreateExtractElement(value, index, "lane.sum.local");
    else value = builder.CreateExtractValue(value, index, "lane.sum.field");
  }
  return value;
}
void apply(Plan &p, unsigned wave) {
  IRBuilder<> builder(&*p.exit->getFirstInsertionPt());
  Value *value = extract(builder, p.phi, p.path);
  auto *module = p.phi->getModule();
  Value *lane = builder.CreateCall(Intrinsic::getOrInsertDeclaration(module, Intrinsic::amdgcn_mbcnt_lo),
                                   {builder.getInt32(-1), builder.getInt32(0)}, "lane.sum.lane.lo");
  if (wave == 64)
    lane = builder.CreateCall(Intrinsic::getOrInsertDeclaration(module, Intrinsic::amdgcn_mbcnt_hi),
                              {builder.getInt32(-1), lane}, "lane.sum.lane");
  for (unsigned bit : llvm::reverse(p.sum.bits)) {
    Value *address = builder.CreateShl(builder.CreateXor(lane, builder.getInt32(bit)), builder.getInt32(2));
    Value *bits = builder.CreateBitCast(value, builder.getInt32Ty());
    Value *other = builder.CreateCall(Intrinsic::getOrInsertDeclaration(module, Intrinsic::amdgcn_ds_bpermute),
                                     {address, bits}, "lane.sum.shuffle");
    other = builder.CreateBitCast(other, value->getType());
    value = builder.CreateBinOp(static_cast<Instruction::BinaryOps>(p.update->getOpcode()), value, other, "lane.sum.final");
  }
  Value *packed = nullptr;
  for (Use *use : p.outside) {
    if (use->get()->getType()->isVectorTy()) {
      if (!packed) packed = builder.CreateInsertElement(PoisonValue::get(use->get()->getType()), value, builder.getInt32(0));
      use->set(packed);
    } else use->set(value);
  }
  p.update->setOperand(p.reducedOperand, p.sum.leaf);
  // Changed partials may overflow even if the original group sums did not.
  p.update->dropPoisonGeneratingFlags();
  if (p.scale) p.scale->dropPoisonGeneratingFlags();
  // Delete only the matched tree, including convergent calls. General DCE does
  // not remove an unused ds.bpermute because of its convergence semantics.
  while (!p.sum.nodes.empty()) {
    Instruction *dead = nullptr;
    for (auto *node : p.sum.nodes) if (node->use_empty()) { dead = node; break; }
    assert(dead && "private reduction tree must be acyclic");
    p.sum.nodes.erase(dead); dead->eraseFromParent();
  }
}
} // namespace

bool parseDeferLaneSumOption(StringRef arg, StringRef &mode,
                            DeferLaneSumOptions &options, bool &consumed) {
  consumed = arg.starts_with("--defer-lane-sum");
  if (!consumed) return true;
  if (mode.empty()) mode = "off";
  if (arg.consume_front("--defer-lane-sum=")) {
    if (arg != "off" && arg != "legal" && arg != "approx") return false;
    mode = arg; options.approximate = mode == "approx"; return true;
  }
  if (arg.consume_front("--defer-lane-sum-wave-size="))
    return !arg.getAsInteger(10, options.waveSize) && options.valid();
  return false;
}

PreservedAnalyses DeferLaneSumPass::run(Function &function, FunctionAnalysisManager &am) {
  if (!options.valid() || function.isDeclaration() ||
      function.getCallingConv() != CallingConv::AMDGPU_KERNEL ||
      function.hasFnAttribute(Attribute::StrictFP) || function.hasFnAttribute(Attribute::OptimizeNone))
    return PreservedAnalyses::all();
  unsigned wave = options.waveSize;
  auto features = function.getFnAttribute("target-features").getValueAsString();
  SmallVector<StringRef> tokens; features.split(tokens, ',');
  for (unsigned candidate : {32u, 64u}) {
    std::string feature = "+wavefrontsize" + std::to_string(candidate);
    if (llvm::is_contained(tokens, feature)) {
      if (wave && wave != candidate) return PreservedAnalyses::all();
      wave = candidate;
    }
  }
  if (!wave) return PreservedAnalyses::all();
  if (llvm::is_contained(tokens, "-wavefrontsize" + std::to_string(wave)))
    return PreservedAnalyses::all();
  auto *wg = function.getMetadata("reqd_work_group_size");
  if (!wg || wg->getNumOperands() != 3) return PreservedAnalyses::all();
  unsigned dims[3], threads = 1;
  for (unsigned d = 0; d < 3; ++d) {
    auto *size = mdconst::dyn_extract<ConstantInt>(wg->getOperand(d));
    if (!size || size->isZero() || size->getValue().getActiveBits() > 11) return PreservedAnalyses::all();
    dims[d] = size->getZExtValue(); threads *= dims[d];
    if (threads > 1024) return PreservedAnalyses::all();
  }
  if (threads % wave) return PreservedAnalyses::all();
  Proof proof(function, wave, threads, dims[1] == 1 && dims[2] == 1);
  // A conservative whole-function convergence proof also protects a candidate
  // inside an enclosing divergent region or after an early lane return.
  for (auto &bb : function) {
    if (auto *br = dyn_cast<CondBrInst>(bb.getTerminator())) {
      if (!proof.uniform(br->getCondition(), wave - 1, false)) return PreservedAnalyses::all();
    } else if (!isa<ReturnInst, UncondBrInst>(bb.getTerminator())) return PreservedAnalyses::all();
    for (auto &i : bb) if (isa<CallBase>(i)) {
      auto *call = dyn_cast<CallInst>(&i);
      if (!call) return PreservedAnalyses::all();
      if (auto *assembly = dyn_cast<InlineAsm>(call->getCalledOperand())) {
        // Only known EXEC-preserving templates. Unknown assembly is opaque.
        StringRef code = assembly->getAsmString(), constraints = assembly->getConstraintString();
        bool safe = (code.empty() && (constraints == "=v,0" || constraints == "=v,0,i")) ||
          (code == "s_waitcnt vmcnt(0)" && constraints == "~{memory}") ||
          (code == "v_mmac_f32_16x16x16_f16 $0, $1, $2, $3\n\t" && constraints == "=v,v,v,0") ||
          (code == "s_mov_b32 $0, $1\n\tbuffer_load_dwordx4 $2, $3, 0 offen lds" &&
           constraints == "={m0},s,v,s,~{memory}");
        if (!safe) return PreservedAnalyses::all();
        continue;
      }
      if (!call->getCalledFunction() || !call->getCalledFunction()->isDeclaration())
        return PreservedAnalyses::all();
      // Keep the convergence adapter explicit: unknown calls and target
      // intrinsics may manipulate EXEC even in a uniform-looking LLVM CFG.
      StringRef name = callee(call);
      if (name.starts_with("llvm.amdgcn.")) {
        bool safe = name.starts_with("llvm.amdgcn.workgroup.id.") ||
          name.starts_with("llvm.amdgcn.workitem.id.") ||
          name.starts_with("llvm.amdgcn.mbcnt.") ||
          name.starts_with("llvm.amdgcn.readfirstlane.") ||
          name.starts_with("llvm.amdgcn.exp2.") ||
          name == "llvm.amdgcn.ds.bpermute" || name == "llvm.amdgcn.s.barrier" ||
          name == "llvm.amdgcn.global.load.async.lds" ||
          name == "llvm.amdgcn.asyncmark" || name == "llvm.amdgcn.wait.asyncmark";
        if (!safe || !isa<IntrinsicInst>(call)) {
          errs() << "DeferLaneSum: " << function.getName() << ": skipped (unsupported call " << name << ")\n";
          return PreservedAnalyses::all();
        }
      } else if (name != "llvm.hcu.ds.read.m32x16.b16.v4i32.p3" &&
                 name != "llvm.hcu.global.load.dwordx4.lds") {
        auto *intrinsic = dyn_cast<IntrinsicInst>(call);
        if (!intrinsic || !call->doesNotAccessMemory() || call->isConvergent() ||
            !call->hasFnAttr(Attribute::WillReturn) || !call->doesNotThrow())
          return PreservedAnalyses::all();
      }
    }
  }
  auto &li = am.getResult<LoopAnalysis>(function);
  auto &dt = am.getResult<DominatorTreeAnalysis>(function);
  unsigned states = 0, shuffles = 0;
  for (Loop *loop : li.getLoopsInPreorder()) {
    if (!loop->getSubLoops().empty() || !loop->getLoopPreheader() || !loop->getLoopLatch() ||
        !loop->getExitingBlock() || !loop->getExitBlock() ||
        !loop->getExitBlock()->getSinglePredecessor() ||
        loop->getExitingBlock() != loop->getHeader()) continue;
    for (auto &phi : loop->getHeader()->phis()) {
      if (phi.getNumIncomingValues() != 2) continue;
      std::vector<Path> paths;
      if (!enumerate(phi.getType(), {}, paths)) continue;
      for (auto &path : paths) {
        Plan plan;
        if (!analyze(&phi, path, loop, proof, dt, options.approximate, plan)) continue;
        ++states; shuffles += plan.sum.bits.size(); apply(plan, wave); proof.clearCaches();
      }
    }
  }
  errs() << "DeferLaneSum: " << function.getName() << ": " << states << " states, "
         << shuffles << " loop shuffles deferred (" << (options.approximate ? "approx" : "legal") << ")\n";
  if (!states) return PreservedAnalyses::all();
  auto preserved = PreservedAnalyses::none(); preserved.preserveSet<CFGAnalyses>(); return preserved;
}
void deferLaneSums(Module &module, DeferLaneSumOptions options) {
  FunctionAnalysisManager analyses;
  PassBuilder pb; pb.registerFunctionAnalyses(analyses);
  FunctionPassManager pipeline; pipeline.addPass(DeferLaneSumPass(options));
  for (auto &function : module) if (!function.isDeclaration()) pipeline.run(function, analyses);
}
} // namespace mlir::frisk
