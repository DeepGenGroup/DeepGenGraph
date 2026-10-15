#!/usr/bin/env python3
"""Execute the exported split-D attention LLVM body with NumPy lane values.

Unlike address/provenance checks, this executes loops, PHIs, mask, reductions,
MMAC operands/accumulators, memory traffic and normalization numerically. The
supported ABI is Q/K/V/O in contiguous BHSD storage, one block at a time.
Unsupported operations/control flow fail explicitly. Poison float lanes use NaN.

MMAC uses the documented-in-repository 16x16x16 wave64 mapping. DMA completes
immediately and barriers are collective no-ops: this is a mathematical model,
NOT a model of gfx936 hazards, asynchronous timing, or backend code generation.
A passing result does not establish hardware correctness.
"""
import argparse, re, struct
from pathlib import Path
import numpy as np
from check_gemm_pv_addresses import split_args

class LLVMWorkgroupModel:

    def __init__(self, text, q, k, v, threads=256, head=0, block=0):
        self.t = threads
        self.head = head
        self.block = block
        self.values = {}
        self.mem = {}
        self.active = np.ones(threads, dtype=bool)
        for i, x in enumerate([q, k, v, np.full_like(q, np.nan)]):
            self.mem[i + 1 << 32] = x.ravel()
            self.values['%' + str(i)] = np.full(threads, i + 1 << 32, dtype=np.int64)
        storage = re.search(r'^@shm_0 = .*global \[(\d+) x (i8|half|float)\]', text, re.M)
        assert storage, 'expected explicit shared storage'
        shared_bytes = int(storage[1]) * {'i8': 1, 'half': 2, 'float': 4}[storage[2]]
        self.mem[0] = np.full(shared_bytes // 2, np.nan, dtype=np.float16)
        self.blocks = {}
        current = '4'
        self.blocks[current] = []
        body = text.split('define amdgpu_kernel void ')[1].split('\n}', 1)[0].split('\n', 1)[1]
        for l in body.splitlines():
            if re.match('^[\\w.]+:', l):
                current = l.split(':')[0]
                self.blocks[current] = []
            elif l.startswith('  '):
                self.blocks[current].append(re.sub(', ![\\w.]+ !\\d+', '', l.strip()))
        self.mmac_count = 0

    def parse_type(self, s):
        s = s.strip()
        if s[0] in '<[':
            close = '>' if s[0] == '<' else ']'
            depth = 0
            for i, c in enumerate(s):
                if c == s[0]:
                    depth += 1
                if c == close:
                    depth -= 1
                    if depth == 0:
                        return (s[:i + 1], s[i + 1:].strip())
        m = re.match('(ptr(?: addrspace\\(\\d+\\))?|i\\d+|half|float|double)\\s*(.*)', s)
        assert m, s
        return (m[1], m[2])

    def shape(self, ty):
        if ty[0] in '<[':
            n, sub = ty[1:-1].split(' x ', 1)
            return (int(n),) + self.shape(sub)
        return ()

    def dtype(self, ty):
        return np.float16 if 'half' in ty else np.float32 if 'float' in ty else np.int64

    def typed(self, s):
        ty, val = self.parse_type(s)
        return self.value(val, ty)

    def value(self, s, ty=None):
        if s in self.values:
            return self.values[s]
        if s.startswith('getelementptr'):
            return self.gep(s[s.index('(') + 1:-1])
        shape = (self.t,) + self.shape(ty or 'i64')
        dt = self.dtype(ty or 'i64')
        if s == '@shm_0':
            return np.zeros(self.t, dtype=np.int64)
        if s == 'zeroinitializer':
            return np.zeros(shape, dtype=dt)
        if s in ('poison', 'undef'):
            return np.full(shape, np.nan if dt in (np.float16, np.float32) else -(1 << 60), dtype=dt)
        if s.startswith('splat ('):
            return np.broadcast_to(self.typed(s[7:-1]).reshape(self.t, *[1] * (len(shape) - 1)), shape).copy()
        if s.startswith('<') or s.startswith('['):
            return np.stack([self.typed(x) for x in split_args(s[1:-1])], axis=1)
        if s.startswith('0xH'):
            val = np.array(int(s[3:], 16), dtype=np.uint16).view(np.float16).item()
        elif s.startswith('0x'):
            val = struct.unpack('>d', bytes.fromhex(s[2:]))[0]
        elif s in ('true', 'false'):
            val = int(s == 'true')
        else:
            val = float(s) if dt in (np.float32, np.float16) else int(s)
        return np.full(shape, val, dtype=dt)

    def size(self, ty):
        if ty[0] in '<[':
            n, sub = ty[1:-1].split(' x ', 1)
            return int(n) * self.size(sub)
        return {'i8': 1, 'half': 2, 'float': 4, 'i32': 4}[ty]

    def gep(self, body):
        body = re.sub('^(?:(?:inbounds|nuw|nusw) )*', '', body)
        ty, p, *idx = split_args(body)
        assert len(idx) == 1, body
        return self.typed(p) + self.typed(idx[0]) * self.size(ty)

    def memory(self, ptr, ty, write=None):
        count = int(np.prod(self.shape(ty))) if self.shape(ty) else 1
        addresses = ptr[:, None] + np.arange(count)[None, :] * 2
        bases = addresses // (1 << 32) * (1 << 32)
        base = bases[self.active].flat[0]
        assert np.all(bases[self.active] == base)
        idx = (addresses - bases) // 2
        mem = self.mem[int(base)]
        live = idx[self.active]
        assert live.min() >= 0 and live.max() < len(mem), (live.min(), live.max(), len(mem))
        if write is not None:
            mem[live] = write.reshape(self.t, count)[self.active]
            return
        result = np.zeros((self.t, count), dtype=mem.dtype)
        result[self.active] = mem[live]
        return result.reshape((self.t,) + self.shape(ty))

    def call(self, body):
        if 'asm sideeffect' in body:
            operands = re.search('"\\((.*)\\)', body)[1]
            args = [self.typed(x) for x in split_args(operands)] if operands else []
            if 'buffer_load_dwordx4' in body:
                assert '"={m0},s,v,s,~{memory}"' in body
                lds, offset, resource = args
                assert np.all(resource[:, 3] == 0x20000), 'unexpected buffer format'
                assert np.all(offset[self.active] >= 0)
                assert np.all(offset[self.active] + 16 <= resource[self.active, 2]), 'buffer OOB'
                for wave in range(self.t // 64):
                    lanes = slice(wave * 64, (wave + 1) * 64)
                    assert np.all(resource[lanes] == resource[wave * 64])
                    assert np.all(lds[lanes] == lds[wave * 64])
                source = resource[:, 0] + (resource[:, 1] << 32) + offset
                self.memory(lds + np.arange(self.t) % 64 * 16, '<8 x half>',
                            self.memory(source, '<8 x half>'))
                return lds
            if 's_waitcnt vmcnt(0)' in body:
                assert not args and '"~{memory}"' in body
                return
            if 'v_mmac_' not in body:
                assert 'asm sideeffect "", "=v,0"' in body, body
                assert len(args) == 1
                return args[0]
            assert 'v_mmac_f32_16x16x16_f16 $0, $1, $2, $3' in body, body
            a, b, c = args
            out = c.copy()
            for w in range(self.t // 64):
                aa = a[w * 64:(w + 1) * 64].reshape(4, 16, 4).transpose(1, 0, 2).reshape(16, 16).astype(np.float32)
                bb = b[w * 64:(w + 1) * 64].reshape(4, 16, 4).transpose(1, 0, 2).reshape(16, 16).astype(np.float32)
                prod = aa @ bb.T
                out[w * 64:(w + 1) * 64] += prod.reshape(16, 4, 4).transpose(2, 0, 1).reshape(64, 4)
            self.mmac_count += 1
            return out
        name, args = re.search('@([\\w.]+)\\((.*)\\)', body).groups()
        args = [self.typed(x) for x in split_args(args)] if args else []
        if name.endswith('workgroup.id.x'):
            return np.full(self.t, self.head, dtype=np.int64)
        if name.endswith('workgroup.id.y'):
            return np.full(self.t, self.block, dtype=np.int64)
        if name.endswith('workitem.id.x'):
            return np.arange(self.t, dtype=np.int64)
        if name.endswith('readfirstlane.i32'):
            return args[0][np.arange(self.t) // 64 * 64]
        if name.endswith('mbcnt.lo'):
            return np.arange(self.t, dtype=np.int64) % 32
        if name.endswith('mbcnt.hi'):
            return np.arange(self.t, dtype=np.int64) % 64
        if name.endswith('ds.bpermute'):
            return args[1][np.arange(self.t) // 64 * 64 + args[0] // 4]
        if 'exp2' in name:
            return np.exp2(args[0])
        if name in ('llvm.amdgcn.global.load.async.lds', 'llvm.hcu.global.load.dwordx4.lds'):
            if name.startswith('llvm.hcu.'):
                size = 16
                # HCU encodes width in its name; arguments are offset/aux/bps.
                assert all(np.all(arg == 0) for arg in args[2:])
            else:
                size = int(args[2][0])
                assert size in (4, 16) and np.all(args[3] == 0) and np.all(args[4] == 0)
            packet = f'<{size // 2} x half>'
            self.memory(args[1] + np.arange(self.t) % 64 * size, packet, self.memory(args[0], packet))
            return
        assert name in ['llvm.amdgcn.asyncmark', 'llvm.amdgcn.wait.asyncmark', 'llvm.amdgcn.s.barrier'], name

    def operation(self, expr):
        expr = re.sub(r'^(?:tail|musttail|notail) ', '', expr)
        op, body = expr.split(' ', 1)
        body = re.sub('^(?:(?:nsw|nuw|disjoint|nneg|samesign|arcp|reassoc|contract|afn|fast) )*', '', body)
        if op == 'call':
            return self.call(body)
        if op == 'getelementptr':
            return self.gep(body)
        if op == 'load':
            ty, p, *_ = split_args(body)
            return self.memory(self.typed(p), ty)
        if op == 'store':
            v, p, *_ = split_args(body)
            ty, _ = self.parse_type(v)
            self.memory(self.typed(p), ty, self.typed(v))
            return
        if op == 'fence':
            return
        if op in ('ptrtoint', 'inttoptr', 'sext', 'zext', 'trunc', 'fptrunc', 'bitcast'):
            src, dst = body.split(' to ')
            ty, _ = self.parse_type(src)
            v = self.typed(src)
            if op == 'fptrunc':
                return v.astype(np.float16)
            if op == 'bitcast':
                def storage_dtype(value_type):
                    if 'half' in value_type:
                        return np.dtype('float16')
                    if 'float' in value_type:
                        return np.dtype('float32')
                    return np.dtype('uint' + re.search(r'i(\d+)', value_type)[1])
                # Preserve all bits, including cross-element vector packing.
                source = np.ascontiguousarray(v.astype(storage_dtype(ty)))
                result = source.reshape(self.t, -1).view(storage_dtype(dst))
                return result.reshape((self.t,) + self.shape(dst)).astype(self.dtype(dst))
            if op in ('sext', 'zext', 'trunc'):
                bits = int(re.search(r'i(\d+)', dst if op == 'trunc' else ty)[1])
                v = v.astype(np.int64)
                if bits < 64:
                    v = v % (1 << bits)
                if op == 'sext':
                    v = np.where(v >= 1 << bits - 1, v - (1 << bits), v)
            return v
        if op in ('extractelement', 'extractvalue'):
            v, i = split_args(body)
            i = self.typed(i) if op == 'extractelement' else self.value(i)
            assert np.all(i == i[0])
            return self.typed(v)[:, i[0]]
        if op in ('insertelement', 'insertvalue'):
            v, a, i = split_args(body)
            i = self.typed(i) if op == 'insertelement' else self.value(i)
            assert np.all(i == i[0])
            r = self.typed(v).copy()
            r[:, i[0]] = self.typed(a)
            return r
        if op == 'shufflevector':
            a, b, ix = map(self.typed, split_args(body))
            assert np.all(ix == ix[0])
            valid = ix[0] >= 0
            result = np.concatenate([a, b], axis=1)[:, np.where(valid, ix[0], 0)].copy()
            result[:, ~valid] = np.nan if result.dtype.kind == 'f' else -(1 << 60)
            return result
        if op == 'select':
            c, a, b = map(self.typed, split_args(body))
            return np.where(c, a, b)
        if op == 'icmp':
            pred, body = body.split(' ', 1)
            a, b = split_args(body)
            ty, _ = self.parse_type(a)
            a, b = (self.typed(a), self.value(b, ty))
            if pred[0] == 'u' and ty == 'i32':
                a, b = (a % (1 << 32), b % (1 << 32))
            return {'eq': np.equal, 'ne': np.not_equal, 'slt': np.less, 'sle': np.less_equal, 'sgt': np.greater, 'sge': np.greater_equal, 'ult': np.less, 'ule': np.less_equal, 'ugt': np.greater, 'uge': np.greater_equal}[pred](a, b)
        a, b = split_args(body)
        ty, _ = self.parse_type(a)
        a, b = (self.typed(a), self.value(b, ty))
        bits_match = re.search(r'i(\d+)', ty)
        bits = int(bits_match[1]) if bits_match else None
        if op == 'sdiv':
            if bits < 64:
                a, b = [np.where(x % (1 << bits) >= (1 << (bits - 1)),
                                  x % (1 << bits) - (1 << bits), x % (1 << bits))
                        for x in (a, b)]
            return np.trunc(a / b).astype(np.int64)
        if op == 'lshr' and bits < 64:
            a = a % (1 << bits)
        funcs = {'add': np.add, 'sub': np.subtract, 'mul': np.multiply, 'and': np.bitwise_and, 'or': np.bitwise_or, 'xor': np.bitwise_xor, 'shl': np.left_shift, 'lshr': np.right_shift, 'ashr': np.right_shift, 'fadd': np.add, 'fmul': np.multiply, 'fdiv': np.divide}
        assert op in funcs, expr
        result = funcs[op](a, b)
        if bits and bits < 64:
            result = result % (1 << bits)
        return result

    def run(self):
        current = '4'
        previous = None
        steps = 0
        reconverge = []
        while True:
            if reconverge and current == reconverge[-1][0]:
                _, self.active = reconverge.pop()
                assert not any(' = phi ' in line for line in self.blocks[current]), 'unsupported divergent PHI'
            pending = {}
            for line in self.blocks[current]:
                m = re.match(r'(%[\w.-]+) = (.*)', line)
                name, expr = m.groups() if m else (None, line)
                try:
                    if expr.startswith('phi '):
                        ty, rest = self.parse_type(expr[4:])
                        incoming = split_args(rest)
                        selected = [split_args(x[1:-1])[0] for x in incoming if split_args(x[1:-1])[1] == '%' + previous]
                        assert len(selected) == 1, (previous, expr)
                        pending[name] = self.value(selected[0], ty).copy()
                        continue
                    self.values.update(pending)
                    pending = {}
                    if expr.startswith('br '):
                        args = split_args(expr[3:])
                        if len(args) == 1:
                            target = args[0].split('%')[1]
                        else:
                            cond = self.typed(args[0])
                            live = cond[self.active]
                            if np.all(live == live[0]):
                                target = args[1 if live[0] else 2].split('%')[1]
                            else:
                                # Full-width noKPack uses a result-free if over
                                # complete waves. Accept only a single straight
                                # arm with an explicit reconvergence successor.
                                yes, no = [arg.split('%')[1] for arg in args[1:]]
                                if self.blocks[yes][-1] == 'br label %' + no:
                                    arm, join, mask = yes, no, cond
                                elif self.blocks[no][-1] == 'br label %' + yes:
                                    arm, join, mask = no, yes, ~cond
                                else:
                                    raise AssertionError('unsupported divergent control flow')
                                assert np.all(mask.reshape(-1, 64) == mask.reshape(-1, 64)[:, :1]), 'partial DMA wave'
                                reconverge.append((join, self.active.copy()))
                                self.active &= mask
                                target = arm
                        previous, current = (current, target)
                        break
                    if expr == 'ret void':
                        return self.mem[4 << 32]
                    value = self.operation(expr)
                    if name:
                        self.values[name] = value
                except Exception:
                    print('FAILED BLOCK', current, 'INSTRUCTION', line)
                    raise
            steps += 1
            assert steps < 10000


def reference(q, k, v, head, block, rows=128):
    start, end = block * rows, (block + 1) * rows
    scores = (q[head, start:end].astype(np.float32)
              @ k[head, :end].astype(np.float32).T) / np.sqrt(q.shape[-1])
    scores[np.arange(end)[None, :] > np.arange(start, end)[:, None]] = -np.inf
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ v[head, :end].astype(np.float32)


def report(label, actual, expected):
    actual, expected = actual.astype(np.float32), expected.astype(np.float32)
    error = np.abs(actual - expected)
    passed = np.allclose(actual, expected, atol=0.01, rtol=0.01)
    print(f"{label}: allclose={passed}, max_abs={error.max():.8g}, "
          f"mean_abs={error.mean():.8g}, "
          f"mismatch={np.count_nonzero(~np.isclose(actual, expected, atol=.01, rtol=.01))}",
          flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ir', type=Path)
    parser.add_argument('--snapshot', type=Path,
                        help='Prefix NPZ saved by kernelLauncher --diagnose-attention')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    text = args.ir.read_text()
    threads = int(re.search(r'"amdgpu-flat-work-group-size"="\d+,(\d+)"', text)[1])
    if args.snapshot:
        data = np.load(args.snapshot, allow_pickle=False)
        q, k, v = [data[name].reshape(-1, *data[name].shape[-2:]) for name in ('q', 'k', 'v')]
        cases = [(0, 0)]
    else:
        rng = np.random.default_rng(args.seed)
        q, k, v = [rng.normal(size=(2, 4096, 128)).astype(np.float16) for _ in range(3)]
        cases = [(0, 0), (0, 1), (1, 7), (1, 31)]
    passed = True
    for head, block in cases:
        model = LLVMWorkgroupModel(text, q, k, v, threads=threads, head=head, block=block)
        actual = model.run().reshape(q.shape)[head, block * 128:(block + 1) * 128]
        expected = reference(q, k, v, head, block)
        passed &= report(f'LLVM model head={head} block={block}', actual, expected)
        if args.snapshot:
            report('GPU HSACO vs LLVM model', data['actual'].reshape(actual.shape), actual)
            report('GPU SDPA vs dense reference', data['sdpa'].reshape(actual.shape), expected)
    if not passed:
        raise SystemExit(1)
    print('PASS mathematical LLVM execution; hardware timing/hazards are not simulated.')


if __name__ == '__main__':
    main()
