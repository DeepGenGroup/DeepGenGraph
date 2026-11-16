#!/usr/bin/env python3
"""Lazy softmax base candidates, rewritten from the measured best non-causal IR.

State M (running max) becomes N = -B. N seeds the QK accumulators, so S' = S - B
needs no subtraction. O, L and S' are rescaled only on a rare per-wave slow path.
This rewrites generated IR only; no compiler pass is involved.
"""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'noncausal_cross_bn_k0_lifetimes/k0_sum_anchor.ll'
LLVM = Path('/data2/xsl/install/bin')
MMAC = re.compile(r'^  (%[\w.]+) = call <4 x float> asm "v_mmac_f32_16x16x16_f16 [^"]*", "=v,v,v,0"'
                  r'\(<4 x half> (%[\w.]+), <4 x half> (%[\w.]+), <4 x float> ([\w.%]+)\)')
TOKEN = re.compile(r'%[\w.]+')
INF = '0xFFF0000000000000'
# name: (branch threshold T as an IR float literal, description)
VARIANTS = {
    'lazy_t8': ('8.000000e+00', 'primary candidate'),
    'lazy_t4': ('4.000000e+00', 'more frequent rescale, smaller P bound'),
    # Always rescale: same arithmetic recurrence as the running max, plus the
    # new control flow. Separates structural overhead from the saved ALU work.
    'lazy_always': (INF, 'always takes the slow path'),
}
O_PHIS = ('%i88', '%i91', '%i92', '%i93')
DBG = ', !dbg !94'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one(pattern, text):
    found = re.findall(pattern, text, re.M)
    assert len(found) == 1, (pattern, found)
    return found[0]


def score_rows(lines):
    """Map each zero-seeded QK chain to its row through the final subtraction."""
    next_mmac, starts = {}, []
    for line in lines:
        m = MMAC.match(line)
        if not m:
            continue
        result, _, _, c = m.groups()
        if c == 'zeroinitializer':
            starts.append(result)
        else:
            assert c not in next_mmac, ('accumulator used twice', c)
            next_mmac[c] = result
    assert len(starts) == 16, starts
    subtraction = {}
    for line in lines:
        m = re.match(r'  (%[\w.]+) = fsub <4 x float> (%[\w.]+), (%i450|%i459),', line)
        if m:
            subtraction[m[2]] = m[3]
    assert len(subtraction) == 16, subtraction
    rows = {}
    for start in starts:
        value = start
        while value in next_mmac:
            value = next_mmac[value]
        rows[start] = {'%i450': 0, '%i459': 1}[subtraction[value]]
    assert sorted(rows.values()) == [0] * 8 + [1] * 8, rows
    return rows, subtraction


def slow_path(literal, subtraction):
    head = [
        f'  %lazy.peak = call float @llvm.maxnum.f32(float %i433, float %i446){DBG}',
        f'  %lazy.mask = call i64 @llvm.amdgcn.fcmp.i64.f32(float %lazy.peak, float {literal}, i32 2){DBG}',
        f'  %lazy.any = icmp ne i64 %lazy.mask, 0{DBG}',
        f'  %lazy.first = icmp eq i64 %i87, 0{DBG}',
        f'  %lazy.take = or i1 %lazy.first, %lazy.any{DBG}',
        '  br i1 %lazy.take, label %lazy.slow, label %lazy.join',
        '',
        'lazy.slow:',
    ]
    slow = []
    for row, m in ((0, '%i433'), (1, '%i446')):
        slow += [
            f'  %lazy.pos{row} = call float @llvm.maxnum.f32(float {m}, float 0.000000e+00){DBG}',
            f'  %lazy.d{row} = select i1 %lazy.first, float {m}, float %lazy.pos{row}{DBG}',
            f'  %lazy.neg{row} = fsub float 0.000000e+00, %lazy.d{row}{DBG}',
            f'  %lazy.exp{row} = call float @llvm.amdgcn.exp2.f32(float %lazy.neg{row}){DBG}',
            # First iteration: O = L = 0, and exp2(-d) may be inf, so do not multiply.
            f'  %lazy.alpha{row} = select i1 %lazy.first, float 1.000000e+00, float %lazy.exp{row}{DBG}',
            f'  %lazy.d{row}.b = insertelement <1 x float> poison, float %lazy.d{row}, i64 0{DBG}',
            f'  %lazy.d{row}.v4 = shufflevector <1 x float> %lazy.d{row}.b, <1 x float> poison, <4 x i32> zeroinitializer{DBG}',
            f'  %lazy.alpha{row}.v1 = insertelement <1 x float> poison, float %lazy.alpha{row}, i64 0{DBG}',
            f'  %lazy.alpha{row}.v8 = shufflevector <1 x float> %lazy.alpha{row}.v1, <1 x float> poison, <8 x i32> zeroinitializer{DBG}',
            f'  %lazy.Lo{row} = extractvalue [2 x <1 x float>] %i89, {row}{DBG}',
            f'  %lazy.Ln{row} = fmul <1 x float> %lazy.Lo{row}, %lazy.alpha{row}.v1{DBG}',
            f'  %lazy.Ns{row} = extractelement <1 x float> %lazy.n{row}, i64 0{DBG}',
            f'  %lazy.Nd{row} = fsub float %lazy.Ns{row}, %lazy.d{row}{DBG}',
            f'  %lazy.Nn{row} = insertelement <1 x float> poison, float %lazy.Nd{row}, i64 0{DBG}',
        ]
    slow += [
        f'  %lazy.LA = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.Ln0, 0{DBG}',
        f'  %lazy.LB = insertvalue [2 x <1 x float>] %lazy.LA, <1 x float> %lazy.Ln1, 1{DBG}',
        f'  %lazy.NA = insertvalue [2 x <1 x float>] poison, <1 x float> %lazy.Nn0, 0{DBG}',
        f'  %lazy.NB = insertvalue [2 x <1 x float>] %lazy.NA, <1 x float> %lazy.Nn1, 1{DBG}',
    ]
    for phi in O_PHIS:
        n = phi[1:]
        for row in (0, 1):
            slow += [
                f'  %lazy.Oo.{n}.{row} = extractvalue [2 x <8 x float>] {phi}, {row}{DBG}',
                f'  %lazy.On.{n}.{row} = fmul <8 x float> %lazy.Oo.{n}.{row}, %lazy.alpha{row}.v8{DBG}',
            ]
        slow += [
            f'  %lazy.OA.{n} = insertvalue [2 x <8 x float>] poison, <8 x float> %lazy.On.{n}.0, 0{DBG}',
            f'  %lazy.OB.{n} = insertvalue [2 x <8 x float>] %lazy.OA.{n}, <8 x float> %lazy.On.{n}.1, 1{DBG}',
        ]
    join = ['  br label %lazy.join', '', 'lazy.join:']
    for score, base in subtraction.items():
        row = {'%i450': 0, '%i459': 1}[base]
        n = score[1:]
        slow.append(f'  %lazy.SS.{n} = fsub <4 x float> {score}, %lazy.d{row}.v4{DBG}')
        join.append(f'  %lazy.S.{n} = phi <4 x float> [ {score}, %bb95 ], [ %lazy.SS.{n}, %lazy.slow ]{DBG}')
    for phi in O_PHIS:
        n = phi[1:]
        join.append(f'  %lazy.O.{n} = phi [2 x <8 x float>] [ {phi}, %bb95 ], [ %lazy.OB.{n}, %lazy.slow ]{DBG}')
    join.append(f'  %lazy.L = phi [2 x <1 x float>] [ %i89, %bb95 ], [ %lazy.LB, %lazy.slow ]{DBG}')
    join.append(f'  %lazy.N = phi [2 x <1 x float>] [ %i90, %bb95 ], [ %lazy.NB, %lazy.slow ]{DBG}')
    return head + slow + join


def transform(source, literal):
    lines = source.splitlines()
    assert one(r'^  %i434 = extractvalue \[2 x <1 x float>\] (%i90), 0', source) == '%i90'
    assert one(r'^  %i447 = extractvalue \[2 x <1 x float>\] (%i90), 1', source) == '%i90'
    rows, subtraction = score_rows(lines)
    # Definitions removed outright (old M, alpha, shifted scores, O/L scaling).
    remove = {'%i434', '%i435', '%i436', '%i447', '%i448', '%i449', '%i450', '%i459',
              '%i468', '%i469', '%i470', '%i471', '%i472',
              '%i476', '%i477', '%i478', '%i479', '%i480', '%i1006', '%i1007'}
    rename = {'%i475': '%i474', '%i483': '%i482'}
    remove |= set(rename)
    definitions = dict(re.findall(r'^  (%[\w.]+) = (.*)$', source, re.M))
    for line in lines:
        m = re.match(r'  (%[\w.]+) = fsub <4 x float> (%[\w.]+), (%i450|%i459),', line)
        if m:
            remove.add(m[1])
            rename[m[1]] = '%lazy.S.' + m[2][1:]
        m = re.match(r'  (%[\w.]+) = fmul <4 x float> (%[\w.]+), (%i472|%i480),', line)
        if m:
            # The slow path scales O[row] by alpha[row]: prove each original
            # fmul pairs them the same way, through shuffle <- extractvalue.
            half = re.match(r'shufflevector <8 x float> (%[\w.]+),', definitions[m[2]])
            field = re.match(r'extractvalue \[2 x <8 x float>\] (%[\w.]+), (\d)', definitions[half[1]])
            assert field[1] in O_PHIS and int(field[2]) == {'%i472': 0, '%i480': 1}[m[3]], line
            remove.add(m[1])
            rename[m[1]] = m[2]
    assert sum(v.startswith('%lazy.S.') for v in rename.values()) == 16
    # 4 O phis x 2 rows x <8 x float> = 16 <4 x float> (32 v_pk_mul), plus L x 2.
    assert sum(not v.startswith('%lazy.') for v in rename.values()) == 16 + 2

    region = {phi: '%lazy.O.' + phi[1:] for phi in O_PHIS}
    region['%i89'] = '%lazy.L'
    output, seeded, after_split, in_exit, rewritten = [], 0, False, False, 0
    for line in lines:
        if line.startswith('declare float @__ocml_exp2_f32'):
            output.append('declare i64 @llvm.amdgcn.fcmp.i64.f32(float, float, i32 immarg)')
        if line.startswith('  %i90 = phi '):
            line, count = re.subn(r'\[ %i1007, %k0\.join \], \[ \[[^\]]*\], %bb \]',
                                  '[ %lazy.N, %k0.join ], [ zeroinitializer, %bb ]', line)
            assert count == 1, line
            output.append(line)
            continue
        if line.startswith('bb1021:'):
            in_exit = True
        m = MMAC.match(line)
        if m and m[4] == 'zeroinitializer':
            if not seeded:
                for row in (0, 1):
                    output.append(f'  %lazy.n{row} = extractvalue [2 x <1 x float>] %i90, {row}{DBG}')
                    output.append(f'  %lazy.seed{row} = shufflevector <1 x float> %lazy.n{row}, <1 x float> poison, <4 x i32> zeroinitializer{DBG}')
            line, count = re.subn(r'<4 x float> zeroinitializer\)',
                                  f'<4 x float> %lazy.seed{rows[m[1]]})', line)
            assert count == 1, line
            seeded += 1
        if line.startswith('  %i450 = '):
            output += slow_path(literal, subtraction)
            after_split = True
        d = re.match(r'  (%[\w.]+) = ', line)
        if d and d[1] in remove:
            rewritten += 1
            continue
        if after_split and not in_exit:
            line = TOKEN.sub(lambda t: region.get(t[0], t[0]), line)
        line = TOKEN.sub(lambda t: rename.get(t[0], t[0]), line)
        output.append(line)
    assert seeded == 16 and after_split and in_exit
    result = '\n'.join(output) + '\n'
    for name in remove:
        assert not re.search(re.escape(name) + r'(?![\w.])', result), ('dangling', name)
    return result


def audit(text):
    fast = text[text.index('\nbb95:'):text.index('\nlazy.slow:')] + \
        text[text.index('\nlazy.join:'):text.index('\nbb1021:')]
    slow = text[text.index('\nlazy.slow:'):text.index('\nlazy.join:')]
    checks = dict(
        loop_MMAC=fast.count('v_mmac_f32_16x16x16_f16'),
        fast_fsub_vec4=len(re.findall(r'= fsub <4 x float>', fast)),
        fast_fmul_vec4=len(re.findall(r'= fmul <4 x float>', fast)),
        fast_exp2=fast.count('@llvm.amdgcn.exp2.f32'),
        slow_fsub_vec4=len(re.findall(r'= fsub <4 x float>', slow)),
        slow_exp2=slow.count('@llvm.amdgcn.exp2.f32'),
    )
    assert checks['loop_MMAC'] == 256, checks
    assert checks['fast_fsub_vec4'] == 0 and checks['fast_fmul_vec4'] == 0, checks
    assert checks['fast_exp2'] == 64, checks
    for op in ('load ', 'store ', 'fence ', 's.barrier', 'buffer_load', 's_waitcnt', 'ds.read', 'v_mmac'):
        assert op not in slow, ('slow path must be pure ALU', op)
    return checks


def invariant_ops(text):
    """Operations that the rewrite must keep exactly, in order."""
    lines = [re.sub(r', ![\w.]+ !\d+', '', l.strip()) for l in text.splitlines()]
    fixed = [l for l in lines if any(x in l for x in (
        'buffer_load_', 's_waitcnt', 'fence ', '@llvm.amdgcn.s.barrier',
        '@llvm.amdgcn.sched.barrier', '@llvm.hcu.ds.read.', ' = load ', 'store '))]
    # A and B of each MMAC are unchanged; C differs only by the seed and O scale.
    mmac = [(m[2], m[3]) for m in map(MMAC.match, ('  ' + l for l in lines)) if m]
    packs = [l for l in lines if 'fptrunc <4 x float>' in l]
    return fixed, mmac, packs


def main():
    prior = json.loads((SOURCE.parent / 'manifest.json').read_text())
    pinned = prior['variants']['k0_sum_anchor']['sha256']
    assert digest(SOURCE) == pinned, 'k0_sum_anchor.ll changed since it was measured'
    source = SOURCE.read_text()
    launch = SOURCE.with_suffix('.launch.json')
    base_ops = invariant_ops(source)
    records = {}
    shutil.copyfile(SOURCE, HERE / 'control.ll')
    for name, literal in [('control', None)] + [(k, v[0]) for k, v in VARIANTS.items()]:
        path = HERE / (name + '.ll')
        if literal is not None:
            text = transform(source, literal)
            assert invariant_ops(text) == base_ops, 'memory/sync/MMAC operands changed'
            path.write_text(text)
        launch_out = path.with_suffix('.launch.json')
        launch_out.write_bytes(launch.read_bytes())
        subprocess.run([LLVM / 'llvm-as', path, '-o', '/dev/null'], check=True)
        opt = path.with_suffix('.o3.ll')
        subprocess.run([LLVM / 'opt', '-S', '-passes=default<O3>,verify', path, '-o', opt], check=True)
        checks = {}
        if literal is not None:
            checks = audit(path.read_text())
            assert '\nlazy.slow:' in opt.read_text(), 'O3 removed the slow path'
            checks['O3'] = audit(opt.read_text())
        records[name] = dict(sha256=digest(path), o3_sha256=digest(opt),
                             branch_threshold=None if literal is None else literal,
                             note=VARIANTS[name][1] if name in VARIANTS else 'byte copy of k0_sum_anchor.ll',
                             LLVM_and_O3_verified=True, checks=checks)
        print('PASS generated', name, checks, flush=True)
    (HERE / 'manifest.json').write_text(json.dumps(dict(
        variants=records, source=str(SOURCE), source_sha256=pinned, target='gfx936',
        shape=[1, 32, 4096, 128], causal=False, gpu_tested=False, target_codegen_tested=False,
        note='IR rewrite only; no compiler pass changed. Non-causal only.'), indent=2) + '\n')


if __name__ == '__main__':
    main()
