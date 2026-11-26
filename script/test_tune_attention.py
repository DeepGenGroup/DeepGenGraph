#!/usr/bin/env python3
"""Input isolation, parameter wiring, filtering and failed-export bookkeeping."""
import contextlib
import io
import json
from pathlib import Path
import shlex
import shutil
import signal
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import TuneAttention as tune
import generate_attention_ir as generator


def completed_results(run):
    return sorted((json.loads(path.read_text()) for path in run.glob("**/result.json")),
                  key=lambda row: row["candidate_index"])


class TuneAttentionTest(unittest.TestCase):
    def test_cross_bn_k0(self):
        _, text, _ = self.invoke('--sweep-base', 'online-vdb-k64', '--sweep',
                                  'cross-bn-k0=0;1', '--list')
        self.assertEqual({r['parameters']['cross-bn-k0'] for r in json.loads(text)}, {'0','1'})
        _, text, _ = self.invoke('--set', 'cross-bn-k0=1', '--dry-run')
        self.assertIn('--cross-bn-k0=1', text)
        for value in ('-1', '2', 'on', '01', '+1'):
            with self.assertRaises(SystemExit):
                self.invoke('--set', 'cross-bn-k0='+value, '--list')
        facts = tune.ir_facts('', 'Cross BN K0: packets=4, slot-bytes=8192, overlap-mmac=32, sum-anchors=64, function=example\n')
        self.assertEqual(facts['cross_bn_k0_reports'][0]['sum_anchors'], 64)

    def test_k_tail_prefetch(self):
        _, text, _ = self.invoke('--sweep-base', 'online-vdb-k64', '--sweep',
                                  'k-tail-prefetch=0;2;4;8', '--list')
        self.assertEqual({r['parameters']['k-tail-prefetch'] for r in json.loads(text)}, {'0','2','4','8'})
        _, text, _ = self.invoke('--case', 'online-vdb-k64', '--set', 'k-lds-prefetch=8',
                                  '--set', 'k-tail-prefetch=4', '--dry-run')
        self.assertIn('--k-lds-prefetch=8', text)
        self.assertIn('--k-tail-prefetch=4', text)
        for v in ('-1', '9', 'on'):
            with self.assertRaises(SystemExit):
                self.invoke('--set', 'k-tail-prefetch='+v, '--list')
        facts = tune.ir_facts('', 'K tail LDS prefetch: distance=4, regions=1, pairs=8, moved=7, anchors=5, unpaired=0, reduction-regions=3, function=example\n')
        self.assertEqual(facts['k_tail_prefetch_reports'][0]['pairs'], 8)

    def test_k_lds_prefetch(self):
        _, text, _ = self.invoke('--sweep-base','online-vdb-k64','--sweep','k-lds-prefetch=0;1;2;4','--list')
        self.assertEqual({r['parameters']['k-lds-prefetch'] for r in json.loads(text)}, {'0','1','2','4'})
        _, text, _ = self.invoke('--case','online-vdb-k64-lds2','--dry-run')
        self.assertIn('--k-lds-prefetch=2',text)
        for v in ('-1','9','on'):
            with self.assertRaises(SystemExit):self.invoke('--set','k-lds-prefetch='+v,'--list')
        facts=tune.ir_facts('', 'K LDS prefetch: distance=2, regions=3, pairs=24, moved=21, anchors=18, unpaired=0, function=example\n')
        self.assertEqual(facts['k_lds_prefetch_reports'][0]['anchors'],18)

    def test_k_dma_sweep_and_validation(self):
        _, preset, _ = self.invoke('--case', 'online-vdb-k64', '--list')
        cfg=json.loads(preset)[0]['parameters']
        self.assertEqual({key:cfg[key] for key in ('workgroup-reverse','local-reduce','defer-lane-sum',
                         'mask-add-bypass','k-dma','v-double-buffer','pv-interleave')},
                         {'workgroup-reverse':'0','local-reduce':'approx','defer-lane-sum':'approx',
                          'mask-add-bypass':'approx','k-dma':'64-swizzle','v-double-buffer':'1','pv-interleave':'4'})
        _, text, _ = self.invoke('--sweep', 'k-dma=128;64;64-swizzle', '--list')
        self.assertEqual({r['parameters']['k-dma'] for r in json.loads(text)},
                         {'128', '64', '64-swizzle'})
        _, preview, _ = self.invoke('--case', 'online-baseline', '--set', 'k-dma=64-swizzle',
                                    '--set', 'k-dma-arg=2', '--dry-run')
        self.assertIn('--k-dma=64-swizzle', preview)
        self.assertIn('--k-dma-arg=2', preview)
        for value in ('k-dma=32', 'k-dma=on', 'k-dma-arg=-1', 'k-dma-arg=abc'):
            with self.assertRaises(SystemExit): self.invoke('--set', value, '--list')
        cfg = dict(tune.BASE, **{'k-dma':'64', 'dma-export':'hcu128'})
        self.assertTrue(tune.incompatible_layout(cfg))
        facts = tune.ir_facts('buffer_load_dwordx2 ',
            'K DMA: mode=64-swizzle, argument=1, packets=8->16, reads=64, bases=2, function=kernel\n')
        self.assertEqual(facts['buffer_dma64_sites'], 1)
        self.assertEqual(facts['k_dma_reports'][0]['bases'], 2)

    def test_v_double_buffer_and_pv_interleave_sweeps(self):
        _, text, _ = self.invoke('--sweep', 'v-double-buffer=0;1',
                                '--sweep', 'pv-interleave=0;2;4', '--list')
        configs = [r['parameters'] for r in json.loads(text)]
        self.assertEqual({(r['v-double-buffer'],r['pv-interleave']) for r in configs},
                         {(v,p) for v in ('0','1') for p in ('0','2','4')})
        _, preview, _ = self.invoke('--case','online-baseline','--set','v-double-buffer=1',
                                    '--set','pv-interleave=4','--dry-run')
        self.assertIn('--v-double-buffer=1',preview)
        self.assertIn('--pv-interleave=4',preview)
        for key,value in [('v-double-buffer','2'),('v-double-buffer','true'),('pv-interleave','3')]:
            with self.assertRaises(SystemExit):
                self.invoke('--set',f'{key}={value}','--list')
        facts = tune.ir_facts('', 'V double buffer: 4 slices, slot-bytes=8192, displacement=-8192, removed-barriers=3, LDS=16384, function=kernel\n'
                             'V double buffer: skipped, function=other: no spare LDS slice\n')
        self.assertEqual(facts['v_double_buffer_reports'][0]['displacement'],-8192)
        self.assertEqual(facts['v_double_buffer_skips'],[{'function':'other','reason':'no spare LDS slice'}])

    def invoke(self, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = tune.main(list(args))
        return result, stdout.getvalue(), stderr.getvalue()

    def test_nested_loop_and_unrelated_attributes(self):
        source = '''// scf.for %dk = %a to %b step %s { }
scf.for %outer = %a to %b step %s {
  scf.for %dk = %a to %b step %s {
    "test.op"() {text = "}", value = 1} : () -> ()
    scf.for %inner = %a to %b step %s { } {frisk.pipeline_hint = 3 : i64}
  } {other = {array = [1, 2], text = "http://test"}, // keep other attributes
     "frisk.pipeline_hint" = 2 : i64, frisk.simplify_loop}
} {frisk.pipeline_hint = 4 : i64}
'''
        changed = tune.prepare_input(source, tune.BASE, "dk")
        self.assertIn('other = {array = [1, 2], text = "http://test"}', changed)
        self.assertIn("frisk.pipeline_hint = 3 : i64", changed)
        self.assertIn("frisk.pipeline_hint = 4 : i64", changed)
        self.assertNotIn('"frisk.pipeline_hint" = 2', changed)
        self.assertEqual(changed.count("frisk.simplify_loop"), 1)
        self.assertEqual(tune.prepare_input(changed, tune.BASE, "dk"), changed)

    def test_no_attributes_and_duplicate_iv(self):
        source = "scf.for %dk = %a to %b step %s { }\n"
        cfg = dict(tune.BASE, **{"pipeline-stages": "3", "loop-mark": "0"})
        self.assertIn("{frisk.pipeline_hint = 3 : i64}", tune.prepare_input(source, cfg, "dk"))
        for invalid in ("", source + source):
            with self.assertRaises(ValueError):
                tune.prepare_input(invalid, cfg, "dk")
        self.assertEqual(tune.prepare_input(source, tune.LEGACY_BASE, "missing"), source)

    def test_capacity_does_not_change_instructions(self):
        source = "@shm_0 = addrspace(3) global [16384 x i8] undef, align 16\nbody"
        result = tune.pad_lds(source, 32768)
        self.assertEqual(result, source.replace("[16384", "[32768"))
        self.assertEqual(tune.lds_bytes(result), 32768)
        with self.assertRaises(ValueError):
            tune.pad_lds(source, 8192)
        with self.assertRaises(ValueError):
            tune.pad_lds(source + "\n" + source.replace("@shm_0", "@shm_1"), 65536)

    def test_preview_wires_all_mytest_arguments(self):
        _, preview, _ = self.invoke("--case", "baseline", "--dry-run")
        command = shlex.split(next(line for line in preview.splitlines() if line.startswith("env ")))
        self.assertEqual(command[3], str(tune.default_paths()[1]))
        self.assertEqual(command[7:23],
                         ["1", "1", "3.3", "34816", "approx", "256", "4,1",
                          "1", "1", "1", "0", "0", "useKPack", "asm128", "32", "32"])
        self.assertEqual(command[23:],
                         ["--frag-prefetch=1", "--frag-prefetch-words=4",
                          "--retain-fragment-words=0",
                          "--pv-interleave=0", "--v-double-buffer=0",
                          "--k-dma=128", "--k-dma-arg=1", "--k-lds-prefetch=0",
                          "--k-tail-prefetch=0",
                          "--cross-bn-k0=0",
                          "--simplify-loop-control=1", "--coalesce-waits=0",
                          "--mask-add-bypass=off",
                          "--defer-lane-sum=off",
                          "--local-reduce=off", "--local-reduce-chains=4",
                          "--local-reduce-min-inputs=16", "--local-reduce-ops=all",
                          "--swizzle-q=off", "--swizzle-k=off", "--swizzle-v=off",
                          "--swizzle-q-arg=-1", "--swizzle-k-arg=-1", "--swizzle-v-arg=-1",
                          "--workgroup-group-size=0", "--workgroup-reverse=0"])
        self.assertIn("pipeline-stages=none", preview)
        self.assertIn("loop-mark=1", preview)

    def test_grid_filters_and_override_precedence(self):
        _, text, _ = self.invoke("--sweep", "copy=async;sync",
                                "--sweep", "dma-pack=useKPack;legacy",
                                "--sweep", "pipeline-stages=none;3",
                                "--set", "pipeline-stages=2", "--list")
        rows = [row["parameters"] for row in json.loads(text)]
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["pipeline-stages"] for r in rows}, {"none", "3"})
        self.assertFalse(any(r["copy"] == "sync" and r["dma-pack"] == "useKPack" for r in rows))

    def test_invalid_flags(self):
        for assignment in ("pipeline-stages=0", "pipeline-stages=65", "coalesce-waits=2",
                           "frag-prefetch-words=0", "qk-tile=-1", "alignment=3",
                           "retain-fragment-words=-1", "retain-fragment-words=1.5",
                           "retain-fragment-words=4294967296", "retain-fragment-words=true"):
            with self.subTest(assignment=assignment), self.assertRaises(SystemExit) as error:
                self.invoke("--case", "baseline", "--set", assignment, "--list")
            self.assertEqual(error.exception.code, 2)

    def test_retained32_preset_and_cli_wiring(self):
        _, text, _ = self.invoke("--case", "retained32-baseline", "--list")
        cfg = json.loads(text)[0]["parameters"]
        self.assertEqual(cfg["retain-fragment-words"], "32")
        self.assertEqual((cfg["workgroup-group-size"],cfg["workgroup-reverse"],cfg["swizzle-k"]),
                         ("8","1","8:6"))
        self.assertEqual([cfg[k] for k in tune.POSITIONAL_KEYS],
                         ["1","1","3.3","34816","approx","256","4,1",
                          "1","1","1","0","0","useKPack","asm128","32","32"])
        self.assertEqual((cfg["pipeline-stages"],cfg["loop-mark"],cfg["coalesce-waits"]),
                         ("none","1","0"))
        _, preview, _ = self.invoke("--case", "retained32-baseline", "--dry-run")
        command = shlex.split(next(l for l in preview.splitlines() if l.startswith("env ")))
        self.assertIn("--retain-fragment-words=32", command)
        self.assertIn("--swizzle-k=8:6", command)
        for name in ("baseline","legacy-baseline","group8-reverse-k128-swizzle"):
            _, text, _ = self.invoke("--case",name,"--list")
            self.assertEqual(json.loads(text)[0]["parameters"]["retain-fragment-words"],"0")
        _, preview, _ = self.invoke("--case","mytest-default","--dry-run")
        self.assertNotIn("--retain-fragment-words=",preview)
        for value in ("0","00032","4294967295"):
            _, text, _ = self.invoke("--case","retained32-baseline","--set",
                                    "retain-fragment-words="+value,"--list")
            self.assertEqual(json.loads(text)[0]["parameters"]["retain-fragment-words"],str(int(value)))

    def test_online_preset_matches_native_accumulator_configuration(self):
        _, text, _ = self.invoke("--case", "online-baseline", "--list")
        cfg = json.loads(text)[0]["parameters"]
        self.assertEqual(cfg, dict(tune.RETAINED32_BASE,
                                  **{"workgroup-group-size": "4", "frag-prefetch": "0"}))
        _, preview, _ = self.invoke("--case", "online-baseline", "--dry-run")
        self.assertIn("--frag-prefetch=0", preview)
        self.assertIn("--workgroup-group-size=4", preview)
        self.assertIn("--retain-fragment-words=32", preview)

    def test_mask_add_bypass_option(self):
        for mode in ("off", "approx"):
            _, preview, _ = self.invoke("--case", "online-baseline", "--set",
                                        "mask-add-bypass=" + mode, "--dry-run")
            self.assertIn("--mask-add-bypass=" + mode, preview)
        _, text, _ = self.invoke("--case", "online-baseline", "--list")
        self.assertEqual(json.loads(text)[0]["parameters"]["mask-add-bypass"], "off")
        with self.assertRaises(SystemExit):
            self.invoke("--case", "online-baseline", "--set", "mask-add-bypass=invalid", "--list")
        with self.assertRaises(SystemExit):
            self.invoke("--case", "online-baseline", "--set", "math=strict",
                        "--set", "mask-add-bypass=approx", "--list")
        with self.assertRaises(SystemExit):
            self.invoke("--case", "online-baseline", "--set", "mask-add-bypass=legal", "--list")
        facts = tune.ir_facts("", "MaskAddBypassThread: Attn_p2: 1 groups, 16 additions bypassed (approx)\n")
        self.assertEqual(facts["mask_add_bypass_reports"],
            [dict(function="Attn_p2", groups=1, additions=16, mode="approx", stage="thread")])

    def test_defer_lane_sum_options(self):
        _, preview, _ = self.invoke("--case", "online-baseline", "--set", "defer-lane-sum=approx", "--dry-run")
        self.assertIn("--defer-lane-sum=approx", preview)
        for settings in (("defer-lane-sum=invalid",), ("defer-lane-sum=approx", "math=strict")):
            with self.assertRaises(SystemExit):
                args = [part for setting in settings for part in ("--set", setting)]
                self.invoke("--case", "online-baseline", *args, "--list")
        _, text, _ = self.invoke("--sweep-base", "online-baseline", "--sweep", "defer-lane-sum=off;legal;approx", "--list")
        self.assertEqual([r["parameters"]["defer-lane-sum"] for r in json.loads(text)], ["off", "legal", "approx"])
        facts = tune.ir_facts("", "DeferLaneSum: generic: 2 states, 4 loop shuffles deferred (approx)\n")
        self.assertEqual(facts["defer_lane_sum_reports"],
                         [dict(function="generic", states=2, loop_shuffles=4, mode="approx")])

    def test_local_reduction_options(self):
        _, preview, _ = self.invoke("--case", "online-baseline", "--set", "local-reduce=approx",
                                   "--set", "local-reduce-ops=add,max", "--dry-run")
        self.assertIn("--local-reduce=approx", preview)
        self.assertIn("--local-reduce-ops=add,max", preview)
        for option in ("local-reduce=bad", "local-reduce-chains=1", "local-reduce-chains=17",
                       "local-reduce-min-inputs=3", "local-reduce-ops=sub", "local-reduce-ops=max,"):
            with self.assertRaises(SystemExit):
                self.invoke("--case", "online-baseline", "--set", option, "--list")
        with self.assertRaises(SystemExit):
            self.invoke("--case", "online-baseline", "--set", "math=strict",
                        "--set", "local-reduce=approx", "--list")
        _, text, _ = self.invoke("--sweep-base", "online-baseline", "--sweep", "local-reduce-chains=2;4;16", "--list")
        self.assertEqual([row["parameters"]["local-reduce-chains"] for row in json.loads(text)], ["2", "4"])
        facts = tune.ir_facts("", "LocalReduction: generic: 3 groups, 96 inputs, 4 chains (legal)\n")
        self.assertEqual(facts["local_reduction_reports"],
                         [dict(function="generic", groups=3, inputs=96, chains=4, mode="legal")])

    def test_retention_sweep_preserves_selected_baseline(self):
        _, text, _ = self.invoke("--sweep-base","retained32-baseline",
                                "--sweep","retain-fragment-words=0;8;16;24;32",
                                "--set","retain-fragment-words=40","--list")
        rows = [r["parameters"] for r in json.loads(text)]
        self.assertEqual([r["retain-fragment-words"] for r in rows],["0","8","16","24","32"])
        for row in rows:
            self.assertEqual((row["workgroup-group-size"],row["workgroup-reverse"],row["swizzle-k"]),
                             ("8","1","8:6"))
        _, text, _ = self.invoke("--sweep-base","retained32-baseline",
                                "--sweep","workgroup-group-size=4;8;16","--list")
        self.assertTrue(all(r["parameters"]["retain-fragment-words"]=="32" for r in json.loads(text)))
        _, text, _ = self.invoke("--sweep","retain-fragment-words=0;32","--list")
        self.assertTrue(all(r["parameters"]["swizzle-k"]=="off" for r in json.loads(text)))
        for args in (("--sweep-base","retained32-baseline"),
                     ("--sweep-base","missing","--sweep","retain-fragment-words=32"),
                     ("--sweep-base","mytest-default","--sweep","retain-fragment-words=32")):
            with self.subTest(args=args), self.assertRaises(SystemExit):
                self.invoke(*args,"--list")

    def test_retention_diagnostics_do_not_confuse_budget_with_actual_use(self):
        diagnostics = ("RetainInvariantFragments: 4 operand tiles retained; budget 32 words/thread\n"
                       "frisk.retained_words = 32 : i32\n")
        facts = tune.ir_facts("",diagnostics+"frisk.retained_words = 32 : i32\n"*4)
        self.assertEqual(facts["retention_reports"],[dict(operand_tiles=4,budget_words_per_thread=32)])
        self.assertEqual(facts["retained_loop_words_observed"],[32])
        facts = tune.ir_facts("","RetainInvariantFragments: 0 operand tiles retained; budget 32 words/thread\n")
        self.assertEqual(facts["retention_reports"][0]["operand_tiles"],0)
        self.assertEqual(facts["retained_loop_words_observed"],[])
        self.assertEqual(tune.ir_facts("","")["retention_reports"],[])

    def test_swizzle_presets_and_canonical_values(self):
        _, text, _ = self.invoke("--case", "group8-reverse*", "--list")
        presets = {row["case"]: row["parameters"] for row in json.loads(text)}
        self.assertEqual(len(presets), 4)
        winner = presets["group8-reverse-k128-swizzle"]
        self.assertEqual([winner[k] for k in tune.SWIZZLE_KEYS], ["off", "8:6", "off"])
        self.assertEqual((winner["workgroup-group-size"],winner["workgroup-reverse"]), ("8","1"))
        self.assertEqual(tune.BASE["swizzle-k"], "off")
        self.assertEqual(tune.BASE["workgroup-group-size"], "0")
        _, text, _ = self.invoke("--sweep", "swizzle-k=0;off;1;on;08:06", "--list")
        self.assertEqual([x["parameters"]["swizzle-k"] for x in json.loads(text)], ["off","8:6"])
        _, text, _ = self.invoke("--case", "mytest-default", "--dry-run")
        self.assertNotIn("--swizzle-", text)
        self.assertNotIn("--workgroup-", text)

    def test_swizzle_validation_and_combination_filters(self):
        for setting in ("swizzle-q=7:3", "swizzle-k=8:8", "swizzle-v=08:8",
                        "swizzle-k=31:6", "swizzle-v=garbage", "swizzle-q-arg=-2",
                        "swizzle-k-arg=2147483648", "workgroup-group-size=-1",
                        "workgroup-group-size=9223372036854775808", "workgroup-query-axis=w"):
            with self.subTest(setting=setting), self.assertRaises(SystemExit):
                self.invoke("--case", "baseline", "--set", setting, "--list")
        _, text, _ = self.invoke("--sweep", "swizzle-k=off;on", "--sweep", "pack=0;1",
                                "--sweep", "dma-export=split32;asm128", "--list")
        rows = [r["parameters"] for r in json.loads(text)]
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r["pack"] == "1" and r["dma-export"] == "asm128"
                            for r in rows if r["swizzle-k"] != "off"))
        _, text, _ = self.invoke("--sweep", "workgroup-group-size=0;8",
                                "--sweep", "workgroup-reverse=false;true", "--list")
        self.assertEqual(len(json.loads(text)), 3)
        for settings in (("swizzle-k=on", "alignment=4"),
                         ("swizzle-k=on", "swizzle-k-arg=1", "swizzle-v-arg=1"),
                         ("workgroup-group-size=8", "workgroup-query-axis=x", "workgroup-head-axis=x")):
            with self.subTest(settings=settings), self.assertRaises(SystemExit):
                self.invoke("--case", "baseline", *[a for v in settings for a in ("--set",v)], "--list")

    def test_workgroup_axes_follow_input_semantics(self):
        source = generator.TEMPLATE.read_text()
        cfg = dict(tune.BASE, **{"workgroup-group-size":"8", "workgroup-reverse":"1"})
        current = tune.resolve_workgroup_axes(source, cfg)
        self.assertEqual([current[k] for k in tune.WORKGROUP_AXIS_KEYS], ["x","y"])
        # Axis inference uses output dimensions, not grid sizes or SSA spelling.
        old = source.replace("res_dims = [2]", "res_dims = [TEMP]").replace(
            "res_dims = [1]", "res_dims = [2]").replace("res_dims = [TEMP]", "res_dims = [1]")
        previous = tune.resolve_workgroup_axes(old, cfg)
        self.assertEqual([previous[k] for k in tune.WORKGROUP_AXIS_KEYS], ["y","x"])
        self.assertEqual(cfg["workgroup-query-axis"], "auto")
        with self.assertRaises(ValueError):
            tune.resolve_workgroup_axes("no mapping", cfg)
        explicit = dict(cfg, **{"workgroup-query-axis":"y", "workgroup-head-axis":"x"})
        self.assertEqual(tune.resolve_workgroup_axes("no mapping", explicit), explicit)
        with self.assertRaises(ValueError):
            tune.resolve_workgroup_axes(source, dict(cfg, **{"workgroup-query-axis":"y"}))
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/"current.mlir", Path(tmp)/"old.mlir"
            first.write_text(source); second.write_text(old)
            _, preview, _ = self.invoke("--input",str(first),str(second),
                                       "--case","group8-reverse-k128-swizzle","--dry-run")
            commands = [shlex.split(l) for l in preview.splitlines() if l.startswith("env ")]
            self.assertIn("--workgroup-query-axis=x", commands[0])
            self.assertIn("--workgroup-query-axis=y", commands[1])
            self.assertTrue(all("--swizzle-k=8:6" in command for command in commands))

    def test_swizzle_diagnostics_distinguish_static_ir_from_pmc(self):
        text = ("buffer_load_dwordx4 v0\nbuffer_load_dword v1\n"
                "%x = load <4 x half>, ptr addrspace(3) %p\n")
        log = ("Shared swizzle: argument 1 byte bit 6 ^= 8, DMA=16 bytes, aliases=2\n"
               "CanonicalizeLDSGEPs: Attn_p2: 124 proven constant offsets\n"
               "frisk.workgroup_order = array<i64: 8, 0, 1, 1>\n" * 2)
        facts = tune.ir_facts(text, log)
        self.assertEqual(facts["buffer_dma128_sites"],1)
        self.assertEqual(facts["buffer_dma32_sites"],1)
        self.assertEqual(facts["lds_b64_half_load_sites"],1)
        self.assertEqual(facts["lds_gep_reported_offsets"],248)
        self.assertEqual(facts["workgroup_orders"],[[8,0,1,1]])
        self.assertEqual(facts["shared_swizzle_plans"][0],
                         dict(argument=1,source_bit=8,target_bit=6,dma_bytes=16,aliases=2))
        self.assertNotIn("bank_conflict", facts)

    def test_generated_inputs_and_portable_tool(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            portable = directory / "portable tools"
            portable.mkdir()
            shutil.copy2(ROOT / "TuneAttention.py", portable / "TuneAttention.py")
            tool = portable / "MyTest"
            tool.write_text('''#!/usr/bin/env python3
import sys
from pathlib import Path
source = Path(sys.argv[1]).read_text()
assert "deepgengraph.kernel @Attn_p2" in source
Path(sys.argv[3]).write_text("define amdgpu_kernel void @Attn_p2() {\\n  ret void\\n}\\n")
''')
            tool.chmod(0o755)
            template = generator.TEMPLATE.read_text()
            sources = []
            for index, bk in enumerate((None, 32)):
                # 同名输入放在不同目录中，且路径包含空格。
                path = directory / f"input {index}" / "attention.mlir"
                path.parent.mkdir()
                path.write_text(generator.generate_ir(template, 64, 32, bk)[0])
                sources.append(path)
            generated_stdout = "\n".join(map(str, sources + [sources[0]])) + "\n"
            args = [sys.executable, str(portable / "TuneAttention.py"),
                    "--input-list", "-", "--case", "baseline",
                    "--output-dir", str(directory / "results")]
            completed = subprocess.run(args, input=generated_stdout, text=True, cwd=directory,
                                       capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr + completed.stdout)
            run = next((directory / "results").glob("run-*"))
            records = json.loads((run / "run.json").read_text())
            self.assertEqual(records["tool"], str(tool))
            self.assertEqual(len(records["inputs"]), 2)
            self.assertEqual([r["loop_iv"] for r in records["inputs"]], [None, "dk"])
            rows = completed_results(run)
            self.assertEqual([r["status"] for r in rows], ["exported", "exported"])
            self.assertEqual([r["parameters"]["loop-mark"] for r in rows], ["keep", "1"])
            for index, record in enumerate(records["inputs"]):
                self.assertEqual(Path(record["snapshot"]).read_text(), sources[index].read_text())
            self.assertIn("无 %dk", completed.stderr)
            # 预览命令可重放，保留多输入目录隔离。
            preview = subprocess.run(args + ["--dry-run"], input=generated_stdout, text=True,
                                     cwd=directory, capture_output=True, check=True)
            subprocess.run(["bash", "-e", "-c", preview.stdout], cwd=directory, check=True)
            self.assertEqual(len(list((directory / "OUTPUT").glob("*/baseline/kernel.ll"))), 2)

    def test_directory_list_and_explicit_loop_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            first, second = directory / "a.mlir", directory / "b.mlir"
            first.write_text("scf.for %dk = %a to %b step %s { }\n")
            second.write_text("scf.for %n = %a to %b step %s { }\n")
            (directory / "ignored.ll").write_text("")
            _, preview, _ = self.invoke("--input-dir", str(directory), "--case", "baseline", "--dry-run")
            self.assertEqual(preview.count("\nenv "), 2)
            listing = directory / "paths.txt"
            listing.write_text(f"{first}\n\n{second}\n{first}\n")
            _, from_list, _ = self.invoke("--input-list", str(listing), "--case", "baseline", "--dry-run")
            self.assertEqual(preview, from_list)
            _, from_files, _ = self.invoke("--input", str(first), str(second),
                                           "--case", "baseline", "--dry-run")
            self.assertEqual(preview, from_files)
            with self.assertRaises(SystemExit) as error:
                self.invoke("--input", str(second), "--loop-iv", "dk", "--case", "baseline", "--dry-run")
            self.assertEqual(error.exception.code, 2)
            with mock.patch("sys.stdin", io.StringIO("")), self.assertRaises(SystemExit) as error:
                self.invoke("--input-list", "-", "--case", "baseline", "--dry-run")
            self.assertEqual(error.exception.code, 2)

    def test_launch_metadata_uses_ir_block_and_reverses_grid(self):
        template = generator.TEMPLATE.read_text()
        source, _ = generator.generate_ir(template, 64, 32, 16)
        ir = ("define amdgpu_kernel void @Attn_p2() !reqd_work_group_size !7 {\n}\n"
              "!7 = !{i32 128, i32 1, i32 1}\n")
        config = tune.launch_config(source, ir)
        self.assertEqual(config["shape"], [1, 32, 4096, 128])
        self.assertEqual(config["grid"], [64, 32, 1])
        self.assertEqual(config["block"], [128, 1, 1])
        self.assertEqual(config["shared_mem"], 0)
        self.assertEqual(config["k_layout"], "bhsd")
        with self.assertRaises(ValueError):
            tune.launch_config(source, ir.replace("!reqd_work_group_size", "!other"))

    def test_backend_benchmark_and_failure_isolation(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            original = directory / "input.mlir"
            original.write_text(generator.generate_ir(generator.TEMPLATE.read_text(), 64, 32, 32)[0])
            tool = directory / "MyTest"
            tool.write_text('''#!/usr/bin/env python3
from pathlib import Path
import sys
Path(sys.argv[3]).write_text(
    "define amdgpu_kernel void @Attn_p2() !reqd_work_group_size !7 {\\n ret void\\n}\\n"
    "!7 = !{i32 128, i32 1, i32 1}\\n")
''')
            tool.chmod(0o755)
            build = directory / "GenHsaco.sh"
            build.write_text('''#!/bin/bash
set -e
test "$1" = kernel.ll
test "$2" = 3
test -f "$1"
echo object > kernel.o
if [[ "$PWD" == *grid-000003 ]]; then
    if [[ -n "${TUNE_TEST_PAUSE:-}" ]]; then
        echo $$ > "$TUNE_TEST_PAUSE"
        sleep 60
    fi
    exit 9
fi
echo hsaco > "$3"
''')
            launcher = directory / "launcher.py"
            launcher.write_text('''import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
for flag in ("hsaco", "config", "json-output", "device", "warmup", "repeat", "seed"):
    p.add_argument("--" + flag, required=True)
a = p.parse_args()
c = json.loads(Path(a.config).read_text())
assert c["grid"] == [64, 32, 1] and c["block"] == [128, 1, 1]
assert c["shape"] == [1, 32, 4096, 128] and c["shared_mem"] == 0
assert Path(a.hsaco).is_file()
assert a.warmup == "0" and a.repeat == "2"
passed = "grid-000001" not in a.hsaco
last = "grid-000004" in a.hsaco
Path(a.json_output).write_text(json.dumps({
    "passed": passed, "kernel_median_ms": 1.0 if last else 0.5,
    "sdpa_median_ms": 2.0 if last else 0.25}))
raise SystemExit(0 if passed else 1)
''')
            interpreter = directory / "gpu-python"
            interpreter.symlink_to(sys.executable)
            args = ["--input", str(original), "--tool", str(tool), "--benchmark",
                    "--gen-hsaco", str(build), "--launcher", str(launcher),
                    "--launch-python", str(interpreter),
                    "--warmup", "0", "--repeat", "2", "--output-dir", str(directory / "results")]
            code, _, _ = self.invoke(*args, "--sweep", "carrier=0;24576;28672;34816")
            self.assertEqual(code, 1)
            run = next((directory / "results").glob("run-*"))
            rows = completed_results(run)
            self.assertEqual([r["status"] for r in rows],
                             ["incorrect", "benchmark-passed", "hsaco-failed:9", "benchmark-passed"])
            self.assertFalse((run / "grid-000003/benchmark.json").exists())
            self.assertFalse((directory / "kernel.o").exists())
            best = json.loads((run / "best.json").read_text())
            self.assertEqual(best[str(original)]["case"], "grid-000004")
            self.assertEqual(best[str(original)]["speedup"], 2.0)
            self.assertEqual(best[str(original)]["parameters"]["carrier"], "34816")
            self.assertFalse(list(run.glob("*.tsv")))
            record = json.loads((run / "run.json").read_text())
            self.assertTrue(record["hardware_tested"])
            config = json.loads((run / "grid-000002/config.json").read_text())
            self.assertEqual(config["launch_command"][0], str(interpreter))
            # 完整预览可重放，GenHsaco 的临时文件始终位于候选目录内。
            _, preview, _ = self.invoke(*args, "--case", "baseline", "--dry-run")
            subprocess.run(["bash", "-e", "-c", preview], cwd=directory, check=True)
            self.assertTrue((directory / "OUTPUT/baseline/kernel.o").is_file())
            self.assertTrue((directory / "OUTPUT/baseline/benchmark.json").is_file())
            # Stop the wrapper during candidate 3. Candidate 2's checkpoint must
            # already be readable, and cancellation must reach backend children.
            entry = ROOT / "run_tuning.py"
            if not entry.exists():
                entry = ROOT / "portable/attention-tuning/run_tuning.py"
            wrapper = directory / "wrapper.py"
            wrapper.write_text(f'''import importlib.util, os, signal, sys
spec = importlib.util.spec_from_file_location("runner", {str(entry)!r})
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
signal.signal(signal.SIGTERM, runner.interrupted)
try:
    sys.exit(runner.run_tuning(sys.argv[1:], os.environ.copy()))
except KeyboardInterrupt:
    sys.exit(130)
''')
            session = directory / "interrupted-session"
            marker = directory / "backend-started"
            command = [sys.executable, str(wrapper), sys.executable, str(ROOT / "TuneAttention.py"),
                       *args, "--sweep", "carrier=0;24576;28672;34816",
                       "--output-dir", str(session / "tuning"), "--summary-dir", str(session)]
            with (directory / "interruption.log").open("w") as log:
                process = subprocess.Popen(command, stdout=log, stderr=log,
                                           env=dict(os.environ, TUNE_TEST_PAUSE=str(marker)))
                try:
                    deadline = time.monotonic() + 15
                    while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
                        time.sleep(0.05)
                    self.assertTrue(marker.exists(), (directory / "interruption.log").read_text())
                    progress = json.loads((session / "summary.json").read_text())
                    self.assertEqual(progress["state"], "running")
                    self.assertEqual(progress["completed_candidates"], 2)
                    winner = json.loads((session / "best_by_shape.json").read_text())["1x32x4096x128"]
                    self.assertEqual(winner["case"], "grid-000002")
                    self.assertEqual(winner["sdpa_median_ms"], 0.25)
                    process.send_signal(signal.SIGTERM)
                    self.assertEqual(process.wait(timeout=10), 130)
                    progress = json.loads((session / "summary.json").read_text())
                    self.assertEqual(progress["state"], "interrupted")
                    self.assertEqual(progress["completed_candidates"], 2)
                    self.assertEqual(progress["best_by_shape"]["1x32x4096x128"]["case"], "grid-000002")
                    self.assertFalse(Path("/proc").joinpath(marker.read_text().strip()).exists())
                    self.assertFalse(list(session.glob("**/*.tsv")))
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)

    def test_input_snapshot_and_failed_padding(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            original = directory / "original.mlir"
            source = "scf.for %dk = %a to %b step %s { } {frisk.pipeline_hint}\n"
            original.write_text(source)
            tool = directory / "fake_mytest"
            tool.write_text('''#!/usr/bin/env python3
import sys
from pathlib import Path
Path(sys.argv[3]).write_text("@shm_0 = addrspace(3) global [16384 x i8] undef, align 16\\n"
                           "define amdgpu_kernel void @Attn_p2() {\\n  ret void\\n}\\n")
print("---------- after frisk-auto-pipeline ---------\\nmodule {\\n}\\n")
print("SimplifyLoopControl: 1 small loops specialized")
''')
            tool.chmod(0o755)
            code, _, _ = self.invoke("--case", "baseline", "--case", "nohint-unrolled-32k",
                                    "--input", str(original), "--tool", str(tool),
                                    "--output-dir", str(directory))
            self.assertEqual(code, 0)
            run = next(directory.glob("run-*"))
            rows = completed_results(run)
            self.assertEqual([r["static_lds_bytes"] for r in rows], [16384, 32768])
            self.assertEqual(original.read_text(), source)
            candidate = run / "baseline"
            cfg = json.loads((candidate / "config.json").read_text())
            self.assertEqual(cfg["command"][4], str(candidate / "input.mlir"))
            self.assertNotIn("pipeline_hint", (candidate / "input.mlir").read_text())
            self.assertIn("frisk.simplify_loop", (candidate / "input.mlir").read_text())
            _, preview, _ = self.invoke("--case", "nohint-unrolled-32k", "--dry-run",
                                        "--input", str(original), "--tool", str(tool))
            subprocess.run(["bash", "-e", "-c", preview], cwd=directory, check=True,
                           stdout=subprocess.DEVNULL)
            self.assertEqual(tune.lds_bytes((directory / "OUTPUT/nohint-unrolled-32k/kernel.ll").read_text()),
                             32768)
            code, _, _ = self.invoke("--case", "baseline", "--set", "lds-bytes=8192",
                                    "--input", str(original), "--tool", str(tool),
                                    "--output-dir", str(directory / "failed"))
            self.assertEqual(code, 1)
            failed = next((directory / "failed").glob("run-*"))
            result = json.loads((failed / "baseline/result.json").read_text())
            self.assertEqual(result["status"], "invalid-lds-control")


if __name__ == "__main__":
    unittest.main()
