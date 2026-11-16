# ThreadIR mask-add bypass 验证

`bypassMaskAdds` 已从 LLVM 层移除。统一使用 `frisk-bypass-mask-adds`，在 thread loop 展开、寄存器 SSA 清理之后，LLVM lowering 之前运行。`--mask-add-bypass` 仅支持 `off|approx`；`legal` 明确报错。其他优化的 `legal` 模式保持原有行为。

支持 Frisk 已融合的 uniform-mask 分支，以及单结果 `scf.if` / 标量或逐元素 `arith.select` 产生的零 mask；跟踪静态 extract、shuffle、reshape、broadcast、transpose、slice 和 from_elements，覆盖多个加法消费者。分支形式将加法放入 masked 分支，clear 分支返回 score；select 形式保留 select，不保证硬件跳过加法。跨内存/同步操作、动态索引、无法证明为正零的 clear lane、逃逸消费者等情况不处理。

`approx` 允许匹配加法的正负零差异，不保证 NaN 位模式或浮点异常等价。保留 StrictFP、OptNone 和显式 denormal 环境保护；不对其他算术设置 fast-math。

## 指定调优参数生成 IR

命令来自 `../causal_mask_lowering/mask_finite_pk_rtz.command.sh`，对应 `pmc_test.log` 中的 `mask_finite_pk_rtz.o3.ll`。调优参数未改变，只修改输出/工作目录。

- [最终 O3 IR](mask_finite_pk_rtz.o3.ll)、[未优化 LLVM IR](mask_finite_pk_rtz.ll)
- [完整命令](mask_finite_pk_rtz.command.sh)、[launch 配置](mask_finite_pk_rtz.o3.launch.json)
- [哈希、参数与对比结果](manifest.json)、[生成脚本](generate.py)

该配置为 `uniform-masks=false`、`causal-mask=finite`，已有直接 select，因此新 pass 命中为 0。原始 LLVM IR 与本轮修改前的 baseline 完全相同；O3 IR 忽略 ModuleID 后相同。保留 32 对 PKRTZ pack。历史 pmc 产物与当前 baseline 已有差异，不能将本次产物描述为历史 IR 的逐字节复现。

LLVM 与 O3 IR 均完成 verifier 验证。未运行目标 GPU codegen 或 GPU 性能测试。

## 命中验证与回归

- [真实 Frisk 多消费者输入](multi_mask.input.mlir)、[处理前](multi_mask.before.mlir)、[处理后](multi_mask.after.mlir)、[命令](multi_mask.commands.json)：1 个 mask 分支，8 个加法消费者；使用不同 score，避免消费者被 CSE 合并。
- `unfused.before.mlir` / `unfused.after.mlir`：未融合 Frisk mask，1 个分支，4 个加法消费者。
- `vector_tests.log`：59 个结构用例及 58,080 次 CPU 数值对照；包含不同 view、分支/select、操作数顺序、clear 分支方向、特殊浮点值以及保守回退。NaN 按类别比较，允许正负零差异。
- `frisk_tests.log`：44 个直接/typed Frisk lowering 用例。
- `check_*.log`：mask fusion、uniform mask、causal mask、semantic/typed fragments、thread math 回归全部通过。
- `tune_tests.log`：27 个调优工具单元测试通过。
- `removed_mode_tests.log`：MyTest 拒绝 legal，旧 LLVM 导出入口拒绝 mask-add-bypass 并提示迁移。
- `build.log`：ThreadTilingTest、MyTest、LegacyLLVMExportTest 构建成功。

本次修改文件的 `git diff --check` 通过。工作区其他已有修改包含空白告警，未调整这些无关文件。
