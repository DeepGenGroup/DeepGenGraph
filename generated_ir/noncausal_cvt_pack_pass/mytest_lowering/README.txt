MyTest 完整 lowering 日志

lowering.log 是 command.sh 从 input.mlir 开始执行 MyTest 的 stdout + stderr，
已开启 --dump-llvm-passes=1，包含 MLIR 到 LLVM 全流程以及每个 LLVM Pass 后的 IR。
stages.txt 列出各阶段行号。LazySoftmaxBase 和 SoftmaxCvtPack 都在 MyTest 内执行。
kernel.ll 是该次 MyTest 导出；kernel.o3.ll 经 default<O3>,verify 生成。
generation.json 记录命令、哈希、阶段索引和命中信息。input.mlir 是原输入的快照。

这是当前 MyTest 全流水线的新导出。先前 ../pass_pk_rtz.o3.ll 是从历史 lazy_t8.ll
经独立 Pass 生成，两者并非同一条生成路径，新产物与历史文件不逐字节相同。
原有 pass_pk_rtz.o3.ll 未覆盖；本日志对应的是本目录 kernel.ll / kernel.o3.ll。

重现（仓库根目录）：
  bash generated_ir/noncausal_cvt_pack_pass/mytest_lowering/command.sh > generated_ir/noncausal_cvt_pack_pass/mytest_lowering/lowering.log 2>&1
