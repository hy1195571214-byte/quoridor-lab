# 训练、验证与测试数据

完整冻结语料和原始数据通过 [GitHub Release](https://github.com/hy1195571214-byte/quoridor-lab/releases/tag/training-data-2026-11-23) 分发。仓库保留来源、审计、第一轮清单及固定划分说明，大附件不写入 Git 历史。

| 分区 | 棋谱 | 逐步样本 |
| --- | ---: | ---: |
| 训练 | 168,730 | 6,559,801 |
| 验证 | 20,848 | 809,950 |
| 测试 | 21,281 | 829,255 |
| 合计 | 210,859 | 8,199,006 |

在仓库根目录执行：

```bash
python training/examples/install_training_data.py
```

或先在 Release 下载 `quoridor-training-data-round2-v1.tar.gz`，再执行：

```bash
python training/examples/install_training_data.py --archive /path/to/quoridor-training-data-round2-v1.tar.gz
```

脚本校验下载包、完整棋谱和冻结清单的 SHA-256，恢复全部分区，并生成本机 `active-dataset.json`。安装不会启动训练。可用 `--data-dir` 指定数据存储目录；当前 Mac 的运行数据仍保存在外接 HY。

独立测试集另有 `quoridor-test-set-round2-v1.jsonl.gz`，每行保留 `split=test`、来源与完整棋谱，21,281 盘。该附件只用于评价，不应混入训练。日常训练安装完整包即可，原始 1.95 GB ZIP 不必再次解压。

`release.json` 提供所有附件地址、大小和校验值。上游 Quoridor-Zero 数据的许可仍为未知/未核实；项目代码许可不替代第三方数据许可，发布保留来源与现有许可状态。

训练方案、数据限制及网页手动启动方式见[合并数据说明](../../../docs/COMBINED_TRAINING_DATA.md)。
