# Barricade 共享棋谱与使用规范

## Master 对 Master 正式数据集

2026 年 11 月 17 日发布。红蓝双方均为 Barricade Master，标准开局，红方先手。正式数据集包含来源批次按完成时间排序的前 3,000 盘自然完局，共 181,157 手。红方胜 1,689 盘，蓝方胜 1,311 盘。 [下载完整棋谱 JSONL](../training/data/datasets/master-vs-master-3000-20261117/episodes.jsonl)

棋谱为 UTF-8 JSONL，每行一盘，包含完整走法、双方档位、胜方和终局。SHA-256 为 `25ee84a40228ab544115c5f332b53e857582d99ec2ff7b037e7a61431b333a93`。

2026 年 11 月 16 日发布的 500 盘 Master 对 Master 样本全部包含在这 3,000 盘里，原记录逐条相同。训练或统计时直接使用正式数据集即可。 [此前的 500 盘样本](../training/data/shared-samples/master-vs-master-500-20261116.jsonl)

## 2026 年 11 月 16 日发布的共享样本

2026 年 11 月 16 日发布的共享样本共 1,000 局。两组各含 500 盘自然完局，均从各自的已保存棋局中随机无放回抽取。在同一个 GitHub Release 的 Assets 中可下载两份数据和校验文件。[统一下载入口](https://github.com/hy1195571214-byte/quoridor-lab/releases/tag/barricade-samples-20261116)

| 红方 / 蓝方 | 抽样池 | 文件与用途 |
| --- | ---: | --- |
| Master / Master | 858 局 | `master-vs-master-500-20261116.jsonl`。每行一盘完整棋谱，共 30,188 步。可用于回放、走法分析和离线重建局面。 |
| Master / Expert | 736 局 | 文件名包含 `e2-v3-random-500` 的 `.tar.gz`。包含完整棋谱和 23,009 条逐步示范，其中 11,505 条请求 Master，11,504 条请求 Expert。 |

两份文件以原始字节发布。Master / Master 原文件在提交说明中记录了抽样种子及 SHA-256，Master / Expert 的抽样种子及时间在压缩包的 `manifest.json` 中。两组所有对局都按项目规则回放并核对胜方、终局和步数。当前快照在两组之间没有重复的 `episode_id` 或完整走法序列。 [Master / Master 来源提交](https://github.com/hy1195571214-byte/quoridor-lab/commit/f6b81e4428ea4767ba5c125476c7f59b49fffe52) · [Master / Expert 来源 Release](https://github.com/hy1195571214-byte/quoridor-lab/releases/tag/barricade-red-master-blue-expert-451c58f65f6b40bfb7a2bd772573476d-e2-v3-random-500-20261116T054627Z)

## 下载与校验

从 Release 的 Assets 下载两个数据文件及 `SHA256SUMS.txt`，放在同一目录。也可在仓库根目录通过 GitHub CLI 下载并解包。

```bash
mkdir -p data/barricade-samples-20261116
(
  set -e
  cd data/barricade-samples-20261116
  gh release download barricade-samples-20261116 --repo hy1195571214-byte/quoridor-lab
  shasum -a 256 -c SHA256SUMS.txt
  mkdir -p master-vs-expert
  tar -xzf *-e2-v3-random-500-*.tar.gz -C master-vs-expert
)
```

校验通过后再使用数据。Windows PowerShell 可用 `Get-FileHash -Algorithm SHA256 <文件名>`，逐项与清单比对。Master / Expert 压缩包含有 `episodes.jsonl`（整局）、`transitions.jsonl`（逐步示范）、`manifest.json`（抽样信息）和 `batch.json`（导出时的批次配置）。逐局档位以该局 `config.remote_sides` 为准。

## 回放和读取

两组的棋谱都是 UTF-8 JSONL，每行一个对象。按 `moves` 原顺序回放，红方先走。 `e2` 是棋子目标格，`ha2`、`va2` 分别是横墙、竖墙。下例读取第一盘 Master / Master 棋谱并打印本地回放网址。切换到 Master / Expert 时，将参数替换成 `data/barricade-samples-20261116/master-vs-expert/episodes.jsonl`。回放前按项目说明启动本地网页。 [网页启动说明](../README.md#网页启动)

```bash
python3 - data/barricade-samples-20261116/master-vs-master-500-20261116.jsonl <<'PY'
import json
import sys
from urllib.parse import urlencode

with open(sys.argv[1], encoding="utf-8") as source:
    game = json.loads(next(source))

query = urlencode({
    "moves": ",".join(game["moves"]),
    "ai": ",".join(game["config"]["remote_sides"]),
})
print("http://127.0.0.1:4173/analysis?" + query)
PY
```

## 数据字段与训练

- 整局记录以 `episode_id` 为主键。`game_number` 表示来源批次内的编号。`ply_count` 是棋子移动与放墙的总步数。`winner` 的 1 为红方，2 为蓝方。两组的 `status` 均为 `completed`，`finished_reason` 均为 `goal`。
- 逐步记录通过 `episode_id` 关联整局，`ply` 从 0 起，`player` 为该步行动方。`history`、`observation` 和 `action_mask` 对应落子前局面，`notation` 和 `action_id` 对应该步走法，`after` 是走完后的状态。`observation` 形状为 `8 × 9 × 9`，掩码长度为 209，目标动作的掩码必须为 1。 [训练接口与动作编码](../training/README.md)
- Master / Master 这份快照只有整局记录。可以用规则引擎逐步重建局面、动作编号和掩码，并记录这些字段由回放生成。原始请求、响应、缓存状态、时延和 e2 复核证据需要来自原始逐步日志。
- Master / Expert 可直接供项目的模仿学习流程使用。如果只想模仿请求 Master 的应手，先按 `source == "barricade:master"` 筛选。保留原始文件，使用项目默认的 `verified-e2` 训练策略，派生文件写到新位置。 [E2 训练策略](../training/README.md#模仿专家走法)

下载解包后，从仓库根目录执行。请先按项目安装说明创建训练环境并安装 `learning` 依赖。 [训练环境安装](TRAINING_GUIDE.md)

```bash
cd training
.venv/bin/python - ../data/barricade-samples-20261116/master-vs-expert <<'PY'
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1])
with (folder / "transitions.jsonl").open(encoding="utf-8") as source, \
     (folder / "master-transitions.jsonl").open("x", encoding="utf-8") as target:
    for line in source:
        if json.loads(line)["source"] == "barricade:master":
            target.write(line)
PY

.venv/bin/python examples/train_imitation.py \
  --data ../data/barricade-samples-20261116/master-vs-expert/master-transitions.jsonl \
  --e2-policy verified-e2 --seed 0 --epochs 20 \
  --output ../data/barricade-samples-20261116/master-imitation.pt
```

## 合并、比较和复现

1. 记录 Release 标签、文件 SHA-256、筛选规则、划分种子和最终样本数。难度名称来自请求参数，服务端实际模型及版本未独立确认。
2. 先按 `episode_id` 去重，再检查完整走法是否重复。Master / Expert 随机样本与正式采集分片来自同一批次，合并时先去重。按整盘棋划分训练集和验证集，避免同局的相邻局面进入两边。两组对手和开局分布可能不同，报告结果时注明对阵组合与执色。
3. 区分棋谱合法性与应手质量。两份样本均未出现全局前五步恰为 `e2,e8,e3,e7,e2` 的开局。跳过墙步后，红方前三次移动为 `e2 → e3 → e2` 的整局分别有 14 盘（Master / Master）和 23 盘（Master / Expert）。回走是否合理须结合墙和当时局面判断。训练时保留筛选记录，并用独立对局评估棋力。
