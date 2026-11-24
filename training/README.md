# Quoridor Lab 本地专家采集与训练接口

本项目可以直接向 barricade.gg 的人机专家请求走法，在本地自动完成对局并保存训练数据。无需操控浏览器，不会操作你正在下的棋，也不会把采集局提交到官网的对局历史、排行榜或账号系统。

接口依据官网公开前端使用的 Socket.IO `/ai` 命名空间和 `ai:get_move` 事件实现。它提供的是专家的行为样本，无法取得专家内部算法、模型权重或训练数据。当前未获得服务端模型版本信息，因此记录难度和采集时间，不把难度名称当作固定模型版本。

## 已验证的结果

2026-11-05 已跑完两盘真实专家自动对局，专家执红、执蓝各一盘：

- 25、26 个动作后分别自然结束，专家均获胜；对手是本地仅走最短路、不主动放墙的简单基线。
- 共记录 51 个动作，其中 26 个为实际远程专家决策，没有用本地策略替代专家。
- 数据位于 `data/live-smoke/`；`episodes.jsonl` 保存胜负，`transitions.jsonl` 保存逐步样本。
- 11 项离线检查通过，另验证了 HTTP 查询、完整回合、非法棋谱拒绝。
- 两轮模仿学习和两盘本地策略梯度示例用于检查训练链路，不能据此声称模型已学会专家策略。

## 启动本地服务

在终端执行：

```bash
cd training
.venv/bin/python -m quoridor_lab serve --port 8765
```

接口只监听本机 `127.0.0.1`。打开 [交互式 API 文档](http://127.0.0.1:8765/docs)，可直接查看和调用接口。若已有服务占用 8765，可使用 `--port 8766`。

新环境安装：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install .
# 运行神经网络示例时再安装：
.venv/bin/python -m pip install '.[learning]'
```

原始验证环境复用了已安装的 NumPy、Gymnasium、FastAPI 和 PyTorch；在其他机器上请按上面的安装命令准备依赖。源码修改后，项目目录内的 `python -m quoridor_lab` 会直接使用新源码；从其他目录运行安装包时，请重新 `pip install .` 更新安装副本。

## 请求一次官网专家决策

命令行：

```bash
.venv/bin/python -m quoridor_lab probe --moves e2
```

HTTP：

```bash
curl http://127.0.0.1:8765/expert/move \
  -H 'Content-Type: application/json' \
  -d '{"moves":["e2"]}'
```

返回 `action_id`、`notation`、`source`、耗时、是否缓存和官网原始响应。请求前会在本地严格重放并校验整条棋谱；不接受非法局面或终局后继续查询。

Python：

```python
from quoridor_lab.rules import State
from quoridor_lab.opponents import RemoteExpert

teacher = RemoteExpert(difficulty="expert", data_dir="data")
try:
    state = State.from_moves(["e2"])
    decision = teacher.choose(state)
    print(decision.to_dict())
finally:
    teacher.close()
```

默认从握手开始直接使用 WebSocket，不需要浏览器。2026-11-16 的交叉实验将频繁返回错误 `e2` 的路径定位到绕过环境代理的 Polling 初始化；WebSocket 首连在使用和绕过代理时均通过同局面对照。CLI 可用 `--transport polling` 或 `--transport upgrade` 复现旧路径；Python 使用 `transport="websocket"`。连接失败不自动回退到 Polling。适配器默认不使用环境代理，且 HTTP 与 WebSocket 均遵守这一设置；需要使用现有环境代理时加入 `--proxy-env`，或在 Python 中设 `proxy_env=True`。不修改系统网络设置。详见[排查记录](../docs/BARRICADE_E2_INVESTIGATION_20261116.md)。

## 自动批量对战采集

### 本地采集面板

完整启动步骤、回放、并发上限和无缓存实测见 [本地采集 Dashboard 指南](../docs/COLLECTION_DASHBOARD.md)。

在 `training/` 目录启动 `.\.venv\Scripts\python.exe -m quoridor_lab.dashboard`，
打开 [本地面板](http://127.0.0.1:8770/) 后选择红蓝各自的 Barricade 难度和盘数。
“同时运行盘数”设置可推进的棋局数。各局轮流提交走法，共用一条连接和同一限速，
面板分别显示每盘的进度和回放。远端应手最多一个请求在途。
可指定固定合法开局；开局进入棋谱，后续远程应手进入逐步示范数据。
面板显示进度、终局、重复棋谱、重复局面、重复示范和缓存数。
点击“打开回放”可在 Quoridor Lab 的原有回放页查看棋谱。
启动网页服务的说明见 [组员使用指南](../docs/DELIVERY_GUIDE.md)；若网页端口不同，
启动面板时用 `--replay-base` 指定地址。

每批保存在 `data/collection/dashboard/<运行编号>/`。面板可下载批量棋谱与示范的
JSONL 文件，单盘也会保存成独立 JSON。远程请求失败时保留进度，可检查错误后恢复。
两边共用一条远程连接和同一请求间隔。相同难度与完整历史会复用已观测的应手。
勾选“每步重新请求”可关闭本批缓存，用于观察同一配置的新应手是否重复。
比较数据多样性时，面板分别统计完整走法序列、棋子和墙等局面信息，以及加上档位和走法后的示范。

也可以用命令行采集双远端对弈：

```sh
python -m quoridor_lab collect --red-difficulty expert --blue-difficulty hard --games 2 --output data/collection/expert-hard
```

### 与本地基线对战

```bash
.venv/bin/python -m quoridor_lab collect \
  --games 20 --difficulty expert --expert-side alternate \
  --policy shortest --max-plies 256 --interval 1 \
  --output data/collection
```

`--policy` 是与专家对抗的本地采集策略，可选 `shortest` 和 `random`。默认策略只沿最短路径走棋，不能代表一个强搜索 AI。更强的自定义策略可以直接接入 `QuoridorEnv.step()`；不同采集策略会产生不同分布的训练局面。

### 搜索型 AI（Minimax）

实现位于 `quoridor_lab/search.py`，对外类为 `SearchAgent`，与 `LocalPolicy` 一样提供 `choose(state) -> Decision`。

```bash
PYTHONPATH=training training/.venv/bin/python training/examples/serve_custom_agent.py \
  --port 8765 --data-dir ./handoff-local-data
```

浏览器将「搜索型 AI」指向 `http://127.0.0.1:8765/agents/search/move`。三档难度注册为 `search_easy`、`search_medium`、`search_hard`。

三档难度两两对战基线（默认每组 20 局并交换先后手）：

```bash
PYTHONPATH=training training/.venv/bin/python training/examples/search_baseline.py --games 20 --output data/search_baseline
```

输出 `baseline_<时间>.csv`、`*_steps.jsonl`（每步耗时）与 JSON 摘要。

- `--expert-side alternate` 交换专家先后手，也可固定 `1` 或 `2`。
- 所有官网调用串行执行，默认至少间隔 1 秒。同一输出目录一次运行一个采集进程。
- 设备 ID 保存在 `data/device-id.txt`，持续复用；不读取浏览器账号或轮换身份。
- 默认缓存已请求的“难度＋完整棋谱”。命中时 `cached=true`，该条是复用先前观测，不是一次新专家采样。研究随机应手分布时可指定 `--no-cache`。
- 已知异常 e2、非法缓存及带回退/往返信号的 e2 缓存不再直接复用：保留原缓存文件，重新请求远端。新采集的正常首手 e2 和前进 e2 不受此限制。
- 遇到限流、访问拒绝、非法专家响应或网络错误时停止当前采集，保留检查点，不把错误算成输赢，不换成本地 AI。
- 每个动作后落盘。中断后用同样命令加 `--resume` 恢复 `pending/` 中的棋局；恢复时沿用检查点的先后手、策略、随机数状态和最大步数。
- 达到步数上限属于 `truncated`，不是自然和棋。只有到达目标边线才标记 `completed` 和胜方。

输出文件：

```text
transitions.jsonl        每个动作一行，包含专家和本地对手
episodes.jsonl           每个完成/截断棋局一行
episode-<id>.json        可独立读取的完整棋谱与结果
pending/<id>.json        可恢复的中途棋局
last-error.json         最近一次中断原因（如有）
```

每条 transition 含 `episode_id`、`ply`、当前玩家、完整历史、观察张量、合法动作掩码、动作编号、记谱、`is_teacher`、来源、缓存标记、决策耗时和专家响应。`is_teacher=true` 表示采集器中被指定为教学方；确认它来自官网还需检查 `source` 以 `barricade:` 开头。胜负需按 `episode_id` 关联 `episodes.jsonl`，中途检查点没有胜负标签。

## 模仿专家走法

两组各 500 局的共享样本及 Master 应手筛选示例见 [Barricade 共享棋谱与使用规范](../docs/SHARED_BARRICADE_DATASETS.md)。

```bash
.venv/bin/python examples/train_imitation.py \
  --data data/collection/transitions.jsonl \
  --epochs 20 --output data/imitation.pt
```

示例使用 PyTorch 的小型策略网络，只训练官网专家动作，屏蔽非法动作，并按整盘棋划分训练与验证集，避免同一盘棋的相邻局面被拆到两边。只有一盘时不生成验证指标。`*.metrics.json` 保存数据划分、损失和动作准确率；动作准确率不等于对战胜率。大量重复开局仍可能跨棋局出现，正式实验还应做局面去重与独立对手评估。

训练示例默认使用 `--e2-policy verified-e2`：排除四种已匹配验证的故障前缀、造成完整盘面第三次重复的 e2，以及缺少原局面新会话复核证据的距离疑点 e2。`--e2-policy quarantine-signatures` 只排除前两类，`--e2-policy raw` 保留全量对照；三者输出各自的 `e2_quality` 计数，不删除或修改原始棋谱。缺少证据不等于已证明坏棋。正常首手、前进 e2 和仅因放墙产生的距离前进型往返均保留。完整盘面比较双方棋子、墙、余墙和行动方；墙变化后的绕路不算同盘面循环。

对没有缩短距离或存在更短棋子走法的 e2，在线适配器会另建会话，在相同档位下重新请求原始完整棋谱一次；若给出不同应手，使用第二次新鲜远端结果并保留第一次候选，若仍为 e2，则经过连接检查后记录两次一致的原局面回复。每次调用最多做一次原局面复核，额外请求遵守共享限速并纳入请求计数；检查请求本身不入棋谱。训练筛选从两条原始请求/回复核对历史、档位、不同会话编号和关联编号，不单凭一个“已确认”标签放行。该策略改变了可疑候选的采样分布，需用全量模式做对照，不能据此宣称棋力提升。

独立只读审计命令（输出目录必须尚不存在）：

```bash
.venv/bin/python -m quoridor_lab audit-e2 \
  --source data/collection/transitions.jsonl --output data/e2-audit
```

它产生 `report.json` 和 `annotations.jsonl`，记录读取范围、原始字节前缀 SHA-256、每个 e2 的分类及高风险前缀数量，支持采集中途的完整行快照。

如需供其他训练程序直接读取，可导出独立快照：

```bash
.venv/bin/python -m quoridor_lab export-teacher \
  --source data/collection/transitions.jsonl --output data/verified-teacher
```

输出 `teacher-transitions.jsonl`、排除原因清单和哈希清单。选中行保留原字节、历史、动作和来源，不重写棋谱，也不覆盖既有目录。默认使用 `verified-e2`，另可通过 `--e2-policy` 选择较宽松或全量对照策略。

在线防护在异常时保持设备身份、档位、原始棋谱和共享请求间隔：已知故障前缀的 e2 会先隔离并重连，旧 Polling/升级路径上的可疑 e2 自动切换为 WebSocket 首连重求解。疑似动作按原即时重试额度处理（至少一次恢复重试，最多两次）；持续异常以 `suspect_e2` 暂停并保留断点，不按弃局补开。非法 e2 也会重置连接；其后仍按原非法应手冷却规则处理。WebSocket 首连上的其他可疑 e2 会使用新鲜的蓝方首手局面检查连接，检查通过时保留候选和复核标记；检查样本不写入棋谱或缓存，真实请求数包含该额外请求。限流和拒绝立即停止，不触发换路规避。所有这些恢复都使用真实远端结果，不本地补棋。

同一请求通道或共享队列累计连续 6 次非法 e2 时触发 `e2_service_unhealthy` 并暂停，防止远端持续故障造成大量弃局。成功的新鲜棋局应手会重置对应连续计数；缓存命中和额外健康检查不重置。某个健康通道的成功不会抹去另一个通道的连续故障。

此前生成过 `data/imitation-smoke.pt` 作为链路样例；模型权重不随仓库或交付包分发，可使用上述命令重新生成。小样本仅供检查模型加载与训练流程。

## 强化学习环境

```python
from quoridor_lab.env import QuoridorEnv
from quoridor_lab.opponents import RemoteExpert

env = QuoridorEnv(opponent=RemoteExpert(), learner_player=1, max_plies=256)
try:
    obs, info = env.reset(seed=0)
    action = int(obs["action_mask"].nonzero()[0][0])  # 换成模型的 masked policy
    obs, reward, terminated, truncated, info = env.step(action)
finally:
    env.close()
```

一次 `step()` 包含学习方行动和专家回应。学习方到终点奖励 `+1`，专家到终点奖励 `-1`，其他情况为 `0`；没有额外路径奖励。学习方可以执红或执蓝，执蓝时 `reset()` 会先取得专家开局。`info.transitions` 保留实际发生的双方动作。

- 观察为 `board: float32[8,9,9]` 和 `action_mask: int8[209]`。
- 通道顺序：红棋、蓝棋、横向阻断、纵向阻断、红墙数/10、蓝墙数/10、当前玩家减一、已走步数/步数上限。
- 坐标固定，不随学习方旋转；学习方编号在 `info.learner_player` 中。
- 动作 `0..80` 为目标格（`row*9+col`）；`81..144` 为横墙（`81+row*8+col`）；`145..208` 为竖墙。
- 所有采样必须应用掩码。普通 PPO 不会自动读取这个字段；可自行屏蔽 logits，或接入支持 action masking 的实现。`env.action_masks()` 也可供 MaskablePPO 一类工具使用。
- 非法行动抛出异常，不会悄悄替换走法。专家调用失败会抛 `RemoteError`；学习方动作可能已落下，可查看状态并重置，不能直接把同一动作再次提交。

一个使用合法动作掩码的 REINFORCE 示例：

```bash
# 先用本地对手检查策略更新，不查询官网
.venv/bin/python examples/train_policy_gradient.py \
  --episodes 20 --warm-start data/imitation.pt --output data/reinforce.pt

# 需要真实专家作为环境对手时明确加入 --remote
.venv/bin/python examples/train_policy_gradient.py --remote --episodes 2
```

这是接口示范，不是经过调参的训练算法或已达到可用强度的模型。

## HTTP 游戏接口

- `GET /health`：状态、动作空间和查询计数。
- `POST /state`：输入 `{"moves":[...]}`，严格重放，返回状态和掩码。
- `POST /expert/move`：输入棋谱，取得官网应手。
- `POST /games`：输入 `{"opponent":"remote","learner_player":1}` 创建本地会话；也可选 `shortest`、`random`。
- `GET /games/{id}`：当前局面与观察张量。
- `POST /games/{id}/step`：提交 `{"notation":"e2"}` 或 `{"action_id":67}`，二选一；包含对手回应。
- `POST /games/{id}/reset`：重置会话。
- `DELETE /games/{id}`：丢弃内存中的会话。

HTTP 游戏会话在服务重启时清空；持久化训练数据由采集器负责。服务端专家应手的算法属于第三方服务，接口与难度行为未来可能变化。

## 检查

```bash
.venv/bin/python -m unittest discover -s tests -v
```

检查覆盖 209 个动作编解码、开局掩码、直跳/斜跳、接缝穿墙、重叠/交叉/封路、之前的 46 步专家实战、环境边界、缓存、限流错误、HTTP 行为及中断续采。实际远程对局与 HTTP 冒烟记录见 `data/live-smoke/` 和 `data/api-smoke.json`。

## 项目名称与旧版兼容

安装包名为 `quoridor-lab`，Python 入口为 `quoridor_lab`，Gymnasium 环境类为 `QuoridorEnv`，命令行提供 `quoridor-lab`。旧版 `barricade_lab` 导入、`BarricadeEnv` 和 `barricade-lab` 命令通过兼容入口调用同一实现。
