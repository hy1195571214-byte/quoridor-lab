# Barricade 本地专家采集与训练接口

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
.venv/bin/python -m barricade_lab serve --port 8765
```

接口只监听本机 `127.0.0.1`。打开 [交互式 API 文档](http://127.0.0.1:8765/docs)，可直接查看和调用接口。若已有服务占用 8765，可使用 `--port 8766`。

新环境安装：

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install .
# 运行神经网络示例时再安装：
.venv/bin/python -m pip install '.[learning]'
```

原始验证环境复用了已安装的 NumPy、Gymnasium、FastAPI 和 PyTorch；在其他机器上请按上面的安装命令准备依赖。源码修改后，项目目录内的 `python -m barricade_lab` 会直接使用新源码；从其他目录运行安装包时，请重新 `pip install .` 更新安装副本。

## 请求一次官网专家决策

命令行：

```bash
.venv/bin/python -m barricade_lab probe --moves e2
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
from barricade_lab.rules import State
from barricade_lab.opponents import RemoteExpert

teacher = RemoteExpert(difficulty="expert", data_dir="data")
try:
    state = State.from_moves(["e2"])
    decision = teacher.choose(state)
    print(decision.to_dict())
finally:
    teacher.close()
```

默认使用 Socket.IO 的 polling 传输，不需要 WebSocket 浏览器。原始验证环境的代理曾使 Python 连接重置，因此本适配器默认直连；需要使用环境代理时加入 `--proxy-env`，或在 Python 中设 `proxy_env=True`。不修改系统网络设置。

## 自动批量对战采集

```bash
.venv/bin/python -m barricade_lab collect \
  --games 20 --difficulty expert --expert-side alternate \
  --policy shortest --max-plies 256 --interval 1 \
  --output data/collection
```

`--policy` 是与专家对抗的本地采集策略，可选 `shortest` 和 `random`。默认策略只沿最短路径走棋，不能代表一个强搜索 AI。更强的自定义策略可以直接接入 `BarricadeEnv.step()`；不同采集策略会产生不同分布的训练局面。

- `--expert-side alternate` 交换专家先后手，也可固定 `1` 或 `2`。
- 所有官网调用串行执行，默认至少间隔 1 秒。同一输出目录一次运行一个采集进程。
- 设备 ID 保存在 `data/device-id.txt`，持续复用；不读取浏览器账号或轮换身份。
- 默认缓存已请求的“难度＋完整棋谱”。命中时 `cached=true`，该条是复用先前观测，不是一次新专家采样。研究随机应手分布时可指定 `--no-cache`。
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

```bash
.venv/bin/python examples/train_imitation.py \
  --data data/collection/transitions.jsonl \
  --epochs 20 --output data/imitation.pt
```

示例使用 PyTorch 的小型策略网络，只训练官网专家动作，屏蔽非法动作，并按整盘棋划分训练与验证集，避免同一盘棋的相邻局面被拆到两边。只有一盘时不生成验证指标。`*.metrics.json` 保存数据划分、损失和动作准确率；动作准确率不等于对战胜率。大量重复开局仍可能跨棋局出现，正式实验还应做局面去重与独立对手评估。

已生成的链路样例：`data/imitation-smoke.pt`。其数据很少，仅供检查模型加载与训练流程。

## 强化学习环境

```python
from barricade_lab.env import BarricadeEnv
from barricade_lab.opponents import RemoteExpert

env = BarricadeEnv(opponent=RemoteExpert(), learner_player=1, max_plies=256)
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
