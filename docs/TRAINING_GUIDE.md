# 本地深度学习与强化学习操作流程

## 目标

采用监督式模仿学习，让官网专家为局面提供动作标签，在本地训练神经网络近似其策略。随后可以用本地对手、自我对弈或官网专家作为环境对手继续研究强化学习。

专家提供的是动作，不提供梯度、内部权重或价值网络。训练过程发生在本地；采集完的数据可以反复用于多轮训练，不必每个 batch 访问官网。

## 1. 安装并检查

在仓库根目录执行：

```bash
cd training
python3.11 -m venv .venv
.venv/bin/python -m pip install '.[test,learning]'
.venv/bin/python -m unittest discover -s tests -v
```

不运行神经网络时，只需安装 `.[test]`。Python 版本至少 3.11。

## 2. 用已有样本跑通

```bash
.venv/bin/python examples/train_imitation.py \
  --data data/live-smoke/transitions.jsonl \
  --epochs 2 --output data/pilot.pt
```

输出模型 `data/pilot.pt` 和记录 `data/pilot.metrics.json`。仓库示例只有 26 条官网专家标签，这一步验证数据读取、网络更新和模型保存，不验证棋力。

## 3. 采集新数据

```bash
.venv/bin/python -m barricade_lab collect \
  --games 20 --difficulty expert --expert-side alternate \
  --policy shortest --seed 0 --interval 1 \
  --output data/expert-learning

.venv/bin/python -m barricade_lab collect \
  --games 20 --difficulty expert --expert-side alternate \
  --policy random --seed 100 --interval 1 \
  --output data/expert-learning
```

两种策略的数据追加到同一目录。20+20 盘只是启动实验规模，不是达到专家水平的保证。

检查输出中的 `finished_games`、错误字段及每盘 `status`。遇到限流 / 连接错误时停止；在原命令增加 `--resume` 恢复已有 pending 棋局。默认缓存会复用同一棋谱此前观察的应手，独立网络调用数不等于样本行数，研究采样分布时需区分。

当前采集器只有 `shortest` 与 `random` 两个本地策略入口。最短路策略不主动放墙，会使局面分布偏窄；后续应接入搜索对手或已训练模型，而不是只增加同类对局数量。

## 4. 检查数据与划分

一条专家训练样本是：

```text
局面张量 + 合法动作掩码 → 专家动作编号
```

训练脚本只读取 `is_teacher=true` 且来源以 `barricade:` 开头的记录，并验证标签落在合法动作集合内。用户方、本地策略方的走法不作为专家标签。

现有脚本按整盘棋分组，约 80% 训练、20% 验证；只有一盘时不输出验证准确率。正式实验还应：

- 对规范化局面去重或记录重复权重。
- 保留独立测试棋局 / 局面，不能用于调参或选择模型。
- 记录采集策略、难度、时间、先后手和缓存情况。
- 分别统计移动、横墙、竖墙以及开局 / 中局 / 残局样本。
- 检查中途样本与已完成对局的区别；没有胜负标签时不能伪造结果。

上述正式实验增强并非已全部自动实现。

## 5. 训练第一版

```bash
.venv/bin/python examples/train_imitation.py \
  --data data/expert-learning/transitions.jsonl \
  --epochs 20 --batch-size 64 --seed 0 \
  --output data/expert-policy-v1.pt
```

现有网络为 `Flatten → Linear(648,256) → ReLU → Linear(256,209)`，用交叉熵和 Adam。非法动作 logits 被屏蔽。它是容易检查的全连接基线；后续可以研究 CNN / 残差网络，但先固定数据划分与评估方法。

模型文件保存权重、观察维度、动作数和 seed；指标文件保存划分、每轮损失和验证动作准确率。当前脚本保存最后一轮，未实现自动最佳模型选择、早停或训练恢复。

## 6. 加载并离线决策

在 `training/` 目录执行：

```bash
.venv/bin/python - <<'PY'
import torch
from examples.train_imitation import network
from barricade_lab.rules import State, Action

model = network()
checkpoint = torch.load('data/expert-policy-v1.pt', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
obs = State.from_moves(['e2', 'e8']).observation()
with torch.no_grad():
    scores = model(torch.from_numpy(obs['board']).unsqueeze(0))[0]
    legal = torch.from_numpy(obs['action_mask']).bool()
    action = Action.from_id(int(scores.masked_fill(~legal, float('-inf')).argmax()))
print(action.notation)
PY
```

这一段不访问官网。它可作为网页人机接口或评估脚本中的模型决策函数。

## 7. 评价模型

先观察训练损失和独立验证动作准确率，再检查完整对局。动作一致率不等于胜率：同一局面可能有多个合理走法，少量关键错误也可能决定整盘胜负。

建议保留固定对手和规则，交换先后手，分别报告：

- 自然胜 / 负局数与完成局胜率。
- 截断局数，不能混入自然和棋。
- 平均对局长度、模型决策时间、放墙次数。
- 反复走动、自堵、过早耗墙等失败类型。
- 模型版本、数据版本、随机种子和是否使用缓存。

**当前没有封装好的模型评估 CLI。** `collect --policy shortest/random` 不能冒充模型评估。需要将第 6 步的模型决策接到 `BarricadeEnv.step()`，保存完整对局与结果；后续开发优先级见 [验收清单](VALIDATION.md)。

## 8. 根据模型错误补标

初期：采集基线对手与专家对局。模型可用后：让模型产生自己的合法局面，向专家请求这些局面的标签，再加入训练。这是减少训练局面和模型实际遇到局面不一致的路线，可按 DAgger 思路研究。

目前专家查询接口可以支持这一步，但完整的模型驱动采集循环与数据去重尚未实现。

## 9. 可选强化学习

```bash
.venv/bin/python examples/train_policy_gradient.py \
  --episodes 20 --warm-start data/expert-policy-v1.pt \
  --output data/reinforce.pt
```

这条命令用本地最短路对手，不调用官网。需要用官网专家作为环境对手时明确加入 `--remote`。

示例为带合法动作掩码的 REINFORCE，终局奖励 +1 / -1，截断为 0。它提供接口示范，没有实现 PPO、自我对弈对手池或经调参的高强度模型。普通 PPO 不会自动应用观察中的 action_mask，后续必须接入支持屏蔽的实现或自行处理 logits。

## 实验报告应区分

1. **采集链路验证**：官网能返回合法应手，整盘能完成，记录可重放。
2. **训练链路验证**：参数会更新，模型能保存、加载并输出合法动作。
3. **策略学习效果**：独立局面预测与完整对战表现有所改善。
4. **专家水平**：需要充分对局、稳定评估和统计支持，不能由前两项推出。
