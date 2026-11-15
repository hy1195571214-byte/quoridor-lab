# 专家棋谱行为克隆预训练与强化学习微调

## 目标与当前路线

更新日期为 2026-11-15。先整理专家棋谱，以监督模仿学习中的行为克隆预训练策略网络，完成独立对战评估，再继承策略权重做强化学习微调。主要流程参考 P8 AlphaGo（2016），PPO 的具体更新依据 P2、P3。[实施指南](../references/EXPERIMENT_METHODS.md) · [P8 研读笔记](../references/summaries/08_ALPHAGO_2016.md)

用户报告两台电脑正在采集 master 对 master、master 对 expert。数据导入后记录实际规模、难度分布、重复和质量，历史 26 条专家动作样例继续用于流程检查。首版建议使用 master 行动作为主要示范，expert 行动独立保留。

专家行动用于监督学习，采集完的示范可在预训练中多轮使用。后续 PPO 微调根据当前策略新产生的经历更新。双方合计 256 次单方行动的截断上限沿用已确定设置。

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

## 3. 从现有采集任务接收数据

最新 Dashboard 支持指定红蓝双方难度，并保存逐步示范和完整棋谱。启动、下载、回放及采集配置以最新专门文档为准。[采集 Dashboard](COLLECTION_DASHBOARD.md) · [提速与质量记录](COLLECTION_SPEED_TUNING.md)

通过已完成批次导出或一致性快照取得 `episodes.jsonl`、`transitions.jsonl`、运行配置和相关错误日志。每台电脑使用独立来源标识，保留原批次、原棋局 ID、时间、代码版本、缓存和文件校验值。合并时登记全局标识与原始文件对应关系。

先接收一小批检查数据格式与标签，数据采集可以继续进行。自然完局、步数截断、时间截断、待恢复和错误分别统计。采集配置和采样速度的实测仅按原记录解释，正式模型训练另记数据版本和预算。

## 4. 数据审计、教师选择与固定划分

一条监督示范包含当前局面、合法行动集合和该行动方实际选择的行动。联合逐步与整盘文件，按棋局 ID、步序和公共规则重放，检查观察、行动、掩码和结果。

现有脚本筛选 `is_teacher=true` 且 `source` 以 `barricade:` 开头的记录。双远端采集会把双方均标为示范，master 对 expert 的双方都会通过当前筛选。正式导入需要根据逐步 `metadata.difficulty` 和来源选择难度，核对 `player` 及棋局的 `config.remote_sides`。

首版 master 主集包含 master 对 master 双方，以及 master 对 expert 中的 master 方。expert 行动保留独立标签用于后续混合示范实验。记录胜方与负方示范的数量，按规则校验与来源质量制定选择策略。

采集说明记录了异常 `e2` 应手，其中恰好合法的回复可能进入数据。将拒绝、重试和暂缓日志与批次一起保存，抽查采样窗口和代表棋谱，登记质量限制。

正式数据需保存训练、开发验证和最终测试划分。按完整棋局及重复棋谱分组，报告跨集合重复局面与未见局面子集表现。未完局的已验证动作可用于行为克隆，胜负标签仅关联具有自然结果的记录。

现有脚本会自行按整盘约 80% 训练、20% 验证，保存实际棋局 ID。正式数据适配应使实际划分与冻结清单一致，并将保留测试留在训练输入之外。数据筛选、固定三部分划分和独立测试按后续开发结果登记。[详细数据流程](../references/EXPERIMENT_METHODS.md)

## 5. 训练行为克隆基线

以下命令用于已经导出并完成教师筛选的 `data/expert-learning/transitions.jsonl`。数据导入适配与冻结划分按上一节检查，命令参数对应现有示例。

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
from quoridor_lab.rules import State, Action

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

## 7. 固定行为克隆模型并评价

先观察训练损失和独立验证动作准确率，再检查完整对局。动作一致率不等于胜率：同一局面可能有多个合理走法，少量关键错误也可能决定整盘胜负。

建议保留固定对手和规则，交换先后手，分别报告：

- 自然胜 / 负局数与完成局胜率。
- 截断局数，不能混入自然和棋。
- 平均对局长度、模型决策时间、放墙次数。
- 反复走动、自堵、过早耗墙等失败类型。
- 模型版本、数据版本、随机种子和是否使用缓存。

**当前没有封装好的模型评估 CLI。** `collect --policy shortest/random` 不能冒充模型评估。需要将第 6 步的模型决策接到 `QuoridorEnv.step()`，保存完整对局与结果；后续开发优先级见 [验收清单](VALIDATION.md)。

## 8. 根据模型错误补充示范

当前先使用已采集的双远端棋谱建立基线。模型可用后，可从失败棋谱提取代表局面，向专家请求相应标签，再建立新的数据版本。按 DAgger 思路开展模型驱动补标时，记录采样策略与专家来源，保留原始测试集合。

完整模型驱动补标循环和数据版本管理按实际开发状态登记。

## 9. 继承预训练策略做强化学习微调

先保存行为克隆模型和固定评估结果。以下命令使用同结构网络，并通过现有 `--warm-start` 加载监督预训练权重。

```bash
.venv/bin/python examples/train_policy_gradient.py \
  --episodes 20 --max-plies 256 \
  --warm-start data/expert-policy-v1.pt --output data/reinforce.pt
```

这条命令用本地最短路对手，不调用官网。需要用官网专家作为环境对手时明确加入 `--remote`。

示例为带合法动作掩码的 REINFORCE，终局奖励 +1 / -1，截断为 0。它提供接口示范，没有实现 PPO、自我对弈对手池或经调参的高强度模型。普通 PPO 不会自动应用观察中的 action_mask，后续必须接入支持屏蔽的实现或自行处理 logits。

## 10. 监控、检查点和后续实验

模仿示例每轮输出训练与验证指标，REINFORCE 每盘输出奖励、胜方、截断和长度。两者在运行结束时保存模型及指标文件。采集 Dashboard 已支持采集进度与回放，训练侧持续日志、TensorBoard、定期模型评估及完整中断恢复属于后续接入工作。

行为克隆与强化学习分别保存配置、数据版本、起始权重、模型和日志。评估使用一致对手与先后手，冻结的行为克隆模型持续作为微调对照。奖励组合从共同预训练快照进行比较，五档示范顺序另设课程实验，均按约定预算执行。

## 实验报告应区分

1. **采集链路验证**：官网能返回合法应手，整盘能完成，记录可重放。
2. **训练链路验证**：参数会更新，模型能保存、加载并输出合法动作。
3. **策略学习效果**：独立局面预测与完整对战表现有所改善。
4. **专家水平**：需要充分对局、稳定评估和统计支持，不能由前两项推出。
