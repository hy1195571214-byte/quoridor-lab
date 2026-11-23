# 专家棋谱行为克隆预训练与强化学习微调

## 目标与当前路线

更新日期为 2026-11-23。先整理专家棋谱，以监督模仿学习中的行为克隆预训练策略网络，完成独立对战评估，再继承策略权重做强化学习微调。主要流程参考 P8 AlphaGo（2016），PPO 的具体更新依据 P2、P3。[实施指南](../references/EXPERIMENT_METHODS.md) · [P8 研读笔记](../references/summaries/08_ALPHAGO_2016.md)

当前默认训练语料为 `round2-cpp208k-v1`：保留上一版冻结的 2,500 盘，新增逐盘重放检查后的 QuoridorCPP PGN 208,359 盘，合计 210,859 盘、8,199,006 手。新增来源移除了 4 份完整走法重复记录，隔离了 256 份尚未到达终点的记录。Master 对 Master 与 Master 对 Expert 两个已知批次继续排除。PGN 的玩家身份和档位未知。[合并数据、划分与启动说明](COMBINED_TRAINING_DATA.md) · [训练数据选择清单](../training/data/datasets/training-selection.json)

第一轮的 2,500 盘清单及原有划分保留不变。新版本分为 168,730 盘训练、20,848 盘验证和 21,281 盘测试，完整走法及其左右镜像没有跨分区；跨分区重复局面仍待专门统计。当前训练入口默认读取 `active-dataset.json`，大语料采用分批生成样本，不一次装入全部浮点数组。本次数据导入没有启动正式训练。原基线的网络、超参数、模型恢复和评估预算见 [第一轮训练方案](ROUND1_TRAINING_PLAN.md)。

专家行动用于监督学习，采集完的示范可在预训练中多轮使用。后续 PPO 微调根据当前策略新产生的经历更新。双方合计 256 次单方行动的截断上限沿用已确定设置。

## 实时训练看板与棋谱回放

当前入口 `python -m quoridor_lab.supervised` 默认读取已选中的冻结清单，支持 MLP/CNN32、合法动作掩码、AdamW、学习率调度、早停、完整检查点恢复和大语料流式读取。`examples/train_monitored.py` 为包装入口。下方各节的 `train_imitation.py` 为原有逐步数据示例，两种入口分别保留。新语料的本机启动命令见[合并数据说明](COMBINED_TRAINING_DATA.md)。

在 `training/` 目录启动本地 API。

```bash
.venv/bin/python -m quoridor_lab serve --training-dir data/training-runs
```

另一个终端在 `frontend/` 目录执行 `npm run dev -- --port 4173`，访问 <http://127.0.0.1:4173/training>。训练入口将进度自动写入同一目录。

```bash
# 仅验证联通。每个分区最多 64 个逐步样本，页面标记为试运行。
.venv/bin/python examples/train_monitored.py \
  --run-dir data/training-runs/cnn32-smoke \
  --architecture cnn32 --epochs 2 --sample-limit 64

# 正式训练命令，启动前完成方案中的数据审计和评价准备。
.venv/bin/python examples/train_monitored.py \
  --run-dir data/training-runs/cnn32-seed0 \
  --architecture cnn32 --seed 0 --epochs 40 --batch-size 256
```

训练样本按批次抽样，看板跟随最近报告的样本及其落子前局面。关闭“跟随训练样本”即可手动选择训练或验证棋谱、拖动步数、播放原谱，并查看下一手 Top-3 概率。预测读取最近一次完成保存的 epoch 检查点，界面注明它的轮次；同一检查点交替预测双方共最多 8 手，用“播放模型预测”查看。它表示策略网络的贪心续走，不能解释为搜索主变化或胜率。

“在完整回放中打开预测分支”将整盘原谱保留为主线，将预测附为变化分支，并停在当前展示的步数。预测响应核对检查点版本和局面，跨版本结果不会显示在当前棋盘。看板只提供训练和验证棋谱，保留测试分区不通过这些接口展示。

每次运行产生 `status.json`、`latest.pt` 和 `best.pt`。同一运行目录只允许一个训练写入者；恢复时使用相同架构、种子及样本选择，附加 `--resume`。`--warm-start path/to/best.pt` 在新目录继承权重并建立新的优化器。当前已实现验证 NLL、Top-1/Top-3 和棋子/墙预测准确率，方案中的完整分项指标、跨分区观察审计与固定对手实战评估仍待补齐。小样本联通结果不作为正式棋力成绩。

2026-11-24 起，`latest.pt` 在首批结束、之后默认每 100 批、验证开始前及异常退出时的安全批次边界保存，包含批次位置和完整训练状态。看板只在已有模型的失败/停止运行中显示“继续训练”，点击后才恢复。`best.pt` 仍依据完成验证后的最低损失选择。没有旧检查点的运行不能从日志批次数重建模型权重。恢复实现和版本兼容说明见[合并数据说明](COMBINED_TRAINING_DATA.md)。

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

现有脚本筛选 `is_teacher=true` 且 `source` 以 `barricade:` 开头的记录。这个通用条件也能匹配 Master 行动，因此构建正式输入时须按训练数据选择清单选取来源，核对逐步 `metadata.difficulty`、`player` 和整局 `config.remote_sides`。公开 PGN 的原始来源标签须随格式转换保留，教师导入适配仍待实现。

首版比较 Expert 对 Expert 与公开 PGN 的独立训练结果，再比较组合效果。训练输入使用按完整走法去重的棋谱，记录来源及双方胜负分布。训练选择清单记录当前决策，导入代码需显式执行该清单的来源筛选。

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
