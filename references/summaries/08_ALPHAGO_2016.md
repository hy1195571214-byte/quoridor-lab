# AlphaGo（2016）研读笔记

**本项目的训练决策**　以这篇论文为主要参考，先用示范动作对策略网络做监督式模仿学习，再把预训练权重交给强化学习，用实际对局的胜负优化策略。

[Nature 正式发表版](https://doi.org/10.1038/nature16961) · [DeepMind 官方 PDF](https://storage.googleapis.com/deepmind-media/alphago/AlphaGoNaturePaper.pdf) · [BibTeX 条目](../references.bib)

## 原文怎样训练

### 1. 用人类走法预训练策略网络

每个样本是一副棋盘和高手实际选择的一步。网络输出下一步的概率分布，训练提高示范行动的概率。Methods 记载数据来自 KGS 六至九段玩家的约 16 万盘棋，共 2,940 万个局面。网络在留出的数据上预测实际走法，完整特征的准确率为 57.0%。图 2a 还通过实际对战测量策略网络对棋力的影响。

**阅读位置**　Nature 第 485 页、图 1 和图 2a。实现细节见 Methods 的 *Policy network: classification*。

### 2. 继承策略权重，用自我对弈做 REINFORCE

强化学习策略和监督策略结构相同，直接用监督策略权重初始化。训练中的当前策略与从池中抽出的历史版本对局，每 500 次迭代收录一次当前版本。比赛以终局胜负提供学习信号，获胜为 +1，失败为 −1，中间回报为零。REINFORCE 将行动的对数概率梯度按终局回报与基线的差加权。第一遍用零基线，第二遍才引入价值网络作为基线。论文报告，强化后的策略与监督策略直接对弈，胜率超过 80%。

**阅读位置**　Nature 第 485 页 *Reinforcement learning of policy networks*，Methods 的 *Policy network: reinforcement learning*。

### 3. 用强化后策略产生的对局训练价值网络

价值网络学习预测给定局面下当前行动方的最终结果。作者观察到同盘相邻局面高度相关，训练容易记住整盘结果，因此新的训练数据每盘只取一个局面，并以实际终局胜负作为标签。Methods 还说明如何在生成棋谱时加入随机合法动作，增加局面多样性。

**阅读位置**　Nature 第 486 页 *Reinforcement learning of value networks*，Methods 的 *Value network: regression*。

### 4. 实际下棋时用网络引导树搜索

完整系统用监督策略网络为合法候选走法提供搜索先验。到达搜索叶节点时，结合价值网络估计和快速模拟所得结果，再把信息回传给路径上的节点。最终选择根节点下访问次数最多的走法。论文分别测量了策略网络单独对弈和完整搜索系统的棋力。

**阅读位置**　Nature 第 486–487 页、图 3 和图 4。

## Quoridor 怎样借鉴

1. **先模仿示范。** 从远程对手采集局面、合法行动掩码和应手。用已有的 `train_imitation.py` 拟合这些动作，按整盘棋拆分训练与验证集。统计先后手、放墙与走棋比例，记录样本的档位和来源。
2. **再用胜负更新策略。** 用已有的 `train_policy_gradient.py --warm-start` 载入同一结构的策略权重。在本地环境按合法掩码采样动作，胜 +1、负 −1，双方合计 256 次行动后仍无赢家则记录为截断。首轮可使用固定本地对手，历史版本对手池和价值网络可逐项评估。
3. **检验预训练和课程顺序的收益。** 冻结预训练模型和各阶段强化模型，用相同对手与开局评估并交换先后手，保存胜负、截断和完整棋谱。比较随机初始化与模仿初始化，衡量预训练收益。若检验 `easy → medium → hard → expert → master`，再比较同一批示范的递增顺序与混合顺序，匹配各档样本使用量和更新预算。AlphaGo 的历史版本抽样给自我对弈提供了先例。五档示范排序是本项目自己的实验假设。

[模仿学习实现](../../training/examples/train_imitation.py) · [强化学习实现](../../training/examples/train_policy_gradient.py) · [Quoridor 实验方法](../EXPERIMENT_METHODS.md)
