# Reference 与实验阅读资料

本目录收录 Quoridor 强化学习实验的论文原文、无公式中文概括，以及训练、测试和优化方法。整理日期为 2026-11-13，项目状态依据提交 `49a23f8` 的代码。

## 从哪里开始阅读

先阅读基础概念，理解策略、回报、价值和策略梯度。接着阅读 REINFORCE、PPO 与行动屏蔽的概括，再根据实验流程安排训练和测试。

- [基础概念与 PPO 通俗解释](CONCEPTS.md)
- [实验方法、测试与优化流程](EXPERIMENT_METHODS.md)
- [Agent 工作规范与首轮任务](AGENT_WORKFLOW.md)

中文说明使用文字和游戏例子讲解，数学推导可在英文原文中查阅。各篇概括包含论文研究问题、方法、作者发现、项目用途和原文阅读位置。

## 论文目录

| 编号 | 论文 | 阅读用途 | 中文概括 | 原文 |
|---|---|---|---|---|
| P1 | Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning，Williams，1992 | 理解 REINFORCE 与当前策略梯度示例 | [通俗概括](summaries/01_REINFORCE.md) | [PDF](papers/01_REINFORCE_1992.pdf) |
| P2 | Proximal Policy Optimization Algorithms，Schulman 等，2017 | 理解 PPO 如何改进基础策略梯度更新 | [通俗概括](summaries/02_PPO.md) | [PDF](papers/02_PPO_2017.pdf) |
| P3 | A Closer Look at Invalid Action Masking in Policy Gradient Algorithms，Huang、Ontañón，2020 起 | 理解合法行动屏蔽 | [通俗概括](summaries/03_ACTION_MASKING.md) | [PDF](papers/03_ACTION_MASKING.pdf) |
| P4 | Empirical Design in Reinforcement Learning，Patterson 等，2024 | 设计训练、比较和最终测试 | [通俗概括](summaries/04_EMPIRICAL_DESIGN.md) | [PDF](papers/04_EMPIRICAL_DESIGN_2024.pdf) |
| P5 | Curriculum Learning for Reinforcement Learning Domains — A Framework and Survey，Narvekar 等，2020 | 安排任务难度、训练对手和课程顺序 | [通俗概括](summaries/05_CURRICULUM_LEARNING.md) | [PDF](papers/05_CURRICULUM_LEARNING_2020.pdf) |
| P6 | Deep Reinforcement Learning that Matters，Henderson 等，AAAI 2018 | 理解训练随机性、结果波动与复现 | [通俗概括](summaries/06_RELIABLE_EXPERIMENTS.md) | [PDF](papers/06_DEEP_RL_THAT_MATTERS.pdf) |
| P7 | Implementation Matters in Deep Policy Gradients — A Case Study on PPO and TRPO，Engstrom 等，ICLR 2020 | 分析网络、优化器等实现细节的影响 | [通俗概括](summaries/07_IMPLEMENTATION_MATTERS.md) | [PDF](papers/07_IMPLEMENTATION_MATTERS_2020.pdf) |

P3 的本地原文为 2022 年的 arXiv v3，P6 的本地原文为 2019 年的 arXiv v3。完整作者、出处、版本和文件校验信息见文献与来源记录。

- [BibTeX 引用条目](references.bib)
- [原文来源、页数与 SHA-256](sources.json)

## 论文如何用于当前项目

| 内容 | 当前依据 | 在实验中的角色 |
|---|---|---|
| 合法行动与胜负判定 | [Python 规则](../training/quoridor_lab/rules.py) | 所有对手、训练和评估共用的规则基础 |
| 观察、行动与奖励 | [Gymnasium 环境](../training/quoridor_lab/env.py) | 训练程序与游戏交互的接口 |
| masked REINFORCE | [现有训练示例](../training/examples/train_policy_gradient.py) | 已有的策略梯度基线，对应 P1、P3 |
| 专家行为学习 | [模仿学习示例](../training/examples/train_imitation.py) | 用已记录的专家动作训练策略，可作为初始化实验 |
| PPO / MaskablePPO | P2、P3 与选定框架版本文档 | 后续算法候选，需完成适配与评估 |
| 课程学习与历史对手 | P5 | 后续训练分布设计候选 |
| 实验比较与可靠性 | P4、P6、P7 | 约束预算、种子、日志、模型选择和结论 |
| 网页模型接入 | [AI 接口说明](../docs/AI_INTERFACE.md) | 加载策略后提供合法行动，连接试玩与回放 |

当前网页中的 `dl`、`rl`、`search` 服务示例使用最短路占位策略。已有的训练示例、后续算法方案和实际实验结果，在实验记录中各自写明状态。

## 基础读物与实现文档

强化学习中的环境、行动、策略、回报和价值，可进一步阅读 Sutton 与 Barto 的教材《Reinforcement Learning: An Introduction》第二版第 3 章。策略梯度可结合第 13 章阅读。实现时对照实际安装版本核对框架接口。

- [教材作者网站](http://incompleteideas.net/book/the-book-2nd.html)
- [Stable-Baselines3 自定义环境文档](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html)
- [SB3-Contrib MaskablePPO 文档](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
- [现有训练操作指南](../docs/TRAINING_GUIDE.md)

## 如何在项目报告中引用

算法依据引用原作者和论文。项目自己的奖励、棋盘编码、对手组合和测试结果写入方法与实验章节，并注明对应代码与运行记录。这里的中文概括属于学习笔记，项目建议需要通过本地实验检验。
