# Quoridor / Barricade Lab

步步为营（Quoridor）项目的参考站点拆解、复刻规范，以及可连接 Barricade 官网专家的本地训练接口。

当前可运行的部分是 **Python 规则引擎、官网专家适配器、批量数据采集、Gymnasium 环境、HTTP API 和 PyTorch 训练示例**。网页前端复刻处于规范阶段，尚未实现。

## 文档入口

- [网站定位、入口与功能地图](docs/WEBSITE_BREAKDOWN.md)
- [本地对战、回放、变化分支和导出的交互规范](docs/INTERACTION_SPEC.md)
- [规则、数据格式、模块架构和接口契约](docs/ARCHITECTURE.md)
- [深度学习操作流程与实验规范](docs/TRAINING_GUIDE.md)
- [实测结果、证据等级与功能验收清单](docs/VALIDATION.md)
- [课程项目初始计划](docs/PROJECT_PLAN.md)
- [训练接口完整使用说明](training/README.md)
- [公开前端脚本来源索引](docs/evidence/source-index.json)

## 快速开始

在仓库根目录执行：

```bash
cd training
python3.11 -m venv .venv
.venv/bin/python -m pip install '.[test]'
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m barricade_lab serve --port 8765
```

启动后打开 [本机 API 文档](http://127.0.0.1:8765/docs)。服务只监听 `127.0.0.1`。

请求专家的下一步：

```bash
curl http://127.0.0.1:8765/expert/move \
  -H 'Content-Type: application/json' \
  -d '{"moves":["e2"]}'
```

也可直接使用 Python：

```python
from barricade_lab.rules import State
from barricade_lab.opponents import RemoteExpert

expert = RemoteExpert(data_dir="data")
try:
    decision = expert.choose(State.from_moves(["e2"]))
    print(decision.to_dict())
finally:
    expert.close()
```

## 已完成的验证

截至 2026-11-05：

- 11 项离线检查通过，覆盖规则边界、动作编码、环境、缓存、限流错误、HTTP 与中断续采。
- 通过网页真实完成一盘专家对战：红方先手，23 回合 / 46 个动作后专家蓝方到达 i1 获胜；全部动作已独立验证合法。
- 通过适配器自动完成两盘官网专家对局，交换先后手，共 51 个动作、26 次实际远程专家决策。
- HTTP 专家查询与完整回合检查通过；该项 HTTP 冒烟复用了已观察的专家缓存。
- 模仿学习和策略梯度示例均跑通；这只证明训练链路可用，不代表已达到专家水平。

真实的小样本数据位于 [training/data/live-smoke](training/data/live-smoke)。模型文件可用脚本重新生成，未随仓库分发。

## 项目结构

```text
docs/                         拆解、规范、架构、训练流程与验收
  evidence/                   来源清单、UI / 源码观察和实战记录
training/
  barricade_lab/              规则、专家适配器、采集、Gym 环境和 HTTP API
  examples/                  模仿学习与带动作掩码的 REINFORCE 示例
  tests/                     无需访问官网的自动检查
  data/live-smoke/            两盘真实专家对局与逐步训练样本
.github/workflows/tests.yml   离线 CI；不调用官网专家
```

## 研究边界

本仓库独立编写本地实现，并依据 [Barricade](https://barricade.gg) 的公开网页、浏览器可获取脚本和交互实测分析其行为；不是该网站的官方项目。第三方完整脚本、浏览器账号、设备标识、运行缓存及本机虚拟环境均未打包。

专家适配器使用官网前端已使用的第三方服务，接口未来可能变化。访问拒绝或限流会停止采集并保留进度，不会切换身份或用本地策略冒充专家应手。内部专家权重和算法不可见；本项目通过应手样本做行为学习。

网站全部主要入口已编目，但登录、支付、真人匹配、四人实战等尚未全部实测，详见各文档的证据等级。
