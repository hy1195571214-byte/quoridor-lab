# Quoridor / Barricade Lab

步步为营（Quoridor）项目的本地对战与回放网页、站点拆解，以及深度学习、强化学习与搜索策略接口。

当前可运行 **React 网页、浏览器 / Python 规则、策略接口和 Gymnasium 环境**。网页包含本地双人、基线人机、回放、变化、原站压缩链接、完整 JSON 棋谱和浏览器视频导出。此前训练示例保留，当前开发阶段聚焦平台与接口。使用及交付状态请看 [组员使用指南](docs/DELIVERY_GUIDE.md) 和 [交付验收矩阵](docs/DELIVERY_AUDIT.md)。

## 网页启动

不修改程序时，可直接打开 [离线 HTML](frontend/offline/index.html)。传给组员请使用 [发布包](https://github.com/hy1195571214-byte/quoridor-barricade-lab/releases/latest) 中的 ZIP，解压后双击 `offline/index.html`，无需安装开发环境。

```bash
cd frontend
npm ci
npm run dev -- --port 4173 --strictPort
```

打开 [本地双人](http://127.0.0.1:4173/local)、[电脑](http://127.0.0.1:4173/computer) 或 [回放](http://127.0.0.1:4173/analysis?moves=KYJgNMAcYBbALEA)。网页规则和最短路 / 随机基线直接在浏览器运行。开发服务将 `/api` 转发到 `127.0.0.1:8765`，供模型服务使用。

- `npm run build` 生成网页构建及 `frontend/offline/index.html` 单文件；`npm run build:sites` 额外准备站点构建。
- `npm test` 检查棋谱、链接、策略与过期响应。
- 本地对局在浏览器内自动记录，可在记录库恢复；刷新会开启新局。通过“保存棋谱”可导出含终局原因和变化的 JSON。浏览器记录与后端 SQLite 记录库独立，不会自动同步。
- 回放支持左右键单步、上键开局、下键末步、Space / End 播放，0.5× / 1× / 2× / 3× / 5×，旋转与试走变化。
- 视频优先使用浏览器原生 MP4，编码不可用时使用 WebM，弹窗标明格式。导出前 100 个动作，支持声音、速度、视角和录制取消。
- 当前可运行的对手是简单基线，DL / RL / 搜索槽需要接入自己的策略；不将难度标签当成已训练模型。详见 [AI 接口](docs/AI_INTERFACE.md)。
- 首页按本地项目需求精简为人机与本地对战，参考站全部入口的调查记录见网站拆解文档。

## 最新补充

- [公开源码、尺寸、配色与拖放细节](docs/REPLICA_SOURCE_AUDIT.md)
- [桌面与 390×844 手机现场观察](docs/REPLICA_LIVE_AUDIT.md)
- [资源来源清单](docs/REPLICA_ASSET_MANIFEST.json)
- [浏览器、Worker 与 Python 策略接入](docs/AI_INTERFACE.md)

## 文档入口

- [组员使用、传输与开发指南](docs/DELIVERY_GUIDE.md)
- [本次交付验收矩阵](docs/DELIVERY_AUDIT.md)

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

## 历史验证记录

以下是截至 2026-11-05 的历史结果，并非当前交付包的最终验收：

- 11 项离线检查通过，覆盖规则边界、动作编码、环境、缓存、限流错误、HTTP 与中断续采。
- 通过网页真实完成一盘专家对战：红方先手，23 回合 / 46 个动作后专家蓝方到达 i1 获胜；全部动作已独立验证合法。
- 通过适配器自动完成两盘官网专家对局，交换先后手，共 51 个动作、26 次实际远程专家决策。
- HTTP 专家查询与完整回合检查通过；该项 HTTP 冒烟复用了已观察的专家缓存。
- 模仿学习和策略梯度示例均跑通；这只证明训练链路可用，不代表已达到专家水平。

真实的小样本数据位于 [training/data/live-smoke](training/data/live-smoke)。模型文件可用脚本重新生成，未随仓库分发。

## 项目结构

```text
frontend/                     React 棋盘、对战、回放、分享、视频、策略适配
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

本仓库独立编写本地实现，并依据 [Barricade](https://barricade.gg) 的公开网页、浏览器可获取脚本和交互实测分析其行为；不是该网站的官方项目。第三方完整脚本、浏览器账号、设备标识、运行缓存及本机虚拟环境均未打包。项目包含来自参考站的图标和走棋、放墙音效，具体来源见 [资源清单](docs/REPLICA_ASSET_MANIFEST.json)；其再分发许可尚未核实。

专家适配器使用官网前端已使用的第三方服务，接口未来可能变化。访问拒绝或限流会停止采集并保留进度，不会切换身份或用本地策略冒充专家应手。内部专家权重和算法不可见；本项目通过应手样本做行为学习。

网站全部主要入口已编目，但登录、支付、真人匹配、四人实战等尚未全部实测，详见各文档的证据等级。
