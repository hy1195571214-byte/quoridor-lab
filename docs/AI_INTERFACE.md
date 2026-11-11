# AI 接口

棋盘规则是 `frontend/src/game/rules.js`，坐标的行从上至下为 0–8，列从左至右为 0–8。`parseAction` 接受行动编号、记谱（如 `e2`、`ha8` 或 `a8h`）或 `{kind,row,col}`。编号 0–80 为棋子目标格，81–144 为横墙，145–208 为竖墙。状态保留 `pawns`、`walls`、`remaining`、`turn`、`winner`、`history`；`applyAction` 返回新状态。`observation` 返回八个 9×9 平面和长度为 209 的合法行动掩码，与 Python 训练规则对齐。

## 浏览器策略

```js
import { agentRegistry, createAgentRunner, httpAgent } from './game/agents.js';
const runner = createAgentRunner(agentRegistry);
const result = await runner.choose('shortest', { state, requestId: 'turn-1' });
// result.action、result.action_id、result.notation
runner.cancel();
// 有可用本地模型服务时：
agentRegistry.register('my-model', httpAgent('http://127.0.0.1:8000/agents/my-model/move'));
```

`shortest` 是只移动棋子并选择最短路径的简单基线；`random` 从所有合法行动随机选择。它们不是训练模型。可注册 `async ({state, signal, requestId}) => action`；返回值可以是 action 对象、编号、记谱或带 `action`、`action_id`、`notation` 的对象。`workerAgent(worker)` 可包装 Worker，消息包含 `{state,requestId}`，响应需返回相同 `requestId` 和行动。每局使用独立 runner；新请求或 `cancel()` 使之前的结果失效，并通过 AbortSignal 请求取消。所有结果经过合法性检查。接入其他来源时应由应用提供适当的跨域设置。

## Python 本地服务

原有端点保持兼容。`POST /state` 接受 `{"moves":[...]}`，返回状态、行动掩码以及 `observation`（8×9×9）。`GET /agents` 列出可用的基线与未配置的 `dl`、`rl`、`search` 占位。`POST /agents/{id}/move` 接受相同的历史，返回 `action_id`、`notation` 等；未配置的模型返回 503。自定义本地策略可用 `create_app(agents={"my-model": policy})` 注入，`policy.choose(state)` 须返回带合法 `Action` 的 `Decision`。行动会在服务端再次验证。以上不启动训练，也不自动请求外部专家。

## 本地运行与测试

在项目根目录运行时，应显式指定当前项目的 `training`，以免导入同环境中安装的旧版本：

```sh
python3.11 -m venv training/.venv
training/.venv/bin/python -m pip install './training[test]'
PYTHONPATH=training training/.venv/bin/python -m barricade_lab serve --port 8765
PYTHONPATH=training training/.venv/bin/python -m unittest discover -s training/tests -v
```

该服务默认只监听本机；启动本身不会请求外部专家。Windows 将 `training/.venv/bin/python` 换为 `training\.venv\Scripts\python.exe`，并可在安装后直接运行模块。离线网页的策略地址使用 `http://127.0.0.1:8765/agents/{id}/move`；开发网页可使用 `/api/agents/{id}/move`。

## 本地记录服务

服务在其数据目录创建 `records.sqlite3`，提供 `GET /records`、`POST /records`、`GET /records/{id}`、`PUT /records/{id}`、`DELETE /records/{id}`。POST 和 PUT 的 JSON 请求体直接使用棋谱对象（`moves`、`variations`、`result` 等），服务会重演主线及各变化、检查结果并规范化记谱；无效棋谱返回 400。记录独立保存在本机，与浏览器 localStorage 不会自动同步。仅本机启动的服务可被 `file://` 页面（Origin 为 `null`）及 localhost/127.0.0.1 网页跨域访问；因此应只打开信任的本地页面，并保持服务绑定 127.0.0.1。

## 自定义策略服务示例

安装依赖后，在项目根目录运行 `PYTHONPATH=training training/.venv/bin/python training/examples/serve_custom_agent.py --port 8765`。示例将 `dl`、`rl`、`search` 注册为三个**最短路基线占位**，没有加载模型权重；启动或调用这些策略不会请求外部专家。组员可修改示例中的 `choose(state)`，从 `state.observation()` 读取棋盘和合法行动掩码，返回包含合法 `Action` 的 `Decision`。各模式的响应 `source` 会明确标为 `placeholder:...:shortest`。
