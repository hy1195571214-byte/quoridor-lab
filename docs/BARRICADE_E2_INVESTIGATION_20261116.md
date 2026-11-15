# Barricade e2 深入排查（2026-11-16）

> 后续已补上复发识别、自动换连接、持续故障暂停、缓存绕过、历史训练筛选和专项监测，详见 [e2 防护与恢复验收](BARRICADE_E2_GUARD_20261116.md)。以下保留最初定位与匹配实验过程。

本次在原工作区 `b109a4e` 加未提交改动上继续调查。未重置工作区、未更换设备身份、未清理采集数据。正式 Master 对 Master 批次经 Dashboard 正常暂停后，所有实测单独输出，避免与正式采集争用接口。

**当前结论：连接修复已部署，并补齐合法异常 e2 的专项验证：匹配局面回退由旧路径 37/40 降至修复路径 0/40，三盘全新棋局无非首手 e2 或 e2 往返。** 故障集中于本机绕过环境代理、以 Polling 初始化会话的路径。WebSocket 首连在明确关闭代理和使用现有代理时均恢复正常应手。最初的正式续跑窗口没有遇到精确开局回退的待测局面，不能单凭自然完局或合法率验收 e2；本报告末尾的专项对照直接验证了这一点。专项结束后正式批次已恢复运行，847 / 5,000 盘。有限样本不保证所有未来时段都不复发；服务端内部究竟是路由、会话初始化还是引擎状态问题，客户端证据不能进一步定论。

## 已核实的本地完整性

暂停时，正式批次有 835 盘自然完局、3,615 盘弃局和 295 份断点。逐条重放 68,223 条 transitions，核对前缀、手数、行动方和后继局面；重放全部最终记录核对状态与胜方；核对全部断点与逐步记录；将 91,486 条带实际发送负载的失败请求与其所属棋局前缀比较。上述检查无不一致。另有 67,773 条早期失败记录没有原始负载，不能补证这些请求的实际发送内容。

新增 `test_remote_game_isolation.py` 使用真实 RemoteExpert、假 Socket.IO 客户端和真实线程调度，覆盖：两盘不同历史、单连接轮转与双连接并发、三个恢复位置、蓝方非法 e2 重试不落盘、红方正常首手 e2 保存，以及传输中断后完整前缀恢复。未复现串盘或恢复丢失历史。

证据：[完整性报告](../training/data/collection/e2-investigation-20261116/local-integrity-audit.json)，[可重跑脚本](../training/data/collection/e2-investigation-20261116/audit_local.py)。脚本只读正式批次，要求其已暂停。

## 首轮线索：WebSocket 首连（存在代理混杂因素）

此前试验的 WebSocket 先使用 `transports=["polling", "websocket"]` 建立 Polling 会话，再升级。本次新增 `transports=["websocket"]`，从握手开始就使用 WebSocket。这两条路径不能合并统计为同一种连接。

先用独立原始 Engine.IO / Socket.IO 帧客户端测试，绕过 python-socketio，12 次回复全部合法。随后两轮使用全新连接，按相同历史随机交叉发送，保持相同设备身份、Master 难度、完整棋谱、关闭缓存，回复后全局间隔至少 1 秒。

| 路径 | 第一轮合法应手 | 换新连接第二轮 |
| --- | ---: | ---: |
| 原始协议直连 WebSocket | 8 / 8 | 8 / 8 |
| python-socketio 直连 WebSocket | 8 / 8 | 8 / 8 |
| Polling | 4 / 8 | 0 / 8 |
| Polling 后升级 WebSocket | 3 / 8 | 1 / 8 |

四个局面包括蓝方首手、红方第二手、蓝方第二手及已有的 19 手含墙中盘；这些局面的 e2 都非法。错误回复仍然是 `ok: true`、关联编号匹配、落点 e2。没有显式限流或访问拒绝。

两轮同时间交叉对照不支持“仅是时段变好”的解释，但随后发现一个重要混杂因素：本机存在环境代理，`websocket-client` 会自行读取代理变量，**不受 `requests.Session.trust_env=False` 控制**。上述成功的 WebSocket 首连经过现有环境代理，Polling 则绕过代理；升级连接先绕过代理建立会话，再通过代理升级。因此上表不能单独证明传输协议导致差异，必须继续做“协议 × 环境代理”完整因子对照。这里的“直连 WebSocket”只指不经过 Polling 握手，不代表网络连接绕过代理。

证据：[第一轮](../training/data/collection/e2-investigation-20261116/transport-crosscheck-2.json)、[第二轮](../training/data/collection/e2-investigation-20261116/transport-crosscheck-3.json)、[原始协议](../training/data/collection/e2-investigation-20261116/raw-wire-probe.json)。编号 1 的交叉试验在连接阶段结束，没有发应手请求，不计入上表。

Python 直连还暴露了一个具体兼容问题：`websocket-client` 自动产生 Origin，叠加显式 Origin 会形成重复头。连接时设置 `websocket_extra_options={"suppress_origin": True}`，保留唯一的官方 Origin，直连成功。此修复只处理握手，不能用它单独解释 Polling 的非法应手。

## 其他排查

当前官网公开人机页面显示 easy / medium / hard / advanced / expert，远端调用路径直接传递难度。本站采集的 master 不在当前页面选项中，但同局面随机交叉的 48 次 Polling 请求中，master、expert、advanced、hard 全部返回 e2；更换难度不是本次修复。正式批次继续按原要求发送 master，不将其他难度数据混入。

公开客户端来源：[人机页面](https://barricade.gg/computer)、[AI 连接及请求模块](https://barricade.gg/_next/static/chunks/93703ceb315724a1.js)、[人机调用与难度模块](https://barricade.gg/_next/static/chunks/68db35702298edd7.js)。公开脚本快照只留在本地诊断目录，不加入发布源码。

## 实现与验证状态

- RemoteExpert 支持 polling / websocket / upgrade 显式选择，保留身份、关联编号、规则校验和限流停止语义；`proxy_env=False` 现在同时禁止 HTTP 和 WebSocket 自行使用环境代理，`True` 则允许现有代理，不修改系统设置。
- 成功和失败都记录实际发送前缀、行动方、手数、匿名连接编号、连接内请求序号、请求模式、实际传输和关联编号校验；设备标识不写入请求诊断。
- Dashboard 的新批次、正式批次及恢复配置支持 `request_transport` 与 `proxy_env`；CLI 支持 `--transport` 和已有的 `--proxy-env`。
- 38 项采集/适配器相关测试和 38 项 Dashboard 相关测试通过，Dashboard 内联脚本语法检查及 `git diff --check` 通过。全套初次运行另有 7 项无关环境/既有清理失败：6 项 Windows SQLite 文件占用清理错误、1 项当前 Node 不支持测试命令参数。不能把全套测试说成全绿。
- 两个五分钟完整棋局对照窗口和两个三分钟正式恢复验收窗口均已完成，均持续新增自然完局。

## 第一轮完整棋局对照

五分钟窗口，两组均使用真实 collect / RemoteExpert，标准开局、Master 对 Master、关闭缓存、最多 256 手、现有同一设备身份，共享每秒一次的限速且请求串行。每组目标三盘自然完局。保持既有即时重试、冷却、弃局规则，不添加本地替代走法，不过滤规则合法的回退。输出与正式批次隔离。注意：此轮使用修正代理开关语义之前的适配器，成功组实际为 WebSocket 首连加环境代理，对照为绕过代理的 Polling。

| 指标 | Polling | 直连 WebSocket |
| --- | ---: | ---: |
| 远端回复 | 70 | 197 |
| 合法应手 | 16（22.9%） | 197（100%） |
| 非首手合法步数 | 13 | 194 |
| 自然完局 | 0 | 3（67 / 75 / 55 手） |
| 弃局 | 0 | 0 |
| 精确开局回退入库 | 1 | 0 |

两组请求量不同是原有错误冷却机制的结果，不代表给直连组额外提高限速。直连组自然完成三盘即达到试验目标；对照组冷却中的三份断点保留。逐步记录的实际发送前缀、行动方、后继局面和最终胜方均通过重放验证。

证据：[第一轮完整结果](../training/data/collection/e2-investigation-20261116/natural-trial-1/comparison.json)、[可重跑对照脚本](../training/data/collection/e2-investigation-20261116/natural_game_trial.py)。

第二个五分钟窗口使用全新连接并反转启动顺序，WebSocket 组 191 / 191 应手合法，再完成 3 盘（62 / 61 / 68 手），精确回退 0。Polling 组 57 次回复仅 12 次合法，0 完局，精确回退 0。两组均无弃局。两窗口合计成功组 388 / 388 合法、6 盘完局；对照组 28 / 127 合法、0 完局。第二轮仍是代理开关修正前的实现，所以成功组使用了环境代理。这两轮证明该运行方案可以恢复完整棋局，不能单独证明关闭代理时的正式吞吐。

证据：[第二轮完整结果](../training/data/collection/e2-investigation-20261116/natural-trial-2/comparison.json)。脚本随后显式设置了同样的路由选项，避免修正代理语义后重跑得到不同实验条件。

## 排除代理混杂因素后的因子对照

每轮四个局面、每局面两次，每轮新建连接；相同 Master 难度、设备身份和完整棋谱，随机交叉发送，回复后至少间隔一秒。WebSocket 绕过代理组显式传入 `http_no_proxy=["*"]`，不再依赖 requests 的代理开关。

| 实际路径 | 第一轮合法应手 | 第二轮合法应手 |
| --- | ---: | ---: |
| Polling，绕过环境代理 | 1 / 8 | 0 / 8 |
| Polling，使用现有环境代理 | 8 / 8 | 8 / 8 |
| WebSocket 首连，绕过环境代理 | 8 / 8 | 8 / 8 |
| WebSocket 首连，使用现有环境代理 | 8 / 8 | 8 / 8 |

这支持两个可操作的替代路径：更换首次握手为 WebSocket，或让 Polling 使用本机已有代理。正式恢复选择前者，并明确关闭环境代理；其余任务配置保持 Master / Master、5000 自然完局、标准开局、无缓存、256 手上限，原重试和弃局策略保留。并发从 16 降为 2，共享间隔由 0.1 秒改为 1 秒，以保守速率验证修复；没有轮换身份或更改系统网络设置。

证据：[因子对照第一轮](../training/data/collection/e2-investigation-20261116/proxy-factor-1.json)、[第二轮](../training/data/collection/e2-investigation-20261116/proxy-factor-2.json)、[脚本](../training/data/collection/e2-investigation-20261116/proxy_factor_probe.py)。没有显式限流或访问拒绝。

## 正式批次自然完局恢复验收（已通过，不替代 e2 专项验收）

暂停时的 295 份原始断点及 run.json 已另行备份；五个主要 JSONL 文件记录了原始长度和 SHA-256，恢复后核对相同长度的文件前缀，并逐一核对旧断点历史是否保留为当前或最终棋局的前缀。原始 835 盘完局和 3,615 盘弃局不重新分类、不删除、不混入实验棋局。

验收起点为北京时间 01:08:06：835 盘自然完局、68,223 条样本。后续两个连续三分钟窗口使用已恢复的真实 Dashboard 控制器；独立读数据核对实际传输、代理开关、动作、行动方、后继局面和胜方。

| 北京时间窗口 | 自然完局 | 新增逐步记录 | 新增弃局 |
| --- | ---: | ---: | ---: |
| 01:08:06–01:11:06 | 835 → 837（+2） | 172 | 0 |
| 01:11:06–01:14:06 | 837 → 841（+4） | 179 | 0 |

验收期间共新增 351 条记录，全部是旧棋局的非首手推进，全部新鲜远端应手，无缓存；实际连接均为 WebSocket 首连、`proxy_env=false`，共 2 个匿名连接编号。新增拒绝日志和延后失败日志均为 0；精确 `e2,e8,e3,e7 → e2` 入库数为 0。六盘新完局及所有新步的行动方、实际发送前缀、后继局面和胜方重放检查均无错误。

五个主要数据文件的原始字节前缀 SHA-256 全部保持一致；295 份旧断点的历史全部保留为当前或已结束棋局的前缀。对照试验的六盘棋局未混入正式批次计数。正式目标仍为 5,000 盘，验收结束时服务继续运行，并未声称目标已经全部采满。

证据：[正式验收报告](../training/data/collection/e2-investigation-20261116/production-acceptance.json)、[只读验收脚本](../training/data/collection/e2-investigation-20261116/verify_production.py)、[原数据哈希及断点清单](../training/data/collection/e2-investigation-20261116/production-baseline/manifest.json)、[采集测试日志](../training/data/collection/e2-investigation-20261116/collector-tests.log)、[Dashboard 测试日志](../training/data/collection/e2-investigation-20261116/dashboard-tests.log)。

后续若调整请求频率或并发，须重新观察自然完局和拒绝记录；本次没有验证恢复到 16 连接、0.1 秒间隔的安全吞吐。原始历史数据中的合法可疑回退也没有被删除或重写，训练导出若要降权应另行保留明确标记。

## 补充专项验收：合法但异常的 e2 回退

用户指出重点是解决频繁 e2，而非只提高合法率。本次修正验收口径：必须在 e2 本来合法的同一个局面比较实际回复，并单独检查全新棋局的往返。前述 351 步正式续跑没有遇到一次 `e2,e8,e3,e7`，其“精确回退 0”缺少局面暴露量，不能单独用于证明回退修复。

### 两轮同局面交叉试验

使用真实 RemoteExpert，Master、同一原设备身份、完整历史、关闭缓存、明确绕过环境代理。正式批次正常暂停保存断点后，旧 Polling 和修复后的 WebSocket 首连按随机顺序串行请求，回复后间隔至少一秒。每轮为每组新建连接，每个局面重复五次，共两轮、80 次回复。四个局面从历史高频合法回退前缀中选取，不是先看新结果再挑选。

| 请求历史 | 旧 Polling 返回 e2 | 修复后 WebSocket 返回 e2 |
| --- | ---: | ---: |
| `e2,e8,e3,e7`（无墙） | 8 / 10 | 0 / 10 |
| `e2,e8,e3,hc7` | 10 / 10 | 0 / 10 |
| `e2,e8,e3,ve6` | 10 / 10 | 0 / 10 |
| `e2,e8,e3,hb8` | 9 / 10 | 0 / 10 |
| 合计 | **37 / 40（92.5%）** | **0 / 40** |

两组 80 次回复全部合法，关联编号均匹配。旧路径第一轮回退 19/20，第二轮 18/20；修复路径两轮均为 0/20。修复路径返回 20 次 e4 和 20 次放墙，没有拦截 e2、重试择优或替换远端动作。无显式拒绝或限流。

该对照证明连接方案确实改变了合法异常 e2 的高频行为，而不只是让非法应手过关。它支持在已测场景下解决高频回退，但不证明所有局面、未来所有时段都不再出现回退，也不证明每一个放墙动作都优良。服务端内部原因仍未证实。

证据：[逐请求完整结果](../training/data/collection/e2-investigation-20261116/legal-e2-matched-probe.json)、[专项脚本](../training/data/collection/e2-investigation-20261116/legal_e2_probe.py)。

### 正式续跑中的 e2 与旧历史隔离

截至再次暂停的只读快照，修复后新增 719 步，共收到 8 次 e2：红方非首手 3 次、蓝方 5 次。这 8 次均缩短当前到终点的距离，且达到当时合法棋子移动中的最短距离；红方这 3 次均不闭合 e2 往返。它们均发生于含墙局面，不能只因落点为 e2 就判为异常。

按“忽略放墙、仅看红方棋子移动”的旧指标，新增动作仍闭合 6 次四步交替。但逐条核对发现，6 次全是旧历史中 `e2→e3→e2` 后的新动作 e3，且新动作均缩短距离；没有一组四步全部发生在修复后。不能把旧断点里的回退归因于新连接，也不能因为混合指标仍非零就认定修复失败。此快照仍没有精确无墙开局的应手机会，其效果由上面的匹配试验直接验证。

证据：[暂停时质量快照](../training/data/collection/e2-investigation-20261116/e2-quality-172152.json)、[六次交替逐项核对](../training/data/collection/e2-investigation-20261116/production-e2-loop-closures.json)。

### 修复路径的三盘全新棋局

匹配试验结束后，使用正式运行同样的 WebSocket 首连、`proxy_env=false`、Master 对 Master、关闭缓存和共享一秒间隔，从标准初始盘面独立采集三盘。没有指定开局走法、过滤 e2 或替换应手。

- 三盘全部自然完局，分别 58、38、59 手；155 次新鲜远端回复，无弃局或请求错误。
- e2 共 3 次，全部是各盘红方首手；非首手 e2 为 0。
- 红方 51 次棋子移动中，涉及 e2 的三步往返、四步交替均为 0。
- 三盘没有自然走到精确前缀 `e2,e8,e3,e7`，所以其精确回退率仍不能用 0/0 论证；该局面的证据来自上面独立的 10 次匹配请求。
- 全部 155 条逐步记录的发送历史、实际传输、代理开关、关联编号、行动方和后继局面，以及三盘最终状态与胜方，均经重放核对。

证据：[三盘专项报告](../training/data/collection/e2-investigation-20261116/fresh-direct-e2/report.json)、[逐请求记录](../training/data/collection/e2-investigation-20261116/fresh-direct-e2/requests.jsonl)、[脚本](../training/data/collection/e2-investigation-20261116/fresh_direct_e2_trial.py)。这些试验棋局没有混入正式批次。

专项结束后，经 Dashboard 恢复原正式批次，确认状态为 running、847 盘自然完局、3,615 盘弃局、68,942 条样本；继续使用 WebSocket 首连、关闭代理、两条连接共享一秒间隔。保留全部原始修改、棋谱和断点，不删除历史异常 e2。
