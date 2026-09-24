# DSH 接入本地大模型：插件怎么发现端点、写进模型列表、以及它怎么自检

> 2026-09-24：由「同步本地模型」（`local-models-sync.v1`）升级为「接入本地大模型」
> （`local-models-connect.v1`）。本文前半是两端共有的问题背景，后半是新一代的行为、配置与运维。
> v1 的实测记录留在 §5，因为它是「为什么不能信手写声明」的第一手证据。

## 1. 问题：为什么必须让引擎自己说

DSH 的 `llm-pi-ai` provider（`~/.dsh/settings.yaml` 里的 `qwen-local`）是**手写**的模型声明。
引擎换档（150K / 163840 / 172032 / 262144）之后，这份声明不会自己跟着变。2026-09-12 查到的实际状态：

```yaml
models:
  - id: Qwen3.8-27B-Q6-dual-5060ti
    name: Qwen3.8-27B (本地 2x5060Ti 192K)   # 名字里写着 192K
    contextWindow: 196608                     # 引擎实际只服务 150000
    maxTokens: 16384
```

**它不是「显示不准」这么轻**：`contextWindow` 直接决定自动压缩阈值 ——
`dsh-compaction-basic` 挂载时没有 config，取默认 `thresholdRatio = 0.8`：

| 声明值 | 自动压缩阈值 | 后果 |
|---|---|---|
| 196608（旧） | 157,286 | 会话长到 150,000–157,286 之间时，DSH 认为还能塞 → 请求被 vLLM 以超长拒掉 |
| 150000（现在） | 120,000 | 加 `maxTokens` 16,384 = 136,384 < 150,000 ✓ 安全 |
| 缺 `contextWindow` 时的回落 | 262,144 → 阈值 209,715 | 缺口更大 |

## 2. 为什么 DSH 自带的「获取可用模型」救不了（源码级）

`dsh-client-ui-settings-models` 的 `fetchModels` → `api.llm.discoverModels` → `dsh-llm-pi-ai`
的 `discoverModels()`，三条都卡住：

1. `readListing()` 只认 `context_window` / `context_length`，而 vLLM 的 `/v1/models` 给的是
   **`max_model_len`** → 上下文长度根本读不到；
2. `adoptPicked()` 对**已存在**的 id 不替换（`byId.get(id) ?? adopt(candidate)`）→ 我们的 id
   已经在列表里，点了等于没点；
3. 即使采纳了新 id，缺 `contextWindow` 会回落到 `defaultContextWindow = 262144`，比 196608 更糟。

## 3. 新一代：`local-models-connect.v1`

源码：[`../dsh-plugin/local-models-connect.v9.mjs`](../dsh-plugin/local-models-connect.v9.mjs)
测试：[`../dsh-plugin/tests/local-models-connect.test.mjs`](../dsh-plugin/tests/local-models-connect.test.mjs)（81 条，`node dsh-plugin/tests/local-models-connect.test.mjs`）
面板验证台：[`../dsh-plugin/tests/panel-harness.mjs`](../dsh-plugin/tests/panel-harness.mjs)（起一个仿真的会话头部 + 桩路由，供真浏览器驱动；真 GUI 要 token，而凭据不进 agent 命令）

装到 `~/.dsh/plugins/local-models-connect.v9.mjs`，并在
`~/.dsh/profiles/web/cordis.patch.yml` 里挂一条（见 §6）。**它取代了 `local-models-sync.v1`**。

它做三件事：

### 3.1 发现：种子主机 × 端口 → 「哪个端口、什么模型、多少上下文」

对配置里的每个 `host:port` 探 `{origin}/v1/models`，读出模型列表。上下文字段各引擎不统一，
按**可信度分两档**收集：

| 档 | 来源 | 说明 |
|---|---|---|
| **可写进声明** | vLLM / SGLang `/v1/models` → `max_model_len` | 服务端真实上限 |
| | OpenAI 兼容网关 `/v1/models` → `context_length` / `context_window` / `limit.context` / `max_input_tokens` | |
| | llama.cpp `GET /props` → `default_generation_settings.n_ctx` | `-c` 的**实参** |
| | Ollama `POST /api/show` → `model_info.*.context_length`，退化到 `parameters` 里的 `num_ctx` | |
| **只作证据，永不写进声明** | llama.cpp `/v1/models` → `meta.n_ctx_train` | 这是**训练**长度，不是服务端上限 |

最后一档是整个插件最容易写错的地方，所以单独隔离：llama256 档服务端 `-c 262144`，而同一份权重
`n_ctx_train` 可能只有 32768。取哪个都能过 schema，取错方向一个让压缩过早（长任务被无谓截断），
另一个就是 §1 那个「声明 196608 / 实际 150000」的事故。所以 `n_ctx_train` 只出现在自检报告的
说明文字里，**永不参与合并**。

引擎识别也只看证据：`owned_by` 自报 → `max_model_len` 出现 = vLLM 家族 → `/props` 通 = llama.cpp。

### 3.2 接入：没有就整条建出来，有就只更新引擎说了算的字段

| 规则 | 为什么 |
|---|---|
| baseURL 精确匹配到已有 provider → **复用**，绝不改名、绝不新建重复路由 | 人工起的名字（`qwen-local`）在文档、脚本、铁律 11 里都被引用 |
| 没有对应 provider → 建整条：`api` / `baseURL` / `compat` / `streamIdleTimeoutMs` / `models` | 这是「新环境装上插件即可用」的那一步 |
| `contextWindow` / `maxTokens` 以引擎自报值为准 | §1 |
| `name` / `displayName` / `compat` / `headers` / `apiKeyEnv` 等**已有值一律不动** | 人工声明优先 |
| 引擎不再广告的旧模型条目**保留**（`prune: false`） | 只增不删；可能是手写但当前档位暂未服务的 |
| 上下文读不到的模型 → **不建这一条**（`adoptUnknownContext: false`） | 缺 `contextWindow` 会回落 262144；少一条能选是响的、可逆的，错一个值是哑的 |

**「不建」与「建」的取舍**：新 provider 还会带上 `defaultContextWindow` = 该端点**已知上下文的
最小值**，让兜底不再是 262144。要强行建上下文未知的条目，设 `adoptUnknownContext: true`。

**新建时不给端点挂凭据名**（`apiKeyEnv` 默认空）。这不是洁癖：`dsh-llm-pi-ai` 的 `resolveApiKey`
在 `apiKeyEnv` 存在、而凭据服务与环境变量里都没有该值时**直接抛 `MISSING_CREDENTIAL`**
（`lib/index.js` 的 `resolveApiKey`）。给一个无鉴权的本地端点挂上 `QWEN_LOCAL_API_KEY`，
在配了那份凭据的机器上能用、换台机器整个模型不可用 —— 正好破坏「新环境装上就能用」。

### 3.3 自检：只读，把事实逐条判成 ok / warn / fail

`POST /local-models-connect/selfcheck`（GET 也行）**一个字节都不写**。判定项：

| 检查 | 判据 | 档 |
|---|---|---|
| `plugin` | 版本、自动探测/自动写入开关 | ok |
| `targets` | 候选端点总数 / 存活数 | 0 存活 → **fail** |
| `reachability` | 一个都没通时给出种子主机与端口、并指向 `modelctl status`；按连接层错误码区分「主机通但端口没监听」（`ECONNREFUSED`，多半在启动/重启窗口）与「连不上主机」 | **fail** |
| `model:<host:port>/<id>` | 该模型有没有可用的上下文长度（有则报 `n_ctx_train` 作证据） | 无 → warn |
| `health:<host:port>` | `/health` 是否 200 | 非 200 → warn |
| `link:<origin>` | 端点有没有接进模型列表 | 没有 → warn |
| `declare:<provider>/<id>` | 引擎广告了、声明里没有 | → warn |
| `context:<provider>/<id>` | 声明值与引擎自报值不一致 | → warn（跑一次「接入」即修正） |
| `context-missing:` | 声明里缺 `contextWindow`（会回落 262144） | → warn |
| `headroom:<provider>/<id>` | `0.8 × contextWindow + maxTokens > contextWindow` | **fail**，并给出建议的 maxTokens 上限 |
| `stale:<provider>/<id>` | 声明里有、引擎当前没广告 | → warn |
| `linkage` | 汇总「已接入 / 待接入」 | 有待接入 → warn；**一个端点都没通时不给绿灯**（「已接入 0 个 / 待接入 0 个 ✓」是空转的 ✓），改报「无法判定」 |
| `credential:<provider>` | `apiKeyEnv` 在凭据服务/环境变量里取不到值 | **fail**（用时会抛 `MISSING_CREDENTIAL`） |
| `credentials` | 所有本地 provider 的 `apiKeyEnv` 都能取到值 | ok，但标题里明说**与端点是否连得通无关** |

`verdict` = 有 fail 则 `fail`，否则有 warn 则 `warn`，否则 `ok`。

`headroom` 那条刻意**不自动修**：`maxTokens` 是使用偏好而不是引擎事实，插件不替你改，
只把「能塞下的最大值」算给你（`contextWindow − 0.8 × contextWindow`）。

`credential` 那条的口径与 `resolveApiKey` 对齐（先凭据服务、后启动环境）。没有可用凭据服务时
只报 `unknown`（warn）而不误判成 `missing` —— 判不出来就别吓人。

## 4. 输入输出

| 方法 | 路径 | 作用 |
|---|---|---|
| `GET` | `/local-models-connect/state` | 插件配置 + 最近一次的报告缓存，**不打网络** |
| `POST` | `/local-models-connect/run` | 发现 + 接入；加 `?dry=1` 只看不写 |
| `GET`/`POST` | `/local-models-connect/selfcheck` | 只读自检 |
| `POST` | `/local-models-connect/panel` | **面板用的那一条**：run + selfcheck 跑完，直接回"三个块"的视图（见 §4.1） |
| `POST` | `/local-models-sync/run` | **v1 的旧路径留作别名**，行为同 `run` |

```bash
curl -sS -X POST http://127.0.0.1:3080/local-models-connect/run | python3 -m json.tool
curl -sS "http://127.0.0.1:3080/local-models-connect/run?dry=1" -X POST | python3 -m json.tool
curl -sS http://127.0.0.1:3080/local-models-connect/selfcheck | python3 -m json.tool
```

页面上是**一个**按钮，**挂在会话头部的 utilities 行里、「在本地打开」那个分体控件的左边**
（拿不到头部时退回右下角浮动，按钮不会凭空消失）：

* **一次点击 = 接入 + 自检，报告合一**。原来「接入本地模型」与「自检」是两个按钮：它们探的是
  同一批端点、走的是同一段探测代码，功能明显重合（2026-09-24 用户点出来的）。现在点一下打
  **一个** `POST /panel`（宿主侧先 run 再 selfcheck，顺序不能反 —— 自检要反映**写完之后**的声明），
  回一份排好版的视图。只读能力没有消失，只是不再占一个按钮：CLI 的 `GET /selfcheck`、
  `?dry=1`、`config.autoApply:false` 都还在，开机自动探测也仍旧跑。
* 按钮带一个状态点（最近一次自检的 verdict：绿/黄/红）。
* 度量照抄 `dsh-client-ui-open-in-app` 的 28px 高 / 14px 圆角 / `.5px` `border-l4` /
  11px·16px 字 / 同 padding，并复用同一批 `--dsw-alias-*` 令牌 —— 深浅色主题、hover、
  disabled 都跟着 DSH 走。
* 报告弹窗是 `position: fixed`（**不参与布局、不顶开页面**）。收起方式：**右上角的叉**、
  按 Esc、滚页面、超时（`autoHideMs`，默认 12 秒）。
  **整块弹窗不可点**（`cursor` 是 `auto`，不是 `pointer`）—— 早先它自己就是关闭热区，
  于是报告末尾还得附一行「点这里…」的说明，看着像个大按钮（2026-09-24 用户要求换掉）。
  叉是 22×22 的绝对定位按钮（`top/right: 4px`），挂在 `#lmc-box` 上。
  **高度自适应、默认没有滑动条**（用户 2026-09-25 的要求）：不再钉 `max-height: 42vh`，
  短报告就是短的；`#lmc-body` 是**另一个元素**（这样叉不会跟着正文滚走），
  只有「内容比视口还高」时才在 JS 里临时给它开 `overflow` —— 见 §4.2。
  旧版把报告留在文档流里，点一次就永久占一块地方 —— 也是 2026-09-24 点名要改的。

### 4.1 报告排版：三个块、一根状态点、主机只写一次

用户 2026-09-24 的第二轮反馈是纯排版：「IP 怎么重复显示 / 通不通要用绿点红点 / 排版统一 /
是不是该分三个块」。于是把排版从页面脚本里**搬到宿主侧**（`buildPanelView`，纯函数）——
注入脚本是一整段字符串，Node 侧断言不了它渲染出什么，而排版恰恰是反复被提的东西。

三个块各回答一个问题，所以不合并：

```
检查于 2026-09-25 02:35:41
【接入检查】            ← 有哪些端点、什么模型、多少上下文（发现的**事实**）
  192.168.0.119         ← 主机只写一次（旧版每行都重复 192.168.0.119:xxxx）
  ● 8080   vllm · Qwen3.8-27B-Q6-dual-5060ti @150000     ← 绿点
  ● 8000   连接被拒                                      ← 红点
  ● 30000  连接被拒
  连接被拒 = 主机是通的，只是那个端口上没进程在听（服务没起，或正在启动 / 重启窗口里）
                        ← 这句解释只说一次，不是每个端口重复一遍
【模型列表】            ← 我把声明更新了吗（对 settings.yaml 的**动作**）
  ● qwen-local  已是最新（1 个模型）
  ● 未写入设置（无需改动）
【自检】                ← 现在还有哪里不对（声明与引擎的**判定**）
  ● OK · 6 通过 · 0 提醒 · 0 失败
```

* **行 = 状态点 + 定宽行首列 + 正文**：绿 `ok` / 黄 `warn` / 红 `fail` / 灰 `neutral`（跳过、
  无改动、dry-run）；`host` 与 `detail` 两种行不打点，分别是淡色的主机小标题与缩进 12px 的注解。
  颜色用 `--dsw-alias-state-*-primary`，跟随主题。
* 对齐靠 **flex**（`.lmc-key` 定宽 + 正文 `flex:1`），不靠空格 —— 正文一换行，空格对齐就散。
* 屏幕上短、**悬停不丢信息**：每一行的完整原文（连接层原始报文、change 的逐条 diff、
  未采纳原因、自检 detail）都挂在 DOM 的 `title` 上。
* 「没变化」不再写成光秃秃的「未写入设置」，而是「未写入设置（无需改动）」——
  它和「dry-run 没写」是两回事，早先那句含糊话已经被问过一次。
* 纯函数 ⇒ 可用普通单测钉住（`81 条` 里有 15 条专门测排版：主机只出现一次、点的颜色、
  解释只说一次、各 action 的文案、出错也成块）。

### 4.2 高度自适应（不要滑动条）

用户 2026-09-25 的要求。原来 `#lmc-box` 钉 `max-height: 42vh`、`#lmc-body` 钉
`max-height: calc(42vh - 18px); overflow: auto` —— 报告一长就出滑动条，而且短报告也要
先按 42vh 想一遍。现在 CSS 里**没有任何高度上限**，高度由内容决定；`#lmc-body` 默认不滚。

放不下时按三步处置（都在 `placeBox()` 里，每次都先清掉上一次的限高再量真实高度）：

| 情形 | 处置 | 会不会有滑动条 |
|---|---|---|
| 内容放得下工具条下方 | 直接贴在工具条下面 | 不会 |
| 放不下、但比整个视口矮 | **往上挪**（`top = 视口底 − 高度`） | 不会 |
| 比整个视口还高 | 夹到视口内（`top: 8`）+ 临时给正文开 `overflow: auto` | 会 —— 这是唯一一种 |

弹窗是 `position: fixed`，所以往上挪不会推挤页面（也没有布局抖动）。

真浏览器实测（视口 900px 高，工具条底 36px）：

| 端口行数 | 盒子高度 | top | 默认位置(44) 是否上挪 | 滑动条 |
|---|---|---|---|---|
| 5（真机那种） | 387 | 44 | 否 | 无 |
| 32 | 799 | 44 | 否 | 无 |
| 37 | 882 | **10** | 是 | **无** |
| 62 | 884（夹住） | 8 | 是 | 有（唯一一次） |

`hasScrollbar` 是量 `body.scrollHeight > body.clientHeight` 得出的，不是看截图猜的。

样式走独立的 `style` 注入行、脚本走 `script` 行（v1 把 CSS 塞在模板字符串里，改一个颜色
都要数反斜杠）。页面加载时只读 `/state` 的缓存，不打网络。

### 锚点：为什么不只是「找 `_split`」

两处都不是风格问题，是**必须**这么做：

1. **先限定在头部容器内**（`[class*="_headerUtilities"]`）。全页面另有 2 个包也定义了 `_split`
   （`dsh-client-ui-deliverables`、`dsh-client-ui-trajectory`），全局匹配会挂到别人家去。
2. **爬到容器的直接子项再插**。容器是 `display:flex; gap:8px`（`dsh-client-ui-conversation`），
   而这 8px 只作用在**它的直接子项**之间；`<div class="…_split">` 并不是直接子项 ——
   slot 条目外面还有包裹（open-in-app 的 `Menu` 包着它的 anchor）。插在 `_split` 前面
   = 插进包裹层内部，容器的 gap 够不着 → **与左边邻居贴成 0px**
   （2026-09-24 用户的实测：`接入本地模型` 与「打开终端」之间是 0）。

   注：用户看到的「打开终端」就是这条分体控件 —— 它的 Tooltip label 是「在本地打开」，
   而主按钮的 `aria-label` 是 `open.title` = 「在 {app} 中打开工作目录」，当前记住的 app 是终端。

定位仍按**行为特征**：`div[class*="_split"]` + 内含 `button[aria-haspopup="menu"]`
（CSS module 的哈希随构建变，`_split` / `_headerUtilities` 这些本地名不变）。

## 5. v1 的实测记录（保留，作为「不能信手写声明」的第一手证据）

* 挂载后 `dsh --profile web --dump-config` 里能看到该条目；
* `POST /local-models-sync/run` 返回 `changed: false`（值已正确）；
* 故意把 `contextWindow` 改成 `999999` 再 POST → 返回
  `{"field":"contextWindow","from":999999,"to":150000}`，`settings.yaml` 里随即变成 150000；
* 浏览器实地：页面快照里出现 `button "同步本地模型"`，点击后回显
  `qwen-local：已经是最新（Qwen3.8-27B-Q6-dual-5060ti=150000）`，控制台 0 报错、无失败请求。

v1 只强制引擎**广告出来的事实**（`contextWindow` / `maxTokens` 上限），人工写的 `name`
不会被覆盖。

### v1 那句「没有可同步的 provider」到底是什么意思

它一直被当成含糊话问，所以把触发条件写死在这里。**只有一种情况会打它**：
`llm-pi-ai.providers` 是个**空字典**（一条 provider 都没配）。

推导：v1 的 `summarize` 只在 `report.results` 为空时打这句，而 `syncAll` 对
`Object.entries(section.providers ?? {})` 的**每一项都恰好产出一行**
（没配 baseURL / 不是内网端点 / 拉不到列表 / 没变化 / 已更新，各一条 `continue` 前都 push），
所以 `results.length === 0 ⟺ providers` 为空。

**它是正常还是不正常**：不是报错，但也**不等于「一切正常」**。它的字面意思是「没东西可改」，
而这恰恰是最需要有人替你发现端点的时候（新机器 / 什么都没声明）—— v1 只会改已存在的声明，
不会发现也不会新建，所以那时它帮不上忙，却只说了这么一句。真正让人误读的是它把三种处境
压成同一句话：① 一条 provider 都没配（**只有这种**打这句）② 配了但都不是内网端点
③ 是内网但端点不通（②③ 其实 v1 会逐条列出跳过/失败原因）。

新一代去掉了「汇总成一句话」这条路径：报告恒定按「每个候选端点一行 + 每个 provider 一行
+ 未采纳清单 + 是否写入」输出，所以「什么都没有」与「有但不通」在字面上就是不同的行；
自检另有 `link:<origin>`（还没接进模型列表）与 `linkage`（已接入 0 个 / 待接入 N 个）兜住
「一条都没接入」这个状态。

## 6. 配置与安装

`~/.dsh/profiles/web/cordis.patch.yml` 里的挂载点：

```yaml
- insert:
    - id: local-models-connect
      name: "/home/lcy/.dsh/plugins/local-models-connect.v9.mjs"
      config:
        hosts: ['192.168.0.119']          # 种子主机：唯一的扫描范围，要加机器改这一行
        ports: [8080, 8000, 30000, 8081, 11434, 1234]
        providerId: qwen-local             # 第一个命中的端点用这个路由名（铁律 11 依赖它）
        displayName: Qwen3.8-27B 本地
```

**只扫 `hosts` 里列出的主机**（外加已配 provider 中属于内网的 baseURL，可用
`includeConfigured: false` 关掉）—— 不做网段扫描。全部字段都有默认值，`config:` 整块不写也能工作。

其余可调项与默认值：

| 键 | 默认 | 说明 |
|---|---|---|
| `includeConfigured` | `true` | 顺带刷新已配 provider 里的内网端点（只读它们已有的 baseURL） |
| `autoProbe` | `true` | 插件加载后自动跑一次「发现 + 接入」 |
| `autoApply` | `true` | 自动跑时是否真写设置；`false` = 只报告（等价于一次开机自检） |
| `prune` | `false` | 引擎不再广告的旧模型条目是否删掉 |
| `adoptUnknownContext` | `false` | 上下文未知的模型是否照建 |
| `probeTimeoutMs` | `1500` | 单个探测请求超时 |
| `concurrency` | `6` | 并发探测数 |
| `streamIdleTimeoutMs` | `600000` | 新建 provider 的流空闲超时 |
| `defaultMaxTokens` | `16384` | 引擎没说最大输出时新条目的 `maxTokens` |
| `allowPublic` | `false` | 是否允许探测公网端点 |
| `autoProbeDelayMs` | `4000` | 自动探测的延迟（让 dsh web 先把页面服务起来） |
| `autoHideMs` | `12000` | 报告弹窗自动收起的时间；`0` = 不自动收（仍可点掉 / 按 Esc） |

`providerId` / `displayName` / `apiKeyEnv` 支持**显式写空**表示「关掉/自动命名」
（写成 `''`、`false` 或 `null`）——「没写这个键」才走默认值。

### 安装后的那一次重启（重要）

`~/.dsh/profiles/web/package.json` 里的 `patchReload: "live"` 是 **2026-09-23 17:47:48** 写进去的，
而当时那个 `dsh web` 进程起于 **17:46:04** —— 早于它。也就是说**那个进程启动时并没有挂上
cordis-plugin-hmr**，`cordis.patch.yml` 的改动不会被重放（实测：改完 patch、请求新路由仍 404，
`journalctl -u dsh-web` 里一条相关日志都没有）。源码见 `lib/profile-boot-*.js`：
`patchReload === "live"` 的判断在 boot 时做一次，而且整段包在静默的 try/catch 里。

所以**装完必须重启一次**（2026-09-24 已执行，之后新路由立即 200）：

```bash
sudo systemctl restart dsh-web
sleep 8
curl -sS http://127.0.0.1:3080/local-models-connect/state | python3 -m json.tool
```

**重启之后** `patchReload: live` 才真正生效，此后改 patch 才是「保存即生效」；
在那之前它只是 manifest 里的一行字。重启会打断正在进行的会话 —— 挑个空的时候做。
新 URL（带 token）会写进 `~/.dsh/last-web-url.txt`；页面也要刷新一次才会出现新面板。

### 回滚

```bash
cp ~/.dsh/profiles/web/cordis.patch.yml.bak-before-local-models-connect-<ts> \
   ~/.dsh/profiles/web/cordis.patch.yml
rm ~/.dsh/plugins/local-models-connect.v9.mjs
```

`settings.yaml` 的写入是幂等的（值没变就不写），回滚插件不会把设置改回去。
旧插件 v1 的源码仍在 `dsh-plugin/local-models-sync.v1.mjs`（已从 `~/.dsh/plugins` 删除）。

## 7. 为什么是宿主插件，不是客户端插件

| | 客户端插件 | 宿主插件（本方案） |
|---|---|---|
| 装载 | `dsh-client-modules` 只扫**宿主 Loader 里已知**的包，包元数据按名字缓存且**永不过期** → 新增包**必须重启 `dsh web`** | 挂在 `cordis.patch.yml` 上，Cordis HMR 重放配置树 → 保存即生效（前提见 §6） |
| 调设置 | 浏览器 → RPC（`settings.mutate` 属于客户端可调方法，但要自己拼协议） | 直接在宿主里 `ctx.settings.mutate()`，官方校验 + 原子写 |
| 跨域 | 需要端点给 CORS（vLLM 给的是 `access-control-allow-origin: *`，能用但脆） | 宿主侧 `fetch`，不涉及 CORS |

而那个要重启的进程**正是宿主本体** —— 重启它会打断正在进行的会话，所以客户端插件这条路径
从一开始就不该走。

## 8. 迭代注意

Cordis 只在 `name`（也就是文件路径）变化时才重新 `import` 模块（ESM 有模块缓存），
所以**改插件内容要装成新文件名**（`v2`、`v3`…）并改 `cordis.patch.yml` 里那一行；
只重存同一个文件不会重新加载。

2026-09-24 把这条**实测确认**了一遍，而且是在 `patchReload: live` 真正生效之后（见 §6）：

| 动作 | `/state` 报的版本 | 结论 |
|---|---|---|
| 只改文件内容（版本号 1.0.0 → 1.1.0），patch 不动 | 仍 `1.0.0` | 内容改动**不会**重载，HMR 不管模块内容 |
| 新文件名 + 改 patch 里那一行 | 立刻 `1.1.0`，路由全程 200 | patch 一变就重载，**不需要重启 dsh web** |

所以 `~/.dsh/plugins/` 里那个软链指向仓库实体只解决「文件同步」，不解决「模块重载」——
要生效仍须换文件名。仓库里只保留当前那一版（旧版交给 git 历史），否则 `v1`/`v2`/`v3`
同目录很容易看错哪份是在跑的。

## 9. 2026-09-24 的实测（新一代）

隔离验证（`DSH_HOME=/tmp/lmc-home`，一套只含 `dsh-base` + `dsh-web-app` 的临时 profile，
配一台本机 18080 上的假 vLLM）—— 不碰真 `settings.yaml`：

* 开机自动探测 → 真 `settings.mutate` → 真 `settings.yaml` 里出现新建的 provider：
  `defaultContextWindow: 150000`、`models[0].contextWindow: 150000`、**没有 `apiKeyEnv`**；
* `POST /selfcheck` → `verdict: ok`（5 条 ok，0 warn 0 fail）；
* 把该 provider 的 `apiKeyEnv` 改成不存在的 `LMC_NOT_SET` → `verdict: fail`，报
  `credential:scratch-qwen ... 凭据服务与环境变量里都没有取到值`（说明真凭据服务确实被查到了，
  不是降级成 unknown）；
* 换成 `LMC_LIVE_KEY` 并从环境变量导出该名 → `verdict: ok`（`credentials` 那条 ok）；
* 带 token 取 `/` → 渲染出的 HTML 里有注入的 `<style>`（`#lmc-tools` / `#lmc-box`）与面板脚本
  （`接入本地模型`、`__localModelsConnectInstalled`），且注入脚本里没有提前闭合的 `</script>`；
* `POST /run?dry=1` → `wrote: false`，一个字节都没写。

回归测试 81 条全过（纯逻辑 39 + 假引擎探测 6 + 端到端 19 + 面板排版 16 + 实机冒烟 1 —— 数出来的
是这样，不是估的）。真机 192.168.0.119:8080 冒烟：识别为 `vllm`，`Qwen3.8-27B-Q6-dual-5060ti`，
上下文 150000，KV 池 157,824（= 文档里那个数，见下面「一次误判」）。

真机 `dsh web` 侧（`sudo systemctl restart dsh-web` 之后，PID 658997）：

* `GET /state` → 版本与 `autoHideMs` 都对得上；开机自动探测已跑完：候选 6 个、存活 1 个
  （8080 → vllm / `Qwen3.8-27B-Q6-dual-5060ti`@150000），`qwen-local: unchanged`、
  `wrote: false` —— **没动那份手写声明**（它本来就对）；
* `POST /selfcheck` → `verdict: ok`，6 条 ok / 0 warn / 0 fail，含「1 个本地 provider 的凭据
  都能取到值」（真凭据服务里确实有 `QWEN_LOCAL_API_KEY`）；
* `POST /run?dry=1` → `qwen-local: unchanged`、`wrote: false`；
* v1 旧路径 `/local-models-sync/run` → 200，且回的是**新**结构（`providers`，不再是 `results`）；
* `settings.yaml` 的 md5 与安装前一致（`23c40e46…`）。

### 面板（2026-09-24，真浏览器实测）

用 [`panel-harness.mjs`](../dsh-plugin/tests/panel-harness.mjs) 起仿真头部 + 桩路由驱动，
避开把 GUI 的 token 写进对话记录；头部的**形状与令牌值都抄自 DSH 自己的源码**
（容器 `.…_headerUtilities{display:flex;gap:8px}`、分体控件 `.…_split{28px/14px/.5px}`，
以及**分体控件外面那层 slot 条目包裹** —— 少了这层就验不出下面那个 bug）。

v5（单按钮版）：

* `#lmc-tools` 是容器的**直接子项**、且是 open-in-app 那条 slot 条目的**前一个兄弟**；
  与左邻条的间距 **8px**、与右侧分体控件条目的间距 **8px** —— 都是容器自己的 `gap`；
* 只有一个 `.lmc-btn`（标签「接入本地模型」），计算样式 `28px | 14px | 11px | 16px | 5px 10px | 1px`，
  与原生分体控件容器同档（它也是 28px 高 / 14px 圆角 / `.5px` 边框渲染成 1px）；
* 一次点击 = 3 个请求（`state` → `run` → `selfcheck`），报告是**一段**：接入行在上、
  `—— 自检 WARN：4 通过 · 2 提醒 · 0 失败` 汇总行在下，**ok 的条目被滤掉**、warn 的明细保留；
  状态点转 `warn`；
* 弹窗 `position: fixed`：展示前后头部高度都是 45、文档高度不变 —— **不顶开页面**；
  右边缘与按钮组右边缘**逐像素对齐**；叉 22×22、距外框右上角各 5px，
  正文内容右缘离叉左边 5px（**文字不会钻到叉底下**）；
* 收起：**点正文不再关闭**（`stillOpenAfterBodyClick: true`）、点叉关闭、按 Esc 关闭；
  超时关闭是**精确**的 —— 页面内埋 MutationObserver 记录 class 变化，`shown→hidden`
  的时间差是 **2500ms**（当时 `autoHideMs=2500`），不是"大概"；
* 把 `#lmc-tools` 从 DOM 里删掉 → MutationObserver 300ms 后重挂，且**不重复**（1 个实例）；
* 0 条 console 错误。

**红绿对照（0px 那个 bug）**：把 HEAD 里那一版 v4 交给同一个验证台跑 ——
`tools.parentElement` 是包裹层 `Ss22bb_entry`（不是容器），`gapToSplitPx` **0**，两个按钮；
换成 v5 后 `directChildOfContainer: true`、`gapRightPx/gapLeftPx` 都是 **8**，一个按钮。
用户报的现象与红侧完全一致，所以这个验证台是能抓住它的，不是"跑了个寂寞"。

### v7 的排版实测（2026-09-25，真浏览器）

同一个验证台（桩路由现在直接用**真插件**的 `buildPanelView`，所以验的就是要上线的那份排版）：

* 块标题依次 `接入检查 / 模型列表 / 自检`，块间是 `border-top: .5px`（渲染 1px）+ 9px 间距；
* `hostRowHasDot: false`、主机行文本就是 `192.168.0.119`；
* 6 个端口行：可达那行 `rgb(34,197,94)`（绿）、5 个被拒的行 `rgb(239,68,68)`（红）——
  就是用户要的「通的绿点、不通的红点」；
* `keyColumnAligned: true`：6 行的正文左缘都是 **924px**，端口成列（旧版没有列，靠空格自然对不齐）；
* 三级文字色真的分开了：主机行 `rgb(107,114,128)`（label-secondary）、
  `detail` 注脚 `rgb(129,133,140)`（label-tertiary）、普通行 `rgb(28,28,30)`（label-primary）；
  `detail` 行缩进 **12px**（`detailRowLeft 833` vs `detailTextLeft 845`）；
* 「连接被拒」的解释在整块里只出现 **1 次**（`notes.length === 1`），端口行只留短标签；
* 自检块：结论行 + 只列非 ok 的明细，detail 各占一行且同样淡色缩进；
* 叉仍然关得掉（`closedByX: true`），动态渲染后 **0 条 console 错误**。

真机 `POST /panel` 的返回与上面逐字一致（见 §9 开头那三段 `【…】`）。

**上一批修掉的一个真 bug**：弹窗原本用 `max-width` + shrink-to-fit，于是「可用宽度」会被上一次
写下的 `left` 截断 —— 量到的是「剩下的空间」而不是内容宽度，每次重排都把弹窗再往左推一截
（实测右边缘差了 **59px**）。改成 CSS 里写死 `width: min(72vw, 560px)` + `box-sizing: border-box`
后，`offsetWidth` 与 `left` 无关，重复展示两次的 `left` 完全相同（821px = 1381 − 560）。

### 一次误判：别把「别人的实验」当成「表过时了」

`local-models-connect` 的自检在 09:16 前后报了 **6/6 端点不通**，看起来像服务没了；实测是
**另一台控制机（192.168.0.107）上的会话正在把 GPU 机回滚 vLLM 0.30 → 0.29**：
`/home/lcy/rollback_029.sh` 恢复 `~/.modelctl.env`、把 `vllm-current` 指回 `vllm-venv-029`、
`systemctl restart modelctl`，期间端口本来就是关的。判据是 `systemctl show modelctl` 的
`NRestarts=0` / `Result=success` —— **systemd 的 `Restart=always` 一次都没触发**，
是外部的显式 `systemctl restart`。

回滚完成（09:20:10）后 KV 池回到 **157,824**：所以中途看到的 `KVBYTES=4050000000` / 池 153,905 /
`vllm-venv-030` 都是那次实验的临时状态，**AGENTS.md 与 skill 里的 0.29 / 15.7 万数字并没有过时**。
差一点就把实验残值写成"文档过时"。

处置：**一个字节都没动那台机器**（没启停、没改配置）—— 别人正在那台机器上作业时，
只读排查、别动手，判断依据是 `servers` skill 的「先看有没有别人在同一套设施上」。
自检这边补了两处，正是被这次绊出来的：

* 探测失败时带上连接层错误码（`ECONNREFUSED` / `ETIMEDOUT` / …），
  `reachability` 那条据此说清是「**主机是通的，但端口上没有进程在监听**（服务没起或正在
  启动/重启窗口）」还是「连不上主机」—— 而不是笼统一句"一个候选端点都没通"；
* 一个端点都没通时，`linkage` 那条不再报绿灯（原来会显示「已接入 0 个 / 待接入 0 个 ✓」，
  那是个**空转的 ✓**：没有任何端点可关联，它什么都没验证）；凭据那条也标注了
  「与端点是否连得通无关」，免得和两条 ✗ 并排时被读成「其余正常」。

插件换版后**刷新一次页面**才会看到新面板 —— 已渲染的页面里还是上一版注入进去的脚本
（2026-09-24 从「两个按钮」变成「一个按钮」时就是这样）。
