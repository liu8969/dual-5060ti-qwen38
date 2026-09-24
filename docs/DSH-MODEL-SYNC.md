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

源码：[`../dsh-plugin/local-models-connect.v1.mjs`](../dsh-plugin/local-models-connect.v1.mjs)
测试：[`../dsh-plugin/tests/local-models-connect.test.mjs`](../dsh-plugin/tests/local-models-connect.test.mjs)（62 条，`node dsh-plugin/tests/local-models-connect.test.mjs`）

装到 `~/.dsh/plugins/local-models-connect.v1.mjs`，并在
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
| `reachability` | 一个都没通时给出种子主机与端口、并指向 `modelctl status` | **fail** |
| `model:<host:port>/<id>` | 该模型有没有可用的上下文长度（有则报 `n_ctx_train` 作证据） | 无 → warn |
| `health:<host:port>` | `/health` 是否 200 | 非 200 → warn |
| `link:<origin>` | 端点有没有接进模型列表 | 没有 → warn |
| `declare:<provider>/<id>` | 引擎广告了、声明里没有 | → warn |
| `context:<provider>/<id>` | 声明值与引擎自报值不一致 | → warn（跑一次「接入」即修正） |
| `context-missing:` | 声明里缺 `contextWindow`（会回落 262144） | → warn |
| `headroom:<provider>/<id>` | `0.8 × contextWindow + maxTokens > contextWindow` | **fail**，并给出建议的 maxTokens 上限 |
| `stale:<provider>/<id>` | 声明里有、引擎当前没广告 | → warn |
| `credential:<provider>` | `apiKeyEnv` 在凭据服务/环境变量里取不到值 | **fail**（用时会抛 `MISSING_CREDENTIAL`） |

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
| `POST` | `/local-models-sync/run` | **v1 的旧路径留作别名**，行为同 `run` |

```bash
curl -sS -X POST http://127.0.0.1:3080/local-models-connect/run | python3 -m json.tool
curl -sS "http://127.0.0.1:3080/local-models-connect/run?dry=1" -X POST | python3 -m json.tool
curl -sS http://127.0.0.1:3080/local-models-connect/selfcheck | python3 -m json.tool
```

页面上是右下角常驻的一小块：一个状态点（最近一次自检的 verdict）、「自检」与「接入本地模型」
两个按钮、以及一个可滚动的报告区。样式走独立的 `style` 注入行、脚本走 `script` 行
（v1 把 CSS 塞在模板字符串里，改一个颜色都要数反斜杠）。页面加载时只读 `/state` 的缓存，
不打网络。

## 5. v1 的实测记录（保留，作为「不能信手写声明」的第一手证据）

* 挂载后 `dsh --profile web --dump-config` 里能看到该条目；
* `POST /local-models-sync/run` 返回 `changed: false`（值已正确）；
* 故意把 `contextWindow` 改成 `999999` 再 POST → 返回
  `{"field":"contextWindow","from":999999,"to":150000}`，`settings.yaml` 里随即变成 150000；
* 浏览器实地：页面快照里出现 `button "同步本地模型"`，点击后回显
  `qwen-local：已经是最新（Qwen3.8-27B-Q6-dual-5060ti=150000）`，控制台 0 报错、无失败请求。

v1 只强制引擎**广告出来的事实**（`contextWindow` / `maxTokens` 上限），人工写的 `name`
不会被覆盖。

## 6. 配置与安装

`~/.dsh/profiles/web/cordis.patch.yml` 里的挂载点：

```yaml
- insert:
    - id: local-models-connect
      name: "/home/lcy/.dsh/plugins/local-models-connect.v1.mjs"
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

`providerId` / `displayName` / `apiKeyEnv` 支持**显式写空**表示「关掉/自动命名」
（写成 `''`、`false` 或 `null`）——「没写这个键」才走默认值。

### 安装后的那一次重启（重要）

`~/.dsh/profiles/web/package.json` 里的 `patchReload: "live"` 是 **2026-09-23 17:47:48** 写进去的，
而当前 `dsh web` 进程起于 **17:46:04** —— 早于它。也就是说**这个进程启动时并没有挂上
cordis-plugin-hmr**，`cordis.patch.yml` 的改动不会被重放（实测：改完 patch、请求新路由仍 404，
`journalctl -u dsh-web` 里一条相关日志都没有）。源码见 `lib/profile-boot-*.js`：
`patchReload === "live"` 的判断在 boot 时做一次，而且整段包在静默的 try/catch 里。

所以**装完必须重启一次**：

```bash
sudo systemctl restart dsh-web
sleep 8
curl -sS http://127.0.0.1:3080/local-models-connect/state | python3 -m json.tool
```

重启之后 `patchReload: live` 就生效了，此后**再改 patch 才是真的保存即生效**。
重启会打断正在进行的会话 —— 挑个空的时候做。新 URL（带 token）会写进
`~/.dsh/last-web-url.txt`。

### 回滚

```bash
cp ~/.dsh/profiles/web/cordis.patch.yml.bak-before-local-models-connect-<ts> \
   ~/.dsh/profiles/web/cordis.patch.yml
rm ~/.dsh/plugins/local-models-connect.v1.mjs
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
* 带 token 取 `/` → 渲染出的 HTML 里有注入的 `<style>`（`#lmc-panel …`）与面板脚本
  （`接入本地模型`、`__localModelsConnectInstalled`），且注入脚本里没有提前闭合的 `</script>`；
* `POST /run?dry=1` → `wrote: false`，一个字节都没写。

回归测试 62 条全过（纯逻辑 41 + 假引擎探测 5 + 端到端 15 + 实机冒烟 1）。
真机 192.168.0.119:8080 冒烟：识别为 `vllm`，`Qwen3.8-27B-Q6-dual-5060ti`，上下文 150000。

真机 `dsh web` 侧已确认 `dsh --profile web --dump-config` 的合成树里有新插件条目
（`id: local-models-connect`，config 解析成 6 个端口）、stderr 0 行警告；
但**要等 §6 那次重启**新路由才会出现在 3080 上。
