# DSH 侧的本地模型声明：为什么它会过期，以及那个「同步本地模型」按钮

## 问题

DSH 的 `llm-pi-ai` provider（`~/.dsh/settings.yaml` 里的 `qwen-local`）是**手写**的模型声明。
引擎换档（150K / 163840 / 172032 / 262144）之后，这份声明不会自己跟着变。2026-09-12 查到的实际状态：

```yaml
models:
  - id: Qwen3.8-27B-Q6-dual-5060ti
    name: Qwen3.8-27B (本地 2x5060Ti 192K)   # 名字里写着 192K
    contextWindow: 196608                     # 引擎实际只服务 150000
    maxTokens: 16384
```

**它不是"显示不准"这么轻**：`contextWindow` 直接决定自动压缩阈值 ——
`dsh-compaction-basic` 挂载时没有 config，取默认 `thresholdRatio = 0.8`：

| 声明值 | 自动压缩阈值 | 后果 |
|---|---|---|
| 196608（旧） | 157,286 | 会话长到 150,000–157,286 之间时，DSH 认为还能塞 → 请求被 vLLM 以超长拒掉 |
| 150000（现在） | 120,000 | 加 `maxTokens` 16,384 = 136,384 < 150,000 ✓ 安全 |
| 缺 `contextWindow` 时的回落 | 262,144 → 阈值 209,715 | 缺口更大 |

## 为什么 DSH 自带的「获取可用模型」救不了（源码级）

`dsh-client-ui-settings-models` 的 `fetchModels` → `api.llm.discoverModels` → `dsh-llm-pi-ai`
的 `discoverModels()`，三条都卡住：

1. `readListing()` 只认 `context_window` / `context_length`，而 vLLM 的 `/v1/models` 给的是
   **`max_model_len`** → 上下文长度根本读不到；
2. `adoptPicked()` 对**已存在**的 id 不替换（`byId.get(id) ?? adopt(candidate)`）→ 我们的 id
   已经在列表里，点了等于没点；
3. 即使采纳了新 id，缺 `contextWindow` 会回落到 `defaultContextWindow = 262144`，比 196608 更糟。

## 解决办法：一个宿主插件 + 一个按钮

源码（版本化副本）：[`../dsh-plugin/local-models-sync.v1.mjs`](../dsh-plugin/local-models-sync.v1.mjs)
安装到 `~/.dsh/plugins/local-models-sync.v1.mjs`，并在
`~/.dsh/profiles/web/cordis.patch.yml` 末尾挂一行。

它做两件事：

1. **`POST /local-models-sync/run`** —— 对每个**本地** provider（baseURL 命中
   `localhost` / `127.` / `10.` / `192.168.` / `172.16-31.`）拉 `{baseURL}/models`，
   用引擎自报的 `max_model_len` 改写该 provider 的 `models` 数组，走官方
   `settings.mutate`（schema 校验 + 原子写 + 写完热重载）。
   不动手的情形：端点不通、返回空、值没变、provider 不是本地端点 —— 一律只回报，不写文件。
2. **往页面 head 注入一小段脚本**（`webserver/index-inject`，和 `dsh-client-modules` 用同一个
   注入表），渲染右下角一个常驻按钮「同步本地模型」；点它调上面那个端点，把 diff 显示在旁边。
   纯 DOM + 内联样式，加 MutationObserver 防前端重渲染冲掉节点。

### 为什么是宿主插件，不是客户端插件（`dsh.client` 那套）

| | 客户端插件 | 宿主插件（本方案） |
|---|---|---|
| 装载 | `dsh-client-modules` 只扫**宿主 Loader 里已知**的包，包元数据按名字缓存且**永不过期** → 新增包**必须重启 `dsh web`** | 挂在 `cordis.patch.yml` 上，Cordis HMR 重放配置树 → **保存即生效** |
| 调设置 | 浏览器 → RPC（`settings.mutate` 属于客户端可调方法，但要自己拼协议） | 直接在宿主里 `ctx.settings.mutate()`，官方校验 + 原子写 |
| 跨域 | 需要端点给 CORS（vLLM 给的是 `access-control-allow-origin: *`，能用但脆） | 宿主侧 `fetch`，不涉及 CORS |

而那个要重启的进程**正是宿主本体** —— 重启它会打断正在进行的会话，所以这条路径从一开始就不该走。

### 命令行兜底

同一个实现，两个入口 —— 不用再写第二份脚本：

```bash
curl -s -X POST http://127.0.0.1:3080/local-models-sync/run | python3 -m json.tool
```

### 迭代注意

Cordis 只在 `name`（也就是文件路径）变化时才重新 `import` 模块（ESM 有模块缓存），
所以**改插件内容要装成新文件名**（`v2`、`v3`…）并改 `cordis.patch.yml` 里那一行；
只重存同一个文件不会重新加载。

### 回滚

```bash
cp ~/.dsh/profiles/web/cordis.patch.yml.bak-before-local-models-sync-<ts> \
   ~/.dsh/profiles/web/cordis.patch.yml     # 撤掉挂载点
rm ~/.dsh/plugins/local-models-sync.v1.mjs  # 可选
```

`settings.yaml` 的写入是幂等的（值没变就不写），回滚插件不会把设置改回去。

## 实测（2026-09-12）

* 挂载后 `dsh --profile web --dump-config` 里能看到该条目；
* `POST /local-models-sync/run` 返回 `changed: false`（值已正确）；
* 故意把 `contextWindow` 改成 `999999` 再 POST → 返回
  `{"field":"contextWindow","from":999999,"to":150000}`，`settings.yaml` 里随即变成 150000；
* 浏览器实地：页面快照里出现 `button "同步本地模型"`，点击后回显
  `qwen-local：已经是最新（Qwen3.8-27B-Q6-dual-5060ti=150000）`，控制台 0 报错、无失败请求。

插件只强制引擎**广告出来的事实**（`contextWindow` / `maxTokens` 上限），
人工写的 `name` 不会被覆盖 —— 上面那次测试里我特意改坏的 name 保持原样，是设计如此。
