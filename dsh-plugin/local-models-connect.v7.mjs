/**
 * local-models-connect —— 「接入本地大模型」：自动发现端口 / 模型名 / 上下文，自动接入，自带自检。
 *
 * ## 它和上一代（local-models-sync.v1）差在哪
 *
 * v1 只做一件事：把**已经手写在 `llm-pi-ai` 里**的本地 provider，按引擎自报的
 * `max_model_len` 修正 `contextWindow`。它不发现、不新建。
 *
 * 于是新环境上会撞到 v1 那句最难读的话：**「没有可同步的 provider」**。它的触发条件是确定的
 * —— `summarize` 只在 `report.results` 为空时打它，而 `syncAll` 对 `llm-pi-ai.providers` 的
 * **每一项都恰好产出一行**（没配 baseURL / 不是内网 / 拉不到 / 没变化 / 已更新，各一条），
 * 所以 `results` 为空 ⟺ **`providers` 是个空字典（一条都没配）**。
 *
 * 它既不是报错，也不等于「一切正常」，只说明「没东西可改」—— 而这恰恰是最需要有人替你发现
 * 端点的时候。歧义在于它不区分三种处境：① 一条 provider 都没配（**只有这一种**会打这句）
 * ② 配了但都不是内网端点 ③ 是内网但端点不通（②③ 其实 v1 会逐条列出跳过/失败原因）。
 * 新一代不再有「汇总成一句话」那条路径：报告恒定按「每个候选端点一行 + 每个 provider 一行
 * + 未采纳清单 + 是否写入」输出，所以「什么都没有」与「有但不通」在字面上就是不同的行。
 *
 * 这一代把口径从「同步已有声明」改成「**接入本地大模型**」：
 *
 *   1. **发现**：对配置里的种子主机 × 候选端口逐个探 `{origin}/v1/models`，
 *      读出「哪个端口、以什么 model name、有多少上下文」；
 *   2. **接入**：没有对应 provider 就**整条建出来**（baseURL / api / models / compat），
 *      已经有就只更新引擎说了算的字段 —— 所以新环境装上插件即可用，不必先手写声明；
 *   3. **自检**：一条只读通道，把「端点通不通 / 认出什么模型 / 声明和引擎对不对得上 /
 *      自动压缩阈值安不安全」逐条判出 ok·warn·fail，并在页面上常驻显示。
 *
 * ## 为什么上下文一定要从引擎自报值来（这是整个插件的承重点）
 *
 * `contextWindow` 决定 `dsh-compaction-basic` 的自动压缩阈值（0.8 × contextWindow）。
 * 声明大于引擎实际 = 压缩不来 = 会话长到中间某段被 vLLM 直接拒掉。DSH 自带的
 * 「获取可用模型」救不了这件事，三条都是源码级的（见 `docs/DSH-MODEL-SYNC.md`）：
 * `readListing` 只认 `context_window` / `context_length`（vLLM 给的是 `max_model_len`）；
 * `adoptPicked` 对已存在的 id 不替换；缺 `contextWindow` 会回落到 262144。
 *
 * 各引擎自报字段并不统一，本插件按可信度分两档收集（见 `readContext` / `readTrainContext`）：
 *
 *   **可用作声明**（服务端真实上限）
 *     · vLLM / SGLang  `/v1/models` → `max_model_len`
 *     · OpenAI 兼容网关 `/v1/models` → `context_length` / `context_window`
 *     · llama.cpp      `/props`     → `default_generation_settings.n_ctx`（`-c` 的实参）
 *     · Ollama         `/api/show`  → `model_info.*.context_length`
 *
 *   **只作证据、绝不写进声明**
 *     · llama.cpp `/v1/models` → `meta.n_ctx_train`（**训练**长度）
 *
 * 最后一档是本插件最容易写错的地方，所以单独拉出来：llama256 档服务端 `-c 262144`，
 * 而同一份权重 `n_ctx_train` 可能只有 32768 —— 取哪个都能过，取错方向一个让压缩过早
 * （长任务被无谓截断），另一个就是 v1 文档里那个「声明 196608 / 实际 150000」的事故。
 * 因此 `n_ctx_train` 只出现在自检报告的说明里，**永不参与合并**；上下文拿不到时宁可
 * 不建条目（见下）。
 *
 * ## 「拿不到上下文」怎么办：宁可少一条，不给错值
 *
 * 缺 `contextWindow` 的模型条目会让 pi-ai 回落到 `defaultContextWindow = 262144` ——
 * 这正是 v1 文档里最糟的那一档。所以自动新建时：
 *   · 新 provider 会带上 `defaultContextWindow` = 该端点**已知上下文的最小值**（兜底不再是 262144）；
 *   · 单个模型上下文未知且无安全兜底 → **不建这一条**，在报告里明说（要强行建就设
 *     `adoptUnknownContext: true`）。少一条能选是响的、可逆的；错一个值是哑的、会在
 *     150K 处炸会话。
 *
 * ## 为什么还是宿主插件（而不是客户端插件 / 一个外部脚本）
 *
 * 与 v1 同因，一字未改：客户端模块包元数据按名字缓存且永不过期，新增客户端包**必须重启
 * `dsh web`**，而那个进程正是宿主本体；宿主插件挂在 `cordis.patch.yml` 上，Cordis HMR
 * 重放配置树 → 保存即生效。而 `settings.mutate` 本来就是宿主侧服务，直接调 = 官方 schema
 * 校验 + 原子写 + 热重载，既不用拼客户端 RPC，也不涉及浏览器跨域。
 *
 * ## 写设置的边界（「自动」不等于「乱写」）
 *
 *   · 只碰命中 `PRIVATE_HOST` 的端点（内网 / 回环），远端 API 一律不探；
 *   · **只增不删**：`prune` 默认关，引擎不再广告的旧模型条目原样保留（可能是你手写的）；
 *   · 人工写的字段优先：`name` / `displayName` / `apiKeyEnv` / `compat` / `headers` 等
 *     已有值一律不被覆盖，只补引擎能证明的 `contextWindow` / `maxTokens`；
 *   · 已有 provider 按 **baseURL 精确匹配**复用，绝不改名、绝不新建重复路由；
 *   · 新建时才写 `apiKeyEnv`，且**只写配置里显式给的那个** —— `dsh-llm-pi-ai` 的
 *     `resolveApiKey` 在 apiKeyEnv 存在但凭据缺失时直接抛 `MISSING_CREDENTIAL`，
 *     给一个无鉴权的本地端点挂上凭据名，会把能用的端点变成用不了的；
 *   · 任一步失败只回报，不写文件；写失败也不回滚已验证的其它 provider。
 *
 * ## 迭代提示
 *
 * Cordis 只在 `name`（= 文件路径）变化时才重新 `import` 模块（ESM 有模块缓存），
 * 所以改内容要装成新文件名（v2、v3…）并改 `cordis.patch.yml` 里那一行；
 * 只重存同一个文件不会重新加载。
 */

export const name = 'local-models-connect'
export const inject = ['webServer', 'settings']

/** 版本号 —— 自检报告里回显，方便确认页面上跑的是哪一版。 */
const VERSION = '1.6.0'

/** 写入的设置命名空间（`llm-pi-ai` 的注册者见 dsh-llm-pi-ai）。 */
const NS = 'llm-pi-ai'

/** 三条路由：状态、接入、自检；外加 v1 的旧路径做兼容别名。 */
const ROUTE_STATE = '/local-models-connect/state'
const ROUTE_RUN = '/local-models-connect/run'
const ROUTE_SELFCHECK = '/local-models-connect/selfcheck'
/** 面板用的一条路：run + selfcheck 跑完，直接回「排版好的三块视图」（见 buildPanelView）。 */
const ROUTE_PANEL = '/local-models-connect/panel'
const LEGACY_ROUTE_RUN = '/local-models-sync/run'

/** 只碰内网 / 回环端点，避免拿没配 key 的远端 API 去试。 */
const PRIVATE_HOST = /^(localhost|127\.|\[?::1\]?|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/i

/** 可作为声明的上下文长度字段，按可信度从高到低。 */
const CONTEXT_PATHS = [
  ['max_model_len'],                        // vLLM / SGLang
  ['context_length'],                       // OpenAI 兼容通用；Ollama model_info
  ['context_window'],
  ['max_context_length'],
  ['n_ctx'],
  ['default_generation_settings', 'n_ctx'], // llama.cpp 服务端 `-c` 实参
  ['max_input_tokens'],
  ['limit', 'context']
]
/** **只作证据**的训练长度：出现在报告里，永不参与合并。理由见文件头。 */
const TRAIN_CONTEXT_PATHS = [['meta', 'n_ctx_train'], ['n_ctx_train']]
/** 单次最大输出 token 的候选字段。 */
const MAXTOKENS_PATHS = [
  ['max_output_tokens'],
  ['max_completion_tokens'],
  ['max_tokens'],
  ['limit', 'output'],
  ['top_provider', 'max_completion_tokens']
]

/**
 * 默认配置。`cordis.patch.yml` 里那个 `config:` 块按字段覆盖这里。
 * 全部字段都有默认值 —— 一行 `config` 都不写也能工作（这是「新环境装上就能用」的前提）。
 */
const DEFAULT_CONFIG = {
  /** 种子主机：只扫这里列出的主机（要改就改这一行，默认 119）。 */
  hosts: ['192.168.0.119'],
  /** 每个主机上依次试的端口。 */
  ports: [8080, 8000, 30000, 8081, 11434, 1234],
  /** 除种子主机外，是否也顺带刷新「已配 provider 里的内网端点」（只读它们已有的 baseURL）。 */
  includeConfigured: true,
  /**
   * 第一个命中的端点用这个 provider 路由名（留空则按 `local-<ip>-<port>` 自动命名）。
   * 默认指向 qwen-local：这个路由名在文档、脚本、铁律 11 里到处被引用。
   */
  providerId: 'qwen-local',
  /** 新建 provider 时的显示名（留空则用 `<路由名>（本地）`）。 */
  displayName: 'Qwen3.8-27B 本地',
  /**
   * 新建 provider 时挂的凭据名。**默认空 = 不挂**。
   *
   * 这是「新环境装上就能用」的关键一条：`dsh-llm-pi-ai` 的 `resolveApiKey` 在
   * `apiKeyEnv` 存在、而凭据服务与环境变量里都没有该值时**直接抛 MISSING_CREDENTIAL**
   * —— 一个无鉴权的本地端点被挂上 `QWEN_LOCAL_API_KEY`，在配了那份凭据的机器上能用、
   * 换台机器就整个模型不可用。已存在的 provider 不会被改（见 mergeProfile），所以
   * 本机那份手写的 qwen-local 照旧。
   */
  apiKeyEnv: '',
  /** 插件加载后是否自动跑一次「发现 + 接入」（这是「新环境装上就能识别」的那一步）。 */
  autoProbe: true,
  /** 自动跑时是否真的写设置。设 false = 只发现、只报告，等价于一次开机自检。 */
  autoApply: true,
  /** 引擎不再广告的旧模型条目是否删掉。默认 false（只增不删）。 */
  prune: false,
  /** 上下文未知的模型是否照建（用兜底值）。默认 false —— 从源头挡住 262144 那类错值。 */
  adoptUnknownContext: false,
  /** 单个探测请求的超时。 */
  probeTimeoutMs: 1500,
  /** 并发探测数（种子端点少，6 足够且不打扰网络）。 */
  concurrency: 6,
  /** 本地模型冷启动慢，流空闲超时给宽一点（沿用 qwen-local 的 600000）。 */
  streamIdleTimeoutMs: 600000,
  /** 引擎没说最大输出时，新模型条目用的 maxTokens。 */
  defaultMaxTokens: 16384,
  /** 是否允许探测公网端点。默认 false（本插件的用途就是内网本地模型）。 */
  allowPublic: false,
  /** 自动探测的延迟，让 dsh web 先把页面服务起来。 */
  autoProbeDelayMs: 4000,
  /**
   * 详细报告自动收起的时间（毫秒）。0 = 不自动收（只留手动点掉）。
   * 报告是 fixed 定位、不占布局，但一块大字盖在页面上同样烦人 —— 默认 12 秒自己消失。
   */
  autoHideMs: 12000
}

const msg = (error) => (error && error.message) || String(error)

/**
 * 把 fetch 的失败说清楚。`fetch` 自己只给一句没用的 `fetch failed`，
 * 真正的原因在 `error.cause.code` 里 —— 而那几个码正好分出完全不同的处境：
 *   · `ECONNREFUSED` → 主机是通的，**端口上没有进程在听**（服务没起 / 正在启动或重启窗口）
 *   · `ETIMEDOUT` / `EHOSTUNREACH` / `ENETUNREACH` → 连不到主机或网络
 *   · `ENOTFOUND` → 名字解析不了
 * 2026-09-24 就是这么被问住的：自检只说"一个候选端点都没通"，而当时服务其实正在重启（
 * 09:16:00 health down → 09:16:51 UP），分不出来是"机器没了"还是"刚好在重启"。
 */
function describeFetchError(error) {
  const code = error?.cause?.code ?? error?.code
  const detail = error?.cause?.message ?? error?.message ?? String(error)
  return code === undefined ? detail : `${code}：${detail}`
}

/** 401/500 这类"连上了但回错"的错误带 status，要跟连接层失败分开。 */
const errorCodeOf = (error) => error?.cause?.code ?? error?.code ?? null

// ─────────────────────────────────────────────────────────────────────────
// 配置解析（手写校验：宿主插件的第二参数是 patch yml 里的裸对象，没有 schema 兜底）
// ─────────────────────────────────────────────────────────────────────────

const asArray = (value, fallback) => {
  if (Array.isArray(value)) return value
  if (typeof value === 'string' && value.trim() !== '') {
    return value.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean)
  }
  return fallback
}

const asString = (value, fallback = '') =>
  (typeof value === 'string' && value.trim() !== '' ? value.trim() : fallback)

/**
 * 「没写」与「显式写空」是两件事：
 *   · 键不存在 → 用默认值；
 *   · 写成 `''` / `false` / `null` → **就是空**（providerId 空 = 自动命名，apiKeyEnv 空 = 不挂凭据）。
 * 否则「留空以关闭」这个最自然的写法会静默回落到默认值 —— 冒烟测试里就是这么被抓到的。
 */
const asOptionalString = (source, key, fallback) => {
  const value = source[key]
  if (value === false || value === null) return ''
  if (typeof value === 'string') return value.trim()
  return fallback
}

const asPositiveInt = (value, fallback) => {
  const n = Number(value)
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : fallback
}

const asBool = (value, fallback) => (typeof value === 'boolean' ? value : fallback)

/** 把 patch yml 的 config 收成一份带默认值、字段类型可靠的配置。 */
export function resolveConfig(raw) {
  const source = raw && typeof raw === 'object' ? raw : {}
  const ports = asArray(source.ports, DEFAULT_CONFIG.ports)
    .map((p) => Number(p))
    .filter((p) => Number.isInteger(p) && p > 0 && p < 65536)
  const hosts = asArray(source.hosts, DEFAULT_CONFIG.hosts)
    .map((h) => String(h).trim())
    .filter(Boolean)
  return {
    hosts: hosts.length > 0 ? hosts : [...DEFAULT_CONFIG.hosts],
    ports: ports.length > 0 ? ports : [...DEFAULT_CONFIG.ports],
    includeConfigured: asBool(source.includeConfigured, DEFAULT_CONFIG.includeConfigured),
    providerId: asOptionalString(source, 'providerId', DEFAULT_CONFIG.providerId),
    displayName: asOptionalString(source, 'displayName', DEFAULT_CONFIG.displayName),
    apiKeyEnv: asOptionalString(source, 'apiKeyEnv', DEFAULT_CONFIG.apiKeyEnv),
    autoProbe: asBool(source.autoProbe, DEFAULT_CONFIG.autoProbe),
    autoApply: asBool(source.autoApply, DEFAULT_CONFIG.autoApply),
    prune: asBool(source.prune, DEFAULT_CONFIG.prune),
    adoptUnknownContext: asBool(source.adoptUnknownContext, DEFAULT_CONFIG.adoptUnknownContext),
    probeTimeoutMs: asPositiveInt(source.probeTimeoutMs, DEFAULT_CONFIG.probeTimeoutMs),
    concurrency: asPositiveInt(source.concurrency, DEFAULT_CONFIG.concurrency),
    streamIdleTimeoutMs: asPositiveInt(source.streamIdleTimeoutMs, DEFAULT_CONFIG.streamIdleTimeoutMs),
    defaultMaxTokens: asPositiveInt(source.defaultMaxTokens, DEFAULT_CONFIG.defaultMaxTokens),
    allowPublic: asBool(source.allowPublic, DEFAULT_CONFIG.allowPublic),
    autoProbeDelayMs: asPositiveInt(source.autoProbeDelayMs, DEFAULT_CONFIG.autoProbeDelayMs),
    autoHideMs: Number.isFinite(Number(source.autoHideMs)) && Number(source.autoHideMs) >= 0
      ? Math.floor(Number(source.autoHideMs))
      : DEFAULT_CONFIG.autoHideMs
  }
}

// ─────────────────────────────────────────────────────────────────────────
// 纯逻辑：URL / 字段读取 / 列表解析 / 合并 / diff  —— 不依赖 Cordis，可直接单测
// ─────────────────────────────────────────────────────────────────────────

/** 从 baseURL 剥出 `http://host:port`，非法则 null。 */
export function originOf(baseURL) {
  try {
    const url = new URL(baseURL)
    return `${url.protocol}//${url.host}`
  } catch {
    return null
  }
}

/** `/v1` 归一：配置里写不写 `/v1` 都能用。 */
export function normalizeBaseURL(baseURL, origin) {
  const raw = asString(baseURL)
  const base = raw !== '' ? raw : `${origin}/v1`
  return base.replace(/\/+$/, '')
}

export function isPrivate(baseURL) {
  const origin = originOf(baseURL)
  if (origin === null) return false
  try {
    return PRIVATE_HOST.test(new URL(origin).host)
  } catch {
    return false
  }
}

/** 按候选路径逐个取值，返回第一个正整数。 */
function readNumber(source, paths) {
  for (const path of paths) {
    let cursor = source
    for (const key of path) {
      if (cursor === null || typeof cursor !== 'object') { cursor = undefined; break }
      cursor = cursor[key]
    }
    const n = typeof cursor === 'string' ? Number(cursor) : cursor
    if (typeof n === 'number' && Number.isFinite(n) && n > 0) return Math.floor(n)
  }
  return undefined
}

/** 引擎自报的、**可写进声明**的上下文长度。 */
export function readContext(raw) {
  return readNumber(raw, CONTEXT_PATHS)
}

/** 模型的训练长度 —— 只作证据，永不参与合并（见文件头）。 */
export function readTrainContext(raw) {
  return readNumber(raw, TRAIN_CONTEXT_PATHS)
}

/** 引擎自报的最大输出 token。 */
export function readMaxTokens(raw) {
  return readNumber(raw, MAXTOKENS_PATHS)
}

/**
 * 认引擎。只看**证据**，不猜：
 *   · `owned_by` 自报；· 出现 `max_model_len` = vLLM 家族；· `/props` 通 = llama.cpp。
 */
export function detectEngine(entries) {
  const owned = new Set(entries.map((e) => String(e.ownedBy ?? '').toLowerCase()).filter(Boolean))
  if (owned.has('vllm')) return 'vllm'
  if (owned.has('sglang')) return 'sglang'
  if (entries.some((e) => e.sawMaxModelLen)) return 'vllm'
  if (entries.some((e) => e.sawProps)) return 'llama.cpp'
  return 'unknown'
}

/**
 * 读一份模型列表响应。`data` 数组优先，其次 `models` 映射（键即端点认的 id）——
 * 与 `dsh-llm-pi-ai` 的 `readListing` 同口径，免得同一份响应两边读出不同结果。
 * @returns 归一后的条目数组；结构完全不是列表时抛错（由调用方转成一条 error）。
 */
export function parseListing(body) {
  const listing = body
  let listed
  const data = listing?.data
  if (Array.isArray(data)) {
    listed = data.map((raw) => ({ raw }))
  } else {
    const models = listing?.models
    if (models === null || typeof models !== 'object' || Array.isArray(models)) {
      throw new Error('返回里既没有 data 数组也没有 models 映射')
    }
    listed = Object.entries(models)
      .filter(([, raw]) => raw !== null && typeof raw === 'object' && !Array.isArray(raw))
      .map(([key, raw]) => ({ key, raw }))
  }

  const entries = []
  for (const { key, raw } of listed) {
    const entry = raw ?? {}
    const id = asString(
      typeof key === 'string' && key !== '' ? key : entry.id ?? entry.model ?? entry.name
    )
    if (id === '') continue // 一条坏行不该毁掉整个端点
    entries.push({
      id,
      name: asString(entry.name ?? entry.display_name ?? entry.displayName, id),
      contextWindow: readContext(entry),
      trainContext: readTrainContext(entry),
      maxTokens: readMaxTokens(entry),
      ownedBy: asString(entry.owned_by ?? entry.ownedBy),
      // 证据位：这两条参与 detectEngine，不进最终声明。
      sawMaxModelLen: readNumber(entry, [['max_model_len']]) !== undefined,
      sawProps: false
    })
  }
  return entries
}

/**
 * Ollama 的 `/api/show`：`model_info` 里 `<family>.context_length` 是主来源，
 * `parameters` 里的 `num_ctx`（Modelfile 里设的）次之。
 */
export function contextFromShow(show) {
  const direct = readContext(show)
  if (direct !== undefined) return direct
  const info = show?.model_info
  if (info !== null && typeof info === 'object' && !Array.isArray(info)) {
    let best
    for (const [key, value] of Object.entries(info)) {
      if (!/\.context_length$/i.test(key)) continue
      const n = Number(value)
      if (Number.isFinite(n) && n > 0) best = best === undefined ? Math.floor(n) : Math.max(best, Math.floor(n))
    }
    if (best !== undefined) return best
  }
  const parameters = show?.parameters
  if (typeof parameters === 'string') {
    const hit = /(?:^|\n)\s*num_ctx\s+(\d+)/.exec(parameters)
    if (hit !== null) {
      const n = Number(hit[1])
      if (Number.isFinite(n) && n > 0) return Math.floor(n)
    }
  }
  return undefined
}

/** `local-192-168-0-119-8000` —— 稳定、可读、能当路由名。 */
export function autoProviderId(host, port, prefix = 'local') {
  const slug = String(host).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '')
  return `${prefix}-${slug || 'host'}-${port}`
}

/** 该端点所有**已知**上下文里最小的那个 —— 新 provider 的安全兜底值。 */
export function safeDefaultContext(models) {
  const known = models.map((m) => m.contextWindow).filter((n) => Number.isInteger(n) && n > 0)
  return known.length === 0 ? undefined : Math.min(...known)
}

/**
 * 合并一条 provider。
 * **机器说了算的只有 contextWindow / maxTokens，其余已有值一律不动。**
 * @returns `{ profile, skipped }`；`skipped` 是「拒绝自动新建」的模型条目（上下文未知）。
 */
export function mergeProfile(discovered, existing, naming, config) {
  const previous = existing && typeof existing === 'object' ? existing : {}
  const isNew = existing === undefined
  const profile = { ...previous }
  const skipped = []
  const fallbackContext = safeDefaultContext(discovered.models)

  if (isNew) {
    // 新建时只写「不写就不对」的字段。
    profile.api = 'openai-completions'
    const compat = localCompat(discovered.engine)
    if (compat !== undefined) profile.compat = compat
    profile.streamIdleTimeoutMs = config.streamIdleTimeoutMs
    if (naming.apiKeyEnv !== '') profile.apiKeyEnv = naming.apiKeyEnv
    // 兜底不再是 262144：缺 contextWindow 的条目最多按这个端点已知的最小上下文算。
    if (fallbackContext !== undefined) profile.defaultContextWindow = fallbackContext
  }
  profile.baseURL = discovered.baseURL
  if (asString(profile.displayName) === '' && naming.displayName !== '') profile.displayName = naming.displayName

  // ── 模型数组：并集语义 ──
  const declared = Array.isArray(previous.models) ? previous.models : []
  const byId = new Map(declared.filter((m) => m && typeof m === 'object' && m.id).map((m) => [m.id, m]))
  const merged = []
  const seen = new Set()

  for (const found of discovered.models) {
    const prior = byId.get(found.id)
    seen.add(found.id)

    if (prior === undefined) {
      const cw = found.contextWindow ?? (fallbackContext !== undefined && config.adoptUnknownContext ? fallbackContext : undefined)
      if (cw === undefined) {
        skipped.push({ id: found.id, reason: found.trainContext === undefined
          ? '引擎没报上下文长度'
          : `引擎只报了训练长度 ${found.trainContext}（不等于服务端上限），不猜` })
        continue
      }
      const mt = found.maxTokens ?? config.defaultMaxTokens
      merged.push({ id: found.id, name: found.name || found.id, contextWindow: cw, maxTokens: Math.min(mt, cw) })
      continue
    }

    const entry = { ...prior }
    if (asString(entry.name) === '') entry.name = found.name || found.id
    const cw = found.contextWindow ?? (Number.isInteger(entry.contextWindow) ? entry.contextWindow : undefined)
    if (cw !== undefined) {
      entry.contextWindow = cw
      // maxTokens 是**请求时**的最大输出，必须留在上下文之内，否则长提示必然越界。
      const mt = Number.isInteger(entry.maxTokens) ? entry.maxTokens : (found.maxTokens ?? config.defaultMaxTokens)
      entry.maxTokens = Math.min(mt, cw)
    } else if (entry.maxTokens === undefined && found.maxTokens !== undefined) {
      entry.maxTokens = found.maxTokens
    }
    merged.push(entry)
  }

  // 只增不删：引擎没广告的旧条目保留（可能是手写但当前档位暂未服务的）。
  if (!config.prune) {
    for (const [id, entry] of byId) if (!seen.has(id)) merged.push(entry)
  }

  profile.models = merged
  return { profile, skipped }
}

/** 本地端点的 compat 偏置：这些引擎不认 `max_completion_tokens`。 */
function localCompat(engine) {
  if (engine === 'vllm' || engine === 'sglang' || engine === 'llama.cpp') {
    return { maxTokensField: 'max_tokens' }
  }
  return undefined
}

/** 逐字段对比两条 provider，产出给人看的 diff 行（只比本插件会碰的字段）。 */
export function diffProfile(before, after) {
  const changes = []
  const created = before === undefined
  const prev = before ?? {}

  for (const field of ['baseURL', 'displayName', 'api', 'streamIdleTimeoutMs', 'defaultContextWindow']) {
    if (created) {
      if (after[field] !== undefined) changes.push({ field, from: null, to: after[field] })
    } else if (prev[field] !== after[field]) {
      changes.push({ field, from: prev[field] ?? null, to: after[field] ?? null })
    }
  }

  const beforeModels = new Map((Array.isArray(prev.models) ? prev.models : []).map((m) => [m?.id, m]))
  const afterModels = new Map((after.models ?? []).map((m) => [m.id, m]))

  for (const [id, next] of afterModels) {
    const prior = beforeModels.get(id)
    if (prior === undefined) {
      changes.push({ id, field: '(新增模型)', from: null, to: next.contextWindow ?? null })
      continue
    }
    for (const field of ['name', 'contextWindow', 'maxTokens']) {
      if (prior[field] !== next[field]) {
        changes.push({ id, field, from: prior[field] ?? null, to: next[field] ?? null })
      }
    }
  }
  for (const [id, prior] of beforeModels) {
    if (!afterModels.has(id)) {
      changes.push({ id, field: '(移除模型)', from: prior.contextWindow ?? null, to: null })
    }
  }
  return changes
}

/** 自动压缩阈值安全性：0.8 × contextWindow 之上还要塞得下 maxTokens。 */
export function compactionHeadroom(contextWindow, maxTokens, ratio = 0.8) {
  if (!Number.isFinite(contextWindow) || contextWindow <= 0) return null
  const threshold = Math.floor(contextWindow * ratio)
  const mt = Number.isFinite(maxTokens) && maxTokens > 0 ? maxTokens : 0
  return { ratio, threshold, maxTokens: mt, headroom: contextWindow - threshold - mt }
}

/**
 * 由探测结果算出「这个端点该用哪个 provider 路由名」。
 * 优先级：baseURL 已存在的 provider > 命名配置（只给第一个命中者）> 自动名。
 */
export function pickProviderId(discovered, existingByOrigin, naming, config, taken) {
  const existing = existingByOrigin.get(discovered.origin)
  if (existing !== undefined) return existing

  if (naming.providerId !== '' && !taken.has(naming.providerId)) return naming.providerId

  let host = discovered.host
  try { host = new URL(discovered.origin).hostname } catch { /* 用 discovered.host 兜底 */ }
  const base = autoProviderId(host, discovered.port)
  let candidate = base
  let n = 2
  while (taken.has(candidate)) candidate = `${base}-${n++}`
  return candidate
}

// ─────────────────────────────────────────────────────────────────────────
// 网络探测
// ─────────────────────────────────────────────────────────────────────────

async function getJson(url, timeoutMs) {
  const response = await fetch(url, {
    method: 'GET',
    headers: { accept: 'application/json' },
    signal: AbortSignal.timeout(timeoutMs)
  })
  const status = response.status
  if (!response.ok) {
    const error = new Error(`${url} 回了 ${status}`)
    error.status = status
    throw error
  }
  return { status, body: await response.json() }
}

/** 有并发上限的 map —— 一台机器超时不该拖住其余全部。 */
async function mapLimited(items, limit, worker) {
  const out = new Array(items.length)
  let cursor = 0
  const runners = new Array(Math.min(limit, items.length)).fill(null).map(async () => {
    for (;;) {
      const index = cursor++
      if (index >= items.length) return
      out[index] = await worker(items[index], index)
    }
  })
  await Promise.all(runners)
  return out
}

/**
 * 探一个端点：`/v1/models` 拿模型与上下文，缺上下文时按引擎补 `/props` 或 `/api/show`。
 * 任何失败都收成 `{ reachable:false, error }`，绝不抛。
 */
export async function probeTarget(target, config) {
  const started = Date.now()
  const base = { baseURL: target.baseURL, origin: target.origin, host: target.host, port: target.port }
  let listing
  try {
    listing = await getJson(`${target.baseURL}/models`, config.probeTimeoutMs)
  } catch (error) {
    return {
      ...base,
      reachable: false,
      status: error.status ?? null,
      code: errorCodeOf(error),
      latencyMs: Date.now() - started,
      error: error.status === undefined ? describeFetchError(error) : msg(error)
    }
  }

  const latencyMs = Date.now() - started
  let models
  try {
    models = parseListing(listing.body)
  } catch (error) {
    return { ...base, reachable: true, status: listing.status, latencyMs, error: `列表读不动：${msg(error)}` }
  }
  if (models.length === 0) {
    return { ...base, reachable: true, status: listing.status, latencyMs, models: [], engine: 'unknown' }
  }

  let engine = detectEngine(models)

  // 上下文缺失时补一次。llama.cpp 只有 `/props` 说得出服务端 `-c` 实参
  // （`/v1/models` 的 `n_ctx_train` 是训练长度，已在 parseListing 里单独隔离）。
  if (models.some((m) => m.contextWindow === undefined)) {
    const props = await tryProps(target.origin, config)
    if (props !== null) {
      engine = 'llama.cpp'
      const nCtx = readNumber(props, [['default_generation_settings', 'n_ctx'], ['n_ctx']])
      for (const model of models) {
        if (model.contextWindow === undefined && nCtx !== undefined) model.contextWindow = nCtx
        model.sawProps = true
      }
    }
  }
  if (models.some((m) => m.contextWindow === undefined) && engine === 'unknown') {
    // 可能是 Ollama：`/api/show` 逐模型问一次。
    for (const model of models) {
      if (model.contextWindow !== undefined) continue
      const show = await tryShow(target.origin, model.id, config)
      if (show !== null) {
        engine = 'ollama'
        model.contextWindow = contextFromShow(show) ?? model.contextWindow
      }
    }
  }

  return {
    ...base,
    reachable: true,
    status: listing.status,
    latencyMs,
    engine,
    models: models.map(({ id, name, contextWindow, trainContext, maxTokens }) => ({
      id,
      name,
      ...(contextWindow === undefined ? {} : { contextWindow }),
      ...(trainContext === undefined ? {} : { trainContext }),
      ...(maxTokens === undefined ? {} : { maxTokens })
    }))
  }
}

async function tryProps(origin, config) {
  try {
    return (await getJson(`${origin}/props`, config.probeTimeoutMs)).body
  } catch {
    return null
  }
}

async function tryShow(origin, modelId, config) {
  try {
    const response = await fetch(`${origin}/api/show`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', accept: 'application/json' },
      body: JSON.stringify({ model: modelId, name: modelId }),
      signal: AbortSignal.timeout(config.probeTimeoutMs)
    })
    if (!response.ok) return null
    return await response.json()
  } catch {
    return null
  }
}

/** `/health`：vLLM / SGLang / llama.cpp 都有；拿不到就记 ok:false，不算致命。 */
async function probeHealth(origin, config) {
  try {
    const response = await fetch(`${origin}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(config.probeTimeoutMs)
    })
    return { ok: response.ok, status: response.status }
  } catch (error) {
    return { ok: false, status: null, error: msg(error) }
  }
}

/** 凭据名必须长成 POSIX 环境变量名，否则 `credentialRef` 会抛（与 dsh-credentials 同一条正则）。 */
const CREDENTIAL_REF = /^[A-Za-z_][A-Za-z0-9_]*$/

/**
 * 判一个 `apiKeyEnv` 取不取得到值。口径与 `dsh-llm-pi-ai` 的 `resolveApiKey` 对齐
 * （先凭据服务、后启动环境），因为它抛不抛 MISSING_CREDENTIAL 就是这个判据。
 * @returns `{ state: 'none'|'ok'|'missing'|'unknown', from?, note? }`
 */
export async function credentialStatus(ctx, ref) {
  if (ref === '') return { state: 'none' }
  if (!CREDENTIAL_REF.test(ref)) return { state: 'missing', note: `"${ref}" 不是合法的凭据名（须匹配 ${String(CREDENTIAL_REF)}）` }

  let serviceChecked = false
  try {
    const credentials = ctx?.get?.('credentials')
    if (credentials !== undefined && typeof credentials.resolve === 'function') {
      serviceChecked = true
      const hit = await credentials.resolve(ref)
      if (hit !== undefined && typeof hit.value === 'string' && hit.value.length > 0) {
        return { state: 'ok', from: '凭据服务' }
      }
    }
  } catch (error) {
    return { state: 'unknown', note: `查凭据服务出错：${msg(error)}` }
  }

  const ambient = process.env[ref]
  if (typeof ambient === 'string' && ambient.length > 0) return { state: 'ok', from: '环境变量' }
  if (!serviceChecked) return { state: 'unknown', note: '没有可用的凭据服务，只查了环境变量' }
  return { state: 'missing', note: '凭据服务与环境变量里都没有取到值' }
}

// ─────────────────────────────────────────────────────────────────────────
// 候选端点收集
// ─────────────────────────────────────────────────────────────────────────

/**
 * 种子主机 × 端口，去重后按「用户写的顺序」返回 —— 顺序有意义：
 * 第一个命中的端点会拿到固定的 `providerId`。
 */
export function candidateTargets(config, section) {
  const targets = []
  const seen = new Set()

  const push = (host, port) => {
    const origin = `http://${host}:${port}`
    if (seen.has(origin)) return
    seen.add(origin)
    targets.push({ baseURL: `${origin}/v1`, origin, host, port })
  }

  for (const host of config.hosts) for (const port of config.ports) push(host, port)

  if (config.includeConfigured) {
    // 已经配好的内网 provider：把它的 host:port 也纳入刷新（用户已经声明过它们，
    // 不算「扫新主机」）。公网端点不纳。
    for (const provider of Object.values(section?.providers ?? {})) {
      const origin = originOf(asString(provider?.baseURL))
      if (origin === null || !isPrivate(origin)) continue
      try {
        const url = new URL(origin)
        push(url.hostname, Number(url.port || (url.protocol === 'https:' ? 443 : 80)))
      } catch { /* originOf 已经挡过坏输入，这里只是不给它第二次机会 */ }
    }
  }
  return targets
}

/** 现有 provider 按 origin 建索引，用来「复用而不是新建」。 */
export function indexProvidersByOrigin(section) {
  const map = new Map()
  for (const [providerId, provider] of Object.entries(section?.providers ?? {})) {
    const origin = originOf(asString(provider?.baseURL))
    if (origin !== null && !map.has(origin)) map.set(origin, providerId)
  }
  return map
}

// ─────────────────────────────────────────────────────────────────────────
// 接入（run）
// ─────────────────────────────────────────────────────────────────────────

/** 单飞：自动探测与手动点击可能重叠，重叠会让同一次改动被写两遍。 */
let inFlight = null
/** 最近一次的报告缓存，供页面 `/state` 免网络读取。 */
const cache = { run: null, selfCheck: null }

async function probeAll(config, section) {
  const targets = candidateTargets(config, section)
  const results = await mapLimited(targets, config.concurrency, (target) => {
    if (!config.allowPublic && !isPrivate(target.baseURL)) {
      return { ...target, skipped: '不是内网端点（allowPublic=false）' }
    }
    return probeTarget(target, config)
  })
  return { targets, results }
}

/**
 * 发现 + 接入。写设置是逐 provider 的：一条失败不影响其余，且只回报不回滚。
 * @param options.dryRun - true 时只算不写（`?dry=1`）。
 */
async function runConnect(ctx, config, options = {}) {
  const section = ctx.settings.section(NS) ?? {}
  const dryRun = options.dryRun === true
  const { targets, results } = await probeAll(config, section)

  const existingByOrigin = indexProvidersByOrigin(section)
  const taken = new Set(Object.keys(section.providers ?? {}))
  const naming = { providerId: config.providerId, displayName: config.displayName, apiKeyEnv: config.apiKeyEnv }
  const providers = []
  let wrote = false

  for (const found of results) {
    if (found.skipped !== undefined || found.reachable !== true) continue
    if (!Array.isArray(found.models) || found.models.length === 0) continue

    const providerId = pickProviderId(found, existingByOrigin, naming, config, taken)
    const existing = section.providers?.[providerId]
    const displayName = asString(existing?.displayName) !== ''
      ? existing.displayName
      : (providerId === config.providerId && config.displayName !== '' ? config.displayName : `${providerId}（本地）`)

    const { profile, skipped } = mergeProfile(found, existing, { ...naming, displayName }, config)
    const changes = diffProfile(existing, profile)
    const action = existing === undefined ? 'created' : (changes.length > 0 ? 'updated' : 'unchanged')
    // 面板要能说「已是最新（N 个模型）」，所以把结果里的模型条数一并带出来。
    const modelCount = (profile.models ?? []).length

    if (skipped.length > 0) {
      // 全部被拒 → 不建一个空 provider（空 models 会被 llm-pi-ai 判为「resolves no models」直接报错）。
      if ((profile.models ?? []).length === 0) {
        providers.push({
          provider: providerId,
          baseURL: found.baseURL,
          action: 'skipped',
          changes: [],
          modelCount,
          skipped,
          error: `广告的 ${found.models.length} 个模型上下文都读不到，未建 provider（要强行建请设 adoptUnknownContext: true）`
        })
        continue
      }
    }

    // 先判「有没有变化」，再判 dry-run —— 否则 dry-run 会把已经最新的 provider
    // 也报成「待写入」，读的人以为每次都有东西要改。
    if (changes.length === 0) {
      providers.push({ provider: providerId, baseURL: found.baseURL, action, changes, modelCount, ...(skipped.length ? { skipped } : {}) })
      taken.add(providerId)
      continue
    }
    if (dryRun) {
      providers.push({ provider: providerId, baseURL: found.baseURL, action: 'would-change', changes, modelCount, ...(skipped.length ? { skipped } : {}) })
      taken.add(providerId)
      continue
    }

    try {
      await ctx.settings.mutate(NS, [{ op: 'set', path: ['providers', providerId], value: profile }])
      wrote = true
      taken.add(providerId)
      providers.push({ provider: providerId, baseURL: found.baseURL, action, changes, modelCount, ...(skipped.length ? { skipped } : {}) })
    } catch (error) {
      providers.push({ provider: providerId, baseURL: found.baseURL, action: 'failed', changes, modelCount, error: msg(error) })
    }
  }

  const report = {
    ok: providers.every((p) => p.action !== 'failed'),
    at: new Date().toISOString(),
    version: VERSION,
    dryRun,
    wrote,
    candidates: targets.map((t) => `${t.host}:${t.port}`),
    targets: results,
    providers
  }
  cache.run = report
  return report
}

// ─────────────────────────────────────────────────────────────────────────
// 自检（selfcheck）—— 只读，一个字节都不写
// ─────────────────────────────────────────────────────────────────────────

/**
 * 逐条判。每条 check 都有稳定的 id、ok/warn/fail 三档、以及给人看的一句话。
 * 判定口径全部落在「可复现的事实」上，不做主观评价。
 */
async function selfCheck(ctx, config) {
  const section = ctx.settings.section(NS) ?? {}
  const checks = []
  const add = (id, level, title, detail) => checks.push({ id, level, title, detail })

  add('plugin', 'ok', `插件已加载 v${VERSION}`,
    `命名空间 ${NS}；自动探测 ${config.autoProbe ? '开' : '关'}，自动写入 ${config.autoApply ? '开' : '关'}`)

  const { targets, results } = await probeAll(config, section)
  const live = results.filter((r) => r.reachable === true)
  const withModels = live.filter((r) => Array.isArray(r.models) && r.models.length > 0)

  add('targets', live.length > 0 ? 'ok' : 'fail', `候选端点 ${targets.length} 个，存活 ${live.length} 个`,
    targets.map((t) => `${t.host}:${t.port}`).join('、') || '（没有任何候选）')
  if (live.length === 0) {
    // 「没通」有三种完全不同的处境，而连接层的错误码正好把它们分开。
    // 不区分的话，读者只能看到"一个都没通"，然后对着"服务应该还在呀"发呆。
    const codes = [...new Set(results.map((r) => r.code).filter(Boolean))]
    const onlyRefused = codes.length > 0 && codes.every((c) => c === 'ECONNREFUSED')
    const hint = codes.length === 0
      ? '端点都没回应，原因未识别'
      : onlyRefused
        ? '端口拒绝连接（ECONNREFUSED）：**主机是通的，但那些端口上没有进程在监听** —— 服务没起，'
          + '或正好落在启动/重启窗口里（冷启动 45–60s，systemd 重新拉起还要 +15s）'
        : `连不到主机或网络（${codes.join('、')}）：查网段、路由与防火墙`
    add('reachability', 'fail', '一个候选端点都没通',
      `种子主机 ${config.hosts.join('、')}，端口 ${config.ports.join('、')}；${hint}。`
      + `用 srvctl run gpu "bash ~/deploy-5060ti/modelctl status" 看服务本身`)
  }

  const targetRows = []
  for (const result of results) {
    if (result.skipped !== undefined || result.reachable !== true) {
      targetRows.push({ ...result, health: null })
      continue
    }
    const health = await probeHealth(result.origin, config)
    targetRows.push({ ...result, health })
    for (const model of result.models ?? []) {
      const cw = model.contextWindow
      add(`model:${result.host}:${result.port}/${model.id}`, cw === undefined ? 'warn' : 'ok',
        `${result.host}:${result.port} · ${model.id}`,
        cw === undefined
          ? `引擎（${result.engine ?? 'unknown'}）没给出可用的上下文长度${model.trainContext === undefined ? '' : `，只有训练长度 ${model.trainContext}（不等于服务端上限，故不采用）`}`
          : `上下文 ${cw}，引擎识别为 ${result.engine ?? 'unknown'}，${result.latencyMs}ms`)
    }
    if (result.error !== undefined) {
      add(`probe:${result.host}:${result.port}`, 'warn', `${result.host}:${result.port} 探测异常`, String(result.error))
    }
    if (health.ok !== true) {
      add(`health:${result.host}:${result.port}`, 'warn', `${result.host}:${result.port} 的 /health 不是 200`,
        `status ${health.status ?? '（连不上）'}；/v1/models 能读但仍建议查 modelctl status`)
    }
  }

  if (withModels.length === 0 && live.length > 0) {
    add('models', 'fail', '端点通了但没读到任何模型', '引擎可能还在加载权重（启动 90–210 秒），或 /v1/models 未就绪')
  }

  // ── 声明 vs 引擎：这是本插件存在的理由，判据要最严 ──
  const existingByOrigin = indexProvidersByOrigin(section)
  const linked = new Set()
  for (const result of withModels) {
    const providerId = existingByOrigin.get(result.origin)
    if (providerId === undefined) {
      // 标题里用 host:port 而不是完整 origin —— 面板第一块已经列过主机了，
      // 这里再写一遍 http://… 就是同一句「IP 重复显示」的毛病。
      add(`link:${result.origin}`, 'warn', `${result.host}:${result.port} 还没接进模型列表`,
        `点「接入本地模型」即可建出 provider；当前广告 ${result.models.length} 个模型`)
      continue
    }
    const provider = section.providers?.[providerId] ?? {}
    const declared = new Map((Array.isArray(provider.models) ? provider.models : []).map((m) => [m?.id, m]))
    linked.add(providerId)

    for (const found of result.models) {
      const entry = declared.get(found.id)
      if (entry === undefined) {
        add(`declare:${providerId}/${found.id}`, 'warn', `${providerId} 里没有 ${found.id}`,
          `引擎在 ${result.origin} 广告了它${found.contextWindow === undefined ? '' : `（上下文 ${found.contextWindow}）`}`)
        continue
      }
      const declaredCw = Number(entry.contextWindow)
      if (found.contextWindow !== undefined && declaredCw !== found.contextWindow) {
        add(`context:${providerId}/${found.id}`, 'warn', `${providerId}/${found.id} 的上下文对不上`,
          `声明 ${Number.isFinite(declaredCw) ? declaredCw : '缺'}，引擎自报 ${found.contextWindow}（跑一次「接入本地模型」即修正）`)
        continue
      }
      if (!Number.isFinite(declaredCw)) {
        add(`context-missing:${providerId}/${found.id}`, 'warn', `${providerId}/${found.id} 缺 contextWindow`,
          '缺这个字段会让自动压缩阈值回落到 0.8 × 262144')
        continue
      }
      const head = compactionHeadroom(declaredCw, Number(entry.maxTokens))
      if (head !== null && head.headroom < 0) {
        // maxTokens 是使用偏好而不是引擎事实，本插件不替你改；但要给出能塞下的最大值。
        add(`headroom:${providerId}/${found.id}`, 'fail', `${providerId}/${found.id} 的压缩阈值不够塞`,
          `0.8 × ${declaredCw} = ${head.threshold}，加 maxTokens ${head.maxTokens} 超过上下文 ${-head.headroom} token；` +
          `把 maxTokens 降到 ${Math.max(1, declaredCw - head.threshold)} 或更小`)
        continue
      }
      add(`ok:${providerId}/${found.id}`, 'ok', `${providerId}/${found.id} 声明与引擎一致`,
        `上下文 ${declaredCw}${head === null ? '' : `，压缩阈值 ${head.threshold}，余量 ${head.headroom}`}`)
    }

    for (const id of declared.keys()) {
      if (!result.models.some((m) => m.id === id)) {
        add(`stale:${providerId}/${id}`, 'warn', `${providerId} 里的 ${id} 引擎当前没有`,
          `${result.origin} 只广告 ${result.models.map((m) => m.id).join('、')}；默认不删（prune=${config.prune}）`)
      }
    }
  }

  const unlinked = withModels.filter((r) => existingByOrigin.get(r.origin) === undefined)
  if (withModels.length === 0) {
    // 一个端点都没读到模型时，「已接入 0 个 / 待接入 0 个」是**空转的绿灯** —— 它什么都没验证，
    // 却和上面两条 ✗ 并排显示，读起来像"其余正常"（2026-09-24 用户就是被这个绊住的）。
    // 没有可关联的对象时只能报"无法判定"，不能报 ok。
    add('linkage', 'warn', '接入状态无法判定（没有可关联的端点）',
      live.length === 0
        ? '端点一个都没通，先让端点通起来才谈得上接入'
        : '端点通了但没广告任何模型，先等引擎加载完再看')
  } else {
    add('linkage', unlinked.length === 0 ? 'ok' : 'warn',
      `已接入 provider ${linked.size} 个，待接入端点 ${unlinked.length} 个`,
      unlinked.length === 0 ? '所有有模型的端点都在模型列表里' : unlinked.map((r) => r.origin).join('、'))
  }

  // ── 凭据可解析性 ──
  // 声明了 `apiKeyEnv` 却取不到值，会让这个 provider **在发请求时**抛 MISSING_CREDENTIAL
  // —— 配置本身合法、schema 也过，只有在用时才炸。这条最值得自检。
  // 口径见 credentialStatus；只查内网端点（本插件的范围）。
  let declaredCredentials = 0
  let credentialsOk = true
  for (const [providerId, provider] of Object.entries(section.providers ?? {})) {
    if (!isPrivate(asString(provider?.baseURL))) continue
    const ref = asString(provider?.apiKeyEnv)
    if (ref === '') continue
    declaredCredentials++
    const status = await credentialStatus(ctx, ref)
    if (status.state === 'ok') continue
    credentialsOk = false
    add(`credential:${providerId}`, status.state === 'missing' ? 'fail' : 'warn',
      `${providerId} 的凭据 ${ref} 取不到值`,
      `${status.note ?? ''}；用这个模型时会报 MISSING_CREDENTIAL。修法三选一：在 Models 页存一次 ${ref}、把 ${ref} 导出到 dsh web 的环境、` +
      `或删掉该 provider 的 apiKeyEnv（本地端点通常不需要鉴权）`)
  }
  if (declaredCredentials > 0 && credentialsOk) {
    // 标题里明说「与端点连通性无关」：这条绿灯只证明凭据能解析，不证明模型能用。
    // 2026-09-24 它和两条 ✗ 并排出现时，被读成了「其余正常」。
    add('credentials', 'ok', `${declaredCredentials} 个本地 provider 的凭据都能取到值（与端点是否连得通无关）`,
      '解析口径与 dsh-llm-pi-ai 的 resolveApiKey 一致；只看凭据存不存在，不代表模型此刻可用')
  }

  const verdict = checks.some((c) => c.level === 'fail') ? 'fail' : (checks.some((c) => c.level === 'warn') ? 'warn' : 'ok')
  const report = {
    ok: verdict !== 'fail',
    verdict,
    at: new Date().toISOString(),
    version: VERSION,
    config: {
      hosts: config.hosts,
      ports: config.ports,
      providerId: config.providerId,
      includeConfigured: config.includeConfigured,
      autoProbe: config.autoProbe,
      autoApply: config.autoApply,
      prune: config.prune,
      adoptUnknownContext: config.adoptUnknownContext,
      probeTimeoutMs: config.probeTimeoutMs
    },
    summary: {
      candidates: targets.length,
      reachable: live.length,
      withModels: withModels.length,
      models: withModels.reduce((n, r) => n + r.models.length, 0),
      linkedProviders: linked.size,
      pendingEndpoints: unlinked.length,
      ok: checks.filter((c) => c.level === 'ok').length,
      warn: checks.filter((c) => c.level === 'warn').length,
      fail: checks.filter((c) => c.level === 'fail').length
    },
    targets: targetRows,
    checks
  }
  cache.selfCheck = report
  return report
}

/** 单飞包装：同一时刻只允许一次网络全扫。 */
function singleFlight(job) {
  if (inFlight !== null) return inFlight
  inFlight = Promise.resolve().then(job).finally(() => { inFlight = null })
  return inFlight
}

// ─────────────────────────────────────────────────────────────────────────
// 面板视图：把两份报告压成「三个块 + 每行一个状态点」
// ─────────────────────────────────────────────────────────────────────────
//
// 为什么这件排版的事放在**宿主侧**而不是页面脚本里：注入脚本是一整段字符串，Node 侧断言不了
// 它渲染出什么；而排版恰好是用户反复提的东西（2026-09-24 连提三次：弹窗不消失、两个按钮重合、
// 然后是「IP 重复显示 / 要绿点红点 / 排版统一 / 分几个块」）。放这里就能用普通单测钉住。
//
// 三个块各自回答一个问题，所以不合并：
//   ① 接入检查 —— 有哪些端点、什么模型、多少上下文（发现的**事实**）
//   ② 模型列表 —— 我把声明更新了吗（对 settings.yaml 的**动作**）
//   ③ 自检     —— 现在还有哪里不对（声明与引擎的**判定**）
//
// 行的形状：{ level, key?, text, title? }
//   level: 'ok' | 'warn' | 'fail' | 'neutral'（灰点）| 'host'（主机小标题，不打点）| 'detail'（灰字详情，缩进）
//   key  : 有值时按固定宽度成列（端口 / provider 路由名）
//   title: 完整原文，挂在 DOM 的 title 上 —— 屏幕上短，悬停不丢信息

/** 连接层错误码 → 一**短**句人话。
 *  刻意不带「主机通、端口没在听」这类解释：6 个端口里 5 个不通时，那句话会重复 5 遍 ——
 *  和用户抱怨的「IP 重复显示」是同一种噪音。解释只在块尾出现一次（见 buildPanelView）。
 *  原始报文进 row.title，所以一个字都没丢。 */
export function shortUnreachable(target) {
  switch (target?.code) {
    case 'ECONNREFUSED': return '连接被拒'
    case 'ETIMEDOUT': return '连接超时'
    case 'EHOSTUNREACH':
    case 'ENETUNREACH': return '主机 / 网络不可达'
    case 'ENOTFOUND': return '主机名解析不了'
    default: return '不通'
  }
}

/** 只解释一次的那句（出现条件：确实有端口是 ECONNREFUSED）。 */
const REFUSED_NOTE = '连接被拒 = 主机是通的，只是那个端口上没进程在听（服务没起，或正在启动 / 重启窗口里）'

/** ISO → 本机时区的 `2026-09-25 02:19:31`（宿主与用户同一台机器，所以本地时间就是他要的）。 */
export function formatStamp(iso) {
  const date = new Date(asString(iso))
  if (Number.isNaN(date.getTime())) return asString(iso)
  const pad = (value) => String(value).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} `
    + `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}

/** 一条 change 的人话。 */
function describeChange(change) {
  const who = change?.id ? `${change.id} 的 ` : ''
  return `${who}${change?.field}：${change?.from ?? '（空）'} → ${change?.to ?? '（空）'}`
}

const VERDICT_LEVEL = { ok: 'ok', warn: 'warn', fail: 'fail' }

/**
 * @param run - `POST /run` 的报告（可能带 error）
 * @param check - `POST /selfcheck` 的报告（可能带 error）
 * @returns `{ verdict, head, blocks }`，纯数据、可 JSON 化
 */
export function buildPanelView(run, check) {
  const blocks = []

  // ── ① 接入检查：同一台主机只写一次，端口成列 ──
  const endpointRows = []
  const targets = Array.isArray(run?.targets) ? run.targets : []
  if (run?.error !== undefined) {
    endpointRows.push({ level: 'fail', text: `探测失败：${run.error}` })
  } else if (targets.length === 0) {
    endpointRows.push({ level: 'neutral', text: '没有候选端点（检查 hosts / ports）' })
  }
  const byHost = new Map()
  for (const target of targets) {
    const host = asString(target?.host, '（未知主机）')
    if (!byHost.has(host)) byHost.set(host, [])
    byHost.get(host).push(target)
  }
  for (const [host, group] of byHost) {
    endpointRows.push({ level: 'host', text: host })
    for (const target of group) {
      const key = String(target.port)
      if (target.skipped !== undefined) {
        endpointRows.push({ level: 'neutral', key, text: `跳过：${target.skipped}` })
        continue
      }
      if (target.reachable !== true) {
        endpointRows.push({ level: 'fail', key, text: shortUnreachable(target), title: target.error })
        continue
      }
      const models = Array.isArray(target.models) ? target.models : []
      if (models.length === 0) {
        endpointRows.push({ level: 'warn', key, text: '通了，但没广告模型（可能在加载权重）' })
        continue
      }
      const detail = models
        .map((model) => `${model.id}${model.contextWindow ? ` @${model.contextWindow}` : ' @上下文未知'}`)
        .join('，')
      const engine = asString(target.engine) !== '' && target.engine !== 'unknown' ? `${target.engine} · ` : ''
      endpointRows.push({
        level: 'ok',
        key,
        text: `${engine}${detail}`,
        title: `${host}:${target.port} · ${target.latencyMs ?? '?'}ms`
      })
    }
  }
  // 「连接被拒」的解释只说一次（否则 5 个不通的端口就重复 5 遍）。
  if (targets.some((target) => target?.code === 'ECONNREFUSED')) {
    endpointRows.push({ level: 'detail', text: REFUSED_NOTE })
  }
  blocks.push({ key: 'endpoints', title: '接入检查', rows: endpointRows })

  // ── ② 模型列表：动作（新建 / 更新 / 已是最新 / 跳过 / 失败），末尾一行是写没写设置 ──
  const modelRows = []
  const providers = Array.isArray(run?.providers) ? run.providers : []
  if (run?.error === undefined && providers.length === 0) {
    modelRows.push({ level: 'neutral', text: '没有可接入的 provider' })
  }
  for (const entry of providers) {
    const changes = Array.isArray(entry.changes) ? entry.changes : []
    const detail = changes.map(describeChange).join('；')
    const dropped = Array.isArray(entry.skipped) ? entry.skipped : []
    const tail = dropped.length > 0 ? `，${dropped.length} 条未采纳` : ''
    const droppedTitle = dropped.length > 0
      ? `未采纳：${dropped.map((item) => `${item.id}（${item.reason}）`).join('；')}`
      : ''
    const title = [detail, droppedTitle].filter(Boolean).join('\\n')
    const row = { key: entry.provider }
    if (entry.action === 'failed') {
      modelRows.push({ ...row, level: 'fail', text: '写设置失败', title: entry.error })
    } else if (entry.action === 'skipped') {
      modelRows.push({ ...row, level: 'warn', text: `跳过${tail}`, title: entry.error ?? droppedTitle })
    } else if (entry.action === 'created') {
      const added = changes.filter((change) => change.field === '(新增模型)').length
      modelRows.push({ ...row, level: 'ok', text: `已接入（新建 ${added} 个模型）${tail}`, title })
    } else if (entry.action === 'updated') {
      modelRows.push({ ...row, level: 'ok', text: `已更新 ${changes.length} 处${tail}`, title })
    } else if (entry.action === 'would-change') {
      modelRows.push({ ...row, level: 'neutral', text: `待写入 ${changes.length} 处（dry-run）${tail}`, title })
    } else if (entry.action === 'unchanged') {
      const count = Number.isInteger(entry.modelCount) ? `（${entry.modelCount} 个模型）` : ''
      modelRows.push({ ...row, level: 'ok', text: `已是最新${count}${tail}`, title })
    } else {
      modelRows.push({ ...row, level: 'warn', text: `未知状态 ${entry.action}${tail}`, title })
    }
  }
  if (run !== null && run !== undefined && run.error === undefined) {
    if (run.dryRun === true) modelRows.push({ level: 'neutral', text: 'dry-run：没有写入设置' })
    else if (run.wrote === true) modelRows.push({ level: 'ok', text: '已写入设置（settings 热重载）' })
    else modelRows.push({ level: 'neutral', text: '未写入设置（无需改动）' })
  }
  blocks.push({ key: 'models', title: '模型列表', rows: modelRows })

  // ── ③ 自检：一行结论 + **只**列非 ok 的明细（全绿时不刷屏） ──
  const checkRows = []
  if (check?.error !== undefined) {
    checkRows.push({ level: 'fail', text: `自检失败：${check.error}` })
  } else {
    const summary = check?.summary ?? {}
    const verdict = asString(check?.verdict, 'unknown')
    checkRows.push({
      level: VERDICT_LEVEL[verdict] ?? 'neutral',
      text: `${verdict.toUpperCase()} · ${summary.ok ?? 0} 通过 · ${summary.warn ?? 0} 提醒 · ${summary.fail ?? 0} 失败`,
      title: `候选 ${summary.candidates ?? '?'} · 存活 ${summary.reachable ?? '?'} · 已接入 ${summary.linkedProviders ?? '?'}`
    })
    for (const item of Array.isArray(check?.checks) ? check.checks : []) {
      if (item.level === 'ok') continue
      checkRows.push({ level: item.level === 'fail' ? 'fail' : 'warn', text: item.title })
      if (asString(item.detail) !== '') checkRows.push({ level: 'detail', text: item.detail })
    }
  }
  blocks.push({ key: 'checks', title: '自检', rows: checkRows })

  return {
    verdict: asString(check?.verdict, run?.error !== undefined ? 'fail' : 'unknown'),
    head: asString(run?.at) !== '' ? `检查于 ${formatStamp(run.at)}` : '',
    blocks
  }
}

// ─────────────────────────────────────────────────────────────────────────
// 页面脚本（注入 index.html）
// ─────────────────────────────────────────────────────────────────────────

/**
 * 样式走独立的 `style` 行、脚本走 `script` 行 —— 不再把 CSS 塞进模板字符串里，
 * 省掉 v1 那种 `\`` 转义（改一个颜色都要数反斜杠）。
 */
/**
 * 面板：**一个**按钮挂进**会话头部的 utilities 行**、「在本地打开」那个分体控件的**左边**。
 *
 * 为什么只有一个按钮：原来「接入本地模型」与「自检」是两条动作、两条报告，而它们探的是同一批
 * 端点、走的是同一段探测代码 —— 用户一眼就看出「有点重合」。现在一次点击 = 接入（发现 + 写声明）
 * 再接一次自检（只读判定），报告合成一段：接入的结果在上、自检的结论（汇总行 + 只有非 ok 的明细）
 * 在下。只读能力没有消失，只是不再占一个按钮：CLI 的 `GET /selfcheck`、`?dry=1`、
 * `config.autoApply:false` 都还在。
 *
 * 样式不自己发明 —— 直接抄 `dsh-client-ui-open-in-app` 的度量（28px 高 / 14px 圆角 /
 * .5px `border-l4` / 11px·16px 字 / 同 padding），并复用同一批 `--dsw-alias-*` 设计令牌，
 * 所以深浅色主题、hover、disabled 都跟着 DSH 自己的外观走，不用维护第二套配色。
 *
 * 报告区不再常驻占地方：它是 `position: fixed`（不参与布局、不推挤页面），并且
 * 点报告本身 / 再点按钮 / 按 Esc / 滚页面 / 超时（`autoHideMs`，默认 12s）都会收起。
 * 旧版把报告留在文档流里，点一次就永久占一块地方 —— 2026-09-24 用户点名要改的就是这个。
 *
 * 拿不到头部（还没进会话）时退回右下角浮动，按钮不会凭空消失。
 */
const PANEL_STYLE = `
#lmc-tools { display: inline-flex; align-items: center; gap: 8px; }
#lmc-tools.lmc-float { position: fixed; right: 18px; bottom: 18px; z-index: 2147483000; }
#lmc-tools .lmc-btn { box-sizing: border-box; height: 28px; border-radius: 14px;
  border: .5px solid var(--dsw-alias-border-l4); background: transparent;
  font-family: var(--dsw-font-family); font-size: 11px; font-weight: 400; line-height: 16px;
  color: var(--dsw-alias-label-primary); padding: 5px 10px;
  display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; cursor: pointer; }
#lmc-tools .lmc-btn:hover:not(:disabled) { background: var(--dsw-alias-interactive-bg-hover); }
#lmc-tools .lmc-btn:disabled { color: var(--dsw-alias-label-dimmed); cursor: progress; }
#lmc-tools .lmc-btn:focus-visible { outline: none; box-shadow: inset 0 0 0 1px var(--dsw-alias-label-secondary); }
#lmc-tools .lmc-dot { flex: none; width: 6px; height: 6px; border-radius: 50%;
  background: var(--dsw-alias-label-dimmed); }
#lmc-tools .lmc-dot.ok { background: var(--dsw-alias-state-success-primary); }
#lmc-tools .lmc-dot.warn { background: var(--dsw-alias-state-warn-primary); }
#lmc-tools .lmc-dot.fail { background: var(--dsw-alias-state-error-primary); }
/* 弹窗 = 外框（fixed，定宽）+ 右上角的叉 + 可滚动的正文。
   不要把 overflow:auto 放在外框上：那样叉会跟着正文一起滚走。
   也不要给外框 cursor:pointer —— 它不是一个"点一下"的按钮（2026-09-24 用户的原话：
   「不想是可以点的按钮，换成叉叉在右上角」）。 */
#lmc-box { position: fixed; z-index: 2147483000; display: none; text-align: left;
  box-sizing: border-box; width: min(72vw, 560px); max-height: 42vh; overflow: hidden;
  padding: 9px 11px; border-radius: 10px; border: .5px solid var(--dsw-alias-border-l4);
  background: var(--dsw-alias-bg-layer-3); color: var(--dsw-alias-label-primary);
  box-shadow: 0 8px 28px rgba(0, 0, 0, .3); }
#lmc-box.show { display: block; }
#lmc-box .lmc-close { position: absolute; top: 4px; right: 4px; width: 22px; height: 22px;
  display: inline-flex; align-items: center; justify-content: center; padding: 0;
  border: 0; border-radius: 6px; background: transparent; cursor: pointer;
  color: var(--dsw-alias-label-secondary); font-size: 14px; line-height: 1; }
#lmc-box .lmc-close:hover { background: var(--dsw-alias-interactive-bg-hover);
  color: var(--dsw-alias-label-primary); }
#lmc-body { max-height: calc(42vh - 18px); overflow: auto; padding-right: 20px;
  font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11px; line-height: 1.5;
  word-break: break-word; }
/* 三个块（接入检查 / 模型列表 / 自检）：标题淡一档，块与块之间一条细线 + 间距。
   块内每一行是 flex：状态点 + 定宽行首列 + 正文 —— 用 flex 而不是空格对齐，
   因为正文会换行，空格对齐一换行就散了。 */
#lmc-body .lmc-head { color: var(--dsw-alias-label-tertiary); margin-bottom: 6px; }
#lmc-body .lmc-block + .lmc-block { margin-top: 9px; padding-top: 9px;
  border-top: .5px solid var(--dsw-alias-border-l4); }
#lmc-body .lmc-block-title { color: var(--dsw-alias-label-secondary); margin-bottom: 3px; }
#lmc-body .lmc-row { display: flex; align-items: flex-start; gap: 6px; }
#lmc-body .lmc-row .lmc-text { flex: 1; min-width: 0; white-space: pre-wrap; }
#lmc-body .lmc-row .lmc-dot { margin-top: 5px; }
#lmc-body .lmc-row .lmc-key { flex: none; width: 11ch; color: var(--dsw-alias-label-secondary); }
#lmc-body .lmc-row.lmc-host { color: var(--dsw-alias-label-secondary); margin-top: 3px; }
#lmc-body .lmc-row.lmc-detail { color: var(--dsw-alias-label-tertiary); padding-left: 12px; }
`

/**
 * 页面脚本。`autoHideMs` 是唯一的注入参数 —— 用一个函数包着，而不是把配置塞进模块级常量：
 * 常量在模块加载时就定死了，config 改了脚本不会跟着变。
 */
function panelScript(autoHideMs) {
  return `(() => {
  const ID = 'lmc-tools';
  const BOX_ID = 'lmc-box';
  const BODY_ID = 'lmc-body';
  const API = {
    state: ${JSON.stringify(ROUTE_STATE)},
    panel: ${JSON.stringify(ROUTE_PANEL)}
  };
  const AUTO_HIDE_MS = ${Number(autoHideMs)};

  if (window.__localModelsConnectInstalled) return;
  window.__localModelsConnectInstalled = true;

  let tools = null, box = null, body = null, dot = null;
  let runButton = null, runLabel = null;
  let hideTimer = null, observerTimer = null, pendingVerdict = null;

  // ── 报告区：fixed 定位，绝不占布局 ──
  function hideBox() {
    clearTimeout(hideTimer);
    hideTimer = null;
    if (box !== null) box.classList.remove('show');
  }

  function placeBox() {
    if (tools === null || box === null || !box.classList.contains('show')) return;
    // 宽度是 CSS 里写死的（width: min(72vw, 560px)），所以 offsetWidth 不受上一次 left 影响。
    // 这里刻意不用 max-width + shrink-to-fit：那样「可用宽度」会被上一次的 left 截断，
    // 量出来的是「剩下的空间」，每次重排都把弹窗又往左推一截（实测右边缘差 59px）。
    const width = box.offsetWidth;
    const rect = tools.getBoundingClientRect();
    const left = Math.max(8, Math.min(Math.round(rect.right - width), window.innerWidth - width - 8));
    box.style.left = left + 'px';
    box.style.top = Math.round(rect.bottom + 8) + 'px';
  }

  /** 渲染视图 → 显示弹窗 → 重置自动收起计时。 */
  function showView(view, error) {
    if (box === null || body === null) return;
    renderView(view, error);
    box.classList.add('show');
    placeBox();
    clearTimeout(hideTimer);
    hideTimer = AUTO_HIDE_MS > 0 ? setTimeout(hideBox, AUTO_HIDE_MS) : null;
  }

  function setDot(level) {
    pendingVerdict = level || null;
    if (dot !== null) dot.className = 'lmc-dot' + (pendingVerdict ? ' ' + pendingVerdict : '');
  }

  async function post(url) {
    const response = await fetch(url, { method: 'POST', headers: { accept: 'application/json' } });
    return await response.json();
  }

  /**
   * 一行 = 状态点 + 可选行首列 + 正文。
   * 排版全部由 buildPanelView（宿主侧，可单测）决定，这里只负责把它变成 DOM；
   * 屏幕上短，完整原文挂在 title 上，悬停不丢信息。
   */
  function makeRow(row) {
    const line = document.createElement('div');
    line.className = 'lmc-row lmc-' + (row.level || 'neutral');
    if (row.level !== 'host' && row.level !== 'detail') {
      const mark = document.createElement('span');
      mark.className = 'lmc-dot ' + (row.level || 'neutral');
      line.appendChild(mark);
    }
    if (row.key !== undefined && row.key !== null) {
      const key = document.createElement('span');
      key.className = 'lmc-key';
      key.textContent = row.key;
      line.appendChild(key);
    }
    const text = document.createElement('span');
    text.className = 'lmc-text';
    text.textContent = row.text;
    line.appendChild(text);
    if (row.title) line.title = row.title;
    return line;
  }

  function renderView(view, error) {
    if (body === null) return;
    body.textContent = '';
    if (error) {
      body.appendChild(makeRow({ level: 'fail', text: '请求失败：' + error }));
      return;
    }
    if (view === null || typeof view !== 'object') {
      body.appendChild(makeRow({ level: 'fail', text: '没有拿到视图数据' }));
      return;
    }
    if (view.head) {
      const head = document.createElement('div');
      head.className = 'lmc-head';
      head.textContent = view.head;
      body.appendChild(head);
    }
    for (const block of view.blocks || []) {
      const section = document.createElement('div');
      section.className = 'lmc-block';
      const title = document.createElement('div');
      title.className = 'lmc-block-title';
      title.textContent = block.title;
      section.appendChild(title);
      for (const row of block.rows || []) section.appendChild(makeRow(row));
      body.appendChild(section);
    }
  }

  // ── 头部锚点：找「在本地打开」那条 slot 条目，插到它**最外层**前面 ──
  //
  // 为什么必须爬到最外层：头部容器 .…_headerUtilities 是 display:flex; gap:8px
  // （dsh-client-ui-conversation），而这 8px 只作用在**它的直接子项**之间。分体控件的
  // div.…_split 并不是直接子项 —— slot 条目外面还有包裹（open-in-app 的 Menu 包着它的
  // anchor）。插在 _split 前面 = 插进包裹层内部，容器的 gap 够不着 →
  // 与左边邻居贴成 0px（2026-09-24 用户实测到的就是 0）。爬到容器的直接子项再插，两边各得 8px。
  //
  // 为什么先限定在容器内找：页面里另有 2 个包也定义了 _split（deliverables、trajectory），
  // 全局匹配会挂错地方。容器类名的本地名 _headerUtilities 稳定，哈希部分会随构建变。
  //
  // 注意：这段代码整体活在一个模板字符串里，所以**这里不能出现反引号**（会提前闭合）。
  function findHeaderContainer() {
    return document.querySelector('[class*="_headerUtilities"]');
  }

  function findSplitWithin(root) {
    const splits = root.querySelectorAll('div[class*="_split"]');
    for (let i = 0; i < splits.length; i++) {
      const el = splits[i];
      if (el.querySelector('button[aria-haspopup="menu"]') === null) continue;
      if (el.getBoundingClientRect().width === 0) continue;
      return el;
    }
    return null;
  }

  function findAnchor() {
    const container = findHeaderContainer();
    const split = findSplitWithin(container !== null ? container : document);
    if (split === null) return null;
    if (container === null) return split;
    // 爬到容器的直接子项 —— 这一步与我自己的节点无关（我只是它的前一个兄弟），所以幂等。
    let node = split;
    while (node.parentElement !== null && node.parentElement !== container) node = node.parentElement;
    return node.parentElement === container ? node : split;
  }

  function makeButton(label, hint, withDot) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'lmc-btn';
    button.title = hint;
    const text = document.createElement('span');
    text.textContent = label;
    if (withDot) {
      dot = document.createElement('span');
      dot.className = 'lmc-dot' + (pendingVerdict ? ' ' + pendingVerdict : '');
      button.appendChild(dot);
    }
    button.appendChild(text);
    return { button: button, label: text };
  }

  async function invoke(button, labelEl, working, normal, job) {
    button.disabled = true;
    labelEl.textContent = working;
    try {
      await job();
    } catch (error) {
      showView(null, error && error.message ? error.message : String(error));
      setDot('fail');
    } finally {
      button.disabled = false;
      labelEl.textContent = normal;
    }
  }

  function build() {
    if (document.body === null) return;
    if (document.getElementById(ID) !== null) return;

    tools = document.createElement('div');
    tools.id = ID;

    box = document.createElement('div');
    box.id = BOX_ID;

    // 右上角的叉：唯一的"点"目标。整块弹窗不再可点（用户明确不要那种"它像个按钮"的感觉）。
    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'lmc-close';
    close.textContent = '×';
    close.title = '收起';
    close.setAttribute('aria-label', '收起');
    close.addEventListener('click', hideBox);
    box.appendChild(close);

    body = document.createElement('div');
    body.id = BODY_ID;
    box.appendChild(body);

    tools.appendChild(box);

    const main = makeButton('接入本地模型', '发现本地端点并接进模型列表，随后做一次只读自检', true);
    runButton = main.button;
    runLabel = main.label;
    runButton.addEventListener('click', function () {
      invoke(runButton, runLabel, '接入中…', '接入本地模型', async function () {
        // 一个请求拿回「三块视图」；run 与 selfcheck 都在宿主侧按正确顺序跑完了
        // （run 可能写设置，自检要反映**写完之后**的声明）。
        const payload = await post(API.panel);
        const view = payload && payload.view ? payload.view : null;
        setDot(view ? view.verdict : 'fail');
        showView(view, payload && payload.error);
      });
    });

    tools.appendChild(runButton);
  }

  /** 每次都把按钮摆回「在本地打开」左边；拿不到头部就退回右下角。 */
  function ensure() {
    if (document.body === null) return;
    if (tools === null || !tools.isConnected) {
      dot = null;
      tools = null;
      build();
    }
    if (tools === null) return;

    const anchor = findAnchor();
    if (anchor !== null && anchor.parentElement !== null) {
      if (tools.parentElement !== anchor.parentElement || tools.nextElementSibling !== anchor) {
        hideBox();
        tools.classList.remove('lmc-float');
        anchor.parentElement.insertBefore(tools, anchor);
      }
    } else if (tools.parentElement !== document.body) {
      hideBox();
      tools.classList.add('lmc-float');
      document.body.appendChild(tools);
    }
  }

  function schedule() {
    if (observerTimer !== null) return;
    observerTimer = setTimeout(function () { observerTimer = null; ensure(); }, 300);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ensure, { once: true });
  } else {
    ensure();
  }
  new MutationObserver(schedule).observe(document.documentElement, { childList: true, subtree: true });
  document.addEventListener('keydown', function (event) { if (event.key === 'Escape') hideBox(); });
  window.addEventListener('resize', placeBox);
  document.addEventListener('scroll', function (event) {
    if (box !== null && (event.target === box || box.contains(event.target))) placeBox();
    else hideBox();
  }, true);

  // 页面加载时只读缓存（不打网络）：状态点是热的。
  fetch(API.state, { headers: { accept: 'application/json' } })
    .then(function (response) { return response.json(); })
    .then(function (state) {
      if (state && state.lastSelfCheck) setDot(state.lastSelfCheck.verdict);
    })
    .catch(function () {});
})();`
}

function injectPanel(table, config) {
  table.push({ kind: 'style', text: PANEL_STYLE })
  table.push({ kind: 'script', placement: 'head', text: panelScript(config.autoHideMs) })
}

// ─────────────────────────────────────────────────────────────────────────
// 宿主胶水
// ─────────────────────────────────────────────────────────────────────────

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload)
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
    'cache-control': 'no-store'
  })
  res.end(body)
}

export { runConnect, selfCheck, ROUTE_STATE, ROUTE_RUN, ROUTE_SELFCHECK, ROUTE_PANEL, LEGACY_ROUTE_RUN, VERSION }

export function apply(ctx, pluginConfig) {
  const config = resolveConfig(pluginConfig)
  ctx.logger.info(
    'local-models-connect: v%s 就绪 —— 种子主机 %s，端口 %s',
    VERSION, config.hosts.join('、'), config.ports.join('、')
  )

  const guard = (job) => async (req, res) => {
    try {
      sendJson(res, 200, await job(req))
    } catch (error) {
      ctx.logger.warn('local-models-connect: %s', msg(error))
      sendJson(res, 200, { ok: false, error: msg(error) })
    }
  }

  ctx.effect(() => ctx.webServer.register({
    kind: 'exact',
    path: ROUTE_STATE,
    handler: guard(async () => ({
      ok: true,
      version: VERSION,
      config: {
        hosts: config.hosts,
        ports: config.ports,
        providerId: config.providerId,
        displayName: config.displayName,
        includeConfigured: config.includeConfigured,
        autoProbe: config.autoProbe,
        autoApply: config.autoApply,
        prune: config.prune,
        adoptUnknownContext: config.adoptUnknownContext,
        autoHideMs: config.autoHideMs,
        probeTimeoutMs: config.probeTimeoutMs
      },
      lastRun: cache.run,
      lastSelfCheck: cache.selfCheck
    }))
  }), 'local-models-connect: state route')

  const runHandler = guard(async (req) => {
    if (req.method !== 'POST') throw new Error(`${ROUTE_RUN} 只接受 POST（加 ?dry=1 可只看不写）`)
    const dryRun = new URL(req.url ?? '/', 'http://localhost').searchParams.get('dry') === '1'
    return await singleFlight(() => runConnect(ctx, config, { dryRun }))
  })

  ctx.effect(() => ctx.webServer.register({
    kind: 'exact',
    path: ROUTE_RUN,
    handler: runHandler
  }), 'local-models-connect: run route')

  // v1 的旧路径留作别名：文档、脚本、肌肉记忆都还指着它。
  ctx.effect(() => ctx.webServer.register({
    kind: 'exact',
    path: LEGACY_ROUTE_RUN,
    handler: runHandler
  }), 'local-models-connect: legacy run route')

  ctx.effect(() => ctx.webServer.register({
    kind: 'exact',
    path: ROUTE_SELFCHECK,
    handler: guard(async () => await singleFlight(() => selfCheck(ctx, config)))
  }), 'local-models-connect: selfcheck route')

  // 面板那一个按钮打的就是这条路：一次请求拿回「三块视图」。
  // 顺序不能反 —— run 可能写设置，自检要反映**写完之后**的声明。
  ctx.effect(() => ctx.webServer.register({
    kind: 'exact',
    path: ROUTE_PANEL,
    handler: guard(async (req) => {
      if (req.method !== 'POST') throw new Error(`${ROUTE_PANEL} 只接受 POST`)
      return await singleFlight(async () => {
        const run = await runConnect(ctx, config, { dryRun: false })
        const check = await selfCheck(ctx, config)
        return { ok: run.ok !== false, at: new Date().toISOString(), view: buildPanelView(run, check) }
      })
    })
  }), 'local-models-connect: panel route')

  ctx.on('webserver/index-inject', (table) => injectPanel(table, config))

  // 开机自动接入 —— 「新环境装上插件即可用」这一步就在这里。
  if (config.autoProbe) {
    ctx.effect(() => {
      const timer = setTimeout(() => {
        singleFlight(async () => {
          const report = await runConnect(ctx, config, { dryRun: !config.autoApply })
          const check = await selfCheck(ctx, config)
          ctx.logger.info(
            'local-models-connect: 自动接入 %s（%s）；自检 %s（ok %d / warn %d / fail %d）',
            report.wrote ? '已写设置' : '未写设置',
            report.providers.map((p) => `${p.provider}:${p.action}`).join(' ') || '无 provider',
            check.verdict, check.summary.ok, check.summary.warn, check.summary.fail
          )
        }).catch((error) => ctx.logger.warn('local-models-connect: 自动探测失败：%s', msg(error)))
      }, config.autoProbeDelayMs)
      timer.unref?.()
      return () => clearTimeout(timer)
    }, 'local-models-connect: auto probe')
  }
}
