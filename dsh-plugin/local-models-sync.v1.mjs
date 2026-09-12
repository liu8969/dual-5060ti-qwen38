/**
 * local-models-sync — 「同步本地模型信息」按钮
 *
 * 背景：DSH 的 `llm-pi-ai` provider 里手写的模型声明会过期。它自带的
 * 「获取可用模型」对本例没用，三条原因（都是源码级的）：
 *   1. `readListing` 只认 `context_window` / `context_length`，而 vLLM 返回的是
 *      `max_model_len` → 读不到上下文长度；
 *   2. `adoptPicked` 对**已存在**的 id 不替换（`byId.get(id) ?? adopt(candidate)`）
 *      → 我们的 id 已在列表里，点了等于没点；
 *   3. 缺 `contextWindow` 会回落到 `defaultContextWindow` = 262144，比 150000 更离谱。
 * 而 `contextWindow` 直接决定自动压缩阈值（`dsh-compaction-basic`：0.8 × contextWindow），
 * 声明偏大 = 压缩迟迟不触发 = 会话在 150000 以上被引擎拒掉。
 *
 * 为什么做成**宿主**插件而不是客户端插件（`dsh.client` 那套）：
 *   · 客户端模块只扫「宿主 Loader 里已知」的包，且包元数据按名字缓存、永不过期
 *     → 新加一个客户端插件**必须重启 `dsh web`**，而那个进程正是本体；
 *   · 宿主插件挂在 profile 的 cordis.patch.yml 上，Cordis HMR 重放配置树 → 保存即生效；
 *   · `settings.mutate` 本来就是宿主侧服务，插件直接调 = 走官方校验 + 原子写 + 热重载，
 *     既不需要客户端那套 RPC，也不涉及浏览器跨域。
 *
 * 它做两件事：
 *   1. `POST /local-models-sync/run`：对**本地**端点拉 `{baseURL}/models`，用引擎自报的
 *      `max_model_len` 改写该 provider 的 `models` 数组；
 *   2. 往页面 head 注入一小段脚本，渲染右下角一个常驻按钮，点击时调上面那个端点。
 *
 * 不动手的情形（一律只报结果、不写文件）：端点不通、返回空、值没变、provider 不是本地端点。
 *
 * 迭代提示：Cordis 只在 `name` 变化时才重新 import 模块（ESM 有模块缓存），
 * 所以改内容要装成新文件名（v2、v3…）并改 cordis.patch.yml 里那一行。
 */

/** 写入的设置命名空间（`llm-pi-ai` 的注册者见 dsh-llm-pi-ai）。 */
const NS = 'llm-pi-ai'
/** 按钮与脚本共用的路由。 */
const ROUTE = '/local-models-sync/run'
/** 只碰内网/回环端点，避免拿没配 key 的远端 API 去试。 */
const PRIVATE_HOST = /^(localhost|127\.|\[?::1\]?|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/i
/** 拉模型列表的超时。 */
const FETCH_TIMEOUT_MS = 8000

export const name = 'local-models-sync'
export const inject = ['webServer', 'settings']

const msg = (error) => (error && error.message) || String(error)

function isPrivate(baseURL) {
  try {
    return PRIVATE_HOST.test(new URL(baseURL).host)
  } catch {
    return false
  }
}

/** 拉一个 OpenAI 兼容端点的模型列表，只取我们认得的字段。 */
async function fetchModels(baseURL) {
  const url = `${String(baseURL).replace(/\/+$/, '')}/models`
  const response = await fetch(url, {
    method: 'GET',
    headers: { accept: 'application/json' },
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS)
  })
  if (!response.ok) throw new Error(`${url} 回了 ${response.status}`)
  const body = await response.json()
  if (!Array.isArray(body?.data)) throw new Error(`${url} 的返回里没有 data 数组`)
  return body.data
    .map((raw) => ({ id: raw?.id, maxModelLen: raw?.max_model_len ?? raw?.context_length }))
    .filter((entry) => typeof entry.id === 'string' && entry.id.length > 0)
}

/**
 * 由「端点广告的模型」+「当前声明」算出新声明。
 * 规则：contextWindow 取引擎自报值（拿不到就保留旧值，绝不回落到 262144 默认）；
 * name 保留人工写的那个；maxTokens 保留但不超过 contextWindow。
 */
function buildModels(advertised, current) {
  const byId = new Map((Array.isArray(current) ? current : []).map((m) => [m?.id, m]))
  return advertised.map((a) => {
    const prev = byId.get(a.id) ?? {}
    const entry = { ...prev, id: a.id }
    if (!entry.name) entry.name = a.id
    const cw = Number.isInteger(a.maxModelLen) && a.maxModelLen > 0 ? a.maxModelLen : entry.contextWindow
    if (Number.isInteger(cw) && cw > 0) {
      entry.contextWindow = cw
      if (Number.isInteger(entry.maxTokens)) entry.maxTokens = Math.min(entry.maxTokens, cw)
    }
    return entry
  })
}

/** 逐字段对比，产出给人看的 diff 行。 */
function diffModels(before, after) {
  const changes = []
  const beforeById = new Map((before ?? []).map((m) => [m?.id, m]))
  const afterById = new Map(after.map((m) => [m.id, m]))
  for (const id of afterById.keys()) {
    if (!beforeById.has(id)) changes.push({ id, field: '(新增)', from: null, to: afterById.get(id).contextWindow ?? null })
  }
  for (const id of beforeById.keys()) {
    if (!afterById.has(id)) changes.push({ id, field: '(移除)', from: beforeById.get(id).contextWindow ?? null, to: null })
  }
  for (const [id, next] of afterById) {
    const prev = beforeById.get(id)
    if (!prev) continue
    for (const field of ['name', 'contextWindow', 'maxTokens']) {
      if (prev[field] !== next[field]) {
        changes.push({ id, field, from: prev[field] ?? null, to: next[field] ?? null })
      }
    }
  }
  return changes
}

/** 同步所有本地 provider。返回可 JSON 化的报告；任何失败都不写文件。 */
async function syncAll(ctx) {
  const section = ctx.settings.section(NS) ?? {}
  const providers = section.providers ?? {}
  const results = []

  for (const [providerId, provider] of Object.entries(providers)) {
    const baseURL = String(provider?.baseURL ?? '')
    if (!baseURL) {
      results.push({ provider: providerId, skipped: '没配 baseURL' })
      continue
    }
    if (!isPrivate(baseURL)) {
      results.push({ provider: providerId, skipped: `不是本地端点（${baseURL}），不动` })
      continue
    }

    let advertised
    try {
      advertised = await fetchModels(baseURL)
    } catch (error) {
      results.push({ provider: providerId, baseURL, error: `拉不到模型列表：${msg(error)}` })
      continue
    }
    if (advertised.length === 0) {
      results.push({ provider: providerId, baseURL, error: '端点没广告任何模型，未改动' })
      continue
    }

    const current = Array.isArray(provider.models) ? provider.models : []
    const next = buildModels(advertised, current)
    const changes = diffModels(current, next)
    if (changes.length === 0) {
      results.push({
        provider: providerId,
        baseURL,
        changed: false,
        models: next.map((m) => ({ id: m.id, contextWindow: m.contextWindow ?? null }))
      })
      continue
    }

    try {
      await ctx.settings.mutate(NS, [{ op: 'set', path: ['providers', providerId, 'models'], value: next }])
    } catch (error) {
      results.push({ provider: providerId, baseURL, error: `写设置失败：${msg(error)}` })
      continue
    }
    results.push({ provider: providerId, baseURL, changed: true, changes })
  }

  return {
    ok: results.every((r) => r.error === undefined),
    at: new Date().toISOString(),
    results
  }
}

/**
 * 注入到页面 head 的小脚本：右下角一个常驻按钮。
 * 用纯 DOM + 内联样式（不依赖客户端的 React 树，也不进它的 slot 体系），
 * 加 MutationObserver 以防前端重渲染把节点冲掉。
 */
const BUTTON_SCRIPT = `(() => {
  const ROUTE = ${JSON.stringify(ROUTE)};
  const LABEL = '同步本地模型';
  const ID = 'local-models-sync-button';
  if (window.__localModelsSyncInstalled) return;
  window.__localModelsSyncInstalled = true;

  const style = document.createElement('style');
  style.textContent = \`
    #\${ID} { position: fixed; right: 18px; bottom: 18px; z-index: 2147483000;
      display:flex; flex-direction:column; align-items:flex-end; gap:6px;
      font: 12px/1.4 -apple-system, "SF Pro Text", "Helvetica Neue", sans-serif; }
    #\${ID} button { cursor:pointer; border:1px solid rgba(127,127,127,.45); border-radius:999px;
      padding:7px 14px; background:rgba(28,28,30,.86); color:#f2f2f7; opacity:.55;
      transition:opacity .15s ease, transform .1s ease; backdrop-filter: blur(6px); }
    #\${ID}:hover button { opacity:1; }
    #\${ID} button:active { transform: scale(.97); }
    #\${ID} button[disabled] { cursor:progress; opacity:1; }
    #\${ID} .msg { max-width:min(52vw,520px); text-align:right; padding:5px 9px; border-radius:8px;
      background:rgba(28,28,30,.92); color:#f2f2f7; white-space:pre-wrap; word-break:break-word;
      opacity:0; transition:opacity .15s ease; pointer-events:none; }
    #\${ID} .msg.show { opacity:1; }
    #\${ID} .msg.err { background:rgba(120,30,30,.94); }
    #\${ID} .msg.ok  { background:rgba(24,84,44,.94); }
  \`;
  document.head.appendChild(style);

  let box, button, message, hideTimer;

  function say(text, kind) {
    if (!message) return;
    message.textContent = text;
    message.className = 'msg show' + (kind ? ' ' + kind : '');
    clearTimeout(hideTimer);
    if (kind !== 'err') hideTimer = setTimeout(() => { message.className = 'msg'; }, 8000);
  }

  function summarize(report) {
    if (!report) return ['没有返回', 'err'];
    if (report.error) return ['出错了：' + report.error, 'err'];
    const lines = [];
    let failed = false;
    for (const r of report.results || []) {
      if (r.error) { failed = true; lines.push(r.provider + '：' + r.error); continue; }
      if (r.skipped) { lines.push(r.provider + '：跳过（' + r.skipped + '）'); continue; }
      if (!r.changed) {
        const m = (r.models || []).map(x => x.id + '=' + x.contextWindow).join('，');
        lines.push(r.provider + '：已经是最新' + (m ? '（' + m + '）' : ''));
        continue;
      }
      const c = (r.changes || []).map(x => x.id + ' 的 ' + x.field + ' ' + x.from + ' → ' + x.to).join('；');
      lines.push(r.provider + '：已更新 ' + c);
    }
    if (lines.length === 0) lines.push('没有可同步的 provider');
    return [lines.join('\\n'), failed ? 'err' : 'ok'];
  }

  function build() {
    if (document.getElementById(ID) || !document.body) return;
    box = document.createElement('div');
    box.id = ID;
    message = document.createElement('div');
    message.className = 'msg';
    button = document.createElement('button');
    button.type = 'button';
    button.textContent = LABEL;
    button.title = '从本地模型端点拉 /v1/models，用引擎自报的 max_model_len 更新这里的模型声明';
    button.addEventListener('click', async () => {
      button.disabled = true;
      const original = button.textContent;
      button.textContent = '同步中…';
      try {
        const response = await fetch(ROUTE, { method: 'POST', headers: { accept: 'application/json' } });
        const report = await response.json();
        const [text, kind] = summarize(report);
        say(text, kind);
      } catch (error) {
        say('请求失败：' + (error && error.message ? error.message : String(error)), 'err');
      } finally {
        button.disabled = false;
        button.textContent = original;
      }
    });
    box.appendChild(message);
    box.appendChild(button);
    document.body.appendChild(box);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', build, { once: true });
  } else {
    build();
  }
  new MutationObserver(() => { if (document.body && !document.getElementById(ID)) build(); })
    .observe(document.documentElement, { childList: true, subtree: true });
})();`

/** 把按钮脚本挂到页面 head（与 dsh-client-modules 用的是同一个注入表）。 */
function injectButton(table) {
  table.push({ kind: 'script', placement: 'head', text: BUTTON_SCRIPT })
}

export function apply(ctx) {
  ctx.effect(() => ctx.webServer.register({
    kind: 'exact',
    path: ROUTE,
    handler: async (req, res) => {
      if (req.method !== 'POST') {
        res.writeHead(405, { allow: 'POST' })
        res.end()
        return
      }
      let report
      try {
        report = await syncAll(ctx)
      } catch (error) {
        ctx.logger.warn('local-models-sync: %s', msg(error))
        report = { ok: false, error: msg(error), results: [] }
      }
      const body = JSON.stringify(report)
      res.writeHead(200, {
        'content-type': 'application/json; charset=utf-8',
        'content-length': Buffer.byteLength(body),
        'cache-control': 'no-store'
      })
      res.end(body)
    }
  }), 'local-models-sync: route')

  ctx.on('webserver/index-inject', injectButton)
  ctx.logger.info('local-models-sync: 按钮已就绪（POST %s）', ROUTE)
}
