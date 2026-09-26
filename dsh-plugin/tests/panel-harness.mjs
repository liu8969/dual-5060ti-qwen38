/**
 * 面板的浏览器验证台：起一个**仿真的会话头部** + 桩路由，供真浏览器驱动。
 *
 * 为什么要有它：文档里多处写「真浏览器实测」，但真 GUI 要 token（而凭据不该进 agent 命令），
 * 所以那些结论没有可复现的产物。这个文件把头部形状与设计令牌照 DSH 源码搭出来，让结论可复核。
 *
 * 头部形状照抄自（2026-09-24 核对）：
 *   · 容器：`dsh-client-ui-conversation/lib/client.js` 的 `.…_headerUtilities`
 *     → `display:flex; align-items:center; gap:8px; margin-left:20px`
 *   · 分体控件：`dsh-client-ui-open-in-app/lib/client.js` 的 `.…_split/.main/.chevron`
 *     → 28px 高 / 14px 圆角 / `.5px` border-l4 / 11px·16px 字
 *   · **关键**：`_split` 外面还有一层 slot 条目包裹（open-in-app 的 `Menu` 包着它的 anchor）。
 *     少了这层包裹就验不出「插在 `_split` 前面会拿不到容器的 gap」这个 bug。
 *
 * 用法：
 *   node dsh-plugin/tests/panel-harness.mjs [插件路径] [autoHideMs] [端口] [long]
 * 第 5 个参数 = 端口行数（`long` = 60）。用来验高度自适应那两条退路：
 * 30 行左右「放不下但比视口矮」（应当往上挪、没有滑动条），60 行「比视口还高」（唯一会有滑动条的情况）。
 * 然后把打印出的 URL 交给浏览器工具。
 */
import http from 'node:http'
import { writeFileSync } from 'node:fs'
import { fileURLToPath, pathToFileURL } from 'node:url'

const MODULE = process.argv[2] ?? fileURLToPath(new URL('../local-models-connect.v10.mjs', import.meta.url))
const AUTO_HIDE_MS = Number(process.argv[3] ?? 9000)
// 第 3 个参数之后是端口；状态点轮询节奏另给一个环境变量，方便把它调短来验「自己变红/变绿」
const WATCH_MS = Number(process.env.LMC_WATCH_MS ?? 15000)
const PORT = Number(process.argv[4] ?? 18181)
// 第 5 个参数：端口行数（'long' = 60）。用来验「内容高于视口下方空间」与「比整个视口还高」两条退路。
const ROWS_ARG = process.argv[5] ?? ''
const ROWS = ROWS_ARG === 'long' ? 60 : (Number(ROWS_ARG) > 0 ? Number(ROWS_ARG) : 0)
const plugin = await import(pathToFileURL(MODULE).href)

// ── 取真插件导出的注入行 ──
const table = []
const host = {
  logger: { info: () => {}, warn: () => {}, debug: () => {} },
  effect: (fn) => { fn() },
  on: (event, fn) => { if (event === 'webserver/index-inject') fn(table) },
  get: () => undefined,
  webServer: { register: () => () => {} },
  settings: { section: () => undefined, mutate: async () => {} }
}
plugin.apply(host, { autoProbe: false, hosts: ['127.0.0.1'], ports: [1], includeConfigured: false, autoHideMs: AUTO_HIDE_MS, watchIntervalMs: WATCH_MS })

const style = table.find((r) => r.kind === 'style').text
const script = table.find((r) => r.kind === 'script').text
console.log(`注入行: style ${style.length} 字节, script ${script.length} 字节, autoHideMs=${AUTO_HIDE_MS}`)
if (script.includes('</script')) throw new Error('注入脚本里有 </script，会提前闭合')

// 真令牌值取自 dsh-client-ui-theme/lib/client.js（浅色那一档）
const TOKENS = `
:root {
  --dsw-font-family: -apple-system, "Segoe UI", "PingFang SC", sans-serif;
  --dsw-alias-border-l4: #00000029;
  --dsw-alias-label-primary: #1c1c1e;
  --dsw-alias-label-secondary: #6b7280;
  --dsw-alias-label-dimmed: #b0b0b5;
  /* 浅色档：--dsw-alias-label-tertiary: var(--dsw-static-neutral-bluish-600) */
  --dsw-alias-label-tertiary: #81858c;
  --dsw-alias-interactive-bg-hover: #2631480f;
  --dsw-alias-bg-layer-3: #ffffff;
  --dsw-alias-state-success-primary: #22c55e;
  --dsw-alias-state-warn-primary: #f59e0b;
  --dsw-alias-state-error-primary: #ef4444;
}
body { margin: 0; font-family: var(--dsw-font-family); }
#hdr { display: flex; align-items: center; justify-content: space-between; gap: 8px;
  padding: 8px 12px; border-bottom: .5px solid var(--dsw-alias-border-l4); }
/* 容器：照抄 .…_headerUtilities */
#hdr .wSkVaW_headerUtilities { flex: none; align-items: center; gap: 8px; margin-left: 20px; display: flex; }
/* 左边邻居那条 slot 条目 */
.Rr11aa_entry { display: inline-flex; }
.Rr11aa_plain { box-sizing: border-box; height: 28px; border-radius: 14px;
  border: .5px solid var(--dsw-alias-border-l4); background: transparent; color: var(--dsw-alias-label-primary);
  padding: 5px 10px; font-size: 11px; line-height: 16px; cursor: pointer; }
/* open-in-app 那条 slot 条目的包裹层 + 分体控件（照抄 CAgGvG_*） */
.Ss22bb_entry { display: inline-flex; }
.CAgGvG_split { box-sizing: border-box; border: .5px solid var(--dsw-alias-border-l4);
  height: 28px; border-radius: 14px; align-items: stretch; display: inline-flex; overflow: hidden; }
.CAgGvG_main, .CAgGvG_chevron { color: var(--dsw-alias-label-primary); cursor: pointer;
  white-space: nowrap; background: 0 0; border: 0; align-items: center; gap: 5px;
  font-size: 11px; font-weight: 400; line-height: 16px; display: inline-flex; }
.CAgGvG_main { padding: 5px 6px 5px 7px; }
.CAgGvG_chevron { border-left: .5px solid var(--dsw-alias-border-l4);
  color: var(--dsw-alias-label-secondary); padding: 5px 6px 5px 4px; }
`

const html = `<!doctype html><html><head><meta charset="utf-8"><title>lmc panel harness</title>
<style>${TOKENS}</style>
<style>${style}</style>
<script>${script}</script>
</head><body>
<div id="hdr">
  <span>会话标题</span>
  <div class="wSkVaW_headerUtilities">
    <div class="Rr11aa_entry"><button class="Rr11aa_plain">别的动作</button></div>
    <div class="Ss22bb_entry"><div class="CAgGvG_split"><button class="CAgGvG_main" aria-label="在 终端 中打开工作目录" data-state="idle">🖥</button><button class="CAgGvG_chevron" aria-haspopup="menu" aria-expanded="false" aria-label="选择打开方式">▾</button></div></div>
  </div>
</div>
<main style="padding:16px"><p id="marker">LMC_PAGE_READY</p><div style="height:1200px;background:linear-gradient(#fff,#eee)"></div></main>
</body></html>`
writeFileSync('/tmp/lmc-ui/index.html', html)

const SELFCHECK = {
  ok: true, verdict: 'warn', at: '2026-09-24T00:50:00.000Z', version: '1.4.0',
  summary: { candidates: 6, reachable: 1, withModels: 1, models: 1, linkedProviders: 1, pendingEndpoints: 1, ok: 4, warn: 2, fail: 0 },
  checks: [
    { id: 'plugin', level: 'ok', title: '插件已加载 v1.4.0', detail: 'x' },
    { id: 'targets', level: 'ok', title: '候选端点 6 个，存活 1 个', detail: 'x' },
    { id: 'warn-a', level: 'warn', title: '192.168.0.119:8000 还没接进模型列表', detail: '点「接入本地模型」即可建出 provider' },
    { id: 'warn-b', level: 'warn', title: 'x/y 的压缩阈值不够塞', detail: '把 maxTokens 降到 30000 或更小' }
  ],
  targets: []
}
const RUN = {
  ok: true, at: '2026-09-24T18:19:31.615Z', dryRun: false, wrote: false,
  candidates: ['192.168.0.119:8080'], targets: [
    { baseURL: 'http://192.168.0.119:8080/v1', origin: 'http://192.168.0.119:8080', host: '192.168.0.119', port: 8080, reachable: true, status: 200, latencyMs: 9, engine: 'vllm', models: [{ id: 'Qwen3.8-27B-Q6-dual-5060ti', name: 'x', contextWindow: 150000 }] },
    ...(ROWS > 0 ? Array.from({ length: ROWS }, (_, i) => 9000 + i) : [8000, 30000, 8081, 11434, 1234]).map((port) => ({
      baseURL: `http://192.168.0.119:${port}/v1`, origin: `http://192.168.0.119:${port}`, host: '192.168.0.119', port,
      reachable: false, code: 'ECONNREFUSED', error: `ECONNREFUSED：connect ECONNREFUSED 192.168.0.119:${port}`
    }))
  ],
  providers: [{ provider: 'qwen-local', baseURL: 'http://192.168.0.119:8080/v1', action: 'unchanged', changes: [], modelCount: 1 }]
}

/**
 * 当前"后端"的 verdict —— 可以用 `GET /__verdict?level=fail` 实时改，
 * 用来验「状态点会不会自己跟着变」（不用点任何按钮）。
 */
let verdict = 'ok'

const hits = []
const server = http.createServer((req, res) => {
  const url = new URL(req.url, 'http://localhost')
  const path = url.pathname
  hits.push(`${req.method} ${path}`)
  const send = (body) => {
    res.writeHead(200, { 'content-type': 'application/json; charset=utf-8' })
    res.end(JSON.stringify(body))
  }
  // 控制面：把「后端」拨到某个状态（ok / warn / fail），返回上一次的值
  if (path === '/__verdict') {
    const wanted = url.searchParams.get('level')
    const before = verdict
    if (wanted !== null) verdict = wanted
    console.log(`[control] verdict ${before} -> ${verdict}`)
    return send({ before, now: verdict })
  }
  if (path === '/local-models-connect/state') {
    return send({ ok: true, version: '1.8.0', config: {}, lastRun: null, lastSelfCheck: { verdict: 'ok', at: '2026-09-24T00:45:00.000Z' } })
  }
  if (path === '/local-models-connect/watch') {
    // 真插件那条轮询路：小载荷、只读
    return send({ ok: true, version: '1.8.0', verdict, at: new Date().toISOString(), summary: { ok: 6, warn: 0, fail: 0 } })
  }
  if (path === '/local-models-connect/panel') {
    // 用**真插件**的视图构造器，所以这里验的就是要上线的那份排版
    return send({ ok: true, at: RUN.at, view: plugin.buildPanelView(RUN, SELFCHECK) })
  }
  if (path === '/local-models-connect/selfcheck') return send(SELFCHECK)
  if (path === '/local-models-connect/run') return send(RUN)
  if (path === '/') {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' })
    return res.end(html)
  }
  res.writeHead(404); res.end('nope')
})
server.listen(PORT, '127.0.0.1', () => {
  console.log(`面板验证台: http://127.0.0.1:${PORT}/`)
  console.log('期望：#lmc-tools 是 .wSkVaW_headerUtilities 的**直接子项**，')
  console.log('      紧邻 .Ss22bb_entry 之前，且两者之间正好 8px（容器 gap）。')
  console.log('拨后端状态：curl "http://127.0.0.1:' + PORT + '/__verdict?level=fail"')
})
process.on('SIGTERM', () => { console.log('hits:', hits.join(' | ')); server.close(() => process.exit(0)) })
