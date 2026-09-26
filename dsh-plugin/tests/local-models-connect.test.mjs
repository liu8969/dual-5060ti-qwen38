/**
 * local-models-connect 的测试。分四层：
 *   1. 纯逻辑（不碰网络）：配置解析、列表解析、上下文字段优先级、合并、diff、压缩余量。
 *   2. 探测层：用本地起的三台假引擎（vLLM / llama.cpp / Ollama）+ 一个关掉的端口。
 *   3. 端到端：apply() 注册的路由挂到真 http 服务器上，走一遍 state/run/selfcheck/旧别名。
 *   4. 实机冒烟：真打 192.168.0.119:8080（不在线只记 warn，不算失败）。
 *
 * 跑法：node dsh-plugin/tests/local-models-connect.test.mjs [插件路径]
 *       （不给路径就测同仓库的 ../local-models-connect.v10.mjs）
 */
import assert from 'node:assert/strict'
import http from 'node:http'
import { fileURLToPath, pathToFileURL } from 'node:url'

const MODULE_PATH = process.argv[2] ?? fileURLToPath(new URL('../local-models-connect.v10.mjs', import.meta.url))
const plugin = await import(pathToFileURL(MODULE_PATH).href)

let pass = 0
let fail = 0
const failures = []

function ok(name) { pass++; console.log(`  \u001b[32m✓\u001b[0m ${name}`) }
function bad(name, detail) { fail++; failures.push(`${name} — ${detail}`); console.log(`  \u001b[31m✗\u001b[0m ${name}\n      ${detail}`) }
function t(name, fn) {
  try { fn(); ok(name) } catch (error) { bad(name, error.message) }
}
async function ta(name, fn) {
  try { await fn(); ok(name) } catch (error) { bad(name, error.message) }
}
const section = (title) => console.log(`\n\u001b[1m${title}\u001b[0m`)

// ─────────────────────────────────────────────────────────────────────────
// 假引擎
// ─────────────────────────────────────────────────────────────────────────

function json(res, status, body) {
  const text = JSON.stringify(body)
  res.writeHead(status, { 'content-type': 'application/json' })
  res.end(text)
}

/** 起一台假引擎；返回 { origin, port, close, hits }。 */
async function fakeEngine(handler) {
  const hits = []
  const server = http.createServer((req, res) => {
    let body = ''
    req.on('data', (chunk) => { body += chunk })
    req.on('end', () => {
      hits.push(`${req.method} ${req.url}`)
      handler(req, res, body)
    })
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const port = server.address().port
  return {
    origin: `http://127.0.0.1:${port}`,
    port,
    hits,
    close: () => new Promise((resolve) => server.close(resolve))
  }
}

const vllmEngine = () => fakeEngine((req, res) => {
  if (req.url === '/v1/models') {
    return json(res, 200, {
      object: 'list',
      data: [{
        id: 'Qwen3.8-27B-Q6-dual-5060ti', object: 'model', owned_by: 'vllm',
        root: '/home/lcy/Models/Merkyor-W4A4/NVFP4/W4A4', max_model_len: 150000
      }]
    })
  }
  if (req.url === '/health') return json(res, 200, { status: 'ok' })
  json(res, 404, {})
})

/** 慢引擎：用来制造「一个请求还在飞」的窗口（验全局互斥那条坑）。 */
const slowVllmEngine = (delayMs) => fakeEngine((req, res) => {
  if (req.url === '/v1/models') {
    return setTimeout(() => json(res, 200, {
      object: 'list',
      data: [{ id: 'Qwen3.8-27B-Q6-dual-5060ti', object: 'model', owned_by: 'vllm', max_model_len: 150000 }]
    }), delayMs)
  }
  if (req.url === '/health') return json(res, 200, { status: 'ok' })
  json(res, 404, {})
})

/** llama.cpp：/v1/models 只给训练长度，服务端实参只在 /props 里。 */
const llamaEngine = (serverCtx = 262144, trainCtx = 32768) => fakeEngine((req, res) => {
  if (req.url === '/v1/models') {
    return json(res, 200, {
      object: 'list',
      data: [{ id: 'qwen3-27b-gguf', object: 'model', meta: { n_ctx_train: trainCtx } }]
    })
  }
  if (req.url === '/props') {
    return json(res, 200, { default_generation_settings: { n_ctx: serverCtx }, n_ctx: serverCtx, total_slots: 1 })
  }
  if (req.url === '/health') return json(res, 200, { status: 'ok' })
  json(res, 404, {})
})

/** Ollama：/v1/models 什么上下文都不给，只有 /api/show 说得出。 */
const ollamaEngine = () => fakeEngine((req, res, body) => {
  if (req.url === '/v1/models') {
    return json(res, 200, { object: 'list', data: [{ id: 'qwen3:8b', object: 'model', owned_by: 'ollama' }] })
  }
  if (req.url === '/api/show') {
    assert.ok(body.includes('qwen3:8b'), 'ollama show 应该带上模型名')
    return json(res, 200, { model_info: { 'qwen3.context_length': 40960 } })
  }
  json(res, 404, {})
})

/** 一个确定没人听的端口。 */
async function deadPort() {
  const server = http.createServer(() => {})
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const port = server.address().port
  await new Promise((resolve) => server.close(resolve))
  return port
}

// ─────────────────────────────────────────────────────────────────────────
// 假宿主（Cordis ctx）
// ─────────────────────────────────────────────────────────────────────────

function applyOps(section, ops) {
  for (const op of ops) {
    let cursor = section
    for (const key of op.path.slice(0, -1)) {
      if (cursor[key] === undefined || typeof cursor[key] !== 'object') cursor[key] = {}
      cursor = cursor[key]
    }
    const last = op.path.at(-1)
    if (op.op === 'set') cursor[last] = op.value
    else delete cursor[last]
  }
}

function makeHost(initialSection = {}, options = {}) {
  const state = { section: structuredClone(initialSection) }
  const routes = new Map()
  const injects = []
  const effects = []
  const mutations = []
  const logs = []
  const ctx = {
    logger: { info: (...a) => logs.push(['info', a]), warn: (...a) => logs.push(['warn', a]), debug: () => {} },
    effect: (fn, label) => { effects.push({ label, dispose: fn() }) },
    on: (event, fn) => { injects.push({ event, fn }) },
    get: (name) => (name === 'credentials' ? options.credentials : undefined),
    webServer: { register: (route) => { routes.set(route.path, route); return () => routes.delete(route.path) } },
    settings: {
      section: (ns) => state.section[ns],
      mutate: async (ns, ops) => {
        mutations.push({ ns, ops })
        if (state.section[ns] === undefined || state.section[ns] === null) state.section[ns] = {}
        applyOps(state.section[ns], ops)
      }
    }
  }
  return { ctx, routes, injects, effects, mutations, logs, state }
}

/** 把注册的路由挂到真 http 服务器上，走一遍网络路径。 */
async function serveRoutes(routes) {
  const server = http.createServer((req, res) => {
    const path = new URL(req.url, 'http://localhost').pathname
    const route = routes.get(path)
    if (route === undefined) { res.writeHead(404); res.end('no route'); return }
    Promise.resolve(route.handler(req, res)).catch((error) => { res.writeHead(500); res.end(String(error)) })
  })
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve))
  const base = `http://127.0.0.1:${server.address().port}`
  return {
    base,
    get: async (path, init) => {
      const response = await fetch(base + path, init)
      return { status: response.status, body: await response.json() }
    },
    close: () => new Promise((resolve) => server.close(resolve))
  }
}

const baseConfig = (overrides = {}) => plugin.resolveConfig({ probeTimeoutMs: 800, concurrency: 4, autoProbe: false, ...overrides })

// ─────────────────────────────────────────────────────────────────────────
// 1. 纯逻辑
// ─────────────────────────────────────────────────────────────────────────

section('1. 纯逻辑')

t('配置：不写 config 也能工作（默认 119 + 端口表）', () => {
  const config = plugin.resolveConfig(undefined)
  assert.deepEqual(config.hosts, ['192.168.0.119'])
  assert.equal(config.ports[0], 8080)
  assert.equal(config.providerId, 'qwen-local')
  assert.equal(config.autoProbe, true)
  assert.equal(config.autoApply, true)
  assert.equal(config.prune, false)
  assert.equal(config.adoptUnknownContext, false)
})

t('配置：字符串形式的 hosts/ports 也收（方便在 yml 里一行写完）', () => {
  const config = plugin.resolveConfig({ hosts: '192.168.0.119, 10.0.0.5', ports: '8080 8000' })
  assert.deepEqual(config.hosts, ['192.168.0.119', '10.0.0.5'])
  assert.deepEqual(config.ports, [8080, 8000])
})

t('配置：坏值回落到默认，不抛', () => {
  const config = plugin.resolveConfig({ ports: 'abc', hosts: '', probeTimeoutMs: -5, concurrency: 0 })
  assert.deepEqual(config.ports, plugin.resolveConfig(undefined).ports)
  assert.deepEqual(config.hosts, ['192.168.0.119'])
  assert.equal(config.probeTimeoutMs, 1500)
  assert.equal(config.concurrency, 6)
})

t('配置：默认不挂凭据名（新环境没有那份凭据时不会 MISSING_CREDENTIAL）', () => {
  assert.equal(plugin.resolveConfig(undefined).apiKeyEnv, '')
})

t('配置：显式写空 = 关掉，不回落默认（providerId / displayName / apiKeyEnv）', () => {
  const config = plugin.resolveConfig({ providerId: '', displayName: '', apiKeyEnv: '' })
  assert.equal(config.providerId, '')
  assert.equal(config.displayName, '')
  assert.equal(config.apiKeyEnv, '')
  // 而「没写这个键」仍然走默认
  assert.equal(plugin.resolveConfig({}).providerId, 'qwen-local')
})

t('配置：apiKeyEnv 也接受 false 表示关掉', () => {
  assert.equal(plugin.resolveConfig({ apiKeyEnv: false }).apiKeyEnv, '')
})

t('parseListing：vLLM 的 data 数组读到 id 与 max_model_len', () => {
  const entries = plugin.parseListing({
    object: 'list',
    data: [{ id: 'Qwen3.8-27B-Q6-dual-5060ti', owned_by: 'vllm', max_model_len: 150000 }]
  })
  assert.equal(entries.length, 1)
  assert.equal(entries[0].id, 'Qwen3.8-27B-Q6-dual-5060ti')
  assert.equal(entries[0].contextWindow, 150000)
  assert.equal(plugin.detectEngine(entries), 'vllm')
})

t('parseListing：models 映射用键当 id（键优先于嵌套 id）', () => {
  const entries = plugin.parseListing({ models: { 'alias-a': { id: 'canonical', context_length: 8192 } } })
  assert.equal(entries[0].id, 'alias-a')
  assert.equal(entries[0].contextWindow, 8192)
})

t('parseListing：缺 id 的行跳过而不是整份失败', () => {
  const entries = plugin.parseListing({ data: [{ max_model_len: 100 }, { id: 'good' }] })
  assert.deepEqual(entries.map((e) => e.id), ['good'])
})

t('parseListing：既没有 data 也没有 models → 抛（调用方转成一条 error）', () => {
  assert.throws(() => plugin.parseListing({ object: 'list' }), /既没有 data/)
})

t('上下文：n_ctx_train 只进 trainContext，绝不进 contextWindow（llama.cpp 档）', () => {
  const entries = plugin.parseListing({ data: [{ id: 'gguf', meta: { n_ctx_train: 32768 } }] })
  assert.equal(entries[0].contextWindow, undefined, '训练长度不能当声明值')
  assert.equal(entries[0].trainContext, 32768)
})

t('上下文：强字段优先于训练长度', () => {
  const entries = plugin.parseListing({ data: [{ id: 'x', max_model_len: 150000, meta: { n_ctx_train: 32768 } }] })
  assert.equal(entries[0].contextWindow, 150000)
})

t('上下文：limit.context / max_input_tokens 也认', () => {
  assert.equal(plugin.readContext({ limit: { context: 4096 } }), 4096)
  assert.equal(plugin.readContext({ max_input_tokens: '200000' }), 200000)
})

t('readMaxTokens：max_output_tokens / limit.output 都认', () => {
  assert.equal(plugin.readMaxTokens({ max_output_tokens: 8192 }), 8192)
  assert.equal(plugin.readMaxTokens({ limit: { output: 2048 } }), 2048)
  assert.equal(plugin.readMaxTokens({}), undefined)
})

t('contextFromShow：Ollama 的 model_info.<family>.context_length', () => {
  assert.equal(plugin.contextFromShow({ model_info: { 'qwen3.context_length': 40960, 'x.block_count': 64 } }), 40960)
})

t('contextFromShow：退化到 parameters 里的 num_ctx', () => {
  assert.equal(plugin.contextFromShow({ parameters: 'stop "<|im_end|>"\nnum_ctx 32768\n' }), 32768)
})

t('compactionHeadroom：150000 / 16384 的余量算得对', () => {
  const head = plugin.compactionHeadroom(150000, 16384)
  assert.equal(head.threshold, 120000)
  assert.equal(head.headroom, 13616)
})

t('compactionHeadroom：声明的 maxTokens 过大时余量为负（会判 fail）', () => {
  assert.ok(plugin.compactionHeadroom(150000, 60000).headroom < 0)
})

t('isPrivate：内网与回环为真，公网为假', () => {
  assert.equal(plugin.isPrivate('http://192.168.0.119:8080/v1'), true)
  assert.equal(plugin.isPrivate('http://127.0.0.1:8080/v1'), true)
  assert.equal(plugin.isPrivate('http://10.1.2.3/v1'), true)
  assert.equal(plugin.isPrivate('https://api.openai.com/v1'), false)
})

t('mergeProfile：新建 provider 写全必需字段，且不给无鉴权端点挂凭据名', () => {
  const discovered = {
    baseURL: 'http://127.0.0.1:9999/v1', origin: 'http://127.0.0.1:9999', host: '127.0.0.1', port: 9999,
    engine: 'vllm', models: [{ id: 'm1', name: 'm1', contextWindow: 150000 }]
  }
  const { profile, skipped } = plugin.mergeProfile(discovered, undefined, { providerId: 'p', displayName: '本地', apiKeyEnv: '' }, baseConfig())
  assert.equal(profile.api, 'openai-completions')
  assert.equal(profile.baseURL, 'http://127.0.0.1:9999/v1')
  assert.deepEqual(profile.compat, { maxTokensField: 'max_tokens' })
  assert.equal(profile.streamIdleTimeoutMs, 600000)
  assert.equal(profile.defaultContextWindow, 150000, '兜底值取该端点已知上下文的最小值')
  assert.equal('apiKeyEnv' in profile, false)
  assert.deepEqual(skipped, [])
  assert.equal(profile.models[0].maxTokens, 16384)
})

t('mergeProfile：maxTokens 被夹在 contextWindow 之内', () => {
  const discovered = {
    baseURL: 'http://127.0.0.1:9999/v1', origin: 'http://127.0.0.1:9999', host: '127.0.0.1', port: 9999,
    engine: 'unknown', models: [{ id: 'tiny', name: 'tiny', contextWindow: 4096 }]
  }
  const { profile } = plugin.mergeProfile(discovered, undefined, { providerId: 'p', displayName: '', apiKeyEnv: '' }, baseConfig())
  assert.equal(profile.models[0].maxTokens, 4096)
})

t('mergeProfile：上下文未知时默认不建条目（从源头挡住 262144 回落）', () => {
  const discovered = {
    baseURL: 'http://127.0.0.1:9999/v1', origin: 'http://127.0.0.1:9999', host: '127.0.0.1', port: 9999,
    engine: 'unknown', models: [{ id: 'mystery', name: 'mystery', trainContext: 32768 }]
  }
  const { profile, skipped } = plugin.mergeProfile(discovered, undefined, { providerId: 'p', displayName: '', apiKeyEnv: '' }, baseConfig())
  assert.deepEqual(profile.models, [])
  assert.equal(skipped.length, 1)
  assert.match(skipped[0].reason, /训练长度/)
})

t('mergeProfile：adoptUnknownContext=true 时才用兜底值建条目', () => {
  const discovered = {
    baseURL: 'http://127.0.0.1:9999/v1', origin: 'http://127.0.0.1:9999', host: '127.0.0.1', port: 9999,
    engine: 'unknown',
    models: [{ id: 'known', name: 'known', contextWindow: 32000 }, { id: 'mystery', name: 'mystery' }]
  }
  const { profile, skipped } = plugin.mergeProfile(discovered, undefined, { providerId: 'p', displayName: '', apiKeyEnv: '' }, baseConfig({ adoptUnknownContext: true }))
  assert.deepEqual(skipped, [])
  assert.equal(profile.models.find((m) => m.id === 'mystery').contextWindow, 32000)
})

t('mergeProfile：已有条目的 contextWindow 取引擎值、name 保留人工写的', () => {
  const existing = {
    displayName: '我起的名', baseURL: 'http://127.0.0.1:9999/v1',
    models: [{ id: 'm1', name: '人工名', contextWindow: 999999, maxTokens: 16384 }]
  }
  const discovered = {
    baseURL: 'http://127.0.0.1:9999/v1', origin: 'http://127.0.0.1:9999', host: '127.0.0.1', port: 9999,
    engine: 'vllm', models: [{ id: 'm1', name: 'engine-name', contextWindow: 150000 }]
  }
  const { profile } = plugin.mergeProfile(discovered, existing, { providerId: 'p', displayName: '不该覆盖', apiKeyEnv: 'KEY' }, baseConfig())
  assert.equal(profile.displayName, '我起的名')
  assert.equal(profile.models[0].name, '人工名')
  assert.equal(profile.models[0].contextWindow, 150000)
  assert.equal(profile.baseURL, 'http://127.0.0.1:9999/v1')
})

t('mergeProfile：prune=false 保留引擎不再广告的旧条目', () => {
  const existing = { models: [{ id: 'old', name: 'old', contextWindow: 8192, maxTokens: 1024 }] }
  const discovered = {
    baseURL: 'http://127.0.0.1:9999/v1', origin: 'http://127.0.0.1:9999', host: '127.0.0.1', port: 9999,
    engine: 'vllm', models: [{ id: 'new', name: 'new', contextWindow: 150000 }]
  }
  const kept = plugin.mergeProfile(discovered, existing, { providerId: 'p', displayName: '', apiKeyEnv: '' }, baseConfig())
  assert.deepEqual(kept.profile.models.map((m) => m.id).sort(), ['new', 'old'])
  const pruned = plugin.mergeProfile(discovered, existing, { providerId: 'p', displayName: '', apiKeyEnv: '' }, baseConfig({ prune: true }))
  assert.deepEqual(pruned.profile.models.map((m) => m.id), ['new'])
})

t('mergeProfile：已有条目但引擎没报上下文 → 保留声明值，不跳过', () => {
  const existing = { models: [{ id: 'm1', name: 'm1', contextWindow: 150000, maxTokens: 16384 }] }
  const discovered = {
    baseURL: 'http://127.0.0.1:9999/v1', origin: 'http://127.0.0.1:9999', host: '127.0.0.1', port: 9999,
    engine: 'unknown', models: [{ id: 'm1', name: 'm1', trainContext: 32768 }]
  }
  const { profile, skipped } = plugin.mergeProfile(discovered, existing, { providerId: 'p', displayName: '', apiKeyEnv: '' }, baseConfig())
  assert.deepEqual(skipped, [])
  assert.equal(profile.models[0].contextWindow, 150000)
})

t('diffProfile：新增模型 / 改上下文 / 移除模型都出得来', () => {
  const before = { models: [{ id: 'a', contextWindow: 100, maxTokens: 10 }, { id: 'gone', contextWindow: 5 }] }
  const after = { models: [{ id: 'a', contextWindow: 200, maxTokens: 10 }, { id: 'b', contextWindow: 300 }] }
  const fields = plugin.diffProfile(before, after).map((c) => `${c.id ?? ''}:${c.field}`)
  assert.ok(fields.includes('a:contextWindow'))
  assert.ok(fields.includes('b:(新增模型)'))
  assert.ok(fields.includes('gone:(移除模型)'))
})

t('pickProviderId：baseURL 已存在 → 复用，不新建', () => {
  const existing = new Map([['http://192.168.0.119:8080', 'qwen-local']])
  const id = plugin.pickProviderId({ origin: 'http://192.168.0.119:8080', host: '192.168.0.119', port: 8080 },
    existing, { providerId: 'qwen-local' }, baseConfig(), new Set(['qwen-local']))
  assert.equal(id, 'qwen-local')
})

t('pickProviderId：固定名只给第一个命中者，第二个走自动名', () => {
  const naming = { providerId: 'qwen-local' }
  const first = plugin.pickProviderId({ origin: 'http://192.168.0.119:8080', host: '192.168.0.119', port: 8080 },
    new Map(), naming, baseConfig(), new Set())
  assert.equal(first, 'qwen-local')
  const second = plugin.pickProviderId({ origin: 'http://127.0.0.1:1234', host: '127.0.0.1', port: 1234 },
    new Map(), naming, baseConfig(), new Set(['qwen-local']))
  assert.equal(second, 'local-127-0-0-1-1234')
})

t('autoProviderId：点分/冒号都收敛成合法路由名', () => {
  assert.equal(plugin.autoProviderId('192.168.0.119', 8080), 'local-192-168-0-119-8080')
  assert.equal(plugin.autoProviderId('::1', 8000), 'local-1-8000')
})

t('candidateTargets：顺序 = hosts × ports，且去重', () => {
  const targets = plugin.candidateTargets(baseConfig({ hosts: ['192.168.0.119', '127.0.0.1'], ports: [8080, 8000], includeConfigured: false }), {})
  assert.deepEqual(targets.map((t) => `${t.host}:${t.port}`),
    ['192.168.0.119:8080', '192.168.0.119:8000', '127.0.0.1:8080', '127.0.0.1:8000'])
})

t('candidateTargets：includeConfigured 收编已配内网 provider，公网端点不纳', () => {
  const targets = plugin.candidateTargets(
    baseConfig({ hosts: ['192.168.0.119'], ports: [8080], includeConfigured: true }),
    { providers: { 'qwen-local': { baseURL: 'http://192.168.0.119:8080/v1' }, remote: { baseURL: 'https://api.example.com/v1' }, other: { baseURL: 'http://10.9.9.9:9000/v1' } } }
  )
  const keys = targets.map((t) => `${t.host}:${t.port}`)
  assert.deepEqual(keys, ['192.168.0.119:8080', '10.9.9.9:9000'])
})

t('indexProvidersByOrigin：按 origin 建索引', () => {
  const map = plugin.indexProvidersByOrigin({ providers: { a: { baseURL: 'http://192.168.0.119:8080/v1' }, b: { baseURL: 'nonsense' } } })
  assert.equal(map.get('http://192.168.0.119:8080'), 'a')
  assert.equal(map.size, 1)
})

await ta('credentialStatus：空名 = none（不挂凭据，不需要解析）', async () => {
  const status = await plugin.credentialStatus({ get: () => undefined }, '')
  assert.equal(status.state, 'none')
})

await ta('credentialStatus：凭据服务里有 → ok', async () => {
  const ctx = { get: (name) => (name === 'credentials' ? { resolve: async (ref) => (ref === 'HAVE' ? { value: 'secret' } : undefined) } : undefined) }
  const status = await plugin.credentialStatus(ctx, 'HAVE')
  assert.equal(status.state, 'ok')
  assert.equal(status.from, '凭据服务')
})

await ta('credentialStatus：凭据服务与环境变量都没有 → missing（用时会抛 MISSING_CREDENTIAL）', async () => {
  const ctx = { get: (name) => (name === 'credentials' ? { resolve: async () => undefined } : undefined) }
  const status = await plugin.credentialStatus(ctx, 'LMC_DEFINITELY_NOT_SET')
  assert.equal(status.state, 'missing')
})

await ta('credentialStatus：环境变量里有 → ok', async () => {
  process.env.LMC_TEST_CRED = 'from-env'
  try {
    const ctx = { get: (name) => (name === 'credentials' ? { resolve: async () => undefined } : undefined) }
    const status = await plugin.credentialStatus(ctx, 'LMC_TEST_CRED')
    assert.equal(status.state, 'ok')
    assert.equal(status.from, '环境变量')
  } finally { delete process.env.LMC_TEST_CRED }
})

await ta('credentialStatus：名字不合法 → missing 且说明原因', async () => {
  const status = await plugin.credentialStatus({ get: () => undefined }, 'not a valid name!')
  assert.equal(status.state, 'missing')
  assert.match(status.note, /不是合法的凭据名/)
})

await ta('credentialStatus：没有凭据服务、环境变量也没有 → unknown 而不是 missing（不许误判）', async () => {
  const status = await plugin.credentialStatus({ get: () => undefined }, 'LMC_DEFINITELY_NOT_SET')
  assert.equal(status.state, 'unknown')
})

// ─────────────────────────────────────────────────────────────────────────
// 2. 探测层（假引擎）
// ─────────────────────────────────────────────────────────────────────────

section('2. 探测层')

await ta('probeTarget：vLLM 形状 → id + 上下文 + engine', async () => {
  const engine = await vllmEngine()
  try {
    const result = await plugin.probeTarget(
      { baseURL: `${engine.origin}/v1`, origin: engine.origin, host: '127.0.0.1', port: engine.port }, baseConfig())
    assert.equal(result.reachable, true)
    assert.equal(result.engine, 'vllm')
    assert.equal(result.models[0].id, 'Qwen3.8-27B-Q6-dual-5060ti')
    assert.equal(result.models[0].contextWindow, 150000)
    assert.ok(result.latencyMs >= 0)
  } finally { await engine.close() }
})

await ta('probeTarget：llama.cpp 用 /props 的 n_ctx，而不是 /v1/models 的 n_ctx_train', async () => {
  const engine = await llamaEngine(262144, 32768)
  try {
    const result = await plugin.probeTarget(
      { baseURL: `${engine.origin}/v1`, origin: engine.origin, host: '127.0.0.1', port: engine.port }, baseConfig())
    assert.equal(result.engine, 'llama.cpp')
    assert.equal(result.models[0].contextWindow, 262144, '必须取服务端 -c 实参')
    assert.equal(result.models[0].trainContext, 32768)
    assert.ok(engine.hits.includes('GET /props'))
  } finally { await engine.close() }
})

await ta('probeTarget：Ollama 走 /api/show 拿 model_info.context_length', async () => {
  const engine = await ollamaEngine()
  try {
    const result = await plugin.probeTarget(
      { baseURL: `${engine.origin}/v1`, origin: engine.origin, host: '127.0.0.1', port: engine.port }, baseConfig())
    assert.equal(result.engine, 'ollama')
    assert.equal(result.models[0].contextWindow, 40960)
    assert.ok(engine.hits.some((h) => h.startsWith('POST /api/show')))
  } finally { await engine.close() }
})

await ta('probeTarget：端口不通 → reachable:false 且带 error，不抛', async () => {
  const port = await deadPort()
  const result = await plugin.probeTarget(
    { baseURL: `http://127.0.0.1:${port}/v1`, origin: `http://127.0.0.1:${port}`, host: '127.0.0.1', port }, baseConfig())
  assert.equal(result.reachable, false)
  assert.ok(typeof result.error === 'string' && result.error.length > 0)
})

await ta('probeTarget：端口不通时给出连接层错误码，而不是一句没用的 fetch failed', async () => {
  const port = await deadPort()
  const result = await plugin.probeTarget(
    { baseURL: `http://127.0.0.1:${port}/v1`, origin: `http://127.0.0.1:${port}`, host: '127.0.0.1', port }, baseConfig())
  assert.equal(result.reachable, false)
  assert.equal(result.code, 'ECONNREFUSED', '要能区分「主机通、端口没监听」与「连不上主机」')
  assert.match(result.error, /ECONNREFUSED/)
  assert.notEqual(result.error, 'fetch failed')
})

await ta('probeTarget：列表结构不对 → reachable 但带读不动的 error', async () => {
  const engine = await fakeEngine((req, res) => {
    if (req.url === '/v1/models') return json(res, 200, { object: 'list' })
    json(res, 404, {})
  })
  try {
    const result = await plugin.probeTarget(
      { baseURL: `${engine.origin}/v1`, origin: engine.origin, host: '127.0.0.1', port: engine.port }, baseConfig())
    assert.equal(result.reachable, true)
    assert.match(result.error, /列表读不动/)
  } finally { await engine.close() }
})

// ─────────────────────────────────────────────────────────────────────────
// 3. 端到端（apply + 真 http）
// ─────────────────────────────────────────────────────────────────────────

section('3. 端到端')

await ta('apply：注册 6 条路由，注入 style + script 两行，且启动日志有版本号', async () => {
  const host = makeHost()
  plugin.apply(host.ctx, { autoProbe: false })
  assert.deepEqual([...host.routes.keys()].sort(),
    [plugin.ROUTE_RUN, plugin.ROUTE_SELFCHECK, plugin.ROUTE_STATE, plugin.ROUTE_PANEL, plugin.ROUTE_WATCH, plugin.LEGACY_ROUTE_RUN].sort())
  const table = []
  for (const inject of host.injects) if (inject.event === 'webserver/index-inject') inject.fn(table)
  assert.equal(table.length, 2)
  assert.equal(table[0].kind, 'style')
  assert.equal(table[1].kind, 'script')
  assert.ok(!table[1].text.includes('</script'), '注入脚本里不能出现 </script')
  assert.ok(host.logs.some(([, args]) => args.map(String).join(' ').includes(plugin.VERSION)))
})

await ta('端到端：新环境（没有任何 provider）→ run 建出 provider 并写进设置', async () => {
  const vllm = await vllmEngine()
  const host = makeHost()
  plugin.apply(host.ctx, {
    autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false
  })
  const web = await serveRoutes(host.routes)
  try {
    const state = await web.get(plugin.ROUTE_STATE)
    assert.equal(state.body.version, plugin.VERSION)
    assert.equal(state.body.lastRun, null)

    const run = await web.get(plugin.ROUTE_RUN, { method: 'POST' })
    assert.equal(run.body.wrote, true)
    assert.equal(run.body.providers[0].provider, 'qwen-local')
    assert.equal(run.body.providers[0].action, 'created')

    // 写进去的东西必须能被下一次读出来（假宿主模拟了 set 语义）
    const written = host.state.section['llm-pi-ai'].providers['qwen-local']
    assert.equal(written.baseURL, `${vllm.origin}/v1`)
    assert.equal(written.api, 'openai-completions')
    assert.equal(written.defaultContextWindow, 150000)
    assert.equal(written.models[0].contextWindow, 150000)
    assert.equal(written.models[0].maxTokens, 16384)
    assert.equal(written.models[0].name, 'Qwen3.8-27B-Q6-dual-5060ti')
    assert.equal(host.mutations.length, 1)

    // 幂等：再跑一次不该再写
    const again = await web.get(plugin.ROUTE_RUN, { method: 'POST' })
    assert.equal(again.body.wrote, false)
    assert.equal(again.body.providers[0].action, 'unchanged')
    assert.equal(host.mutations.length, 1)
  } finally { await web.close(); await vllm.close() }
})

await ta('面板：单按钮挂到「在本地打开」左边且间距取容器 gap、报告合一、弹窗会自己收起', async () => {
  const host = makeHost()
  plugin.apply(host.ctx, { autoProbe: false, autoHideMs: 4321 })
  const table = []
  for (const inject of host.injects) if (inject.event === 'webserver/index-inject') inject.fn(table)
  const style = table.find((r) => r.kind === 'style').text
  const script = table.find((r) => r.kind === 'script').text

  // ① 位置：定位「在本地打开」那个分体控件的容器（<hash>_split + aria-haspopup=menu），插在它**前面**
  assert.match(script, /div\[class\*="_split"\]/, '要按分体控件的容器类名找锚点')
  assert.match(script, /button\[aria-haspopup="menu"\]/, '要用行为特征确认是那个分体控件，而不是只信哈希类名')
  assert.match(script, /insertBefore\(tools, anchor\)/, '必须是插到锚点前面 = 按钮在它左边')
  assert.ok(!/position: fixed; right: 18px; bottom: 18px/.test(style) || style.includes('lmc-float'),
    '常驻右下角的浮动面板只能作为拿不到头部时的退路')

  // ② 样式：照抄 dsh-client-ui-open-in-app 的度量 + 复用 DSH 设计令牌（深浅色主题自动跟）
  for (const token of ['--dsw-alias-border-l4', '--dsw-font-family', '--dsw-alias-label-primary',
    '--dsw-alias-interactive-bg-hover', '--dsw-alias-bg-layer-3', '--dsw-alias-state-warn-primary']) {
    assert.ok(style.includes(token), `样式里应复用设计令牌 ${token}`)
  }
  assert.match(style, /height: 28px/, '原生分体控件就是 28px 高')
  assert.match(style, /border-radius: 14px/)
  assert.match(style, /font-size: 11px/)

  // ③ 只有一个按钮（2026-09-24 把「自检」并进来了），一个工厂一个类名 = 同一套大小风格
  assert.match(script, /makeButton\('接入本地模型'/, '只留一个按钮')
  assert.doesNotMatch(script, /makeButton\('自检'/, '「自检」不再是独立按钮')
  assert.equal((script.match(/= makeButton\(/g) || []).length, 1, '只调一次 makeButton（函数定义不算）')
  assert.equal((script.match(/className = 'lmc-btn'/g) || []).length, 1)
  assert.match(script, /tools\.appendChild\(runButton\)/)
  assert.doesNotMatch(script, /checkButton|checkLabel/)

  // ④ 一次点击 = 一个请求（面板路由），排版在宿主侧算好；客户端只把 view 变成 DOM
  assert.ok(script.includes(`panel: ${JSON.stringify(plugin.ROUTE_PANEL)}`), '注入的是解析好的面板路径')
  assert.doesNotMatch(script, /API\.run|API\.selfcheck/, '不再分两次请求')
  assert.match(script, /const payload = await post\(API\.panel\);/)
  assert.match(script, /showView\(view, payload && payload\.error\)/)
  assert.match(script, /function renderView\(view, error\)/, '客户端只负责渲染视图')
  assert.match(script, /function makeRow\(row\)/, '一行 = 点 + 行首列 + 正文')
  assert.doesNotMatch(script, /renderRunLines|renderCheckLines|renderReport/, '旧的纯文本拼装已删')

  // ⑤ 锚点：限定在头部容器内、并爬到容器的**直接子项**（容器的 gap 才作用得到）
  assert.match(script, /_headerUtilities/, '先找头部容器')
  assert.match(script, /findSplitWithin\(container !== null \? container : document\)/, '在容器内找分体控件')
  assert.match(script, /while \(node\.parentElement !== null && node\.parentElement !== container\) node = node\.parentElement;/,
    '爬到容器的直接子项再插，否则拿不到容器的 8px')
  assert.match(style, /#lmc-tools \{ display: inline-flex; align-items: center; gap: 8px; \}/,
    '组内间距与容器的 gap 一致')

  // ⑥ 弹窗不常驻：不占布局 + 自动收起 + Esc 收起 + **右上角的叉**（而不是"整块可点"）
  assert.match(style, /#lmc-box \{ position: fixed/, '弹窗必须 fixed，否则会顶开页面')
  assert.match(script, /const AUTO_HIDE_MS = 4321;/, 'autoHideMs 要真的流进注入脚本')
  assert.match(script, /setTimeout\(hideBox, AUTO_HIDE_MS\)/)
  assert.match(script, /event\.key === 'Escape'/)
  assert.match(script, /className = 'lmc-close'/, '要有右上角的叉')
  assert.match(script, /close\.textContent = '×'/)
  assert.match(script, /close\.setAttribute\('aria-label', '收起'\)/)
  assert.match(style, /#lmc-box \.lmc-close \{ position: absolute; top: 4px; right: 4px/, '叉固定在右上角')
  assert.doesNotMatch(style, /#lmc-box \{[^}]*overflow: auto/, '外框不滚，否则叉会跟着正文滚走')
  // 高度自适应：默认没有滚动条（用户 2026-09-25 要求）；只有「内容比视口还高」才在 JS 里临时开
  assert.doesNotMatch(style, /max-height: 42vh/, '42vh 那个固定高度已去掉')
  assert.doesNotMatch(style, /#lmc-body \{[^}]*max-height/, '正文不再钉死 max-height')
  assert.doesNotMatch(style, /#lmc-body \{[^}]*overflow: auto/, '默认不滚')
  assert.match(script, /body\.style\.maxHeight = '';/, '每次展示先清掉上次可能留下的限高')
  assert.match(script, /if \(top \+ height > room\) top = Math\.max\(margin, room - height\);/, '放不下先往上挪')
  assert.match(script, /body\.style\.overflow = 'auto';/, '挪了还放不下才退化成可滚动')
  // 内部滚动不重排：placeBox() 会清限高再量（白做一次强制重排），位置也不需要更新
  assert.match(script, /if \(box !== null && \(event\.target === box \|\| box\.contains\(event\.target\)\)\) return;/,
    '弹窗内部滚动只忽略，不重排（省掉每次滚动一次的强制重排）')
  assert.doesNotMatch(script, /box\.addEventListener\('click', hideBox\)/, '整块弹窗不再可点')
  assert.doesNotMatch(script, /点这里/, '那行「点这里…」提示已去掉')
  assert.doesNotMatch(style, /#lmc-box \{[^}]*cursor: pointer/, '外框不该长得像个按钮')
})

await ta('端到端：dry=1 只看不写', async () => {
  const vllm = await vllmEngine()
  const host = makeHost()
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false })
  const web = await serveRoutes(host.routes)
  try {
    const run = await web.get(`${plugin.ROUTE_RUN}?dry=1`, { method: 'POST' })
    assert.equal(run.body.dryRun, true)
    assert.equal(run.body.wrote, false)
    assert.equal(run.body.providers[0].action, 'would-change')
    assert.equal(host.mutations.length, 0, 'dry-run 一个字节都不能写')
    assert.equal(host.state.section['llm-pi-ai'], undefined)
  } finally { await web.close(); await vllm.close() }
})

await ta('端到端：状态点会自己刷新（watch 轮询）—— 两条护栏 + 打的是只读那条路', async () => {
  const vllm = await vllmEngine()
  // 声明与引擎一致的 provider：这样自检是绿灯，正对"点该是绿的"那个场景
  const host = makeHost({
    'llm-pi-ai': {
      providers: {
        'qwen-local': {
          api: 'openai-completions', baseURL: `${vllm.origin}/v1`,
          models: [{ id: 'Qwen3.8-27B-Q6-dual-5060ti', contextWindow: 150000, maxTokens: 16384 }]
        }
      }
    }
  })
  plugin.apply(host.ctx, {
    autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false,
    autoHideMs: 4321, watchIntervalMs: 7000
  })
  const table = []
  for (const inject of host.injects) if (inject.event === 'webserver/index-inject') inject.fn(table)
  const script = table.find((r) => r.kind === 'script').text

  // ① 节奏真的流进注入脚本，并且确实起了定时器
  assert.match(script, /const WATCH_MS = 7000;/, 'watchIntervalMs 要真的流进脚本')
  assert.match(script, /setInterval\(watchOnce, WATCH_MS\)/)
  assert.match(script, /if \(WATCH_MS <= 0 \|\| watching \|\| busy\) return;/, '三条护栏：关掉 / 不叠 / 按钮在跑时不插队')
  assert.match(script, /document\.visibilityState === 'hidden'\) return;/, '页面不可见时不打')
  // ② 切回来立刻补一次 —— 这正是"不用手动点"的关键
  assert.match(script, /addEventListener\('visibilitychange'/)
  assert.match(script, /addEventListener\('focus', watchOnce\)/)
  assert.match(script, /  startWatch\(\);/)
  // ③ 打的是**只读**那条 watch 路，不是 /panel（会写设置），也不是 /selfcheck（走全局互斥）
  assert.match(script, new RegExp(`watch: ${JSON.stringify(plugin.ROUTE_WATCH).replace(/[/-]/g, '\\$&')}`))
  assert.match(script, /await post\(API\.watch\)/)
  assert.doesNotMatch(script, /await post\(API\.selfcheck\)/)

  // ④ 真的能拿到 verdict（不是只写了段死代码）
  const web = await serveRoutes(host.routes)
  try {
    const watch = await web.get(plugin.ROUTE_WATCH)
    assert.equal(watch.status, 200)
    assert.equal(watch.body.verdict, 'ok')
    assert.ok(watch.body.at, '要带时间戳，否则用户不知道这个点有多旧')
    assert.equal(watch.body.view, undefined, 'watch 只回小载荷，不回三块视图')
  } finally { await web.close(); await vllm.close() }
})

await ta('轮询：TTL 内复用缓存不重探；过期后重探并跟着后端翻转', async () => {
  // 一台可以"拔线"的引擎：拔线后连接被掐（探不到），但仍能记到 hits ——
  // 这样「有没有重探」这件事才量得出来（直接 close 掉引擎就一起把计数弄没了）。
  let engineUp = true
  const vllm = await fakeEngine((req, res) => {
    if (!engineUp) { req.socket.destroy(); return }
    if (req.url === '/v1/models') {
      return json(res, 200, {
        object: 'list',
        data: [{ id: 'Qwen3.8-27B-Q6-dual-5060ti', object: 'model', owned_by: 'vllm', max_model_len: 150000 }]
      })
    }
    if (req.url === '/health') return json(res, 200, { status: 'ok' })
    json(res, 404, {})
  })
  const host = makeHost({
    'llm-pi-ai': {
      providers: {
        'qwen-local': {
          api: 'openai-completions', baseURL: `${vllm.origin}/v1`,
          models: [{ id: 'Qwen3.8-27B-Q6-dual-5060ti', contextWindow: 150000, maxTokens: 16384 }]
        }
      }
    }
  })
  const config = plugin.resolveConfig({
    autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false,
    watchIntervalMs: 60000, probeTimeoutMs: 500
  })
  try {
    // 先跑一次自检，把缓存做热
    const first = await plugin.watchVerdict(host.ctx, config)
    assert.equal(first.verdict, 'ok')
    const hitsAfterFirst = vllm.hits.length

    // 后端"拔线"：TTL 内不该重探 → 仍然回缓存里的 ok，且一次探测都没发
    engineUp = false
    const cached = await plugin.watchVerdict(host.ctx, config)
    assert.equal(cached.verdict, 'ok', 'TTL 内复用缓存，不重探')
    assert.equal(vllm.hits.length, hitsAfterFirst, 'TTL 内一次探测都不该发')

    // TTL 过期（用 0 逼它立刻重探）→ 真的去探了，verdict 跟着翻成 fail
    const forced = await plugin.watchVerdict(host.ctx, { ...config, watchIntervalMs: 0 })
    assert.equal(forced.verdict, 'fail', '重探之后要反映"后端断了"')
    assert.ok(vllm.hits.length > hitsAfterFirst, '过期后必须重探')
  } finally { await vllm.close().catch(() => {}) }
})

await ta('端到端：GET 打 run → 明确报错（写操作只收 POST）', async () => {
  const host = makeHost()
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [], includeConfigured: false })
  const web = await serveRoutes(host.routes)
  try {
    const run = await web.get(plugin.ROUTE_RUN)
    assert.equal(run.body.ok, false)
    assert.match(run.body.error, /只接受 POST/)
  } finally { await web.close() }
})

await ta('端到端：旧路径 /local-models-sync/run 仍然可用（v1 兼容）', async () => {
  const vllm = await vllmEngine()
  const host = makeHost()
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false })
  const web = await serveRoutes(host.routes)
  try {
    const run = await web.get(plugin.LEGACY_ROUTE_RUN, { method: 'POST' })
    assert.equal(run.body.providers[0].provider, 'qwen-local')
  } finally { await web.close(); await vllm.close() }
})

await ta('端到端：自检 verdict=warn（端点活着、声明还没接上）', async () => {
  const vllm = await vllmEngine()
  const host = makeHost()
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false })
  const web = await serveRoutes(host.routes)
  try {
    const check = await web.get(plugin.ROUTE_SELFCHECK)
    assert.equal(check.body.verdict, 'warn')
    assert.equal(check.body.summary.reachable, 1)
    assert.equal(check.body.summary.models, 1)
    assert.equal(check.body.summary.pendingEndpoints, 1)
    assert.ok(check.body.checks.some((c) => c.id === 'link:http://127.0.0.1:' + vllm.port))
    assert.equal(check.body.targets[0].health.ok, true)
    assert.equal(host.mutations.length, 0, '自检只读')
  } finally { await web.close(); await vllm.close() }
})

await ta('端到端：接上之后自检 verdict=ok，且余量算得出来', async () => {
  const vllm = await vllmEngine()
  const host = makeHost()
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false })
  const web = await serveRoutes(host.routes)
  try {
    await web.get(plugin.ROUTE_RUN, { method: 'POST' })
    const check = await web.get(plugin.ROUTE_SELFCHECK)
    assert.equal(check.body.verdict, 'ok', JSON.stringify(check.body.checks.filter((c) => c.level !== 'ok')))
    assert.equal(check.body.summary.linkedProviders, 1)
    assert.equal(check.body.summary.pendingEndpoints, 0)
    assert.ok(check.body.checks.some((c) => c.id === 'ok:qwen-local/Qwen3.8-27B-Q6-dual-5060ti'))
  } finally { await web.close(); await vllm.close() }
})

await ta('端到端：声明被改坏（999999）→ 自检报 warn 且 run 修回来', async () => {
  const vllm = await vllmEngine()
  const host = makeHost({
    'llm-pi-ai': {
      providers: {
        'qwen-local': {
          displayName: '人工写的名字', api: 'openai-completions', baseURL: `${vllm.origin}/v1`,
          models: [{ id: 'Qwen3.8-27B-Q6-dual-5060ti', name: '人工模型名', contextWindow: 999999, maxTokens: 16384 }]
        }
      }
    }
  })
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false })
  const web = await serveRoutes(host.routes)
  try {
    const before = await web.get(plugin.ROUTE_SELFCHECK)
    assert.equal(before.body.verdict, 'warn')
    assert.ok(before.body.checks.some((c) => c.id.startsWith('context:qwen-local/')))

    const run = await web.get(plugin.ROUTE_RUN, { method: 'POST' })
    assert.equal(run.body.providers[0].action, 'updated')
    const fixed = host.state.section['llm-pi-ai'].providers['qwen-local']
    assert.equal(fixed.models[0].contextWindow, 150000)
    assert.equal(fixed.displayName, '人工写的名字', '人工字段不被覆盖')
    assert.equal(fixed.models[0].name, '人工模型名', '人工模型名不被覆盖')

    const after = await web.get(plugin.ROUTE_SELFCHECK)
    assert.equal(after.body.verdict, 'ok')
  } finally { await web.close(); await vllm.close() }
})

await ta('端到端：手写的 maxTokens 大过压缩余量 → 自检 fail 并给出建议值', async () => {
  const vllm = await vllmEngine()
  const host = makeHost({
    'llm-pi-ai': {
      providers: {
        'qwen-local': {
          api: 'openai-completions', baseURL: `${vllm.origin}/v1`,
          models: [{ id: 'Qwen3.8-27B-Q6-dual-5060ti', contextWindow: 150000, maxTokens: 60000 }]
        }
      }
    }
  })
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false })
  const web = await serveRoutes(host.routes)
  try {
    // maxTokens 是使用偏好、不是引擎事实，run 不会替你改（只夹到 contextWindow 之内）——
    // 这种「声明能过 schema、但压缩阈值塞不下」的坑才是自检存在的意义。
    await web.get(plugin.ROUTE_RUN, { method: 'POST' })
    assert.equal(host.state.section['llm-pi-ai'].providers['qwen-local'].models[0].maxTokens, 60000)
    const check = await web.get(plugin.ROUTE_SELFCHECK)
    assert.equal(check.body.verdict, 'fail')
    const headroom = check.body.checks.find((c) => c.id.startsWith('headroom:qwen-local/'))
    assert.ok(headroom, '必须报出 headroom 这一条')
    assert.match(headroom.detail, /把 maxTokens 降到 30000/)
  } finally { await web.close(); await vllm.close() }
})

await ta('端到端：一个端点都没通时，接入状态不能报绿灯（2026-09-24 的空转 ✓）', async () => {
  const port = await deadPort()
  // 复刻用户实际那一屏：一个内网 provider 在册、它的端点却一个都没通
  const host = makeHost({
    'llm-pi-ai': {
      providers: {
        'qwen-local': {
          api: 'openai-completions', baseURL: 'http://192.168.0.119:8080/v1', apiKeyEnv: 'LMC_PRESENT',
          models: [{ id: 'Qwen3.8-27B-Q6-dual-5060ti', contextWindow: 150000, maxTokens: 16384 }]
        }
      }
    }
  }, { credentials: { resolve: async (ref) => (ref === 'LMC_PRESENT' ? { value: 'x' } : undefined) } })
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [port], includeConfigured: false, probeTimeoutMs: 500 })
  const web = await serveRoutes(host.routes)
  try {
    const check = await web.get(plugin.ROUTE_SELFCHECK)
    assert.equal(check.body.verdict, 'fail')
    assert.equal(check.body.summary.reachable, 0)

    const linkage = check.body.checks.find((c) => c.id === 'linkage')
    assert.notEqual(linkage.level, 'ok', '没有可关联的端点时不能给绿灯 —— 这条 ✓ 什么都没验证')
    assert.equal(linkage.level, 'warn')
    assert.match(linkage.title, /无法判定/)

    const reach = check.body.checks.find((c) => c.id === 'reachability')
    assert.match(reach.detail, /ECONNREFUSED/)
    assert.match(reach.detail, /主机是通的/, '要说清是「主机通、端口没听」还是「连不上主机」')

    // 凭据那条仍然是绿的，但标题必须说清它与连通性无关
    const cred = check.body.checks.find((c) => c.id === 'credentials')
    assert.equal(cred.level, 'ok')
    assert.match(cred.title, /与端点是否连得通无关/)
  } finally { await web.close() }
})

await ta('端到端：全部端口不通 → 自检 fail 且指明「一个都没通」', async () => {
  const port = await deadPort()
  const host = makeHost()
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [port], includeConfigured: false, probeTimeoutMs: 500 })
  const web = await serveRoutes(host.routes)
  try {
    const check = await web.get(plugin.ROUTE_SELFCHECK)
    assert.equal(check.body.verdict, 'fail')
    assert.ok(check.body.checks.some((c) => c.id === 'reachability'))
    assert.equal(check.body.summary.reachable, 0)
  } finally { await web.close() }
})

await ta('端到端：上下文读不到 → 不建 provider，报告里明说', async () => {
  const engine = await fakeEngine((req, res) => {
    if (req.url === '/v1/models') return json(res, 200, { data: [{ id: 'mystery', object: 'model' }] })
    json(res, 404, {})
  })
  const host = makeHost()
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [engine.port], includeConfigured: false, probeTimeoutMs: 500 })
  const web = await serveRoutes(host.routes)
  try {
    const run = await web.get(plugin.ROUTE_RUN, { method: 'POST' })
    assert.equal(run.body.providers[0].action, 'skipped')
    assert.match(run.body.providers[0].error, /上下文都读不到/)
    assert.equal(host.mutations.length, 0)
  } finally { await web.close(); await engine.close() }
})

await ta('端到端：开机自动接入（autoProbe=true）确实自己跑了一遍', async () => {
  const vllm = await vllmEngine()
  const host = makeHost()
  plugin.apply(host.ctx, {
    autoProbe: true, autoApply: true, autoProbeDelayMs: 10,
    hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false
  })
  try {
    const deadline = Date.now() + 4000
    while (host.mutations.length === 0 && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 25))
    }
    assert.equal(host.mutations.length, 1)
    assert.equal(host.state.section['llm-pi-ai'].providers['qwen-local'].models[0].contextWindow, 150000)
    const logged = host.logs.map(([, args]) => String(args[0])).join('\n')
    assert.match(logged, /自动接入/)
  } finally {
    for (const effect of host.effects) if (typeof effect.dispose === 'function') effect.dispose()
    await vllm.close()
  }
})

await ta('端到端：autoApply=false 时自动探测只报告不写', async () => {
  const vllm = await vllmEngine()
  const host = makeHost()
  plugin.apply(host.ctx, {
    autoProbe: true, autoApply: false, autoProbeDelayMs: 10,
    hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false
  })
  try {
    const deadline = Date.now() + 4000
    while (host.state.section['llm-pi-ai'] === undefined && Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 25))
    }
    await new Promise((resolve) => setTimeout(resolve, 150))
    assert.equal(host.mutations.length, 0)
    assert.equal(host.state.section['llm-pi-ai'], undefined)
  } finally {
    for (const effect of host.effects) if (typeof effect.dispose === 'function') effect.dispose()
    await vllm.close()
  }
})

await ta('端到端：一台 llama.cpp 一台 vLLM 同时在线 → 两条 provider，固定名给第一个', async () => {
  const vllm = await vllmEngine()
  const llama = await llamaEngine(131072, 32768)
  const host = makeHost()
  plugin.apply(host.ctx, {
    autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port, llama.port], includeConfigured: false
  })
  const web = await serveRoutes(host.routes)
  try {
    const run = await web.get(plugin.ROUTE_RUN, { method: 'POST' })
    const ids = run.body.providers.map((p) => p.provider).sort()
    assert.deepEqual(ids, ['local-127-0-0-1-' + llama.port, 'qwen-local'].sort())
    const providers = host.state.section['llm-pi-ai'].providers
    assert.equal(providers['qwen-local'].models[0].contextWindow, 150000)
    assert.equal(providers['local-127-0-0-1-' + llama.port].models[0].contextWindow, 131072)
    assert.equal(providers['local-127-0-0-1-' + llama.port].compat.maxTokensField, 'max_tokens')
  } finally { await web.close(); await vllm.close(); await llama.close() }
})

await ta('端到端：provider 声明了取不到的凭据 → 自检 fail 并指名 MISSING_CREDENTIAL', async () => {
  const vllm = await vllmEngine()
  const host = makeHost({
    'llm-pi-ai': {
      providers: {
        'qwen-local': {
          api: 'openai-completions', baseURL: `${vllm.origin}/v1`, apiKeyEnv: 'LMC_DEFINITELY_NOT_SET',
          models: [{ id: 'Qwen3.8-27B-Q6-dual-5060ti', contextWindow: 150000, maxTokens: 16384 }]
        }
      }
    }
  }, { credentials: { resolve: async () => undefined } })
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false })
  const web = await serveRoutes(host.routes)
  try {
    const check = await web.get(plugin.ROUTE_SELFCHECK)
    assert.equal(check.body.verdict, 'fail')
    const item = check.body.checks.find((c) => c.id === 'credential:qwen-local')
    assert.ok(item, '必须报出 credential:qwen-local')
    assert.match(item.detail, /MISSING_CREDENTIAL/)
  } finally { await web.close(); await vllm.close() }
})

await ta('端到端：凭据能取到时自检回到 ok', async () => {
  const vllm = await vllmEngine()
  const host = makeHost({
    'llm-pi-ai': {
      providers: {
        'qwen-local': {
          api: 'openai-completions', baseURL: `${vllm.origin}/v1`, apiKeyEnv: 'LMC_PRESENT',
          models: [{ id: 'Qwen3.8-27B-Q6-dual-5060ti', contextWindow: 150000, maxTokens: 16384 }]
        }
      }
    }
  }, { credentials: { resolve: async (ref) => (ref === 'LMC_PRESENT' ? { value: 'x' } : undefined) } })
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false })
  const web = await serveRoutes(host.routes)
  try {
    const check = await web.get(plugin.ROUTE_SELFCHECK)
    assert.equal(check.body.verdict, 'ok', JSON.stringify(check.body.checks.filter((c) => c.level !== 'ok')))
    assert.ok(check.body.checks.some((c) => c.id === 'credentials'))
  } finally { await web.close(); await vllm.close() }
})

await ta('端到端：全新环境建出的 provider 不挂任何凭据名', async () => {
  const vllm = await vllmEngine()
  const host = makeHost()
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false })
  const web = await serveRoutes(host.routes)
  try {
    await web.get(plugin.ROUTE_RUN, { method: 'POST' })
    const provider = host.state.section['llm-pi-ai'].providers['qwen-local']
    assert.equal('apiKeyEnv' in provider, false, '新环境不该挂本地机器才有的凭据名')
  } finally { await web.close(); await vllm.close() }
})

// ─────────────────────────────────────────────────────────────────────────
// 4. 面板视图（排版）—— 用户反复提的就是这块，所以钉在这里
// ─────────────────────────────────────────────────────────────────────────

section('4. 面板视图（排版）')

const HOST = '192.168.0.119'
const panelTarget = (port, over = {}) => ({
  host: HOST, port, baseURL: `http://${HOST}:${port}/v1`, origin: `http://${HOST}:${port}`, ...over
})
const upTarget = (port = 8080) => panelTarget(port, {
  reachable: true, status: 200, latencyMs: 10, engine: 'vllm',
  models: [{ id: 'Qwen3.8-27B-Q6-dual-5060ti', name: 'x', contextWindow: 150000 }]
})
const refusedTarget = (port) => panelTarget(port, {
  reachable: false, status: null, code: 'ECONNREFUSED',
  error: `ECONNREFUSED：connect ECONNREFUSED ${HOST}:${port}`
})
const panelRun = (over = {}) => ({
  ok: true, at: '2026-09-24T18:19:31.615Z', dryRun: false, wrote: false,
  targets: [upTarget(8080), refusedTarget(8000), refusedTarget(30000)],
  providers: [{ provider: 'qwen-local', baseURL: `http://${HOST}:8080/v1`, action: 'unchanged', changes: [], modelCount: 1 }],
  ...over
})
const panelCheck = (over = {}) => ({
  ok: true, verdict: 'ok',
  summary: { candidates: 3, reachable: 1, withModels: 1, models: 1, linkedProviders: 1, pendingEndpoints: 0, ok: 6, warn: 0, fail: 0 },
  checks: [
    { id: 'plugin', level: 'ok', title: '插件已加载' },
    { id: 'warn-a', level: 'warn', title: '端口 8000 还没接进模型列表', detail: '点一下即可建出 provider' }
  ],
  ...over
})
const rowsOf = (view, key) => view.blocks.find((block) => block.key === key).rows
const textsOf = (view) => view.blocks.flatMap((block) => block.rows.map((row) => row.text))

t('视图：三个块，标题与顺序固定（接入检查 / 模型列表 / 自检）', () => {
  const view = plugin.buildPanelView(panelRun(), panelCheck())
  assert.deepEqual(view.blocks.map((block) => block.title), ['接入检查', '模型列表', '自检'])
})

t('视图：同一台主机只写一次 —— 不再每行重复 IPv4', () => {
  const view = plugin.buildPanelView(panelRun(), panelCheck())
  const texts = textsOf(view)
  const hits = texts.join(' ').match(new RegExp(HOST.replace(/\./g, '\\.'), 'g')) || []
  assert.equal(hits.length, 1, '全视图里主机只出现一次')
  const hostRows = rowsOf(view, 'endpoints').filter((row) => row.level === 'host')
  assert.deepEqual(hostRows.map((row) => row.text), [HOST], '那一次就是 host 行')
  // 端口行里不该再夹带 IP（旧版每行都是 192.168.0.119:8000 这样）
  for (const row of rowsOf(view, 'endpoints')) {
    if (row.level === 'host') continue
    assert.ok(!row.text.includes(HOST), `端口行「${row.text}」不该再带主机`)
  }
})

t('视图：可达 = ok 点、被拒 = fail 点，端口在行首列（key）', () => {
  const rows = rowsOf(plugin.buildPanelView(panelRun(), panelCheck()), 'endpoints')
  assert.deepEqual(rows.find((row) => row.key === '8080').level, 'ok')
  assert.deepEqual(rows.find((row) => row.key === '8000').level, 'fail')
  assert.equal(rows.find((row) => row.key === '8080').text, 'vllm · Qwen3.8-27B-Q6-dual-5060ti @150000')
  assert.equal(rows.find((row) => row.key === '8000').text, '连接被拒')
})

t('视图：「连接被拒」的解释只说一次（不是每个端口重复一遍）', () => {
  const rows = rowsOf(plugin.buildPanelView(panelRun(), panelCheck()), 'endpoints')
  const notes = rows.filter((row) => row.level === 'detail' && row.text.includes('连接被拒 = '))
  assert.equal(notes.length, 1)
  assert.equal(rows.filter((row) => row.text === '连接被拒').length, 2, '端口行只留短标签')
})

t('视图：可达但没广告模型 → warn；skipped → neutral 且带原因', () => {
  const run = panelRun({
    targets: [panelTarget(8080, { reachable: true, engine: 'vllm', models: [] }),
      panelTarget(9000, { skipped: '不是内网端点（allowPublic=false）' })]
  })
  const rows = rowsOf(plugin.buildPanelView(run, panelCheck()), 'endpoints')
  assert.equal(rows.find((row) => row.key === '8080').level, 'warn')
  assert.equal(rows.find((row) => row.key === '9000').level, 'neutral')
  assert.match(rows.find((row) => row.key === '9000').text, /跳过/)
})

t('视图：多台主机 → 各自一个 host 行', () => {
  const run = panelRun({ targets: [upTarget(8080), panelTarget(8080, { host: '127.0.0.1', reachable: true, engine: 'vllm', models: [{ id: 'm', contextWindow: 4096 }] })] })
  const hosts = rowsOf(plugin.buildPanelView(run, panelCheck()), 'endpoints').filter((row) => row.level === 'host')
  assert.deepEqual(hosts.map((row) => row.text), ['192.168.0.119', '127.0.0.1'])
})

t('视图：provider 各种 action 都有人话 + 状态点', () => {
  const cases = [
    [{ action: 'unchanged', changes: [], modelCount: 1 }, 'ok', /已是最新（1 个模型）/],
    [{ action: 'created', changes: [{ id: 'm', field: '(新增模型)', from: null, to: 150000 }], modelCount: 1 }, 'ok', /已接入（新建 1 个模型）/],
    [{ action: 'updated', changes: [{ id: 'm', field: 'contextWindow', from: 100, to: 150000 }], modelCount: 1 }, 'ok', /已更新 1 处/],
    [{ action: 'would-change', changes: [{ field: 'baseURL', from: null, to: 'x' }], modelCount: 1 }, 'neutral', /dry-run/],
    [{ action: 'skipped', changes: [], modelCount: 0, error: '上下文都读不到' }, 'warn', /跳过/],
    [{ action: 'failed', changes: [], modelCount: 1, error: '写设置失败：boom' }, 'fail', /写设置失败/]
  ]
  for (const [provider, level, pattern] of cases) {
    const rows = rowsOf(plugin.buildPanelView(panelRun({ providers: [{ provider: 'qwen-local', ...provider }] }), panelCheck()), 'models')
    const row = rows.find((item) => item.key === 'qwen-local')
    assert.equal(row.level, level, `${provider.action} 的等级`)
    assert.match(row.text, pattern, `${provider.action} 的文案`)
  }
})

t('视图：写没写设置也有一行，且「没变」不等于「没写入」', () => {
  const unchanged = rowsOf(plugin.buildPanelView(panelRun({ wrote: false, dryRun: false }), panelCheck()), 'models')
  assert.equal(unchanged.at(-1).level, 'neutral')
  assert.match(unchanged.at(-1).text, /未写入设置（无需改动）/)
  const wrote = rowsOf(plugin.buildPanelView(panelRun({ wrote: true }), panelCheck()), 'models')
  assert.equal(wrote.at(-1).level, 'ok')
  assert.match(wrote.at(-1).text, /已写入设置/)
  const dry = rowsOf(plugin.buildPanelView(panelRun({ dryRun: true }), panelCheck()), 'models')
  assert.match(dry.at(-1).text, /dry-run/)
})

t('视图：自检块 —— 结论一行、ok 明细滤掉、非 ok 带 detail 行', () => {
  const rows = rowsOf(plugin.buildPanelView(panelRun(), panelCheck()), 'checks')
  assert.equal(rows[0].level, 'ok')
  assert.match(rows[0].text, /^OK · 6 通过 · 0 提醒 · 0 失败$/)
  assert.ok(!rows.some((row) => row.text === '插件已加载'), 'ok 的明细不列')
  assert.equal(rows[1].level, 'warn')
  assert.equal(rows[2].level, 'detail')
  assert.match(rows[2].text, /点一下即可建出 provider/)
})

t('视图：自检 verdict 决定结论行的颜色（fail 红 / warn 黄）', () => {
  const warn = rowsOf(plugin.buildPanelView(panelRun(), panelCheck({ verdict: 'warn', summary: { ok: 5, warn: 1, fail: 0 } })), 'checks')
  assert.equal(warn[0].level, 'warn')
  const fail = rowsOf(plugin.buildPanelView(panelRun(), panelCheck({ verdict: 'fail', summary: { ok: 5, warn: 0, fail: 1 } })), 'checks')
  assert.equal(fail[0].level, 'fail')
})

t('视图：出错也成块 —— run 报错进第一块，check 报错进第三块', () => {
  const broken = plugin.buildPanelView({ error: '端点全挂' }, { error: '自检炸了' })
  assert.deepEqual(broken.blocks.map((block) => block.title), ['接入检查', '模型列表', '自检'])
  assert.equal(rowsOf(broken, 'endpoints')[0].level, 'fail')
  assert.match(rowsOf(broken, 'endpoints')[0].text, /端点全挂/)
  assert.equal(rowsOf(broken, 'checks')[0].level, 'fail')
  assert.match(rowsOf(broken, 'checks')[0].text, /自检炸了/)
  assert.equal(broken.verdict, 'fail')
})

t('视图：没有候选端点 / 没有 provider 时给灰点说明，而不是空白块', () => {
  const view = plugin.buildPanelView({ ok: true, at: 'x', targets: [], providers: [], wrote: false }, panelCheck())
  assert.equal(rowsOf(view, 'endpoints')[0].level, 'neutral')
  assert.match(rowsOf(view, 'endpoints')[0].text, /没有候选端点/)
  assert.equal(rowsOf(view, 'models')[0].level, 'neutral')
  assert.match(rowsOf(view, 'models')[0].text, /没有可接入的 provider/)
})

t('视图：整份可 JSON 化（要过 HTTP）', () => {
  const view = plugin.buildPanelView(panelRun(), panelCheck())
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(view)))
})

t('视图：时间戳是本机时区的可读形式，不是 ISO', () => {
  const view = plugin.buildPanelView(panelRun({ at: '2026-09-24T18:19:31.615Z' }), panelCheck())
  assert.match(view.head, /^检查于 \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$/)
  assert.ok(!view.head.includes('T') && !view.head.includes('Z'))
})

t('shortUnreachable：各连接层错误码都有人话，原始报文不丢（进 title）', () => {
  assert.equal(plugin.shortUnreachable({ code: 'ECONNREFUSED' }), '连接被拒')
  assert.equal(plugin.shortUnreachable({ code: 'ETIMEDOUT' }), '连接超时')
  assert.equal(plugin.shortUnreachable({ code: 'EHOSTUNREACH' }), '主机 / 网络不可达')
  assert.equal(plugin.shortUnreachable({ code: 'ENOTFOUND' }), '主机名解析不了')
  assert.equal(plugin.shortUnreachable({ error: 'weird' }), '不通')
  const rows = rowsOf(plugin.buildPanelView(panelRun(), panelCheck()), 'endpoints')
  const row = rows.find((item) => item.key === '8000')
  assert.match(row.title, /ECONNREFUSED：connect ECONNREFUSED 192\.168\.0\.119:8000/, '悬停里是完整原文')
})

await ta('端到端：POST /panel 回来的就是三块的视图（客户端照它渲染）', async () => {
  const vllm = await vllmEngine()
  const host = makeHost()
  plugin.apply(host.ctx, { autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false })
  const web = await serveRoutes(host.routes)
  try {
    const panel = await web.get(plugin.ROUTE_PANEL, { method: 'POST' })
    assert.equal(panel.status, 200)
    assert.deepEqual(panel.body.view.blocks.map((block) => block.title), ['接入检查', '模型列表', '自检'])
    assert.equal(panel.body.view.verdict, 'ok')
    const rows = panel.body.view.blocks.flatMap((block) => block.rows)
    assert.ok(rows.some((row) => row.level === 'host' && row.text === '127.0.0.1'))
    assert.ok(rows.some((row) => row.level === 'ok' && row.key === String(vllm.port)), '可达端口是绿点行')
    assert.ok(rows.some((row) => row.key === 'qwen-local' && /已接入|已是最新/.test(row.text)))
    assert.equal(host.mutations.length, 1, '面板那次确实跑了接入（新建了 provider）')

    const notPost = await web.get(plugin.ROUTE_PANEL)
    assert.equal(notPost.body.ok, false)
    assert.match(notPost.body.error, /只接受 POST/)
  } finally { await web.close(); await vllm.close() }
})

await ta('端到端：轮询在飞时点按钮，/panel 不会被轮询那份结果劫持（全局互斥的坑）', async () => {
  // singleFlight 是**没有分键的全局互斥**：谁先飞，后来者直接复用同一个 promise。
  // 所以如果 /watch 也走 singleFlight，轮询在飞时点按钮，/panel 就会拿到
  // { verdict, at, summary } 而不是 { view } —— 面板渲染成空。这条测试钉住"不套"。
  const vllm = await slowVllmEngine(300)
  const host = makeHost()
  plugin.apply(host.ctx, {
    autoProbe: false, hosts: ['127.0.0.1'], ports: [vllm.port], includeConfigured: false,
    watchIntervalMs: 0, probeTimeoutMs: 3000   // 0 = 每次都真探，保证 watch 真的在飞
  })
  const web = await serveRoutes(host.routes)
  try {
    const flying = web.get(plugin.ROUTE_WATCH)
    await new Promise((resolve) => setTimeout(resolve, 60))   // 让 watch 先进入飞行状态
    const panel = await web.get(plugin.ROUTE_PANEL, { method: 'POST' })
    const watch = await flying

    assert.ok(panel.body.view, '/panel 必须拿回三块视图，而不是被轮询的结果顶掉')
    assert.equal(panel.body.view.verdict, 'ok')
    assert.equal(typeof watch.body.verdict, 'string', '/watch 回自己的小载荷')
    assert.equal(watch.body.view, undefined)
  } finally { await web.close(); await vllm.close() }
})

// ─────────────────────────────────────────────────────────────────────────
// 5. 实机冒烟
// ─────────────────────────────────────────────────────────────────────────

section('5. 实机冒烟（192.168.0.119:8080）')

await ta('真实端点：读出 model id 与 max_model_len', async () => {
  const result = await plugin.probeTarget(
    { baseURL: 'http://192.168.0.119:8080/v1', origin: 'http://192.168.0.119:8080', host: '192.168.0.119', port: 8080 },
    baseConfig({ probeTimeoutMs: 4000 }))
  if (!result.reachable) {
    console.log(`  \u001b[33m·\u001b[0m 实机不在线，跳过（${result.error}）`)
    return
  }
  assert.equal(result.engine, 'vllm')
  assert.equal(result.models[0].id, 'Qwen3.8-27B-Q6-dual-5060ti')
  assert.equal(result.models[0].contextWindow, 150000)
})

// ─────────────────────────────────────────────────────────────────────────

console.log(`\n\u001b[1m${pass} 通过，${fail} 失败\u001b[0m`)
if (fail > 0) {
  console.log('\n失败明细：')
  for (const line of failures) console.log('  · ' + line)
  process.exitCode = 1
}
