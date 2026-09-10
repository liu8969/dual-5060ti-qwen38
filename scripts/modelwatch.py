#!/usr/bin/env python3
"""modelwatch — live prefill / decode / queue / cache monitor for vLLM, SGLang and llama.cpp.

Two modes:

  terminal       one line per tick, SSH-friendly
      modelwatch.py --gpu --interval 2
      modelctl watch --gpu

  web dashboard  self-contained single page, live via SSE, no external assets
      modelwatch.py --serve --url http://192.168.0.119:8080
      -> http://127.0.0.1:8090

It reads the engine's own Prometheus endpoint and diffs the counters, so the numbers do not depend
on any engine's log format, and a cumulative cache ratio is never confused with one idle logging
window (an idle 10 s window legitimately reports a 0% hit rate).
"""
from __future__ import annotations

import argparse
import collections
import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# --------------------------------------------------------------------------- engine maps
ENGINES = {
    "vllm": {
        "match": "vllm:prompt_tokens_total",
        "prompt_tokens": "vllm:prompt_tokens_total",
        "generation_tokens": "vllm:generation_tokens_total",
        "running": "vllm:num_requests_running",
        "waiting": "vllm:num_requests_waiting",
        "kv_usage": "vllm:kv_cache_usage_perc",
        "kv_usage_scale": 100.0,
        "cache_hits": "vllm:prefix_cache_hits_total",
        "cache_queries": "vllm:prefix_cache_queries_total",
        "preemptions": "vllm:num_preemptions_total",
        "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
        "accepted_tokens": "vllm:spec_decode_num_accepted_tokens_total",
    },
    "sglang": {
        "match": "sglang:prompt_tokens_total",
        "prompt_tokens": "sglang:prompt_tokens_total",
        "generation_tokens": "sglang:generation_tokens_total",
        "running": "sglang:num_running_reqs",
        "waiting": "sglang:num_queue_reqs",
        "kv_usage": "sglang:token_usage",
        "kv_usage_scale": 100.0,
        "cache_hits": None,
        "cache_queries": None,
        "cache_rate_gauge": "sglang:cache_hit_rate",
        "preemptions": None,
        "draft_tokens": None,
        "accepted_tokens": None,
    },
    "llama": {
        "match": "llamacpp:prompt_tokens_total",
        "prompt_tokens": "llamacpp:prompt_tokens_total",
        "generation_tokens": "llamacpp:tokens_predicted_total",
        "running": "llamacpp:requests_processing",
        "waiting": "llamacpp:requests_deferred",
        "kv_usage": "llamacpp:kv_cache_usage_ratio",
        "kv_usage_scale": 100.0,
        "cache_hits": None,
        "cache_queries": None,
        "preemptions": None,
        "draft_tokens": None,
        "accepted_tokens": None,
    },
}


# --------------------------------------------------------------------------- prometheus
def fetch(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url.rstrip("/") + "/metrics", timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def parse_prometheus(text: str):
    """name -> sum over all label sets."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "{" in line:
            name, rest = line.split("{", 1)
            _, _, value = rest.rpartition("}")
            value = value.strip()
        else:
            parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                continue
            name, value = parts[0], parts[1]
        try:
            val = float(value)
        except ValueError:
            continue
        name = name.strip()
        out[name] = out.get(name, 0.0) + val
    return out


class Sample:
    __slots__ = ("t", "metrics")

    def __init__(self, t, metrics):
        self.t = t
        self.metrics = metrics

    def total(self, name):
        if not name:
            return None
        return self.metrics.get(name)


def rate(prev, cur, name, default=None):
    if not name:
        return default
    a, b = prev.total(name), cur.total(name)
    if a is None or b is None:
        return default
    dt = cur.t - prev.t
    if dt <= 0 or b < a:
        return default
    return (b - a) / dt


def gpu_line():
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return None
    rows = []
    for line in out.splitlines():
        try:
            idx, util, power, mem = [x.strip() for x in line.split(",")]
            rows.append({"index": int(idx), "util": float(util), "power": float(power),
                         "mem_mib": int(mem)})
        except ValueError:
            continue
    return rows or None


def detect_engine(metrics):
    for name, spec in ENGINES.items():
        if spec["match"] in metrics:
            return name
    return None


# --------------------------------------------------------------------------- row builder
def build_row(prev, cur, spec, engine, with_gpu=False):
    prefill = rate(prev, cur, spec["prompt_tokens"]) if prev else None
    decode = rate(prev, cur, spec["generation_tokens"]) if prev else None
    running = cur.total(spec["running"])
    waiting = cur.total(spec["waiting"])
    kv = cur.total(spec["kv_usage"])
    if kv is not None and spec.get("kv_usage_scale"):
        kv *= spec["kv_usage_scale"]

    hits, queries = cur.total(spec.get("cache_hits")), cur.total(spec.get("cache_queries"))
    if hits is not None and queries:
        cache_pct, cache_kind = 100.0 * hits / queries, "cumulative"
    elif spec.get("cache_rate_gauge"):
        g = cur.total(spec["cache_rate_gauge"])
        cache_pct, cache_kind = (g * 100.0 if g is not None else None), "gauge"
    else:
        cache_pct, cache_kind = None, None

    if prev is not None:
        d = rate(prev, cur, spec.get("draft_tokens"))
        a = rate(prev, cur, spec.get("accepted_tokens"))
        accept_pct = (100.0 * a / d) if (d and a is not None and d > 0) else None
    else:
        accept_pct = None

    preempt = cur.total(spec.get("preemptions"))
    quiet = (running or 0) == 0 and (waiting or 0) == 0 and not prefill and not decode
    row = {
        "t": round(cur.t, 3),
        "clock": time.strftime("%H:%M:%S", time.localtime(cur.t)),
        "engine": engine,
        "running": running,
        "waiting": waiting,
        "prefill_tok_s": round(prefill, 1) if prefill is not None else None,
        "decode_tok_s": round(decode, 1) if decode is not None else None,
        "kv_cache_pct": round(kv, 1) if kv is not None else None,
        "cache_hit_pct": round(cache_pct, 1) if cache_pct is not None else None,
        "cache_hit_kind": cache_kind,
        "cache_hits_total": int(hits) if hits is not None else None,
        "cache_queries_total": int(queries) if queries is not None else None,
        "accept_pct": round(accept_pct, 1) if accept_pct is not None else None,
        "preemptions_total": int(preempt) if preempt is not None else None,
        "quiet": quiet,
    }
    if with_gpu:
        row["gpu"] = gpu_line()
    return row


# --------------------------------------------------------------------------- shared state
class State:
    def __init__(self, keep=1800):
        self.rows = collections.deque(maxlen=keep)
        self.seq = 0
        self.lock = threading.Lock()
        self.engine = None
        self.error = None

    def add(self, row):
        with self.lock:
            self.seq += 1
            row["seq"] = self.seq
            self.rows.append(row)

    def since(self, cursor):
        with self.lock:
            return [r for r in self.rows if r["seq"] > cursor]


def poll_loop(state, url, interval, engine_choice, with_gpu):
    prev = None
    while True:
        now = time.time()
        try:
            metrics = parse_prometheus(fetch(url))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            state.error = str(exc)
            state.add({"t": round(now, 3), "clock": time.strftime("%H:%M:%S"), "error": str(exc)})
            time.sleep(interval)
            continue
        if state.engine is None:
            state.engine = engine_choice if engine_choice != "auto" else detect_engine(metrics)
            if state.engine is None:
                state.error = "unrecognised metrics — is /metrics enabled on this engine?"
                state.add({"t": round(now, 3), "clock": time.strftime("%H:%M:%S"),
                           "error": state.error, "metric_names": len(metrics)})
                time.sleep(interval)
                continue
        state.error = None
        cur = Sample(now, metrics)
        state.add(build_row(prev, cur, ENGINES[state.engine], state.engine, with_gpu))
        prev = cur
        time.sleep(interval)


# --------------------------------------------------------------------------- dashboard page
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>modelwatch</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#0d1117;--fg:#e6edf3;--dim:#8b949e;--line:#21262d;--pre:#58a6ff;--dec:#3fb950;--kv:#d29922;--acc:#bc8cff;--err:#f85149}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:12px 16px;border-bottom:1px solid var(--line);display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
h1{font-size:15px;margin:0;font-weight:600}
.dim{color:var(--dim)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;padding:12px 16px}
.tile{background:#161b22;border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.tile .k{color:var(--dim);font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.tile .v{font-size:24px;font-weight:600;line-height:1.25}
.tile .s{color:var(--dim);font-size:11px}
.chartbox{margin:0 16px 12px;background:#161b22;border:1px solid var(--line);border-radius:8px;padding:10px}
canvas{width:100%;height:200px;display:block}
.legend{display:flex;gap:16px;color:var(--dim);font-size:11px;margin-bottom:6px;flex-wrap:wrap}
.legend i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
table{border-collapse:collapse;width:100%;font-size:12px}
th,td{text-align:right;padding:3px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--dim);font-weight:500;position:sticky;top:0;background:var(--bg)}
td:first-child,th:first-child{text-align:left}
.wrap{margin:0 16px 20px;background:#161b22;border:1px solid var(--line);border-radius:8px;max-height:260px;overflow:auto}
.gpu{display:flex;gap:18px;padding:8px 16px;color:var(--dim);flex-wrap:wrap}
.bar{display:inline-block;width:70px;height:7px;background:#21262d;border-radius:4px;overflow:hidden;vertical-align:middle;margin:0 6px}
.bar>i{display:block;height:100%;background:var(--dec)}
.err{color:var(--err);padding:6px 16px;min-height:20px}
</style></head><body>
<header>
  <h1>modelwatch</h1>
  <span class="dim" id="meta">connecting…</span>
  <span class="dim" id="state"></span>
</header>
<div class="tiles">
  <div class="tile"><div class="k">prefill tok/s</div><div class="v" id="pre">-</div><div class="s">prompt tokens/s</div></div>
  <div class="tile"><div class="k">decode tok/s</div><div class="v" id="dec">-</div><div class="s">generated tokens/s</div></div>
  <div class="tile"><div class="k">running / waiting</div><div class="v" id="run">-</div><div class="s">requests</div></div>
  <div class="tile"><div class="k">KV cache</div><div class="v" id="kv">-</div><div class="s">of pool</div></div>
  <div class="tile"><div class="k">prefix cache</div><div class="v" id="ch">-</div><div class="s">cumulative hits/queries</div></div>
  <div class="tile"><div class="k">spec acceptance</div><div class="v" id="ac">-</div><div class="s">windowed</div></div>
  <div class="tile"><div class="k">preemptions</div><div class="v" id="pe">-</div><div class="s">since start</div></div>
</div>
<div class="chartbox">
  <div class="legend">
    <span><i style="background:var(--pre)"></i>prefill tok/s</span>
    <span><i style="background:var(--dec)"></i>decode tok/s</span>
    <span><i style="background:var(--acc)"></i>accept %</span>
    <span><i style="background:var(--kv)"></i>KV used %</span>
  </div>
  <canvas id="c"></canvas>
</div>
<div class="gpu" id="gpu"></div>
<div class="err" id="err"></div>
<div class="wrap"><table><thead><tr>
<th>time</th><th>prefill</th><th>decode</th><th>run</th><th>wait</th><th>KV%</th><th>cache%</th><th>accept%</th><th>preempt</th>
</tr></thead><tbody id="rows"></tbody></table></div>
<script>
const hist=[],MAX=600,$=id=>document.getElementById(id);
const SERIES=[['prefill_tok_s','#58a6ff'],['decode_tok_s','#3fb950'],['accept_pct','#bc8cff'],['kv_cache_pct','#d29922']];
const fmt=(v,d=1)=>(v===null||v===undefined)?'-':Number(v).toFixed(d);
function draw(){
  const c=$('c'),ctx=c.getContext('2d'),W=c.width=c.clientWidth*2,H=c.height=400,pad=44;
  ctx.clearRect(0,0,W,H);
  let max=1;for(const r of hist)for(const[k]of SERIES){const v=r[k];if(v!=null&&v>max)max=v;}
  ctx.strokeStyle='#21262d';ctx.lineWidth=1;ctx.font='20px monospace';ctx.fillStyle='#8b949e';
  for(let i=0;i<=4;i++){const y=pad+(H-2*pad)*i/4;ctx.beginPath();ctx.moveTo(pad,y);ctx.lineTo(W-pad,y);ctx.stroke();
    ctx.fillText(String(Math.round(max*(1-i/4))),4,y+6);}
  const n=hist.length;
  for(const[key,color]of SERIES){
    ctx.strokeStyle=color;ctx.lineWidth=2;ctx.beginPath();let on=false;
    hist.forEach((r,i)=>{const v=r[key];if(v==null)return;
      const x=pad+(W-2*pad)*(MAX<=1?1:Math.min(1,i/Math.max(1,MAX-1)));
      const y=H-pad-(H-2*pad)*(v/max);on?ctx.lineTo(x,y):(ctx.moveTo(x,y),on=true);});
    ctx.stroke();
  }
  ctx.fillStyle='#8b949e';ctx.fillText('last '+n+' samples · scale max '+Math.round(max),pad+4,28);
}
function addRow(r){
  const tr=document.createElement('tr');
  tr.innerHTML=`<td>${r.clock||''}</td><td>${fmt(r.prefill_tok_s)}</td><td>${fmt(r.decode_tok_s)}</td>
  <td>${r.running??'-'}</td><td>${r.waiting??'-'}</td><td>${fmt(r.kv_cache_pct)}</td>
  <td>${fmt(r.cache_hit_pct)}</td><td>${fmt(r.accept_pct)}</td><td>${r.preemptions_total??'-'}</td>`;
  const b=$('rows');b.insertBefore(tr,b.firstChild);
  while(b.children.length>300)b.removeChild(b.lastChild);
}
function apply(r){
  if(r.error){$('err').textContent='engine error: '+r.error;return;}
  $('err').textContent='';
  $('meta').textContent=`engine ${r.engine||'?'} · ${r.clock} · ${hist.length} samples`;
  $('state').textContent=r.quiet?'quiet (no requests in this window)':'';
  $('pre').textContent=fmt(r.prefill_tok_s);
  $('dec').textContent=fmt(r.decode_tok_s);
  $('run').textContent=`${r.running??'-'} / ${r.waiting??'-'}`;
  $('kv').textContent=fmt(r.kv_cache_pct)+'%';
  $('ch').textContent=fmt(r.cache_hit_pct)+'%';
  $('ac').textContent=fmt(r.accept_pct)+'%';
  $('pe').textContent=r.preemptions_total??'-';
  if(r.gpu)$('gpu').innerHTML=r.gpu.map(g=>
    `GPU${g.index} <span class="bar"><i style="width:${Math.min(100,g.util)}%"></i></span>${g.util.toFixed(0)}% · ${g.power.toFixed(0)}W · ${g.mem_mib}MiB`).join(' &nbsp;&nbsp; ');
  hist.push(r);if(hist.length>MAX)hist.shift();
  addRow(r);draw();
}
async function boot(){
  try{const s=await(await fetch('/snapshot')).json();
    (s.rows||[]).forEach(r=>{hist.push(r);addRow(r);});draw();
    if(s.rows&&s.rows.length)apply(s.rows[s.rows.length-1]);
  }catch(e){}
  const es=new EventSource('/events');
  es.onmessage=e=>{try{apply(JSON.parse(e.data))}catch(_){}};
  es.onopen=()=>{$('state').textContent='live';};
  es.onerror=()=>{$('state').textContent='stream reconnecting…';};
}
boot();addEventListener('resize',draw);
</script></body></html>
"""


def serve(bind, port, state, url, interval, engine):
    threading.Thread(target=poll_loop, args=(state, url, interval, engine, True), daemon=True).start()

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _send(self, code, ctype, body: bytes):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send(200, "text/html; charset=utf-8", PAGE.encode())
            elif path == "/snapshot":
                self._send(200, "application/json",
                           json.dumps({"engine": state.engine, "error": state.error,
                                       "rows": list(state.rows)}).encode())
            elif path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                cursor = 0
                try:
                    while True:
                        rows = state.since(cursor)
                        for r in rows:
                            cursor = max(cursor, r["seq"])
                            self.wfile.write(b"data: " + json.dumps(r).encode() + b"\n\n")
                        if not rows:
                            self.wfile.write(b": keepalive\n\n")
                        self.wfile.flush()
                        time.sleep(0.5)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return
            else:
                self._send(404, "text/plain", b"not found")

    srv = ThreadingHTTPServer((bind, port), H)
    srv.daemon_threads = True
    shown = "127.0.0.1" if bind in ("127.0.0.1", "localhost") else bind
    print(f"modelwatch dashboard: http://{shown}:{port}   (reading {url}/metrics)")
    print("ctrl-c to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print()


# --------------------------------------------------------------------------- CLI mode
def run_cli(args):
    engine = spec = None
    prev = None
    tick = 0
    if not args.json:
        print(f"modelwatch  url={args.url}  interval={args.interval}s   (ctrl-c to stop)", flush=True)
    while True:
        tick += 1
        now = time.time()
        try:
            metrics = parse_prometheus(fetch(args.url))
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            print(json.dumps({"t": round(now, 3), "error": str(exc)}) if args.json
                  else f"  !! cannot read {args.url}/metrics: {exc}", flush=True)
            if args.once:
                return 2
            time.sleep(args.interval)
            continue
        if engine is None:
            engine = args.engine if args.engine != "auto" else detect_engine(metrics)
            if engine is None:
                msg = ("unrecognised metrics (no vllm:/sglang:/llamacpp: counters) — "
                       "is /metrics enabled on this engine?")
                print(json.dumps({"t": round(now, 3), "error": msg, "metric_names": len(metrics)})
                      if args.json else f"  !! {msg}  (saw {len(metrics)} metric names)", flush=True)
                if args.once:
                    return 2
                time.sleep(args.interval)
                continue
            spec = ENGINES[engine]
            if not args.json:
                print(f"detected engine: {engine}", flush=True)
                print(f"{'time':>8} {'run':>4} {'wait':>4} {'prefill':>9} {'decode':>8} "
                      f"{'KV%':>6} {'cache%':>7} {'accept%':>8} {'preempt':>7}", flush=True)

        cur = Sample(now, metrics)
        row = build_row(prev, cur, spec, engine, args.gpu)

        def f(v, w, p=1):
            return "-".rjust(w) if v is None else f"{v:.{p}f}".rjust(w)

        if args.json:
            print(json.dumps(row), flush=True)
        else:
            print(f"{row['clock']:>8} {f(row['running'],4,0)} {f(row['waiting'],4,0)} "
                  f"{f(row['prefill_tok_s'],9)} {f(row['decode_tok_s'],8)} {f(row['kv_cache_pct'],6)} "
                  f"{f(row['cache_hit_pct'],7)} {f(row['accept_pct'],8)} {f(row['preemptions_total'],7,0)}"
                  + ("   [quiet: no requests in this window]" if row["quiet"] else ""), flush=True)
            if args.gpu and row.get("gpu"):
                print(f"{'':>8} " + " | ".join(
                    f"GPU{g['index']} {g['util']:.0f}% {g['power']:.0f}W {g['mem_mib']}MiB"
                    for g in row["gpu"]), flush=True)
        prev = cur
        if args.once or (args.ticks and tick >= args.ticks):
            return 0
        time.sleep(args.interval)


def main():
    ap = argparse.ArgumentParser(description="Live prefill/decode monitor for vLLM, SGLang, llama.cpp")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--engine", choices=["auto", "vllm", "sglang", "llama"], default="auto")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--ticks", type=int, default=0)
    ap.add_argument("--serve", action="store_true", help="web dashboard instead of the terminal view")
    ap.add_argument("--port", type=int, default=8090, help="dashboard port (default 8090)")
    ap.add_argument("--bind", default="127.0.0.1",
                    help="dashboard bind address; use 0.0.0.0 to open it on the LAN")
    args = ap.parse_args()

    if args.serve:
        serve(args.bind, args.port, State(), args.url, args.interval, args.engine)
        return 0
    return run_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
