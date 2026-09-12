# README.ai.md — full operating manual for AI agents

> **Read this before touching anything.** It contains the ground truth, the reason behind every
> non-obvious parameter, a diagnostic decision tree, and the rules that keep this deployment alive.
> Humans want [`README.md`](README.md). `AGENTS.md` is the condensed version of this file.

---

## 1. Ground truth

| Fact | Value | How to verify |
|---|---|---|
| Endpoint | `http://<host>:8080/v1` | `curl -s localhost:8080/health` |
| Served model id | `Qwen3.8-27B-Q6-dual-5060ti` | `curl -s localhost:8080/v1/models` |
| GPUs | 2× RTX 5060 Ti 16 GB, sm_120a, no NVLink | `nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv` |
| Host RAM | 31 GB | `awk '/MemTotal/{printf "%d GB",$2/1024/1024}' /proc/meminfo` |
| OS / kernel | Ubuntu 24.04.4, 7.0.0-31-generic | `uname -r` |
| Driver | 610.57.04 | `nvidia-smi --query-gpu=driver_version --format=csv,noheader` |
| CUDA toolkit | 13.3 (V13.3.73) | `nvcc --version` |
| Python | 3.13.5 (in `~/vllm-venv`) | `~/vllm-venv/bin/python -V` |
| torch | 2.13.0+cu130 | `~/vllm-venv/bin/python -c 'import torch;print(torch.__version__)'` |
| vLLM | 0.29.0 | `~/vllm-venv/bin/python -c 'import vllm;print(vllm.__version__)'` |
| flashinfer | 0.6.18 | `~/vllm-venv/bin/python -c 'import flashinfer;print(flashinfer.__version__)'` |
| Prebuilt kernels | `flashinfer-jit-cache==0.6.18+cu130`, 906 modules | `~/vllm-venv/bin/python -c 'import flashinfer_jit_cache as m;print(m.get_jit_cache_dir())'` |
| KV pool | **157,824 tokens** (DFlash2 K=10 + prefix caching) | `modelctl status` |
| VRAM in use | ~15,600 MiB / card of 16,311 | `nvidia-smi` |
| Install dir | `~/deploy-5060ti` | |
| Logs | `~/modelctl-logs/{vllm-dflash,systemd,supervisor}.log` | |
| State | `~/.modelctl.state`, `~/.modelctl.supervisor` | |

If a measurement contradicts this table, **the table is stale** — measure, report the delta, do not
assume the table is right.

## 2. The serving profile, and why each value is what it is

```
vLLM 0.29.0
  --model            Merkyor W4A4 (NVFP4 compressed-tensors, 18.8 GB)
  --speculative-config  dflash draft (DFlash2-FP8), num_speculative_tokens=5, TP=2
  --kv-cache-dtype   fp8
  --kv-cache-memory  3865470566      # 3.6 GiB, pinned
  --max-model-len    150000
  --max-num-seqs     4
  --max-num-batched-tokens 1024
  --max-num-scheduled-tokens 8192          # 0.29 knob: silences the spec-decode scheduler warning
  --per-request-spec-decode-metrics summary # acceptance stats in every API response
  --kv-cache-metrics --enable-mfu-metrics --cudagraph-metrics
  --max-num-queued-reqs 32                 # admission valve
  --gpu-memory-utilization 0.977
  --attention-backend TRITON_ATTN
  --tensor-parallel-size 2
  --language-model-only
```

| Parameter | Value | Why exactly this |
|---|---|---|
| `--kv-cache-memory` | `3865470566` | Pins the pool base; the realised pool depends on the draft-token slots: **157,824** at K=10, 170,280 at K=5, 172,480 with prefix caching off. Tuning `--gpu-memory-utilization` instead gives a smaller, jittery pool. This is the single most important knob. |
| `--gpu-memory-utilization` | `0.977` | 0.985 and 0.99 both fail with `Engine core init failed`. 0.977 + pinned KV is the stable combination. |
| `--max-num-batched-tokens` | `1024` | At 4096 the KV pool collapses from 4.3 GiB to 2.92 GiB and 150K raises `ValueError` (max 137,376). Do not raise it. |
| `--max-num-seqs` | `4` | Measured optimum: C=4 → 255.5 tok/s aggregate at ~66 tok/s per stream. C=8 adds only 3.6% and queues half the requests. C=4 with `max-num-seqs 3` queues the 4th request and drops to 156 tok/s. |
| `--kv-cache-dtype` | `fp8` | 16 KB/token/card. SGLang's fp8 path costs ~20.2 KB — vLLM's cheaper KV is what makes the 172K pool possible. |
| DFlash2 draft | 5 tokens | Accept length 3.46–4.77, acceptance 49–75%. MTP3 gives a bigger pool (247,150) but decodes at 73.2/58.3 tok/s — DFlash2 wins for interactive use. |
| `--attention-backend` | `TRITON_ATTN` | Most stable on sm_120. |
| `MAX_JOBS` | `2` | Each nvcc TU uses 2.5–3 GB RSS; 18 parallel nvcc OOM-kills a 32 GB host. |
| `--enable-prefix-caching` | **on** | Costs 2,200 tokens of pool (172,480 → 170,280) but cuts TTFT on a repeated long prompt from 11.9 s to 0.7 s at 30K and from 49.5 s to 1.2 s at 90K (16.9× / 40.2×). Verify with `scripts/prefix_cache_test.py`. Was disabled during the first stability pass; seely with a benchmark before/after. |

**KV pool vs concurrency**: the pool is shared. 4 streams average ~43K tokens each. Four simultaneous
100K requests do **not** fit (400K > 157,824) — vLLM will preempt and recompute. For long-document
work use 1–2 streams.

## 3. Hard invariants

1. **All lifecycle operations go through `modelctl`.** Never `pkill`, never `nohup … &` + blind
   `sleep`, never two instances. Overlapping starts produce a "VRAM spikes then zeroes" crash loop.
2. **After every start/stop/restart, run `modelctl status` and report three numbers**: health,
   KV pool, per-card VRAM. "It's up" is not a report.
3. **Never leave the endpoint down at the end of a turn.**
4. **Never change `CUDA_HOME`.** FlashInfer's JIT cache key includes the full nvcc path
   (`/usr/local/cuda/bin/nvcc` vs `/usr/local/cuda-13.3/bin/nvcc`). Changing it triggers a full
   5–10 minute recompile. This has already cost hours once.
5. **Keep the KV bytes pinned and `--max-num-batched-tokens 1024`.**
6. **`pgrep -f` / `pkill -f` patterns must not match your own shell.** `pgrep` excludes only its own
   PID, not the parent `bash -c` whose command line contains the pattern — this produces phantom
   "it is compiling" readings. Put the check in a script file or bracket it: `[c]icc|[c]c1plus`.
7. **Never upgrade flashinfer/vLLM without installing the matching `flashinfer-jit-cache` wheel**;
   otherwise startup silently reverts to JIT compilation.
8. **Do not change the served model id** — clients (including DSH's provider config) bind to it.

## 4. Commands

```bash
bash ~/deploy-5060ti/modelctl status                  # process / health / model / KV pool / VRAM / supervisor
bash ~/deploy-5060ti/modelctl start vllm-dflash       # progress every 30 s; exits with the error if it dies
bash ~/deploy-5060ti/modelctl stop                    # also stops the supervisor
bash ~/deploy-5060ti/modelctl restart vllm-dflash
bash ~/deploy-5060ti/modelctl logs 80
bash ~/deploy-5060ti/modelctl bench                   # short decode test
bash ~/deploy-5060ti/modelctl watch --gpu             # live prefill/decode/queue/cache/acceptance
bash <repo>/bench.sh                                  # full suite → results/bench-<ts>.md
sudo systemctl status|restart modelctl                # systemd layer
tail -20 ~/modelctl-logs/systemd.log
```

Profiles: `vllm-dflash` (production), `vllm-mtp`, `sglang`, `llama168`, `llama256`.
Two layers of supervision: the systemd unit (`Restart=always`, enabled at boot) and
`modelctl supervise` (in-process watchdog). Measured recovery: `kill -9` → healthy again in 90 s.

## 5. Diagnostic decision tree

```
modelctl status
├─ process not running          → §5.2 "process died"
├─ running, health DOWN         → §5.1 "slow startup"
└─ running, health OK           → client-side problem: check model id / port / streamIdleTimeout
```

### 5.1 Slow startup (health DOWN for minutes, GPU ~0%)

```bash
ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E '[n]vcc|[c]icc|[c]c1plus' | head
find ~/.cache/flashinfer -newermt '-5 minutes' -name '*.o' | wc -l   # progress signal
```

- **Compilers present** → FlashInfer JIT compilation. It is a one-off 5–10 minute cost; **wait**.
  Do not kill it — the supervisor used to do exactly that and produced an infinite restart loop
  (see `docs/PITFALLS.md` §6). Permanent fix: install the prebuilt kernel wheel (§6).
- **Nothing compiling and VRAM flat** → genuinely stuck; read the log tail for
  `Engine core init failed` / `CUDA out of memory` / `ValueError`.

Normal startup is **66–200 s** (cudagraph capture + weight load). Weights alone load in ~2 s, so
"safetensors is slow" is never the explanation.

### 5.2 Process died

```bash
grep -oE 'CUDA out of memory[^)]*|ValueError[^\n]{0,160}|RuntimeError[^\n]{0,160}' \
  ~/modelctl-logs/vllm-dflash.log | tail -3
```

| Signature | Meaning | Action |
|---|---|---|
| `ValueError: Free memory ...` | pinned KV bytes exceed this boot's free VRAM (often ~0.3 GB less than usual) | the supervisor retries automatically; if it repeats, lower `KVBYTES` to ~3.45 GiB (still covers 150K) |
| `Engine core init failed` | util too high | restore 0.977 |
| `CUDA out of memory. Tried to allocate …` at cudagraph capture | pool computed too large | keep `--kv-cache-memory` pinned, do not raise util |
| `Killed` with no traceback | host RAM OOM during compilation | `MAX_JOBS=2` |
| VRAM rises to ~15 GB then drops to 0, repeatedly | two instances fighting | `modelctl stop`, then start once |

## 5b. Watching a live workload (the monitor lives in the `gpu-model-dashboard` project)

`modelctl watch [--gpu] [--interval N] [--json] [--once]` runs the monitor from the separate
`~/gpu-model-dashboard/` project (entry point `dashboard.py`): it scrapes the
engine's **Prometheus endpoint** and diffs the counters, so one tool covers all three engines and the
numbers do not depend on any engine's log format.

| Meaning | vLLM | SGLang | llama.cpp |
|---|---|---|---|
| prefill tokens | `vllm:prompt_tokens_total` | `sglang:prompt_tokens_total` | `llamacpp:prompt_tokens_total` |
| decode tokens | `vllm:generation_tokens_total` | `sglang:generation_tokens_total` | `llamacpp:tokens_predicted_total` |
| running / queued | `num_requests_running/waiting` | `num_running_reqs` / `num_queue_reqs` | `requests_processing` / `requests_deferred` |
| KV usage | `kv_cache_usage_perc` | `token_usage` | `kv_cache_usage_ratio` |
| cache hit | `prefix_cache_hits_total ÷ queries_total` | `cache_hit_rate` gauge | — |
| speculation | `spec_decode_num_accepted_tokens_total ÷ ..._draft_tokens_total` | — | — |

Reading the output:

- **`prefill = 0.0` while a request is running is not "no data" — it means every prompt token came
  from the prefix cache.** On a cache-friendly workload this is the normal steady state and it is the
  single clearest signal that prefix caching is doing the work.
- `cache%` is **cumulative since server start** (`hits_total / queries_total`). The engine's own log
  line reports a *per-10-second-window* rate, which legitimately reads `0.0%` whenever that window was
  idle. Never quote the log's window value as "the cache hit rate".
- `accept%` is the windowed speculative acceptance. It is strongly task-dependent — 86-94% on code,
  ~40-52% on summarisation, and lower still on long-context free-form work — so never compare two
  `accept%` values measured on different prompts.
- `[-]` for the first tick is correct: a rate needs two samples.
- Engines with `--metrics` off (llama.cpp default) fail detection explicitly rather than silently
  showing zeros.

Verified against the engine's own accounting: `prefill` and `running` match exactly, and `decode`
agrees in both level and mean (the residual gap is the 10 s window boundary offset).

### Web dashboard

```bash
~/gpu-model-dashboard/dashboard.sh status|start|stop|logs   # the panel is its own project (8090)
```

Self-contained single page (inline CSS/JS, no CDN), live via SSE, reconnect on drop: seven tiles
(prefill / decode / queue / KV / cache / acceptance / preemptions), a 4-series chart with autoscaling,
GPU utilisation-power-VRAM bars, and a rolling table of the last 300 samples.

Run it **on the host that has the GPUs** — a laptop-side instance still reads `/metrics` across the
network but shows no GPU row, because `nvidia-smi` is local-only. Stop it with
`sudo systemctl stop gpu-model-dashboard` (the unit is installed with `Restart=always`, so a
pkill would be undone three seconds later).

## 6. Prebuilt kernels (removes the 5–10 minute compile)

```bash
~/vllm-venv/bin/python -c 'import flashinfer;print(flashinfer.__version__)'   # e.g. 0.6.18
# wheel: flashinfer_jit_cache-<VER>+<cuda-tag>-cp39-abi3-manylinux_2_28_x86_64.whl
# index:  https://flashinfer.ai/whl/<cuda-tag>/flashinfer-jit-cache/   (links → GitHub releases)
# mirror: https://gh-proxy.com/https://github.com/flashinfer-ai/flashinfer/releases/download/v<VER>/<wheel>
# the venv has NO pip:
~/.local/bin/uv pip install --python ~/vllm-venv/bin/python --no-deps <wheel>
```

Direct GitHub is ~60 KB/s in CN, the `gh-proxy.com` mirror ~2.9 MB/s. Verify afterwards that
`get_jit_cache_dir()` lists ~959 module directories including `fp4_gemm_cutlass_sm120`, and that a
restart shows **zero** compiler processes.

Note: the prebuilt and locally-compiled `.so` both contain `sm_120` cubins and are byte-comparable
in size (5,336,880 vs 5,316,288) — there is **no throughput penalty** for using the wheel.

## 7. Benchmark interpretation

- `bench_gsq.py` uses `max_tokens=512` with an English prompt; `bench_concurrency.py` uses
  `max_tokens=256` with a Chinese prompt. **Never compare numbers across the two scripts.**
- The first request after startup is 10–20% slower (cudagraph/autotune warm-up). Run the
  single-stream test twice and report the second run.
- Reference values (production profile): single stream 115.7 tok/s off / 79.1 on; C=4 255.5 tok/s
  aggregate; 100K prefill ~1,396 tok/s; 100K needle recall 5/5.
- Anything more than ~5% below these on a warm run is a regression worth investigating.

## 8. Change protocol

Before changing any parameter:

1. Record the current numbers (`modelctl status` + `bench.sh`).
2. Change **one** parameter.
3. Restart, wait for health, re-measure.
4. Report before/after with the measured delta and the exact revert command.
5. If the change loses the KV pool or the endpoint, revert immediately.

Never claim a fix without a measurement. Never report a benchmark from a cold start.

## 9. What NOT to do

- Do not install `fastsafetensors` / `runai_streamer` / `tensorizer` to "speed up loading".
- Do not raise `MAX_JOBS` above 2 on a 32 GB host.
- Do not run `docker rm -f $(docker ps -aq)` casually — `modelctl`'s `kill_all` does this by design.
- Do not edit files under `$HOME` and restart blindly; edit → restart → measure → report.
- Do not stop the server while leaving the supervisor running — it will resurrect it. Use
  `modelctl stop`.
- Do not leave a stale second supervisor: `~/.modelctl.supervisor` holds the PID.

## 10. File map

| Path | Role |
|---|---|
| `install.sh` | idempotent installer, `DRY_RUN=1` supported |
| `bench.sh` | full benchmark suite → `results/bench-<ts>.md` |
| `scripts/modelctl` | lifecycle CLI (installed to `~/deploy-5060ti/modelctl`) |
| `scripts/modelctl.next` + `scripts/profile_render.py` + `profiles/*.json` | profile-as-data runner (landed, **not** in systemd yet) — see [`docs/PROFILE-DATA.md`](docs/PROFILE-DATA.md) |
| `dsh-plugin/local-models-sync.v1.mjs` | DSH 宿主插件：GUI 右下角的「同步本地模型」按钮 + `POST /local-models-sync/run` — see [`docs/DSH-MODEL-SYNC.md`](docs/DSH-MODEL-SYNC.md) |
| `scripts/vllm-merkyor-dflash-launch.sh` | production launch script; every knob is env-overridable |
| `scripts/bench_gsq.py` | single-stream + needle recall |
| `scripts/bench_concurrency.py` | concurrency sweep |
| `systemd/modelctl.service` | unit template (`/home/USER` rewritten at install time) |
| `docs/PITFALLS.md` | full pitfall log — read before redesigning anything |
| `docs/BENCHMARKS.md` | archived measurements and stack comparisons |
| `AGENTS.md` | condensed rules (this file is the full version) |

## 11. Upgrading vLLM (and rolling back)

The production venv is never modified in place. The active interpreter is a symlink:

```bash
ln -sfn ~/vllm-venv-029 ~/vllm-current     # switch
ln -sfn ~/vllm-venv     ~/vllm-current     # rollback
sudo systemctl restart modelctl            # either way
```

`modelctl` passes `VLLM_BIN=$HOME/vllm-current/bin/vllm`, and the launch script honours it.
**After any switch, confirm the real interpreter path with `ps -eo args | grep '[v]llm serve'`** —
an env var alone does nothing if the script hardcodes the path (this exact mistake happened once).

Upgrade checklist: (1) new venv side by side, (2) install the **matching** `flashinfer-jit-cache`
wheel for whatever flashinfer the new vLLM pins, (3) `vllm serve --help=all` to confirm every flag
this deployment uses still exists, (4) restart, (5) verify health / KV pool / VRAM, (6) re-benchmark
and compare against the numbers in §5. Current: vLLM 0.29.0 + flashinfer 0.6.18.

## 12. Escalation

If two consecutive restart attempts fail with **different** error signatures, stop retrying and
report: the exact log tail, `nvidia-smi` output, `free -g`, the profile in use, and what you already
tried. Do not loop.
