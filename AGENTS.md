# AGENTS.md — operating rules for AI agents

> Audience: an autonomous coding agent operating this repository / the host it was installed on.
> Read this **before** touching the model server. Humans should read `README.md` instead.

## 0. Scope and ground truth

This repo deploys and operates a single local LLM endpoint:

| Fact | Value |
|---|---|
| Endpoint | `http://<host>:8080/v1` (OpenAI-compatible) |
| Model id served | `Qwen3.8-27B-Q6-dual-5060ti` — **must not change**, clients depend on it |
| Stack | vLLM 0.28 + Merkyor W4A4 (NVFP4) + DFlash2 draft + FP8 KV |
| Hardware | 2× 16 GB Blackwell (sm_120), 32 GB RAM |
| KV pool | 172,480 tokens |
| Install dir | `~/deploy-5060ti` |
| Logs | `~/modelctl-logs/{vllm-dflash,systemd,supervisor}.log` |
| State | `~/.modelctl.state`, `~/.modelctl.supervisor` |

If an observed value contradicts this table, the table is stale — measure and report, do not assume.

## 1. Hard invariants (do not violate)

1. **Use `modelctl` for all lifecycle operations.** Never `pkill`, never `nohup ... &` followed by
   blind `sleep`, never start two instances. Overlapping starts produce a "VRAM spikes then zeroes"
   crash loop.
2. **After every start/stop/restart, run `modelctl status` and report three numbers**: health,
   KV pool, per-card VRAM. "It's up" is not an acceptable report.
3. **Never leave the endpoint down** when ending a turn. If you broke it, fix it or restore the
   previous profile.
4. **Never change `CUDA_HOME`** in the systemd unit or launch script. The FlashInfer JIT cache key
   includes the full nvcc path; changing it forces a 5–10 minute full recompile.
5. **Keep `--max-num-batched-tokens 1024`.** Raising it collapses the KV pool below what 150K needs.
6. **Keep `--kv-cache-memory` (KV_BYTES) pinned.** Do not "improve" it by raising
   `--gpu-memory-utilization`; ≥0.985 fails engine init.
7. **`pkill -f` / `pgrep -f` patterns must not match your own shell.** `pgrep` excludes only its own
   PID, not the parent `bash -c` whose command line contains the pattern. Write the check into a
   server-side script file, or bracket it: `pgrep -f '[c]icc|[c]c1plus'`.
8. **Do not modify `--max-num-seqs` without re-benchmarking.** 4 is measured optimal.
9. **Do not upgrade flashinfer/vLLM without also fetching the matching `flashinfer-jit-cache` wheel.**
   A version mismatch silently reverts to JIT compilation.

## 2. Command reference

```bash
# status: process / health / model id / KV pool / VRAM / supervisor
bash ~/deploy-5060ti/modelctl status

# start (prints progress every 30 s; exits immediately with the error if the process dies)
bash ~/deploy-5060ti/modelctl start vllm-dflash

# stop (also stops the supervisor — otherwise it immediately restarts the server)
bash ~/deploy-5060ti/modelctl stop

# logs
bash ~/deploy-5060ti/modelctl logs 80

# benchmark
bash ~/deploy-5060ti/modelctl bench          # short decode test
bash <repo>/bench.sh                         # full suite -> results/bench-<ts>.md

# systemd (double layer: unit + in-process supervisor)
sudo systemctl status modelctl
sudo systemctl restart modelctl
tail -20 ~/modelctl-logs/systemd.log
```

Profiles: `vllm-dflash` (production), `vllm-mtp`, `sglang`, `llama168`, `llama256`.

## 3. Diagnostic decision tree

**Symptom: server not healthy.**

```
1. bash ~/deploy-5060ti/modelctl status
   ├─ process not running  -> read logs, go to "process died"
   ├─ process running, health DOWN -> go to "slow startup"
   └─ process running, health OK   -> the client's problem, check model id / port
```

**Slow startup (health DOWN for minutes, GPU utilisation ~0%):**

```
ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E '[n]vcc|[c]icc|[c]c1plus' | head

├─ compilers present -> FlashInfer JIT compilation (one-off, 5-10 min). WAIT.
│                      Confirm progress: find ~/.cache/flashinfer -newermt '-5 minutes' -name '*.o' | wc -l
│                      Permanent fix: install flashinfer-jit-cache (see §4).
└─ nothing compiling, VRAM flat -> genuinely stuck: read the log tail, look for
                                  "Engine core init failed" / "CUDA out of memory" / "ValueError".
```

**Process died:**

```
grep -oE 'CUDA out of memory[^)]*|ValueError[^\n]{0,160}|RuntimeError[^\n]{0,160}' ~/modelctl-logs/vllm-dflash.log | tail -3
```

| Log signature | Meaning | Action |
|---|---|---|
| `ValueError: Free memory ...` | pinned KV_BYTES > free VRAM this boot (often ~0.3 GB less than usual) | supervisor retries automatically; if repeated, lower `KV_BYTES` to ~3.45 GiB (still covers 150K) |
| `Engine core init failed` | `--gpu-memory-utilization` too high | restore 0.977 |
| `CUDA out of memory. Tried to allocate ...` at cudagraph capture | pool computed too large | keep `--kv-cache-memory` pinned, do not raise util |
| `Killed` (SIGKILL, no traceback) | host RAM OOM during compilation | set `MAX_JOBS=2` |
| VRAM rises to ~15 GB then drops to 0, repeating | two instances fighting | `modelctl stop`, then start once |

## 4. Prebuilt kernels (avoid the 5–10 minute compile entirely)

```bash
# detect the installed version
~/vllm-venv/bin/python -c 'import flashinfer;print(flashinfer.__version__)'

# wheel name pattern
flashinfer_jit_cache-<VER>+<cuda-tag>-cp39-abi3-manylinux_2_28_x86_64.whl

# index (links point at GitHub releases)
https://flashinfer.ai/whl/<cuda-tag>/flashinfer-jit-cache/

# fast mirror when GitHub is slow (CN)
https://gh-proxy.com/https://github.com/flashinfer-ai/flashinfer/releases/download/v<VER>/<wheel>

# install (the venv has NO pip — use uv)
~/.local/bin/uv pip install --python ~/vllm-venv/bin/python --no-deps <wheel>
```

Verification after install: `~/vllm-venv/bin/python -c "import flashinfer_jit_cache as m;print(m.get_jit_cache_dir())"`
should list ~959 module directories including `fp4_gemm_cutlass_sm120`.

A restart after this should show **zero** compiler processes and take 66–200 s.

## 5. Reporting contract

Every status report to the user must contain:

```
health : OK/DOWN
kv pool: <N> tokens
gpu    : <MiB> / <MiB> per card
```

Plus, when you changed something: what changed, the measured before/after number, and how to revert.
Never claim a fix without a measurement. Never report a benchmark from a cold start (first request
after startup is 10–20% slower); run it twice.

## 6. What NOT to do

- Do not install `fastsafetensors` / `runai_streamer` / `tensorizer` to "speed up loading" —
  weights load in 2 s; the time is cudagraph capture.
- Do not raise `MAX_JOBS` above 2 on a 32 GB host.
- Do not change the served model id.
- Do not run `docker rm -f $(docker ps -aq)` unless you know what else runs on the host
  (`modelctl`'s `kill_all` does this by design — be aware).
- Do not tune by editing files in `/home/lcy` and restarting blindly; edit, restart, measure, report.
- Do not leave the supervisor running against a profile you then stop manually — `modelctl stop`
  handles both; killing only the server leaves the supervisor to resurrect it.

## 7. File map

| Path | Role |
|---|---|
| `install.sh` | idempotent installer, supports `DRY_RUN=1` |
| `bench.sh` | full benchmark suite → `results/bench-<ts>.md` |
| `scripts/modelctl` | lifecycle CLI (installed to `~/deploy-5060ti/modelctl`) |
| `scripts/vllm-merkyor-dflash-launch.sh` | production launch script (all knobs are env-overridable) |
| `scripts/bench_gsq.py` | single-stream + needle recall |
| `scripts/bench_concurrency.py` | concurrency sweep |
| `systemd/modelctl.service` | unit template (`/home/lcy` replaced at install time) |
| `docs/PITFALLS.md` | full pitfall log — read before redesigning anything |
| `docs/BENCHMARKS.md` | archived measurements |

## 8. Escalation

If two consecutive restart attempts fail with different error signatures, stop retrying and report:
the exact log tail, `nvidia-smi` output, `free -g`, and what you already tried. Do not loop.
