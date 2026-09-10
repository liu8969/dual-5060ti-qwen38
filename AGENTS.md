# AGENTS.md — condensed rules

**Full manual: [`README.ai.md`](README.ai.md).** Read that before changing anything; this file is
only the part an agent must never get wrong.

## Invariants

1. **All lifecycle operations go through `modelctl`** — never `pkill`, never `nohup … &` + blind
   `sleep`, never two instances.
2. **After every start/stop/restart, report three numbers**: health, KV pool, per-card VRAM.
   `modelctl status` prints all three. "It's up" is not a report.
3. **Never leave the endpoint down** at the end of a turn.
4. **Never change `CUDA_HOME`** — the FlashInfer JIT cache key includes the nvcc path; changing it
   costs a 5–10 minute recompile.
5. **Keep `--kv-cache-memory` pinned and `--max-num-batched-tokens 1024`.** DFlash2 draft length lives in
   `~/.modelctl.env` as `K` — **10 is the measured optimum, 12 is a cliff** (acceptance 86% → 53%).
6. **`pgrep -f` / `pkill -f` must not match your own shell** — use `[c]icc` style patterns or a
   script file.
7. **Upgrading flashinfer/vLLM requires the matching `flashinfer-jit-cache` wheel.**
8. **Do not change the served model id** (`Qwen3.8-27B-Q6-dual-5060ti`).

## Commands

```bash
bash ~/deploy-5060ti/modelctl status | start vllm-dflash | stop | logs 80 | bench
sudo systemctl restart modelctl
bash <repo>/bench.sh          # full suite → results/bench-<ts>.md
```

## Startup is slow — do not panic, do not kill it

Normal: **66–200 s** (cudagraph capture, not compilation). Weights load in 2 s.
If it exceeds that, check for compilers before concluding it is stuck:

```bash
ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E '[n]vcc|[c]icc|[c]c1plus' | head
```

Compilers present → FlashInfer JIT, wait it out (permanent fix in `README.ai.md` §6).

## Reference numbers (warm)

Single stream 162 tok/s (thinking off) / 87 (on) · C=4 aggregate 249.9 tok/s · KV pool 157,824 (K=10 + prefix caching) ·
VRAM ~15,600 MiB/card · recovery after `kill -9` 90 s. More than ~5% below → investigate.

Details, parameter rationale, diagnostic tree and failure signatures: [`README.ai.md`](README.ai.md).
