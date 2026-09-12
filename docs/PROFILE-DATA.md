# Profiles as data — `profiles/*.json` + one generic runner

**Status: phase 1 landed, production still runs `scripts/modelctl`.**
`modelctl.next` renders byte-identical argv to the live process (see §Verification) but is
*not* wired into systemd yet.

## Why

A profile used to live in three places: a `case` branch inside `modelctl`, a dedicated
`*-launch.sh`, and `~/.modelctl.env`. Adding a knob meant editing the engine-specific shell;
the real launch line and the copy in the repo drifted apart (three different versions of
`vllm-merkyor-dflash-launch.sh` were in circulation at one point).

Now a profile is one JSON file. The runner knows nothing about vLLM, SGLang or llama.cpp —
no engine enums, no engine flags.

## Shape

The field names follow SSH Manager's launch projects (`project.work_dir`,
`project.environment_command`, `profiles[].launch_command`), so the same file can be pasted
into that UI later.

```jsonc
{
  "schema": "modelctl/profile@1",
  "project": {
    "work_dir": "/home/lcy/deploy-5060ti",   // 启动前 cd 到这里
    "environment_command": ". $HOME/.modelctl.env"  // 准备命令（例如 activate venv）
  },
  "profile": {
    "id": "vllm-dflash",
    "engine": "vllm",                        // 只是给人看的标签
    "bin": "${VLLM_BIN:-$HOME/vllm-current/bin/vllm}",
    "env": { "CUDA_HOME": "/usr/local/cuda" },
    "args": ["serve", "${MODELS:-/home/lcy/Models}/.../W4A4", "--port", "8080"],
    "pre":    ["docker rm -f \"${NAME:-sglang-merkyor}\" || true"],  // 可选，拉之前跑
    "stop":   ["docker rm -f \"${NAME:-sglang-merkyor}\""],          // 可选，收尾
    "kill":   ["pkill -9 -f 'vllm serve'"],                          // 可选，兜底强清
    "detect": ["docker inspect -f '{{.State.Running}}' x | grep -q true"], // 可选，容器档专用
    "health": "http://127.0.0.1:8080/health",
    "log": "$HOME/modelctl-logs/vllm-dflash.log"
  }
}
```

* `${VAR}` / `${VAR:-default}` expand from the caller's environment, **recursively** (a default
  may reference another variable). `~/.modelctl.env` is sourced with `set -a` first, so it still
  wins over every JSON default — that is the A/B experiment channel and it must keep working.
* A conditional argument is an object: `{"when": "KV_METRICS==1", "args": ["--kv-cache-metrics"]}`.
  Bare `{"when": "SCHED_TOKENS"}` means "non-empty".
* `pre` / `stop` / `kill` / `detect` are **shell lines handed to `eval`** (not word-split), which
  is how a container profile expresses "clear the old container name first".

## Rendering vs running

`scripts/profile_render.py` only turns JSON into text; `scripts/modelctl.next` is the lifecycle
driver (`list | show <p> | start <p> [--dry-run] | stop [p] [--dry-run] | restart | status |
logs | supervise | watch | bench`).

```bash
python3 profile_render.py profiles/vllm-dflash.json --argv      # one argv item per line
python3 profile_render.py profiles/vllm-dflash.json --argv-nul  # NUL-delimited (safe for spaces)
python3 profile_render.py profiles/vllm-dflash.json --env       # KEY=VAL
python3 profile_render.py profiles/vllm-dflash.json --pre-nul   # also --stop-nul/--detect-nul/--kill-nul
python3 profile_render.py profiles/vllm-dflash.json --log --work-dir --env-command
```

`modelctl.next start <p> --dry-run` prints cwd, env, `environment_command`, `pre` and the full
argv without starting anything. **Always dry-run a new profile first.**

## Two safety rules that came out of real incidents

1. **`stop` never guesses which profile is running.** The profile used for the `stop` / `kill`
   sweep comes from an explicit argument, then `~/.modelctl.state`, then `MODELCTL_PROFILE`.
   If none of them resolves, it only reclaims the process group this script launched and runs
   **no** `pkill` — because guessing wrong means killing a healthy service. (A first version
   defaulted to `vllm-dflash` and a self-test dutifully killed production vLLM.)
2. **`start` tears down the *service*, never the supervisor.** `cmd_start` calls
   `teardown_service`, not `cmd_stop`; killing the supervisor from inside makes systemd run
   `ExecStop`, which would kill the service that was just started.

`stop --dry-run` prints the commands it would run, the pgid it would kill, and whether the
supervisor is in the line of fire. Use it before any manual stop.

Self-testing this script requires redirecting **all four** state paths together —
`PROFILES_DIR`, `MODELCTL_STATE`, `MODELCTL_PIDFILE`, `MODELCTL_RUNFILE` — and `unset
MODELCTL_PROFILE`. Changing only `PROFILES_DIR` leaves the test pointing at the production
supervisor's pid file.

## Launched processes are process groups

`setsid nohup <bin> <args> &` puts the child in a new session whose `pgid == pid`, so
`kill -- -$pgid` reclaims the whole tree — including vLLM's `VLLM::Worker_TP0/TP1` children that
survive a plain `kill -9` of the API server and keep ~15.7 GB/card. The pgid is recorded in
`~/.modelctl.run`. This is why the `kill` pkill patterns are a *fallback*, not the primary teardown.

## Verification (2026-09-12)

`scripts/parity_check.py` compares a rendered profile against the live process's own
`/proc/<pid>/cmdline` and environ (`/tmp/parity/live.argv`, `/tmp/parity/live.env`):

* renderer path — `✓ bin 一致` + `✓ argv 逐字一致（54 项）` + 7/7 env keys identical;
* runner path (`modelctl.next start vllm-dflash --dry-run`) — aligned `bin + argv` 55/55
  byte-identical (`/proc` shows the console script's interpreter first, so the comparison drops
  `live[0]`).

Two intentional deltas: `VLLM_BIN` is no longer injected into the child environment, and `PATH`
no longer lists `/usr/local/cuda-13.3/bin` twice.

Lifecycle was exercised with a throwaway `python3 -m http.server` profile while production kept
serving: `pre` ran, `environment_command` took effect, the health check drove the readiness
decision, `stop` ran its hooks, and the process group plus its port were fully released —
production health, supervisor pid and engine pid unchanged throughout.

## Profiles in the box's `profiles/`

| id | engine | what it is |
|---|---|---|
| `vllm-dflash` | vllm | production: Merkyor W4A4 NVFP4 + DFlash2 K=10 + FP8 KV + FLASHINFER, 150K |
| `vllm-mtp` | vllm | comparison: built-in BF16 MTP-3, hard-coded TRITON_ATTN, 150K, 0.28 venv |
| `sglang` | sglang | comparison: SGLang container + DFlash2-FP8, `fp8_e4m3` KV, 163840 ctx |
| `llama168` | llama.cpp | UD-Q6_K + MTP2, `q8_0` KV, 172032 ctx, 4-slot unified pool |
| `llama256` | llama.cpp | GSQ-RCO IQ3_S + DFlash2-Q4_K_M draft + ngram-map-k4v, 262144 ctx, 1 slot |

Converting the last four surfaced two latent breakages: `vllm-mtp` and `llama256` were written as
`MODEL=$MODELS/...` in `modelctl`'s `case` branch, and **`MODELS` is set nowhere** on this box —
so both rendered `--model /Merkyor-...` / `/GSQ-RCO/...` and could not have started. The JSON
form uses `${MODELS:-/home/lcy/Models}`, which restores the intended paths.
