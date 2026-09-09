# dual-5060ti-qwen38

**Serve a 27B hybrid model at 150K context on 2× 16 GB consumer Blackwell — with a 172,480-token KV pool.**
**用 2 张 16 GB 消费级 Blackwell 跑 27B 混合模型、150K 上下文、KV 池 172,480 token。**

`中文` · [English](#english)

![gpus](https://img.shields.io/badge/GPU-2%C3%97%20RTX%205060%20Ti%2016GB-76b900)
![stack](https://img.shields.io/badge/stack-vLLM%200.28%20%2B%20NVFP4%20%2B%20DFlash2-blue)
![context](https://img.shields.io/badge/context-150K-orange)
![kvpool](https://img.shields.io/badge/KV%20pool-172%2C480%20tokens-success)

---

## 中文

### 这是什么

一份**经过实测的**部署配方：在两张 16 GB 显卡（共 32 GB 显存、32 GB 内存）上，把一个 27B 的
hybrid GDN + attention 模型跑成可用的长上下文服务。

- **栈**：vLLM 0.28 + Merkyor W4A4（NVFP4 compressed-tensors）+ DFlash2 投机解码 + FP8 KV
- **结果**：KV 池 **172,480 token**（比已知所有社区配置都大），单流 115.7 tok/s，4 并发 255.5 tok/s
- **运维**：systemd 托管 + 自动重启 + 开机自启，`modelctl` 一条命令看状态

不是"跑得起来"级别的记录，是**每个参数都试过、每个坑都踩过**之后的最终配置。

### 实测结果（先看数字）

| 指标 | 数值 |
|---|---|
| KV 池 | **172,480 tokens**（150K 请求的 1.15×） |
| 显存占用 | 15,459–15,517 MiB / 卡（上限 16,384） |
| 单流解码 | **115.7 tok/s**（关思考）/ **79.1**（开思考） |
| 并发聚合 | C=1 70.4 / C=2 149.4 / **C=4 255.5** / C=8 264.7 tok/s |
| 100K 预填充 | 1,393 tok/s |
| 100K 召回 | **5/5** |
| DFlash2 接受长度 | 3.46–4.77（接受率 49–75%） |
| 启动耗时 | 66–200 秒（cudagraph 捕获，**不是**编译） |
| 崩溃恢复 | `kill -9` 后 **90 秒**自动恢复 |

**和社区配置比：**

| 来源 | 配置 | KV 池 |
|---|---|---|
| darksidewalker | 单卡 5090 + DFlash2 + fp8 | 90.7–94.9K |
| 0xSero | 2×3090 + AWQ-INT4 + DSpark | 118,693 |
| club-5060ti | vLLM | 122,880 |
| **本项目** | **2×5060Ti + W4A4 + DFlash2 + FP8 KV** | **172,480** |

### 硬件要求

| 项 | 要求 | 说明 |
|---|---|---|
| GPU | 2× 16 GB，sm_120（Blackwell 消费级） | 2× RTX 5060 Ti / 5070 Ti / 5080 均可，越强越快 |
| 驱动 | ≥ 580（CUDA 13.x runtime） | 实测 610.57.04 |
| 内存 | ≥ 32 GB（推荐 64 GB+） | 32 GB 时 `MAX_JOBS=2`；内存是编译瓶颈不是 CPU |
| 系统 | Ubuntu 24.04+ | 需要 systemd |
| 磁盘 | ≥ 40 GB 空闲 | 权重 22 GB + venv |
| 其他 | 停掉 gdm（`sudo systemctl stop gdm3`） | 桌面会抢显存 |

### 快速开始

```bash
git clone https://github.com/<you>/dual-5060ti-qwen38.git
cd dual-5060ti-qwen38

# 先干跑一遍看它要做什么（不改任何东西）
DRY_RUN=1 bash install.sh

# 正式安装：建 venv、装 vLLM、下预编译内核、下权重、装 systemd、启动、验证
bash install.sh
```

安装脚本做的事（每一步都幂等，可重复运行）：

1. **preflight** — 检查 GPU 数量 / 显存 / 算力 / 驱动 / 内存 / 显示管理器
2. **venv** — 建 `~/vllm-venv`，装 vLLM 0.28
3. **预编译内核** — 装 `flashinfer-jit-cache`（959 个模块），**启动时不再 JIT 编译**
4. **权重** — 从 HF 镜像下 Merkyor W4A4 + DFlash2-FP8 草稿
5. **脚本 + 服务** — 装 `modelctl`、启动脚本、systemd 单元
6. **验证** — 等 health、打印 KV 池/显存、发一个冒烟请求

### 关键参数为什么这么设

| 参数 | 值 | 为什么 |
|---|---|---|
| `--kv-cache-memory` | `3865470566` | **硬钉 KV 池**，确定性拿到 172,480；比调 `--gpu-memory-utilization` 可靠得多 |
| `--max-num-batched-tokens` | `1024` | 设 4096 会让 KV 从 4.3 GiB 掉到 2.92 GiB，150K 直接报 ValueError |
| `--gpu-memory-utilization` | `0.977` | 0.985/0.99 会 `Engine core init failed` |
| `--max-num-seqs` | `4` | 并发甜点：C=4 聚合 255.5 tok/s；C=8 只多 3.6% 且后半排队 |
| `--kv-cache-dtype` | `fp8` | 16 KB/token，比 SGLang 的 20.2 KB 便宜 → 池子大 62% |
| `--speculative-config` | dflash, 5 tokens | 接受长度 3.46–4.77，实测有效 |
| `--attention-backend` | `TRITON_ATTN` | sm_120 上最稳 |
| `MAX_JOBS` | `2` | 每个 nvcc TU 吃 2.5–3 GB 内存，32 GB 只能开 2 个 |

### 运维

```bash
bash ~/deploy-5060ti/modelctl status          # 进程 / health / KV 池 / 每卡显存 / 守护
bash ~/deploy-5060ti/modelctl start vllm-dflash   # 启动，每 30s 打印进度，死了立刻报错
bash ~/deploy-5060ti/modelctl logs 80
bash ~/deploy-5060ti/modelctl bench
sudo systemctl restart modelctl               # systemd 整体重启
```

- **双层兜底**：systemd（`Restart=always` + 开机自启）+ `modelctl supervise`（进程内守护）
- 换 profile 前先 `stop`；`stop` 会同时关掉守护，否则守护会立刻把服务拉回来

profile 一览：`vllm-dflash`（生产）、`vllm-mtp`、`sglang`、`llama168`、`llama256`。

### 基准测试

```bash
bash bench.sh              # 单流 + 并发 C=1/2/4/8 + 100K 召回
bash bench.sh 50000 100000 # 自定义长文测试长度
SKIP_NEEDLE=1 bash bench.sh
```

报告写到 `results/bench-<时间戳>.md`。

### 故障排查

| 症状 | 原因 | 处理 |
|---|---|---|
| 启动卡在 `still loading... 300s`，GPU 0% | FlashInfer JIT 编译 | `ps -eo args \| grep -E '[n]vcc\|[c]icc'`，有输出就等（5–10 分钟，一次性）；装 jit-cache 轮子可永久消除 |
| 启动日志 `ValueError: Free memory ...` | 空闲显存比平时少 0.3 GB | 守护会重试一次；频繁发生就把 `KV_BYTES` 降到 3.45 GiB |
| `Engine core init failed` | util 太高 | 用 `--kv-cache-memory`，util 保持 0.977 |
| 显存先涨到 15 GB 又归零 | 两个实例在抢卡 | `modelctl stop` 后单实例启动 |
| 第 4 个并发请求很慢 | `--max-num-seqs 3` | 改成 4 |
| 机器被 OOM Killer 打爆 | 并行 nvcc 太多 | `MAX_JOBS=2` |

完整的踩坑记录（硬件 / 量化 / 三套推理栈 / 运维 / 编译）见 **[docs/PITFALLS.md](docs/PITFALLS.md)**。

### 目录结构

```
install.sh             一键安装（幂等，支持 DRY_RUN）
bench.sh               基准测试，输出 markdown 报告
scripts/
  modelctl             运维 CLI（status/start/stop/restart/logs/bench/supervise）
  vllm-merkyor-dflash-launch.sh   生产启动脚本
  bench_gsq.py         单流 + 长文召回
  bench_concurrency.py 并发扫描
systemd/
  modelctl.service     systemd 单元模板（安装时自动替换用户名/路径）
docs/
  PITFALLS.md          踩坑全集
  BENCHMARKS.md        实测数据存档
results/               bench.sh 的输出
AGENTS.md              给 AI agent 的操作规程
```

### 参考

- [FlashInfer 预编译内核](https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/) — 消除启动 JIT
- [vLLM DFlash 支持](https://github.com/vllm-project/vllm) — v0.28 起原生支持
- 权重：[Merkyor Qwen3.8-27B EfficientThink DFlash2](https://huggingface.co/Merkyor)（NVFP4/W4A4）

### 许可

MIT

---

## English

### What this is

A **measured, reproducible recipe** for serving a 27B hybrid GDN + attention model at long context
on two 16 GB consumer Blackwell cards (32 GB VRAM total, 32 GB system RAM).

- **Stack**: vLLM 0.28 + Merkyor W4A4 (NVFP4 compressed-tensors) + DFlash2 speculative decoding + FP8 KV
- **Result**: **172,480-token KV pool** (larger than any published community config), 115.7 tok/s
  single stream, 255.5 tok/s aggregate at C=4
- **Ops**: systemd-managed, auto-restart, boot-persistent, one command to see status

### Measured results

| Metric | Value |
|---|---|
| KV pool | **172,480 tokens** (1.15× a 150K request) |
| VRAM | 15,459–15,517 MiB / card (of 16,384) |
| Single-stream decode | **115.7 tok/s** (thinking off) / **79.1** (thinking on) |
| Aggregate | C=1 70.4 / C=2 149.4 / **C=4 255.5** / C=8 264.7 tok/s |
| 100K prefill | 1,393 tok/s |
| 100K needle recall | **5/5** |
| DFlash2 accept length | 3.46–4.77 (acceptance 49–75%) |
| Startup | 66–200 s (cudagraph capture, **not** compilation) |
| Crash recovery | **90 s** after `kill -9` |

### Requirements

2× 16 GB sm_120 GPUs · driver ≥ 580 (CUDA 13.x) · ≥ 32 GB RAM (64 GB+ recommended) ·
Ubuntu 24.04+ with systemd · ≥ 40 GB free disk · stop the display manager.

### Quick start

```bash
git clone https://github.com/<you>/dual-5060ti-qwen38.git && cd dual-5060ti-qwen38
DRY_RUN=1 bash install.sh     # see what it would do, change nothing
bash install.sh               # install everything and verify
bash bench.sh                 # reproduce the numbers above
```

### The four settings that matter

1. **`--kv-cache-memory 3865470566`** — pin the KV pool instead of tuning
   `--gpu-memory-utilization`. Deterministic 172,480 tokens.
2. **`--max-num-batched-tokens 1024`** — raising it to 4096 shrinks the KV pool from 4.3 GiB to
   2.92 GiB and makes 150K fail outright.
3. **`--max-num-seqs 4`** — the concurrency sweet spot; 8 adds only 3.6% and queues the rest.
4. **Install `flashinfer-jit-cache`** — otherwise the first start JIT-compiles kernels for 5–10
   minutes, and changing `CUDA_HOME` re-triggers the whole compile.

### Ops

```bash
bash ~/deploy-5060ti/modelctl status      # process / health / KV pool / VRAM / supervisor
sudo systemctl restart modelctl           # systemd restart
```

### Troubleshooting

See **[docs/PITFALLS.md](docs/PITFALLS.md)** for the full list (hardware, quantization, all three
serving stacks, ops, and compilation). The three most common:

- **Stuck at `still loading...`, GPU idle** → check for `nvcc`/`cicc`; it is compiling, wait it out,
  or install the prebuilt kernel wheel.
- **`ValueError: Free memory ...`** → pinned KV bytes are slightly too large for this boot's free
  VRAM; the supervisor retries, or lower `KV_BYTES`.
- **VRAM spikes then drops to zero** → two instances fighting; `modelctl stop`, then start one.

### License

MIT
