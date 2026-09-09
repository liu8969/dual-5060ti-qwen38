# dual-5060ti-qwen38

2× 16 GB 消费级 Blackwell 上跑 27B 模型，150K 上下文，KV 池 172,480 token。
Serving a 27B model at 150K context on two 16 GB consumer Blackwell cards.

## Profile

| | |
|---|---|
| **模型** | [Merkyor / Qwen3.8-27B-EfficientThink-…-DFlash2](https://huggingface.co/Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2) |
| **量化** | NVFP4 W4A4（compressed-tensors，18.8 GB，9.3 GB/卡） |
| **解码** | DFlash2 投机解码，草稿 `DFlash2-FP8`（2.4 GB），5 tokens/步，接受长度 3.46–4.77 |
| **KV** | FP8，16 KB/token → 池 172,480 token |
| **上下文 / 并发** | 150,000 / 4 条流共享同一 KV 池 |
| **吞吐** | 115.7 tok/s 单流 · 255.5 tok/s 聚合（C=4） |

## 环境

```
CUDA 13.3 · driver 610.57.04 · Ubuntu 24.04 · kernel 7.0.0-31
Python 3.13.5 · torch 2.13.0+cu130 · vLLM 0.28.0
flashinfer 0.6.16.post3 + flashinfer-jit-cache（预编译内核，启动不编译）
```

## 主机

2× RTX 5060 Ti 16 GB（sm_120a）· **31 GB 内存**

## 快速开始

**方式一：自己装（推荐先干跑）**

```bash
git clone https://github.com/<you>/dual-5060ti-qwen38.git && cd dual-5060ti-qwen38
DRY_RUN=1 bash install.sh     # 看看要做什么
bash install.sh               # 装好并启动
bash bench.sh                 # 复现下面的数字
```

**方式二：整个项目丢给 AI agent**

不用自己读参数，把仓库（或压缩包）交给任意 AI coding agent（Claude Code / Cursor / DSH / Codex…），
粘这段话就行：

```text
先完整读 README.ai.md，然后按它操作。目标：在本机（2× 16GB Blackwell, 31GB 内存）安装并验证
这个推理服务。每一步都要实测，最后报告四个数字：health、KV 池、每卡显存、bench 吞吐。
硬约束：用 modelctl 做所有启停，不要 pkill，不要改 CUDA_HOME，不要动 max-num-batched-tokens。
```

`README.ai.md` 里已经写好：ground truth（每项带验证命令）、每个参数的取值理由、诊断决策树、
失败签名对照表、变更协议（改一个参数 → 重启 → 测量 → 报告前后差值）。AI 不需要猜。

## 实测

| 指标 | 数值 |
|---|---|
| KV 池 | 172,480 token（150K 请求的 1.15×） |
| 显存 | 15,597 MiB / 卡（共 16,311） |
| 单流解码 | 115.7 tok/s（关思考）/ 79.1（开思考） |
| 并发聚合 | C=1 70.4 · C=2 149.4 · **C=4 255.5** · C=8 264.7 tok/s |
| 100K 预填充 / 召回 | 1,396 tok/s / 5-of-5 |
| 启动 / 崩溃恢复 | 66–200 s / 90 s |

## 文件

| | |
|---|---|
| `install.sh` | 一键安装（幂等，支持 `DRY_RUN=1`） |
| `bench.sh` | 基准测试 → `results/bench-<时间戳>.md` |
| `scripts/modelctl` | 运维 CLI（status/start/stop/restart/logs/bench/supervise） |
| `README.ai.md` | **给 AI 的详细手册**（参数原理、诊断树、失败签名、运维规程） |
| `AGENTS.md` | 给 AI 的硬规则摘要 |
| `docs/PITFALLS.md` | 踩坑全集 |
| `docs/BENCHMARKS.md` | 实测数据存档 |

---

## English

Serve a 27B model at 150K context on 2× 16 GB consumer Blackwell — **172,480-token KV pool**.

| | |
|---|---|
| **Model** | [Merkyor / Qwen3.8-27B-EfficientThink-…-DFlash2](https://huggingface.co/Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2) |
| **Quantization** | NVFP4 W4A4 (compressed-tensors), 18.8 GB |
| **Decode** | DFlash2 speculative decoding, `DFlash2-FP8` draft, 5 tokens/step, accept length 3.46–4.77 |
| **KV** | FP8, 16 KB/token → 172,480-token pool |
| **Context / concurrency** | 150,000 / 4 streams sharing one pool |
| **Throughput** | 115.7 tok/s single stream · 255.5 tok/s aggregate at C=4 |

**Environment**: CUDA 13.3 · driver 610.57.04 · Ubuntu 24.04 · Python 3.13.5 · torch 2.13.0+cu130 ·
vLLM 0.28.0 · flashinfer 0.6.16.post3 (+ prebuilt kernel wheel, no JIT at startup).

**Host**: 2× RTX 5060 Ti 16 GB (sm_120a) · 31 GB RAM.

**Quick start**

1. Install it yourself: `DRY_RUN=1 bash install.sh` → `bash install.sh` → `bash bench.sh`.
2. Or hand the whole repo to any AI coding agent and paste:

```text
Read README.ai.md in full first, then follow it. Goal: install and verify this inference server on
this host (2× 16GB Blackwell, 31GB RAM). Measure every step and report four numbers at the end:
health, KV pool, per-card VRAM, bench throughput. Hard constraints: use modelctl for all lifecycle
operations, never pkill, never change CUDA_HOME, never touch --max-num-batched-tokens.
```

`README.ai.md` contains the ground truth (with a verification command per fact), the rationale behind
every parameter, a diagnostic decision tree, a failure-signature table, and a change protocol —
the agent does not have to guess. [`AGENTS.md`](AGENTS.md) is the condensed version.

MIT licensed.
