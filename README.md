# dual-5060ti-qwen38

2× 16 GB 消费级 Blackwell 上跑 27B 模型，150K 上下文，KV 池 170,280 token（开前缀缓存）/ 172,480（关）。
Serving a 27B model at 150K context on two 16 GB consumer Blackwell cards.

## Profile

| | |
|---|---|
| **模型** | [Merkyor / Qwen3.8-27B-EfficientThink-…-DFlash2](https://huggingface.co/Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2) |
| **量化** | NVFP4 W4A4（compressed-tensors，18.8 GB，9.3 GB/卡） |
| **解码** | DFlash2 投机解码，草稿 `DFlash2-FP8`（2.4 GB），5 tokens/步，接受长度 3.46–4.77 |
| **KV** | FP8，16 KB/token → 池 157,824 token（K=10 + 前缀缓存） |
| **上下文 / 并发** | 150,000 / 4 条流共享同一 KV 池（K=10 下池子 157,824，仍是 150K 的 1.05×） |
| **吞吐** | **162 tok/s** 单流 · **249.9** tok/s 聚合（C=4） |

## 环境

```
CUDA 13.3 · driver 610.57.04 · Ubuntu 24.04 · kernel 7.0.0-31
Python 3.13.5 · torch 2.13.0+cu130 · **vLLM 0.29.0**
flashinfer 0.6.18 + flashinfer-jit-cache 0.6.18+cu130（预编译内核，启动不编译）
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

**盯着负载跑（Web 仪表盘）**

```bash
# 在跑服务的机器上（要能访问 nvidia-smi）
bash ~/gpu-model-dashboard/dashboard.sh start   # 面板已独立成项目（默认 8090），本仓库不再维护它
# 浏览器打开 http://<host>:8090
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
| KV 池 | **157,824** token（K=10 + 前缀缓存；K=5 时 170,280） |
| 显存 | 15,597 MiB / 卡（共 16,311） |
| 解码（短提示 49 token） | **162** tok/s（关思考）/ **87**（开思考）— DFlash2 K=10 |
| 解码（长上下文 92K） | **33.5 tok/s** ⚠️ 见下 |
| 预填充 | 1,466 tok/s @92K · 1,257 tok/s @115K |
| TTFT @92K / @115K | 62.6 s / 91.8 s |
| 并发聚合（短提示） | C=1 74.6 · C=2 142.7 · **C=4 249.9** · C=6 193.8（排队）· C=8 243.1 |
| DFlash2 接受度 | 代码类任务 **94%**、平均接受长度 **9.63** token/步（`--per-request-spec-decode-metrics summary`） |
| 长文召回 | 5/5 @96.6K；club 协议 2/2 @115K |
| 前缀缓存 | 重复长提示 TTFT：30K 档 11.9s→0.7s（16.9×）· 90K 档 49.5s→1.2s（**40.2×**） |
| 启动 / 崩溃恢复 | 66–200 s / 90 s |

> ⚠️ **解码速度强烈依赖上下文长度**：短提示 115.7 tok/s，92K 上下文只有 33.5 tok/s（掉到 1/3.5）。
> 原因是长上下文下投机解码接受率下降、且每步要扫更长的 KV。引用数字时务必带上上下文长度。

## 文件

| | |
|---|---|
| `install.sh` | 一键安装（幂等，支持 `DRY_RUN=1`） |
| `bench.sh` | 基准测试 → `results/bench-<时间戳>.md` |
| `modelctl watch --gpu` | 盯住正在跑的负载：每 2 秒一行 prefill/decode/queue/KV/cache/accept |
| `~/gpu-model-dashboard/` | **Web 仪表盘**（独立项目）：实时曲线 + 数字卡片 + GPU，浏览器打开即用 |
| `scripts/modelctl` | 运维 CLI（status/start/stop/restart/logs/bench/**watch**/supervise） |
| `modelctl watch` | **实时监控**：prefill / decode / 队列 / KV / 缓存命中 / 投机接受度，兼容 vLLM·SGLang·llama.cpp（实现在 `~/gpu-model-dashboard/`） |
| `README.ai.md` | **给 AI 的详细手册**（参数原理、诊断树、失败签名、运维规程） |
| `AGENTS.md` | 给 AI 的硬规则摘要 |
| `docs/PITFALLS.md` | 踩坑全集 |
| `docs/BENCHMARKS.md` | 实测数据存档 |

---

## English

Serve a 27B model at 150K context on 2× 16 GB consumer Blackwell — **170,280-token KV pool** (172,480 with prefix caching off).

| | |
|---|---|
| **Model** | [Merkyor / Qwen3.8-27B-EfficientThink-…-DFlash2](https://huggingface.co/Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2) |
| **Quantization** | NVFP4 W4A4 (compressed-tensors), 18.8 GB |
| **Decode** | DFlash2 speculative decoding, `DFlash2-FP8` draft, **10 tokens/step**, measured accept rate 86–94% |
| **KV** | FP8, 16 KB/token → 170,280-token pool |
| **Context / concurrency** | 150,000 / 4 streams sharing one pool |
| **Throughput** | 115.7 tok/s single stream · 255.5 tok/s aggregate at C=4 |

**Environment**: CUDA 13.3 · driver 610.57.04 · Ubuntu 24.04 · Python 3.13.5 · torch 2.13.0+cu130 ·
vLLM 0.29.0 · flashinfer 0.6.18 (+ prebuilt kernel wheel, no JIT at startup) · DFlash2 K=10.

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
