# STRATA —— 在消费级机器上跑 Qwen3.8-Flash-Next（125B MoE）

调研日期 2026-10-03 · 未实测（本文所有"我们这台机器"的数字都是推算，标注了依据）

> 续篇：**换 DDR4 平台值不值**（PCIe 4.0 x8 / 洋垃圾选型 / 内存容量）见 [`PLATFORM-DDR4-PLAN.md`](PLATFORM-DDR4-PLAN.md)。

- 上游：<https://github.com/Niko1221/Strata>（MIT；最新 **v0.1.37**，2026-10-02 发布）
- 近亲仓库：<https://github.com/ExTV/strata-5090-4070>（5090 32G + 4070TiS 16G / **31 GB RAM**，和我们内存同档，最有参考价值）
- 模型：<https://huggingface.co/Qwen/Qwen3.8-Flash-Next>，量化版 ISTA-DASLab GSQ-RCO（**和我们 llama256 profile 用的 GSQ-RCO 是同一家**）

## 1. 它是什么

不是一个新模型，是**一个单模型推理引擎**：只服务 Qwen3.8-Flash-Next（125B MoE，24,576 个专家，每 token 路由 10 个）及其量化/微调版。
基于 llama.cpp/ggml 的部分代码改造，卖点是**把专家分散到 GPU / 内存 / SSD 三层**，让 12–24 GB 显卡也能跑：

| 层 | 放什么 |
|---|---|
| VRAM | attention + DeltaNet mixer、router、shared expert、输出头、MTP 草稿层、KV（64K 起只留热点）、**自适应专家缓存**（对话中自己学哪些专家常用） |
| RAM | 全部 24,576 个专家（pinned），CPU 用 AVX2/AVX-512 内核**并行**算 GPU 没缓存的那些 |
| SSD | 28.8 GB 的 PLE / n-gram 查表，每 token 只读几行 |

其余关键机制：MTP 投机解码（1.6–1.8×，草稿 3 token/轮，平均 2.4–3.2 接受）、提示词查表草稿（重复上下文再 +6–11%）、
8,192 token 的大块预填、前缀缓存、**多卡 layer split（流水线并行，不是 TP，不需要 NVLink/P2P）**、低内存模式（`mmap` 专家 / `resident` 子集）、
KV 流式（64K 起 KV 放内存、只把 attention 要读的那段放 VRAM）、`--kv k8v4` 混合 KV、CJK 草稿词表（0.1.27：中日韩 token 全进草稿子集，中文回答 **+15–38%**）。

接口：OpenAI 兼容 `/v1`（模型名 `qwen3.8-flash-next`）+ Anthropic `/v1/messages`（`ANTHROPIC_BASE_URL=http://host:port`），
浏览器 Chat/Monitor 面板，MCP server，Docker 镜像。**同一时刻只处理一个请求**（无 batching）。

## 2. 官方实测（参考基线）

RTX 5070 12 GB / Ryzen 5 7600 / 64 GB RAM / Windows / engine 0.1.26，32K 提示：

| 量化 | 写答案 (tok/s, 短对话) | 写答案 (128K) | 读提示 (tok/s, 32K) |
|---|---:|---:|---:|
| Q2_0 | 94 | 76 | 2,650 |
| IQ2_XS（推荐） | 79 | 63 | 2,090 |
| IQ3_XXS | 62 | 49 | 1,750 |
| IQ3_S | 53 | 46 | 1,620 |
| Coder（IQ1_M，砍一半专家） | 55 | 43 | 2,180 |

容量表（**RAM + VRAM 合计**要求）：Q2_0 37.6 GB · IQ2_XS 39.2 · IQ3_XXS 47.0 · IQ3_S 54.8；
下载 66–76 GB + MTP 草稿层 6 GB（开视觉 +1 GB）。官方内存规则是 **RAM ≥ shard1 + 10 GB**，64 GB 才能"每个尺寸都放得下"。
32 GB 内存官方只推荐 Coder，或走 low-RAM 模式（resident 变体**没有 layer split，只能单卡**；多卡 + low-RAM 会退化成 mapped 模式）。

## 3. 我们这台（192.168.0.119）能不能上

实测到的底子：

| 项 | 值 | 对 STRATA 的意义 |
|---|---|---|
| GPU | 2× RTX 5060 Ti 16 GB，sm_120，驱动 610.57.04 | ✅ RTX 50 在支持列表；预编译引擎含 **sm_120**；0.1.36 的 int8 tensor-core 预填内核、0.1.36 的 RTX 50 解码簇内核都吃得到；驱动 ≥580 ✅ |
| PCIe | 两张都是 **Gen5 x8** | ✅ layer split 不需要 P2P/NVLink；ExTV 那套第二张卡挂在 **x1** 上都能跑，我们的 hand-off 带宽反而更好 |
| RAM | **31 GB**（已用 10，swap 47） | ⚠️ **唯一的硬伤**。63 GB（31 RAM + 32 VRAM）合计过得了 54.8 GB 的 IQ3_S 门槛，但单看 RAM 远低于官方"shard1 + 10 GB"，必然落在 low-RAM/mapped 的"页错误主导区" |
| CPU | i5-13600KF，14C/20T，**只有 AVX2**（13 代消费级 AVX-512 被熔断） | ⚠️ STRATA 的 CPU 专家池是并行主力，AVX2 比 AVX-512（Ryzen 7000/9000）慢一档；6 P-core + 8 E-core 也谈不上富裕 |
| 盘 | / 1.8 T，**剩 613 GB** | ✅ 模型 70–80 GB + pack（`experts.bin` 23–50 GB）+ MTP 6 GB ≈ 100–140 GB，够但机器已用 65% |
| 工具链 | Docker 29.7.1 + nvidia-container-runtime(CDI)、git、python3.12、cmake 3.28、**已有能编 sm_120 的 llama.cpp** | ✅ 走 Docker 最干净（`--build-arg CUDA_ARCHITECTURES=120` 只编我们这一档）；native 路线也可行，本机已经会编 llama.cpp |
| 下载 | `HF_ENDPOINT=https://hf-mirror.com` 已配 | ✅ v0.1.37 起官方**支持 `HF_ENDPOINT`**（含草稿层），走 hf-mirror 能下 |
| 现状 | 两张卡都是 100% 占用 / 15.8 GB，**8080 被 vLLM 占着** | ⚠️ 试用必须停生产服务；STRATA 默认端口也是 8080，要改 8888/8899 |

### 推算性能（低置信，标了依据）

最接近的公开数据点是 ExTV 的 5090 32G + 4070TiS 16G / **31 GB RAM**：
2K 提示 350–375 tok/s、16K 985、80K 1,796（打了 3 个 patch）、解码 95–108 tok/s。

我们 VRAM 32 GB < 他们 48 GB、CPU 更弱、但 PCIe 更好，且同样卡在 31 GB RAM：

| 场景 | 预期 |
|---|---|
| 短提示（<4K）预填 | **150–400 tok/s** |
| 32K 预填 | **500–1,200 tok/s** |
| 80K+ 预填 | **300–1,500 tok/s**（强依赖 major page fault 数，ExTV 同配置在 640–1,796 之间横跳） |
| 解码（Q2_0/IQ2_XS，双卡 layer split） | **50–90 tok/s**（中文开 CJK 草稿词表可再 +15–38%） |
| 上下文 | 128K 稳，262K 要试（模型原生 262K） |

即：**能跑，但落在这套引擎最难受的"内存不足 + SSD 流式"区间**。预期单流比我们现役 vLLM DFlash2（8K→3,285 预填 / 292 解码）慢 3–4 倍，换来的是一颗 125B MoE 的脑子。

## 4. 和现役生产的关系（重要）

| | 现役 vLLM `vllm-dflash` | STRATA |
|---|---|---|
| 模型 | Qwen3.8-27B Q6 | **Qwen3.8-Flash-Next 125B MoE**（2–3 bit） |
| 引擎 | vLLM 0.29 + DFlash2 K=10 | ggml 系定制引擎 + MTP |
| 并发 | `--max-num-seqs 4`，4 路聚合 307 tok/s | **同时只服务 1 个请求**，无 batching |
| 上下文 | 150K（KV 池 157,824） | 128–262K |
| 上量 | 已跑生产、`modelctl` 托管 | 单机玩具档位，需停生产才能试 |

结论：**STRATA 不是 vLLM 的替代**，是一条"质量优先、单流、慢"的旁路。DSH 的 `qwen-local` provider 依赖 8080 的并发端点，不能拿它顶替。
真正值得试的动机只有一个：**125B MoE 的回答质量是不是明显强过 27B Q6**（尤其代码/长文档），以及中文（CJK 草稿词表只加速、不改质量；但注意 **Coder 版本中文会坏**，社区 issue #438——我们只能选 Q2_0/IQ2_XS/IQ3_S 这类保留全部专家的尺寸）。

## 5. 要试的话（方案已想好，未执行）

1. 选型：**IQ2_XS**（官方推荐）或 Q2_0（更快）起步；不放 IQ3_XXS（47 GB，和我们 31 GB RAM 直接对撞）。
2. 方式：Docker，`docker build -t strata --build-arg CUDA_ARCHITECTURES=120 .`，`-v strata-data:/data`，端口 `8899`，
   环境变量 `MODEL=IQ2_XS FAMILY=qwen CONTEXT=131072 VISION=no`；先用 `-e GPU=0` 单卡 + low-RAM resident 对一次，
   再上 `-e GPUS=0,1` 看 layer split 的 mapped 模式谁快。
3. 前置：`modelctl stop`（同时关掉 supervise 守护）腾出两张卡 —— **必须用户点头**，且跑完要按铁律 `modelctl status` 报 health/KV 池/每卡显存。
4. 预算：下载 70–80 GB（hf-mirror），盘 100–140 GB，首次启动 1–3 分钟内存打满属正常。
5. 验收：`tools/needle_bench.py` 长文召回 + `strata-*.log` 里的 prefill/decode 计时行 + `pgmajfault`/swap 计数（照 ExTV 的 `tools/snapshot.py` 思路，**不看 page fault 就分不清"慢"是配置还是换页**）。

## 6. 风险与坑（从上游文档摘的）

- **31 GB RAM 是真正的天花板**：官方"慢、磁盘灯常亮、engine stopped unexpectedly"三条症状都指向内存不足；Linux 下 OOM 会直接杀引擎。
- low-RAM 的 **resident 变体不支持 layer split**（单卡），多卡一给 `--gpus` 就退化成 mapped，长时间预填时**文件页缓存能把 RAM 吃到 0**。
- 我们 **无 AVX-512**，Q2_0 在 AVX-512 CPU 上还会额外生成 ~40 GB 快速内核副本的那条路径用不上。
- 单请求无 batching；服务端有"引擎静默 300 s 判死"逻辑（0.1.37 新增），长预填期间别误判。
- **默认端口 8080 和生产 vLLM 撞车**。
- 上游 Windows 优先（预编译引擎是 Windows zip），Linux 侧要么 Docker 要么本地编译；0.1.33 那一代多卡/低内存还被社区打了 3 个 patch 才好用（ExTV 仓库），说明这档配置仍在打磨中。

## 7. 来源

- README / MODELS / MULTI_GPU / HOW_IT_WORKS / DETAILS / INSTALL / TROUBLESHOOTING / COMMUNITY_BENCHMARKS（`raw.githubusercontent.com/Niko1221/Strata/main/docs/`）
- 发布说明 v0.1.37：<https://github.com/Niko1221/Strata/releases/tag/v0.1.37>
- 同档硬件实测：<https://github.com/ExTV/strata-5090-4070>
