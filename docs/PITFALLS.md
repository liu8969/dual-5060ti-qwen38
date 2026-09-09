# 5060 Ti × 2 部署经验与踩坑全集

> 机型：2× RTX 5060 Ti 16GB（GB206 / sm_120a）+ i5-13600KF + 32GB DDR5 + Ubuntu 24.04（kernel 7.0.0-31，driver 610.57.04）
> 模型：Qwen3.8-27B（hybrid GDN + attention，65 层 = 48 GDN + 16 full-attention，原生 262,144 上下文）
> 端口 8080，model id 固定 `Qwen3.8-27B-Q6-dual-5060ti`（DSH provider `qwen-local` 依赖）

---

## 0. 最重要的事（先看这 8 条）

1. **KV 池用 `--kv-cache-memory` 硬钉，不要靠调 `--gpu-memory-utilization`**。钉住后池子确定性拿到 170,280（关前缀缓存时 172,480）；调 util 反而在 0.985/0.99 时 `Engine core init failed`。
2. **`--max-num-batched-tokens` 保持 1024**。设 4096 会让 KV 从 4.3 GiB 掉到 2.92 GiB，150K 直接 `ValueError`。
3. **别改 systemd 单元的 `CUDA_HOME`**。FlashInfer 的 JIT 缓存 key 含 nvcc 完整路径，一改就全套重编译 5–10 分钟。
4. **启动慢 ≠ 权重加载慢**。权重 2 秒读完（page cache），45–200 秒全花在 torch.compile / cudagraph 上；fastsafetensors / runai_streamer / tensorizer 都帮不上忙。
5. **进程管理只用 `modelctl` + systemd**，禁止裸 `pkill` + `nohup &` + 盲等。多个启动脚本互抢显卡会造成"显存先上去又归零"的崩溃循环。
6. **`pkill -f` / `pgrep -f` 会匹配到自己的命令行**（`pgrep` 只排除自身 PID，不排除父 shell）。要么写成服务器端脚本执行，要么用 `[c]icc` 这种括号写法。
7. **GGUF 只能跑 llama.cpp**。SGLang 直接报 `GGUF model with architecture qwen35/dflash is not supported yet`。
8. **并发甜点是 4**：C=4 聚合 255.5 tok/s 且每条流仍有 ~66 tok/s；C=8 只多 3.6% 且后 4 条排队掉到 33.7。

---

## 1. 硬件与平台

### 1.1 P2P 是死路（结论：不要用）

- 实测 peer write **6.8 GB/s** vs 显存中转（staged）**24.3 GB/s**。
- NCCL SHM 11.92 GB/s vs 强制 P2P 4.94 GB/s。
- 根因是 **Intel 消费级芯片组根复合体**的 PCIe 交换能力：同一张 A6000 换到 Z890 平台也只有 5.48 GB/s。
- 折腾过的：aikitoria `610.57.04-p2p-v3` 补丁驱动 + `RMForceStaticBar1=1`，`GGML_CUDA_P2P` 只是"存在性检查"。
- **最终决定**：保留补丁驱动不动，生产走显存中转路径。

### 1.2 其他硬件约束

| 项 | 事实 |
|---|---|
| 显存 | 16,384 MiB/卡，生产用 15,459–15,517 MiB |
| 内存 | 31 GiB 可用 → 编译时 `MAX_JOBS` 只能 2（18 个并行 nvcc 直接 OOM） |
| 桌面 | gdm 必须停掉，否则抢显存 |
| 磁盘 | 780 GB 可用，够放多个量化版本 |
| 主板 | Maxsun H770YTX D5（PCH 8086:7a05，ITX，2 个 DIMM 槽） |

### 1.3 编译该买什么（问过的采购问题）

瓶颈是**内存不是 CPU**：每个 nvcc TU 约 2.5–3 GB RSS（`cc1plus` 850 MB + `cicc` 1.9 GB），17 个 TU，31 GB 内存下只能开 2 个。

- **最优**：内存加到 64–96 GB，`MAX_JOBS=8` → 编译快 4–6 倍（¥1200–1800）。13600KF 的 20 线程本来就够。
- 换 CPU：**9950X > 270K Plus ≫ EPYC 7642**。7642 是 2019 年的 Zen2，单线程只有 13600KF 的 40–50%，除非要建真服务器（8 通道 + 128 lanes）否则是倒车。
- **最省事**：装预编译内核轮子，编译成本直接归零（已完成，见 §7）。

---

## 2. 模型与量化

### 2.1 用过的权重

| 权重 | 大小 | 能用在哪 | 备注 |
|---|---|---|---|
| Merkyor W4A4（compressed-tensors NVFP4） | 18.82 GB（9.27 GB/卡） | SGLang ✅ vLLM ✅ | 比 RadixArk NVFP4 省 0.9 GB/卡，lm_head 未量化 |
| Merkyor 自带 DFlash2-FP8 草稿 | 2.41 GB（1.54 GB/卡） | 两栈 ✅ | 需 `--speculative-draft-model-quantization compressed-tensors` |
| GSQ-RCO IQ3_S.gguf | 11.77 GB | 仅 llama.cpp | 社区争议的是另一个 8.42 GB 的 IQ2_XS（Token Embedding 掉到 IQ1_M） |
| Qwen3.8-27B-UD-Q6_K.gguf | 22 GB | 仅 llama.cpp | 168K 档生产用 |
| DFlash2-Q4_K_M.gguf | 1.09 GB | 仅 llama.cpp | 草稿 |
| ikantkode AWQ | — | ❌ 两栈都不可用 | 量化了 lm_head → SGLang 报 `'ParallelLMHead' object has no attribute 'output_size'` |

### 2.2 KV 成本（决定池子大小的关键）

| 栈 | KV 精度 | 每 token 成本 | 结果 |
|---|---|---|---|
| SGLang | fp8_e4m3 | ~20.2 KB/token/卡 | 池 106,357（mfs 0.92） |
| **vLLM** | **fp8** | **16 KB/token** | **池 172,480** |

**vLLM 的 KV 更便宜，同样显存能开更大池子** —— 这是选 vLLM 做 150K 长文档的核心原因。

### 2.3 DFlash2 草稿模型

- vLLM 原生支持：`DFlashModelTypes = Literal["dflash"]`，草稿模型名含 "dflash" 就自动识别。
- 配置：`--speculative-config {"model":..., "method":"dflash","num_speculative_tokens":5,"quantization":"compressed-tensors","draft_tensor_parallel_size":2}`
- 实测接受长度 3.46–4.77，接受率 49–75%。
- llama.cpp 侧 `--spec-type ngram-map-k4v,draft-dflash`。

---

## 3. vLLM（当前生产栈）

### 3.1 生产参数

```
vLLM 0.28.0 + Merkyor W4A4 + DFlash2 + FP8 KV
--kv-cache-memory 3865470566      # 硬钉 3.6 GiB → 池 170,280（关前缀缓存 172,480）
--enable-prefix-caching           # 重复长提示 TTFT 90K 档 49.5s → 1.2s
--max-num-batched-tokens 1024     # 不能调大
--gpu-memory-utilization 0.977
--max-num-seqs 4                  # 并发甜点
--attention-backend TRITON_ATTN
--language-model-only
--max-model-len 150000
```

### 3.2 坑

1. **`--gpu-memory-utilization` 0.985 / 0.99 → `Engine core init failed`**。0.977 是上限，而且必须配合 `--kv-cache-memory` 才能稳定。
2. **`--max-num-batched-tokens 4096` → KV 池 4.3 GiB 掉到 2.92 GiB** → 150K 报 `ValueError`（最大只给到 137,376）。保持 1024。
3. **`--kv-cache-memory` 硬钉 + 空闲显存波动 → `ValueError: Free memory ...`**。启动时 `Initial free memory` 只有 15.17 GiB（比平时少 0.3 GiB）就会失败；守护进程重试一次通常能过。频繁发生就把 KVBYTES 降到 3.45 GiB（仍够 150K）。
4. **`--max-num-seqs 3` 会让第 4 个并发请求排队**：C=4 聚合只有 156 tok/s，改成 4 后 255.5 tok/s。
5. **FlashInfer 编译必须 `MAX_JOBS=2`**，否则 18 个并行 nvcc 把 31 GB 内存打爆。
6. **cudagraph 捕获阶段可能 OOM**：util 0.977 时池子算出来 208,914，结果在 capture 时报 `CUDA out of memory. Tried to allocate 20.00 MiB`。
7. `--safetensors-load-strategy {eager,lazy,prefetch,torchao}` 和 fastsafetensors 都**不影响启动时间**（权重 2 秒就读完了）。

---

## 4. SGLang

1. **`--mem-fraction-static 0.94`（池 121,509）会在 100K prefill 时崩**，余量只剩 0.37 GB；0.92（池 106,357，余量 0.69 GB）才稳。
2. **mamba 槽不是免费的**：`mamba_cap = max_mamba_cache_size // ratio`（ratio 5=eager / 4=lazy）。槽数 5→20 要吃掉 **1.41 GB/卡**，KV 池塌到 21,568。
3. **fp4 KV 是陷阱**：需要 `trtllm_mha` 解码 + `flashinfer` 预填充 + 草稿后端钉死；`trtllm_mha` 的 workspace 吃 ~1 GB，把 fp4 省下的显存全抵消，所有 150K 尝试全部 OOM。
4. 长文档要配 `--disable-prefill-cuda-graph`、`--cuda-graph-max-bs-decode 1`。
5. **GGUF 完全不支持**：`GGUF model with architecture qwen35/dflash is not supported yet`。
6. 启动约 40 秒（比 vLLM 快，因为没有那套 cudagraph）。
7. 跑批脚本的 bug：**草稿模型路径和草稿 token 数不能共用变量名 `D`**（bash 前缀赋值会串值，导致 `TARGET` 解析成 "8"）。

---

## 5. llama.cpp

1. **DFlash2 必须 `--split-mode layer`**；tensor split 直接 assert：`GGML_ASSERT(src_ss[0].axis != GGML_BACKEND_SPLIT_AXIS_0)` @ `ggml-backend-meta.cpp:543`。
2. **KV 必须对称**（`-ctk q4_0 -ctv q4_0`）；非对称会让 K/V 走 CPU 回退，速度腰斩。
3. **进程名是 `llama` 不是 `llama-server`**，`pkill -x llama-server` 完全打不中。
4. 带 NCCL 编译的版本会让生产崩溃 —— `GGML_CUDA_P2P` 只是存在性检查，用 `-DGGML_CUDA_NCCL=OFF` 重编后稳定。
5. 长文档参数：`--ctx-checkpoints`（`-ctxcp`）、`--cache-ram`（`-cram`）、`--no-kv-unified`。
6. 256K 档实测 66.4 / 47.0 tok/s（关/开思考），100K 召回 5/5，显存 12.8/14.4 GB。

---

## 6. 运维与进程管理（血泪最多的一节）

### 6.1 历史事故

| 事故 | 原因 | 修法 |
|---|---|---|
| SSH 会话被自己杀掉 | `pkill -f 'run-merkyor-tests'` / `'EngineCore'` 匹配到自己的命令行 | 写成服务器端脚本执行；或用 `[E]ngineCore` 括号写法 |
| "显存先上去又归零"崩溃循环 | 两个启动脚本（PID 686766 + 703226）同时跑，互抢显卡 | 先杀干净，再单实例启动 |
| 守护进程重启后仍挂 | 崩溃后没等显存排空就重启，第一次必然 OOM | 加 `wait_vram_free()`：轮询到两卡 < 500 MiB 再启动 |
| **无限自杀循环**（本次最严重） | `cmd_start` 等待上限 360 秒，而 JIT 编译要 5–10 分钟 → 超时 → 守护判定"进程死了" → `kill_all` → 重启 → 编译从头再来 | 改成**编译感知等待**：检测到 `nvcc/cicc` 就不计时，上限 2400 秒 |
| 编译永远完不成 | 我给 systemd 单元加了 `CUDA_HOME`，nvcc 路径从 `/usr/local/cuda-13.3/bin` 变成 `/usr/local/cuda/bin`，JIT 缓存 key 失效 | 钉死 `CUDA_HOME`；装预编译轮子彻底消除编译 |
| 重复长提示每次都重新预填充（TTFT 49.5s） | 配置里写的是 `--no-enable-prefix-caching` | 改成 `--enable-prefix-caching`：30K 档 11.9s→0.7s、90K 档 49.5s→1.2s；代价是 KV 池 172,480→170,280 |
| 误报"在编译" | `pgrep -f "cicc|cc1plus"` 匹配到父 shell 的命令行 | 用 `[c]icc|[c]c1plus`，或看缓存目录有无新 `.o` |

### 6.2 现在正确的方式

```bash
ssh <user>@<host> 'bash ~/deploy-5060ti/modelctl status'
ssh <user>@<host> 'bash ~/deploy-5060ti/modelctl start vllm-dflash'   # 每 30s 打印进度，死了立刻报错
ssh <user>@<host> 'bash ~/deploy-5060ti/modelctl logs 60'
ssh <user>@<host> 'bash ~/deploy-5060ti/modelctl bench'
sudo systemctl restart modelctl                     # 整体重启（systemd 托管）
```

- **双层兜底**：systemd（`Restart=always` + `enable`，开机自启）+ 内置 `supervise`。
- 实测：`kill -9` 掉 vLLM → **90 秒自动恢复**；`systemctl restart modelctl` → 66–200 秒。
- `modelctl stop` 会同时关掉守护进程（否则守护会立刻把服务拉回来）。
- profiles：`vllm-dflash`（生产）、`vllm-mtp`、`sglang`、`llama168`、`llama256`。

### 6.3 沟通习惯（用户明确要求的）

- 不要只说"起来了"，要报 **health + KV 池 + 每卡显存** 三个数字。
- 不要盲等 / 盲轮询，要有进度输出和"死掉立刻知道"的机制。
- 不要把端点留在 down 状态。

---

## 7. 编译与加载

### 7.1 启动慢的真实构成

| 阶段 | 耗时 |
|---|---|
| 权重加载 | **2 秒**（page cache 命中；即使 18.8 GB 也一样） |
| torch.compile（AOT 缓存命中） | 0.5 秒 |
| cudagraph 捕获 + KV 分配 + GDN 初始化 | 60–190 秒 |
| **FlashInfer JIT 编译（如果缓存 miss）** | **5–10 分钟** |

→ 想加速"加载"，改 safetensors 读取策略毫无意义，真正的杠杆是消除 JIT 编译。

### 7.2 预编译内核（已装）

```bash
# 轮子索引（实体在 GitHub releases）
https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/
# 国内代理（直连 GitHub 只有 60 KB/s，代理 2.9 MB/s 起）
https://gh-proxy.com/https://github.com/flashinfer-ai/flashinfer/releases/download/...
# venv 里没有 pip，要用 uv
~/.local/bin/uv pip install --python ~/vllm-venv/bin/python <wheel>
```

- 已装 `flashinfer-jit-cache==0.6.16.post3+cu130`：**959 个预编译模块**，含 `fp4_gemm_cutlass_sm120`、`gemm_sm120`、`nvfp4_attention_sm120`、388 个 xqa 变体等。
- `cuobjdump --list-elf` 核对：预编译版与本地编译版 **cubin 都是 `sm_120`**，体积 5,336,880 vs 5,316,288 字节 —— 内核等价，不掉性能。
- 效果：重启 180 秒 → **66 秒**，全程零编译。
- **升级 flashinfer / vLLM 后必须同步换对应版本的轮子**，否则回落到现场编译。

---

## 8. 性能数字速查

| 栈 | 上下文 | KV 池 | 单流（关/开思考） | 并发聚合 | 100K 召回 |
|---|---|---|---|---|---|
| **vLLM + W4A4 + DFlash2**（生产） | 150K | **172,480** | **115.7 / 79.1** | C=4 **255.5** | 5/5 |
| SGLang + W4A4 + DFlash | 106K | 106,357 | 100.5 / 72.9 | — | 5/5 |
| llama.cpp + GSQ IQ3_S + DFlash2 | 256K | — | 66.4 / 47.0 | — | 5/5 |
| vLLM + FP8 KV + MTP3 | 150K | 247,150 | 73.2 / 58.3 | — | 5/5 |

并发明细（vLLM 生产档，`max-num-seqs 4`）：

| 并发 | 聚合 | 每条流 |
|---|---|---|
| C=1 | 70.4 | 70.4 |
| C=2 | 149.4 | 76.3 / 74.7 |
| **C=4** | **255.5** | 68.9 / 68.9 / 67.5 / 63.9 |
| C=8 | 264.7 | 一半掉到 33.7（排队） |

> KV 池共享但总量有限：4 条流平均各 ~43K token；4×100K 放不下（抢占重算），长文建议 1–2 并发。

---

## 9. 社区参考数字

| 来源 | 配置 | 结果 |
|---|---|---|
| darksidewalker | 单卡 5090 + DFlash2 + fp8 | 池 90.7–94.9K |
| 0xSero / qwen38-3090-sglang | 2×3090 + AWQ-INT4 + DSpark | 118,693（bf16） |
| club-5060ti | vLLM | 122,880 |
| 我们（本次） | 2×5060Ti + W4A4 + DFlash2 + FP8 KV | **172,480** |

**我们超过了所有已知社区配置的 KV 池**，关键是 `--kv-cache-memory` 硬钉 + FP8 KV + W4A4 省显存。

---

## 10. 文件索引

| 文件 | 用途 |
|---|---|
| `~/deploy-5060ti/modelctl` | 运维入口（status/start/stop/restart/logs/bench/supervise） |
| `/etc/systemd/system/modelctl.service` | systemd 托管（开机自启 + 崩溃重启） |
| `~/deploy-5060ti/vllm-merkyor-dflash-launch.sh` | 生产启动脚本 |
| `~/deploy-5060ti/launch.sh` / `launch-256k-dflash.sh` | llama.cpp 168K / 256K |
| `~/deploy-5060ti/sglang-merkyor-launch.sh` | SGLang 档 |
| `~/deploy-5060ti/bench_gsq.py` / `bench_concurrency.py` | 单流 / 并发基准 |
| `deploy-5060ti/TEST-MATRIX.md` | 全部参数矩阵与实测数据 |
| `deploy-5060ti/PITFALLS.md` | 本文档 |
| `~/.dsh/skills/model-ops/SKILL.md` | DSH 技能：操作铁律 |
| `~/.dsh/AGENTS.md` | 每次会话自动加载的规则 |
