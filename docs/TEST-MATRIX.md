# 测试矩阵：参数与实测速度（2026-09-09）

> 主机 `lcy-pc` · 2× RTX 5060 Ti 16GB（GB206, sm_120a）· i5-13600KF · 32GB DDR5-4800 · Ubuntu 24.04 · kernel 7.0.0-31 · 驱动 610.57.04 + aikitoria `610.57.04-p2p-v3` · Docker 29.7.1
> 每卡可用显存 16,311 MiB；两张卡均为 CPU 直连 PCIe Gen5 x8（拓扑 PHB）。
> 本文所有数字均为本机实测；失败档保留失败原因，便于复现排查。

---

## 0. 速览：三条栈的可用档

| 栈 | 模型 / 权重 | KV | 上下文（池） | 并发 | 解码 tok/s（关思考 / 开思考） | 预填充 tok/s | 备注 |
|---|---|---|---|---|---|---|---|
| **SGLang + DFlash2** | RadixArk NVFP4（10.17 GB/卡） | fp8_e4m3 | 65,536（池 65,777） | 1 | **117.9 / 68.4–74.9** | 3,044（20K 提示，TTFT 6.6 s） | needle 3/3；短上下文最快 |
| SGLang + DFlash2（大池档） | 同上 | fp8_e4m3 | 池 **90,000** | 1 | **108.7 / 76.2** | — | mfs 0.96，启动余量 0.47 GB |
| vLLM + MTP3 | pottokao NVFP4 | FP8 | **122,880** | 1 | 68.3 | 1,434 / 1,313 / 1,097（8K/32K/128K） | host 版 uv 环境 |
| **llama.cpp + MTP2**（生产默认） | unsloth UD-Q6_K（22 GB） | q8_0 | **168,000** | 4 | 56–60 | 1,069 / 1,051 / 971 / 841 / 786 | 长文档首选 |

> SGLang 与 llama.cpp/vLLM 互斥（显存），切换 = 停对方 + 跑对应脚本。

---

## 1. P2P 测试（同一台机器，2026-09-09）

驱动状态：官方 610.57.04 + aikitoria `610.57.04-p2p-v3`（srcversion `F110D2947200966FBA4D579`）+ `RMForceStaticBar1=1`；`nvidia-smi topo -p2p r` = **OK**。

| 测试 | 参数 | 结果 |
|---|---|---|
| `p2p_bw_test` | 256 MiB，`cudaMemcpyPeer` + `EnablePeerAccess` | **6.79 GB/s**（20 次循环 6.80） |
| `p2p_copy_bw` | 256 MiB，**不开** peer access | **22.40 GB/s** |
| `p2p_verify` | 64 MiB，双向 + 数据校验 | 0 / 16,777,216 不一致 |
| `p2p_peer_sweep` | 1/4/16/64/256/1024 MiB | memcpyPeer 6.40→6.80；SM 内核读对端 11.39→11.43 |
| `p2p_hybrid` | 256 MiB | D2H **26.72** · H2D **26.85** · pipelined 中转 **24.30** · P2P 单向 6.80 · P2P 双向 9.88 · 混合同向 **10.61** · 混合交叉 **10.76** |
| `p2p_vec` | 256 MiB，手写内核 | peer 写 4B 6.80 / 16B 6.80 / 64B 1.70 / 纯写 6.80；peer 读 16B **11.50**；本地读写 190–428 |
| `p2p_link` | 6 s 循环 + NVML 采样 | peer 写 **6.79**（全程 Gen5 x8、SM 2850 MHz、100% 占用）；D2H **28.39**；H2D **27.08** |
| `p2p_slices` | 28×256 MiB（0–7 GiB） | 全部 6.79–6.81；1/2/4 流 6.79–6.80 |
| `p2p_sm_paths` | 256 MiB | 写本地 428 · 写主机 **26.32** · 写对端 **6.79** · 读主机 24.06 · 读对端 11.50 |
| `nccl_bench` | 2 rank allreduce（1–256 MiB） | 默认 SHM **11.92** busbw；`NCCL_P2P_LEVEL=SYS` **4.94**；8/16 通道无变化 |

**结论**：peer 路径被平台侧压在 ~6.8 GB/s（写的 25% / 读的 40%），内存中转 24–28 GB/s；两者混合不叠加（10.6 < 24.3）。生产保持 SHM/中转，**不设** `NCCL_P2P_LEVEL=SYS`。

---

## 2. SGLang 上下文 / 推测解码矩阵

公共参数：镜像 `docker.m.daocloud.io/lmsysorg/sglang:dev-qwen38-27b-dflash2`；target `RadixArk/Qwen3.8-27B-NVFP4`；`--tp 2`；`--attention-backend flashinfer`（fp4 档例外）；`--disable-flashinfer-autotune`；`--disable-custom-all-reduce`；`--language-only`；`--disable-prefill-cuda-graph`；`--cuda-graph-max-bs-decode 1`。

### 2.1 成功启动的档

| 档名 | mfs | KV dtype | 草稿 | D | mamba | ctx 配置 | **KV 池** | 启动余量 | 解码（关/开思考） |
|---|---|---|---|---|---|---|---|---|---|
| `DBG` | 0.90 | fp8_e4m3 | z-lab bf16（2.25 GB/卡） | 8 | 5 | — | 11,367 | 0.80 GB | — |
| `L_fp8_D16` | 0.92 | fp8_e4m3 | FP8（1.52 GB/卡） | 16 | 5 | — | 36,593 | 0.54 GB | — |
| `M_mtp_eagle` | 0.93 | fp8_e4m3 | MTP 头（2.82 GB/卡） | 4 | 5 | — | 48,201 | 2.48 GB | 77.5 / 64.0 |
| **`J_fp8_D8`（已验证档）** | 0.92 | fp8_e4m3 | FP8 | 8 | 5 | 131,072 | **65,289** | 0.54 GB | 117.9 / 68.4–74.9 |
| `fp8_graphs_on` | 0.96 | fp8_e4m3 | FP8 | 8 | 5 | 163,840 | **90,000**（cap） | 0.47 GB | **108.7 / 76.2** |
| `lazy_graphs_off` | 0.96 | fp8_e4m3 | FP8 | 8 | 5 | 163,840 | **105,000**（cap） | 0.39 GB | **43.3 / 28.6** |

> `lazy_graphs_off` 额外加了 `--disable-cuda-graph` + `--mamba-radix-cache-strategy extra_buffer_lazy`。**关掉 decode CUDA 图 = 解码 −60%**（108.7 → 43.3），因此生产必须保留图。
> `M_mtp_eagle` 的 MTP 档：池更小（48,201）、解码更低（77.5/64.0），DFlash2 全面胜出。

### 2.2 FP4 KV 全部尝试（fp8 装不下 150K 时的路线，均失败）

`nvfp4` 需要 `--decode-attention-backend trtllm_mha` + `--prefill-attention-backend flashinfer` + `--speculative-draft-attention-backend trtllm_mha`。

| mfs | D | 池上限 | **KV 池** | 结果 |
|---|---|---|---|---|
| 0.92 | 8 | — | 93,824 | ❌ OOM 256 MB |
| 0.94 | 8 | — | 137,344 | ❌ OOM 32 MB |
| 0.95 | 4 | — | 168,320 | ❌ OOM 20 MB |
| 0.96 | 8 | — | 158,528 | ❌ OOM 512 MB |
| 0.96 | 4 | — | 178,944 | ❌ OOM 512 MB |
| 0.97 | 4 | — | 189,568 | ❌ OOM 512 MB |
| 0.96 | 4 | 150,000 | 149,952 | ❌ OOM 256 MB |
| 0.96 | 8 | 150,000 | 149,952 | ❌ OOM 20 MB |
| 0.95 | 8 | 150,000 | 147,904 | ❌ OOM 20 MB |
| 0.96 | 4 | 150,000（`--disable-cuda-graph`） | 149,952 | ❌ OOM 256 MB |
| 0.96 | 4 | 150,000（draft window 2048） | 149,952 | ❌ OOM 256 MB |
| 0.96 | 4 | 150,000（chunked-prefill 512） | 149,952 | ❌ OOM 256 MB |
| 0.96 | 4 | 60,000 | 59,968 | ❌ trtllm prefill 断言 |

**fp4 的账**：`nvfp4` 的 prefill 走 flashinfer 反量化工作区、decode 走 trtllm_mha，二者合计吃掉 ~1 GB，正好把 fp4 相对 fp8 省下的显存还回去（日志里 `avail mem` 从 1.30 GB 掉到 0.25 GB）。

`fp4_mx_block16`（`_plain` 路径，无反量化工作区）只支持 `triton / torch_native / flex_attention / trtllm_mha`，且 Blackwell 混合 GDN 模型不允许 `fa4`：

| 档 | mfs | D | 池上限 | **KV 池** | 结果 |
|---|---|---|---|---|---|
| `mx_t150k` | 0.96 | 4 | 150,000 | 150,000 | ❌ OOM 294 MB（triton） |

### 2.3 其他失败档（原因归档）

| 档 | 参数 | 失败原因 |
|---|---|---|
| `A_mfs090_D4` / `B_mfs092_D8` / `C_mfs095_D8` | bf16 草稿 | OOM（bf16 草稿 2.25 GB/卡，装不下） |
| `I_replay_D8` | `--enable-linear-replayssm-spec` | `requires a KDA (kimi_linear) model`（Qwen3.8 是 GDN，不可用） |
| `ggufdraft` | `--speculative-draft-load-format gguf` | `GGUF model with architecture dflash is not supported yet` |
| — | target 用 GGUF（Q6/Q5） | `GGUF model with architecture qwen35 is not supported yet` |
| mamba 槽位 4 | `--max-mamba-cache-size 4` + extra_buffer | `mamba_ratio=5, resulting max_num_reqs=0`（并发 1 时槽位必须 ≥5） |

---

## 3. llama.cpp 256K 长上下文尝试（照抄 7900XTX/ROCmFP4 配方）

参考：[QingYis/llama-7900xtx-qwen3.8-27b](https://github.com/QingYis/llama-7900xtx-qwen3.8-27b)（单张 7900XTX 24 GB → 256K 上下文，97 tok/s 峰值）。

| 项 | 参数 |
|---|---|
| 权重 | `/home/lcy/Models/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q6_K.gguf`（22 GB） |
| 草稿 | `Qwen3.8-27B-DFlash2-Q4_K_M.gguf`（1.09 GB） |
| 推测 | `--spec-type ngram-map-k4v,draft-dflash`；`--spec-ngram-map-k4v-size-n 12 --spec-ngram-map-k4v-size-m 48 --spec-ngram-map-k4v-min-hits 1`；`--spec-draft-n-max 5 --spec-draft-p-min 0.4` |
| KV | `-ctk q4_0 -ctv q4_0` |
| 上下文 | `-c 262144`（256K） |
| 省显存 | `--ctx-checkpoints 2`（默认 32 ≈ 10 GiB）、`--cache-ram 4096`、`--no-kv-unified`、`--parallel 1`、`--no-warmup` |
| 其他 | `-ngl 99 -ngld 99 -b 2048 -ub 512 -fa on --jinja --split-mode tensor --tensor-split 50,50` |

**结果**：目标模型（Q6_K + q4_0 KV）加载通过；草稿加载失败——

```
E llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 81, got 58
E srv load_model: failed to load draft model, Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

即生产版 llama.cpp（0.19.0，commit 1692f9e50）的 dflash 加载器与 Q4_K_M 草稿不兼容；新版构建（0.23.0，`~/llama.cpp-new`）未测（当时 SSH 被限流）。
脚本：`deploy-5060ti/launch-256k-dflash.sh`（可用 `BIN=` / `KVT=` / `CTX=` / `CTXCP=` 覆盖）。

---

## 4. 那位 7900XTX 用户的显存账（为什么 24 GB 能塞 256K）

| 项 | 他的开销 | 我们的对应项 |
|---|---|---|
| 权重 | ROCmFP4/STRIX 4-bit ≈ 14–15 GB | NVFP4 mixed 21 GB → 10.17 GB/卡 |
| KV | q4_0 @256K ≈ 6.1 GB（自述"省 ~14 GiB"） | fp8 @150K 需 3.03 GB/卡 |
| 草稿 | DFlash2 Q4_K_M 1.09 GB | FP8 草稿 1.52 GB/卡 |
| **`--ctx-checkpoints 2`** | 默认 32 ≈ 10 GiB | SGLang 无此杠杆 |
| 其他 | `--cache-ram 4096` / `--no-kv-unified` / `--no-warmup` | — |

结论：**不是靠 KV 量化一项**，而是"小权重 + q4_0 KV + ctx-checkpoints 2"三件套；且他用 llama.cpp（无 mamba 中间态、无草稿复制）。

---

## 5. 结论

1. **SGLang 路线**：DFlash2 + fp8 KV 的池子上限 ≈ **90K**（mfs 0.96，保留 decode CUDA 图）；到 105K 必须关图，代价是解码 −60%。**150K + DFlash2 + fp8 KV 不可行**（差 ~0.9 GB）。社区同构参考（单张 5090 32 GB + DFlash2 + fp8）实测 90,663–94,946，同样到不了 150K。
2. **fp4 KV 不是出路**：后端受限（trtllm_mha/triton）、工作区吃掉收益、质量无验证（SGLang 自身警告 + 社区 q4 证明仅"未暴露问题"）。
3. **权重侧没有更小的选择**：RadixArk NVFP4 / cyankiwi AWQ-INT4 都是 21 GB，unsloth NVFP4 22.6 GB；GGUF 在 SGLang 全路径不可加载。
4. **长上下文继续走 llama.cpp**：现有 168K（q8_0 KV）已是生产档；要冲 256K，按第 3 节的参数组合（q4_0 KV + `--ctx-checkpoints 2`）最有希望，只差一个能被 0.19.0/0.23.0 加载的 DFlash2 草稿。
5. **推荐档位**：短上下文高频交互 / 工具调用 → SGLang 65K 或 90K（108–118 tok/s）；长文档 → llama.cpp 168K（56–60 tok/s）；折中 → vLLM 122,880（68 tok/s）。

## 6. 复现

```bash
# SGLang 扫描（每档 2–4 分钟）
bash ~/deploy-5060ti/sweep4-fp4.sh     # fp4 nvfp4：mfs/D 扫描
bash ~/deploy-5060ti/sweep9-fp8.sh     # fp8：CUDA 图开关对照 + 基准
bash ~/deploy-5060ti/sweep10-fp8.sh    # fp8 + lazy mamba
# 单档试验（参数可覆盖）
NAME=t1 MFS=0.96 D=8 CTX=163840 KVDT=fp8_e4m3 ATTN=flashinfer \
  EXTRA="--language-only --disable-prefill-cuda-graph --cuda-graph-max-bs-decode 1" \
  bash ~/deploy-5060ti/try-dflash2.sh
# 基准
python3 ~/deploy-5060ti/bench_local.py   # 短提示解码吞吐（关/开思考）
python3 ~/deploy-5060ti/bench_ctx.py 100000   # 长文埋点召回（可传 token 数）
# 生产档
bash ~/deploy-5060ti/launch.sh            # llama.cpp 168K（默认）
bash ~/deploy-5060ti/sglang-dflash2-launch.sh   # SGLang 65K
bash ~/deploy-5060ti/launch-256k-dflash.sh      # llama.cpp 256K 尝试档
```

原始日志：`~/deploy-5060ti/trials/*.log`、`summary2.txt`–`summary10.txt`。

---

## 7. Merkyor W4A4（compressed-tensors NVFP4，18.82 GB）双栈实测（2026-09-09）

权重 `Merkyor/Qwen3.8-27B-EfficientThink-...-DFlash2` 的 `NVFP4/W4A4`（9.27 GB/卡，比 RadixArk NVFP4 省 0.9 GB/卡，lm_head 未量化）；草稿为其自带静态 FP8 `DFlash2-FP8`（1.54 GB/卡，需 `--speculative-draft-model-quantization compressed-tensors`）。

| 配置 | KV 池 | 解码（关/开思考） | 100K 预填充 | 100K 召回 | 备注 |
|---|---|---|---|---|---|
| SGLang + DFlash2 + fp8 KV，mfs 0.92 | **106,357** | **100.5 / 72.9** tok/s | **1,810** tok/s | **5/5** | 余量 0.69 GB，长文安全档 |
| SGLang + DFlash2 + fp8 KV，mfs 0.94 | 121,509 | 108.4 / 73.3 | — | ❌ | 余量仅 0.37 GB，100K prefill 时崩 |
| **vLLM + FP8 KV + MTP3，max-model-len 150000** | **247,150** | 73.2 / 58.3 | 1,497 tok/s | **5/5** | vLLM 的 KV 更便宜（16 KB/token） |

**结论**：150K+ 上下文 → vLLM（池 247K，还能往 ~240K 开）；≤106K 高频交互 → SGLang + DFlash2（100 tok/s）；256K → llama.cpp + GSQ IQ3_S（66.4 tok/s）。

**踩坑记录**：① vLLM 的 FlashInfer fp4_gemm JIT 在 32 GB 内存机上会被 OOM Killer 杀掉（18 个并行 nvcc），必须 `MAX_JOBS=2`；② ikantkode 的 AWQ 因 lm_head 也量化，SGLang `awq_marlin` 路径报 `'ParallelLMHead' object has no attribute 'output_size'`，不可用；③ 试验脚本里草稿路径与草稿 token 数不能共用变量名 `D`（bash 前缀赋值会串值）。

**生产切换**：`bash ~/deploy-5060ti/sglang-merkyor-launch.sh`（SGLang 档，served-model-name = `Qwen3.8-27B-Q6-dual-5060ti`，与 DSH 对齐）/ `bash ~/deploy-5060ti/launch.sh`（llama.cpp 168K）/ `vllm-merkyor-launch.sh`（vLLM 150K，需 `MAX_JOBS=2`）。

---

## 8. 当前生产配置与运维（2026-09-10 定稿）

### 8.1 生产栈

**vLLM 0.29.0 + Merkyor W4A4（NVFP4）+ DFlash2（K=10）+ FP8 KV + 前缀缓存**，端口 8080，
model id `Qwen3.8-27B-Q6-dual-5060ti`（DSH provider `qwen-local` 依赖此名）。

| 指标 | 数值 |
|---|---|
| KV 池 | **157,824 tokens**（K=10 + 前缀缓存；K=5 时 170,280，关缓存 172,480） |
| 显存 | 15,700–15,800 MiB / 卡（共 16,311；切 FLASHINFER 前是 15,460–15,590） |
| 解码（短提示） | **162** tok/s（关思考）/ **87**（开思考）— DFlash2 K=10 |
| 解码（长上下文） | **以 §8.1b 为准**（DFlash2 K=10：8K→149.9 · 96K→69.2 · 128K→57.9 tok/s）。旧行的 `23K→68.6…92K→33.5` 是 DFlash（非 K=10）时代数据，与本文档 §8.1b 自相矛盾，**已作废** |
| 前缀缓存 | 重复长提示 TTFT：30K 档 11.9s→0.7s（16.9×）· 90K 档 49.5s→1.2s（**40.2×**） |
| DFlash 接受长度 | 3.46–4.77，接受率 49–75% |
| 预填充 (TRITON_ATTN，旧) | 29K→2,619 · 58K→1,932 · 87K→1,522 · 115K→1,257 tok/s |
| 预填充 (FLASHINFER，现生产) | **97K→1,794 tok/s；TTFT 69.5 s→54.2 s（−22%）** — 见 §8.5 |
| 长文召回 | 100K 5/5；club 协议 115K 2/2 |
| 并发聚合 | C=1 74.6 / C=2 142.7 / **C=4 249.9**（甜点，无排队）/ C=6 193.8 ↓ / C=8 243.1 |
| DFlash2 接受度 | 代码类 94%、平均接受长度 9.63 token/步 |
| 启动 | **66–200 秒**（预编译内核后；波动来自 cudagraph） |
| 编译器 | `/home/lcy/vllm-current` 软链 → `vllm-venv-029`（回滚：重指软链） |
| 实验开关 | `~/.modelctl.env`（K、SPEC_METRICS、MAMBA_MODE…），改完重启即生效 |

关键参数：`--kv-cache-memory 3865470566`（硬钉 KV 池）、`--max-num-batched-tokens 1024`、
`--gpu-memory-utilization 0.977`、**`--max-num-seqs 4`**（4 槽共享同一 KV 池，池子仍 170,280）、
**`ATTN=FLASHINFER`**（在 `~/.modelctl.env` 里 `export`；启动脚本有 `ATTN="${ATTN:-TRITON_ATTN}"`，所以不用改脚本。回滚见 §8.5）、`--language-model-only`、**`--enable-prefix-caching`**、`--max-num-scheduled-tokens 8192`、`--per-request-spec-decode-metrics summary`、`--kv-cache-metrics`、`--enable-mfu-metrics`、`--cudagraph-metrics`、`--max-num-queued-reqs 32`。

**K 扫描结论**：K=5→121.6 · K=7→144.5 · **K=10→162（最优）** · K=12→148（接受率断崖 86%→53%）。每多 1 个草稿槽 KV 池少约 2,490 token，K=14 会跌破 150K。

**sm_120 平台限制（源码级确认）**：GDN 的 flashinfer/cutedsl 预填充内核只支持 SM90 与 SM100 家族，我们恒走 Triton；SymmMem all-reduce 白名单是 `['9.0','10.0','10.3','10.7']`；FlashInfer all-reduce 需要 world_size>2。详见 `PITFALLS.md` §12。

> **但这个"回退"不必修（2026-09-12 实测）**：FLA/Triton 的 GDN 预填内核在 T=1024（真实预填步长）× 48 层 = **13.1 ms/步**，
> 对 97K 预填的 69.5 s 只占 **3.4%**；nsys 剖面里它只占 GPU 时间 **1.0%**。也就是说"拿 SM120 手写内核替掉 Triton"
> 的收益上限就是这 3.4% —— **已测算，不要再往这个方向投入**。
> 同理 NVFP4 权重路径也没有可做的事：生产本来就是 ModelOpt NVFP4（`quant_algo: NVFP4` → `quantization=modelopt_fp4`），
> GEMM 走 `FlashInferCutlassNvFp4LinearKernel`，占 GPU 时间 14.2%。详见 §8.5。

**升级/回滚**：`ln -sfn ~/vllm-venv-029 ~/vllm-current`（或指回 ~/vllm-venv）→ `sudo systemctl restart modelctl`。
详见 `PITFALLS.md` §11。

> 并发甜点是 4：C=4 聚合 255.5 tok/s 且每条流仍有 ~66 tok/s；C=8 只多 3.6%（264.7），
> 后 4 条掉到 33.7 tok/s 在排队。KV 池共享但总量有限——4 条流平均各 ~43K token，
> 4×100K 放不下（会抢占重算），长文仍建议 1–2 并发。

### 8.1b 上下文扫描：预填充与解码（冷预填充，唯一 nonce 防前缀缓存命中）

脚本 `context_sweep.py`（每个档位一次冷请求，`max_tokens 256`，关思考）。

| 上下文 | 实际 prompt | TTFT | 预填充 | 解码 | 接受长度 | 接受率 |
|---|---|---|---|---|---|---|
| 8K | 8,182 | 2.44s | **3,356.8** | **149.9** | 5.00 | 40.0% |
| 16K | 16,366 | 5.47s | 2,992.2 | 134.1 | 5.00 | 40.0% |
| 32K | 32,758 | 13.45s | 2,435.9 | 114.8 | 5.00 | 40.0% |
| 64K | 65,516 | 37.04s | 1,768.8 | 86.6 | 5.00 | 40.0% |
| 96K | 98,280 | 70.81s | 1,388.0 | 69.2 | 6.25 | 52.5% |
| 128K | 131,060 | 114.64s | 1,143.3 | 57.9 | 6.25 | 52.5% |

- 预填充衰减 **2.9×**（8K→128K），解码衰减 **2.6×**；TTFT 近似线性 → 可用于预测交互延迟。
- **接受率与任务强相关**：摘要类 40–52.5%，代码类 86–94%。所以"解码速度"不是上下文的单值函数，
  报数字时必须带上 prompt 形态。
- **per-request 指标分模式**：`stream=False` 返回全部 6 项（TTFT/generation/queue/ITL/tok/s/spec）；
  `stream=True` **只返回 `speculative_decoding`**。要服务端 TTFT 就用非流式。

### 8.1c llama.cpp 上下文扫描与 DFlash 对照

模型 `GSQ-RCO IQ3_S`(11.77GB) + DFlash2-Q4_K_M 草稿，q8_0 KV，ctx 262144，**layer split**，
`BIN=~/llama.cpp-new/build/bin/llama`（旧构建加载草稿会报 81 vs 58 张量）。数字为 llama.cpp 原生 `timings`。

| 上下文 | TTFT | 预填充 | 解码 | 草稿接受 |
|---|---|---|---|---|
| 8K | 7.90s | 1,079.2 | 66.2 | 18/26 |
| 16K | 15.75s | 1,067.1 | 62.1 | 18/26 |
| 32K | 33.09s | 1,010.8 | 57.0 | 18/26 |
| 64K | 74.03s | 898.8 | 39.4 | 17/32 |
| 96K | 123.43s | 806.2 | 41.1 | 19/26 |
| 128K | 178.40s | 735.5 | 36.5 | 18/23 |

**衰减更平缓**：预填充 1.47×、解码 1.81×（vLLM 是 2.9×/2.6×）。

**vLLM ÷ llama.cpp 倍率**：8K 预填充 3.11× / 解码 2.26× → 128K **1.55× / 1.59×**。
短上下文 vLLM 大幅领先，长上下文两者趋同（都撞 KV 带宽墙）。

**DFlash 在 llama.cpp 的净效果**（同配置仅去掉 `--spec-type`）：

| 上下文 | 解码 无投机 → DFlash | 预填充 无投机 → DFlash |
|---|---|---|
| 8K | 29.1 → **66.2（+127%）** | 1,334 → 1,079（**−19%**） |
| 32K | 25.5 → **57.0（+124%）** | 1,279 → 1,011（−21%） |
| 64K | 22.0 → **39.4（+79%）** | 1,142 → 899（−21%） |

**结论：llama.cpp 上 DFlash 用 20% 的预填充换 2.2× 解码，生成型负载稳赚；纯"读长文短答"场景会亏。**

### 8.1d 实时监控

**Web 仪表盘（推荐，浏览器看）**：`http://192.168.0.119:8090/`

```bash
bash ~/gpu-model-dashboard/dashboard.sh start|stop|status   # 默认 8090，bind 0.0.0.0
bash ~/gpu-model-dashboard/dashboard.sh logs 40             # 面板日志
python3 ~/gpu-model-dashboard/verify.py                     # 自检（HTTP/SSE/引擎识别/对账）
```

单页自包含（内联 CSS/JS，无 CDN），SSE 实时推送，断线自动重连。含 7 个数字卡片
（prefill / decode / 队列 / KV / 缓存 / 接受度 / 抢占）、4 条自适应曲线、GPU 利用率-功耗-显存条、
最近 300 条采样滚动表。**必须跑在有 nvidia-smi 的机器上**——笔记本端跑能读到指标但 GPU 行是空的。

> **2026-09-12：仪表盘已摘成独立项目 `gpu-model-dashboard`**（服务器 `~/gpu-model-dashboard/`，本地
> `~/Documents/ubuntu/gpu-model-dashboard/`）。现在由 systemd 单元 `gpu-model-dashboard.service`
> （`enable` + `Restart=always`）托管、开机自启——原先那个 `~/start-dashboard.sh` 裸进程已退役。
> `~/deploy-5060ti/` 下与面板相关的残留（旧的 `dashboard.sh` 和 `modelwatch.py` 转发壳）已于
> 2026-09-12 归档到 `~/deploy-5060ti/old/dashboard/`——**本侧只保留 `modelctl watch` 这一行入口**
> （它指向新项目），面板本身不在本仓库维护。
> 项目自带 `verify.py`、`docs/READING-RULES.md`（读数规则完整版）、`README.md`。

**终端方式**：`modelctl watch --gpu`（SSH 里每 2 秒一行）。两者共用同一份 `dashboard.py`，读数规则相同。

`modelctl watch --gpu`（底层 `gpu-model-dashboard/dashboard.py`）抓引擎的 Prometheus 端点做计数器差分，**三个引擎通用**
（vLLM / SGLang / llama.cpp 各自换一套指标名，语义相同）。

```
    time  run wait   prefill   decode    KV%  cache%  accept% preempt
20:17:13    1    0       0.0     20.9   77.1    90.0     11.6       0
         GPU0 100% 78W 15767MiB | GPU1 100% 78W 15767MiB
```

**三条读数规则（容易误判）**：

1. **请求在跑时 `prefill = 0.0` 不是"没数据"，而是"提示 token 100% 命中前缀缓存"** —— 这是缓存友好型
   负载的正常稳态，也是最直观的"前缀缓存在干活"的信号。
2. **`cache%` 是累计值**（`hits_total ÷ queries_total`）。引擎日志里那个是**每 10 秒窗口**的值，空闲窗口
   天然显示 `0.0%`——不要把日志的窗口值当成"缓存命中率"引用（我就误报过一次）。
3. **`accept%` 强依赖任务形态**：代码类 86–94%，摘要类 40–52%，长上下文自由文本只有 5–24%。
   不同 prompt 之间的 accept% 不能横向比。
4. 第一帧显示 `-` 是对的（算速率需要两个采样点）；llama.cpp 若没加 `--metrics` 会明确报"识别不了"而不是显示 0。

**对账结论（2026-09-12 重做，方法已修正）**：与引擎自身日志逐窗口比对（`gpu-model-dashboard/verify.py --reconcile`）：

| 指标 | 实测一致性 |
|---|---|
| `decode` | 均值差 **1–9%**（窗口边界偏移）——硬判据 ≤20% |
| `prefill` | 有真实计算量时 **0.2%**；前缀缓存全命中的窗口我们读 0、日志仍报几千 tok/s |
| `running` | 精确一致 |

**两个坑（原"prefill 精确一致 / decode 差 8%"的结论就是这么得出来的）**：

1. **日志窗口长度不是固定 10 秒**：vLLM 不保证每 10 秒打一行，长预填充时会跳（实测 `04:43:43 →
   04:44:13` 跳了 30 秒），那一行的 `7923.9 tokens/s` 覆盖的是 **30 秒**累计。必须用**日志自己相邻两行**
   的时间戳算窗口长度，否则直接差 3 倍。
2. **两个独立相位的窗口均值不可比**：突发负载下相位误差能到 30%+（实测 decode 差 1.0% 时 prefill 差 33.3%）。
   正确做法：1–2 秒细粒度连续采样 + 按日志时间戳分桶逐窗口比。

另外 `prefill` 在缓存命中窗口会出现"我们 0 / 日志几千"：日志报的是调度器窗口内处理的提示 token（含缓存命中
的部分），而 `vllm:prompt_tokens_total` 只统计真正做了前向的 token。**所以对账别看 prefill，看 decode。**

> 仪表盘的 `cache%` 是**累计** hits÷queries（卡片下标 `cumulative`）；引擎日志里那个才是每 10 秒窗口值。
> 两者别混（§8.1d 规则 2）。完整读数规则：`gpu-model-dashboard/docs/READING-RULES.md`。

### 8.2 运维工具

- `~/deploy-5060ti/modelctl`：`status | start <profile> | stop | restart | logs [n] | bench | supervise <profile>`；
  启动时每 30 s 打印进度，进程死掉**立即报错退出**，不盲等。profiles：`vllm-dflash`（生产）、
  `vllm-mtp`、`sglang`、`llama168`、`llama256`。
- `~/gpu-model-dashboard/`：**独立项目**（2026-09-12 从本仓库摘出）——`dashboard.sh`
  （start/stop/restart/status/logs/install/uninstall）、`dashboard.py`（原 modelwatch.py）、`verify.py`、
  systemd 单元模板。8090 面板的唯一入口；`modelctl watch` 也改为调用它的 `dashboard.py`。
- systemd 单元 `/etc/systemd/system/modelctl.service`（`Restart=always` + `enable`）：崩溃自动重启、开机自启。
  实测 `kill -9` 掉 vLLM → 90 秒自动恢复；`systemctl restart modelctl` → 66 秒恢复。
- 操作规程：`~/.dsh/skills/model-ops/SKILL.md`（DSH 技能目录，会话自动可用）。

### 8.3 三个新踩的坑

1. **JIT 缓存 key 含 nvcc 完整路径**：`CUDA_HOME` 从 `/usr/local/cuda` 变成 `/usr/local/cuda-13.3`
   （或反之）会让 FlashInfer 全套重编译 5–10 分钟。systemd 单元的 `CUDA_HOME` 定死别动。
2. **`cmd_start` 360 秒超时 → 自杀循环**：JIT 编译 > 360 s 时，守护进程判定"进程死了"→ `kill_all` → 重启
   → 编译从头再来，无限循环。已改成**编译感知等待**（检测到 `nvcc/cicc` 就不计时），上限 2400 s。
3. **预编译内核**：`flashinfer-jit-cache==0.6.16.post3+cu130`（1.5 GB 轮子，959 个模块，含
   `fp4_gemm_cutlass_sm120`）装进 venv 后 flashinfer 直接加载 `.so`，**不再现场编译**。
   轮子索引 `https://flashinfer.ai/whl/cu130/flashinfer-jit-cache/`，国内用
   `https://gh-proxy.com/https://github.com/...` 代理（2.9 MB/s vs 直连 60 KB/s）。
   venv 里没有 pip，要用 `uv pip install --python <venv>/bin/python <wheel>`。
   升级 flashinfer/vLLM 后必须同步换对应版本轮子。

### 8.4 编译硬件结论

瓶颈是**内存不是 CPU**：每个 nvcc TU 约 2.5–3 GB RSS，17 个 TU，31 GB 内存下只能 `MAX_JOBS=2`。
- 最优解：内存加到 64–96 GB，`MAX_JOBS=8` → 编译快 4–6 倍（¥1200–1800）。
- 换 CPU：9950X > 270K Plus ≫ 7642；但内存不够时换 CPU 无效。EPYC 7642 单线程只有 13600KF 的 40–50%。
- 最省事：装预编译轮子，编译成本归零（已完成）。

### 8.5 attention 后端切换：TRITON_ATTN → FLASHINFER（2026-09-12）

**起因**：对 97K 预填做逐 kernel 归因（nsys 会话式抓取，`--cuda-graph-trace=node`，
480,324 个 kernel，wall 69.45 s，GPU busy 138.02 s = 两卡占用 **198.7%**，即全程满载无空转）。
结果与直觉完全不同：

| 项 | GPU 时间 | 占比 |
|---|---|---|
| attention（`kernel_unified_attention`，TRITON_ATTN） | 88.76 s | **64.3%** |
| all-reduce（`ncclDevKernel_AllReduce_Sum_bf16`，29,960 次） | 22.79 s | 16.5% |
| NVFP4 GEMM（`cutlass::device_kernel<GemmUniversal>`） | 19.57 s | 14.2% |
| norm / act / elementwise | 1.94 s | 1.4% |
| fp16→fp4 量化转换 | 1.92 s | 1.4% |
| **GDN / linear-attention（原以为的优化对象）** | 1.33 s | **1.0%** |
| spec/draft | ~0 s | ~0% |

分桶看随上下文恶化：最后一个 bin 里 attention 78%、all-reduce 10%。

**改动**：

```bash
echo 'export ATTN=FLASHINFER' >> ~/.modelctl.env
srvctl sudo gpu "systemctl restart modelctl"
```

启动脚本本来就有 `ATTN="${ATTN:-TRITON_ATTN}"`，**不需要改脚本**；改完 `ps -eo args | grep [v]llm`
应能看到 `--attention-backend FLASHINFER`。

**实测（同日、同脚本、同提示词配对）**：

| 指标（~90–100K 上下文） | TRITON_ATTN | FLASHINFER | 变化 |
|---|---|---|---|
| 预填 TTFT | 69.47 s | 54.17 s | **−22.0%** |
| 预填吞吐 | 1,398.8 tok/s | 1,793.7 tok/s | **+28.2%** |
| 解码（`context_sweep.py`，accept 6.25 两边完全相同） | 73.0 tok/s | **152.0** tok/s | **+108%（步率 11.7→24.3）** |

FLASHINFER 同时给出 `decode_backend=xqa`（`arch=sm120` 已 resolve）。

**精度自检**（`fixed_probe.py`：确定性文档 + 5 针长文检索，temperature 0，比 sha256）：

| 档位 | TRITON_ATTN | FLASHINFER |
|---|---|---|
| 29K | 5/5 · `f194e5ec2ad98c1fca809fb341645842` | 5/5 · **同哈希** |
| 100K | 5/5 · `f194e5ec2ad98c1fca809fb341645842` | 5/5 · **同哈希** |

输出逐字节相同。注：`uncalibrated q_scale/prob_scale` 警告**两个后端都会打**，不是切换引入的。

**并发无回归**（`bench_concurrency.py 256`，同日配对；TRITON 自己两轮的波动是 C=1 **18%**、C=4 **11%**）：

| C | TRITON 两轮 | FLASHINFER | 判读 |
|---|---|---|---|
| 1 | 65.1 / 76.9 | 77.4 | 波动内 |
| 2 | 155.2 / 151.3 | 160.7 | +4~6% |
| 4 | 255.1 / 229.1 | 227.5 | 波动内 |
| 8 | 255.3 / 249.3 | 248.2 | 波动内 |

短上下文并发**不区分**这两个后端（50 token 提示词下 attention 工作量本就微不足道）——
先看到"C=4 −9%"时不要下结论，那是噪声。

**回滚**：

```bash
cp ~/.modelctl.env.bak-pre-flashinfer ~/.modelctl.env && srvctl sudo gpu "systemctl restart modelctl"
```

**已关闭的门（别再试）**：

- **custom all-reduce**：去掉 `--disable-custom-all-reduce` 后 `CUSTOM` 确实注册上了
  （`Using ['CUSTOM', 'PYNCCL'] all-reduce backends`），但在 **CUDA Graph 捕获阶段**就崩：
  `Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:164 'invalid argument'` → `Engine core initialization failed`。
  **所以启动脚本里那句 flag 是必需的**，不是遗留。那 16.5% 是 TP=2 的结构成本（每层 2 次 × 95 步 ≈ 29,960 次 AR，
  PHB 拓扑无 P2P），已知选项全关：custom 崩、SymmMem 白名单不含 12.0、FlashInfer AR 需 world_size>2。
- **显存门**：做变体实验时 `UTIL=0.977` 下若空闲显存比 `0.977 × 总可用` 少 ~40 MiB 就直接
  `ValueError: Free memory ... less than desired`（本次连续两次命中，15.12 vs 15.16 GiB），需 `UTIL=0.970` 才起得来。
  systemd 的 `Restart=always` 靠重试绕过这个坑，但手动起变体时要注意。
- **孤儿 worker**：`kill -9` 掉 API server 后，TP worker 会改名成 `VLLM::Worker_TP0/TP1` 继续占 ~15.7 GB/卡，
  而 **`pkill -f 'vllm serve'` 抓不到它们**（所以 `modelctl stop` 后若显存不归零，先看
  `nvidia-smi --query-compute-apps=pid,process_name`，再按 PID kill）。

**系统型开关扫描（2026-09-12）——已扫完，结论是"没有剩余免费收益"**。把启动脚本里所有硬钉开关逐个查了源码语义：

| 旋钮 | 结论 | 依据 |
|---|---|---|
| `--no-enable-flashinfer-autotune` | **死旋钮** | `kernel_warmup.py` 里 autotune 只在 `has_flashinfer() and current_platform.has_device_capability(90)` 分支触发；我们 12.0 **恒不进该分支**，加不加只差一行 `Skipping FlashInfer autotune` 日志 |
| `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=16777216`（16 MB，vLLM 默认 394 MB） | **死旋钮** | 该 buffer 只被 `TRTLLMPrefill` 与 trtllm-gen decode 分支取用（`flashinfer.py:2179`、`:2362`）；我们走 `BatchPrefillWithPagedKVCacheKernel` + XQA，不碰它。当初设小是为了省 378 MB/卡 |
| `MAXNUM`（`--max-num-batched-tokens`）1024→2048 | **OOM 出局** | 97K 预填时 `torch.OutOfMemoryError: Tried to allocate 62.00 MiB`，15.51 GiB 只剩 41.56 MiB 空闲；KV 池只掉 401 token（157,423）但激活缓冲翻倍撑不住 |
| `SEQ` / `SCHED_TOKENS` / `--mamba-*-cache-dtype` | 无收益 | 单请求下不参与调度；GDN 全程只占 1.2%；并发已实测两后端无差（§8.5） |

**所以这一轮唯一真正的收益就是 §8.5 的 attention 后端切换本身。** 再往下压 attention（仍占 54.4%）
需要 fp8 attention 标定级别的内核工作，收益不确定 —— 不要再按"改个开关"的预期去评估它。

**切换后的剖面（2026-09-12 实测）**：同一发 97K 预填请求（唯一 nonce），同样用 nsys 会话式抓取：

| 项 | TRITON_ATTN | FLASHINFER | 新占比 |
|---|---|---|---|
| wall / GPU busy | 69.45 s / 138.02 s | **54.04 s / 107.20 s**（−22%） | — |
| 两卡占用 | 198.7% | 198.4% | 仍满载 |
| attention | 88.76 s · `kernel_unified_attention` | **58.29 s** · `flashinfer::BatchPrefillWithPagedKVCacheKernel` | 64.3% → **54.4%** |
| all-reduce | 22.79 s | 22.49 s | 21.0% |
| NVFP4 GEMM | 19.57 s | 19.55 s | 18.2% |
| fp4 量化转换 | 1.92 s | 1.96 s | 1.8% |
| norm / act / elementwise | 1.94 s | 1.93 s | 1.8% |
| GDN / linear-attention | 1.33 s | 1.30 s | 1.2% |
| kernel 总数 | 480,324 | 477,756 | — |

**只有 attention 那一项动了**（88.76→58.29 s，同样 3424 次调用下 **1.52×**），其余全在噪声内 ——
这是一次干净的受控实验。它也验证了上文的推导（推导 ~58.3 s vs 实测 58.29 s）。

**attention 仍是最大块（54.4%）**，但要再压就得做 fp8 attention 标定级别的内核工作，收益不确定 ——
不要再按"改个开关"的预期去评估它。

**本次产出的分析工具**（本地 `~/Documents/ubuntu/ninfer-gdn-port/`，机器上 `/tmp/`）：

| 文件 | 用途 |
|---|---|
| `long_prefill.py` | 预填 TTFT/吞吐（唯一 nonce 防前缀缓存命中） |
| `long_decode.py` | 解码速率，**按 `usage.completion_tokens` 计**（投机的 SSE chunk 含多 token，按 chunk 数算会差 ~10×） |
| `fixed_probe.py` | 确定性精度对拍（输出 sha256，跨后端比） |
| `nsys_launch.sh` | nsys 会话式启动（`nsys launch` → `start` → 打请求 → `stop`），抓前先起服务、只在请求窗口录制 |
| `run_variant.sh` | 用生产等价环境起配置变体（`ATTN=` / `LAUNCH_SCRIPT=` / `UTIL=` 可覆盖） |
| `nsys_sqlite_report.py` | 逐 kernel 归类 + 时间分桶（名字列可能是 StringIds，需映射） |
| `gdn_prefill_bench.py` | GDN 预填内核 golden rig（改 GDN 内核时用它和 FLA 对拍） |

剖面留在机器上：`/tmp/attn100k.nsys-rep`（58 MB）+ `/tmp/attn100k.sqlite`（143 MB）。

### 8.6 真实代码语料的深度扫描（2026-09-12）

§8.1b 用的是合成填充（重复句子 + 针），这里换成**真实语料**重测，并做了内容偏置对照。

方法：填充 = **真实仓库源码**（vLLM 自己的 2,512 个 `.py`，按路径排序拼接，每个文件加
`# ===== FILE: <相对路径> =====` 头，像一个真实的 repo dump）；token 数用**模型自带 tokenizer
精确计数**（不是 chars/4 估）；末尾接**一道 HumanEval 真题**（`HumanEval/0 has_close_elements`，
每档同一题，只有深度变）；每请求独立 nonce，前缀缓存不可能偷跑。脚本 `code_ctx_sweep.py`。
全程打在生产实例上，**零停机**。

| 深度 | 实际 prompt | TTFT | **预填 tok/s** | 解码 tok/s | finish |
|---|---|---|---|---|---|
| 8K | 8,351 | 2.54 s | **3,285.2** | 292.4 | stop |
| 32K | 32,927 | 11.82 s | **2,785.2** | 267.4 | stop |
| 64K | 65,693 | 30.14 s | **2,179.4** | 246.7 | stop |
| 96K | 98,451 | 55.12 s | **1,786.2** | 204.0 | stop |
| 128K | 128,147 | 83.56 s | **1,533.6** | 236.2 | stop |

预填衰减 **2.14×**（8K→128K），比 §8.1b 记的 2.9× 缓（因为现在跑在 FLASHINFER 上）。

**内容偏置对照**（同深度 ~96K，人造散文填充 vs 真实代码）：

| 填充类型 | prompt_tokens | TTFT | 预填 tok/s |
|---|---|---|---|
| 人造散文填充（旧脚本） | 95,652 | 52.94 s | 1,806.8 |
| 真实代码（vLLM 源码） | 98,451 | 55.12 s | 1,786.2 |

**差 1.2% → 预填耗时与内容无关**（与理论一致：稠密 attention 的代价只由形状/长度决定）。
**但解码强烈依赖任务形态**：同样 ~96K，散文提示词 69–150 tok/s，真实代码任务 **204–292 tok/s**
（接受率 86–94% vs 40–52%）。**报解码数字必须带任务类型，只带上下文长度不够。**

FLASHINFER 收益的三方交叉验证：§8.1b（TRITON，96K，合成）**1,388** → 本表口径的合成对照
（FLASHINFER，96K）**1,806**（+30%）→ 真实代码（FLASHINFER，96K）**1,786** ✓ 一致。

> 注：每档解码只有 43–168 token，单样本 ±15% 波动；要更稳需多跑几道题。

### 8.7 SM120 上游与社区现状（2026-09-12 调研）

**FlashInfer autotune 在 SM120 上不可用，且不是配置问题**：vLLM 0.29 里两处 autotune 驱动
（`fi_utils.autotune(tune_mode=True)` 与 dummy-run）**都在 `flashinfer_autotune()` 函数体内**，
而该函数只被 `elif has_flashinfer() and current_platform.has_device_capability(90)` 调用；
12.0 永远进不去，所以 `--no-enable-flashinfer-autotune` 与我们无关（该 flag 由
[vLLM PR #34006](https://github.com/vllm-project/vllm/pull/34006) 引入）。两条上游硬事实解释了
这个保守取舍：

- [flashinfer #3294](https://github.com/flashinfer-ai/flashinfer/issues/3294)：`flashinfer-cubin` 只发
  Sm100a/Sm100f/Sm103a，**没有 sm_120/121 的 cubin**
- [flashinfer #3569](https://github.com/flashinfer-ai/flashinfer/issues/3569)：RTX 5090 上**开着 autotune
  引擎启动 7/7 全崩**（autotuner 输入探测 `torch.rand` 报 device not ready），关掉才正常

**社区移植：有，且有一个和本机配置几乎同款**：

| 项目 | 内容 |
|---|---|
| ⭐ [seanyourhighness/vllm-sm12x-nvfp4-dflash2](https://github.com/seanyourhighness/vllm-sm12x-nvfp4-dflash2) | SM120/SM121 社区 overlay：**全链路 NVFP4（target 权重 + DFlash2 draft + KV cache 都是 NVFP4）**、DFlash2 K=7、**8 GiB NVFP4 KV → 262K 上下文下 325,139 token 池**。基于 vLLM v0.27.1 + 51 文件 Python-only patch + [FlashInfer PR #4346（SM120 NVFP4 paged-prefill backport）](https://github.com/flashinfer-ai/flashinfer/pull/4346)。作者原话：公开的 5090 配方没有 4-of-4 全 NVFP4 的，**普遍模式是「NVFP4 权重 + FP8 KV」——正是本机现状** |
| [devnen/qwen3.6-windows-server](https://github.com/devnen/qwen3.6-windows-server/blob/v1.3.4/docs/SM120_GDN_CEILING.md) | 独立复现了 SM120 GDN 预填天花板（同三个死路径），结论是换 NVFP4 权重把预填 1,100→5,300 tok/s |
| 上游零散 PR | [#36453](https://github.com/vllm-project/vllm/pull/36453)（SM120 NVFP4 MoE 能力检查）、[#44405](https://github.com/vllm-project/vllm/pull/44405)（consumer Blackwell 默认关 FlashInfer sampler）、#33517（`enable_sm120_or_later`）、#43477（DeepSeek V4 on SM120） |

**GDN 那条：没有可用的社区移植。** [flashinfer #2340](https://github.com/flashinfer-ai/flashinfer/issues/2340)
仍开着，sm_120 不在任何里程碑里（SM120 缺 `cvt.rs.f16x2.f32`，CUTLASS 明说 sm_100a 内核与 RTX 50 系不兼容）。
这与 §8.5 的实测一致：GDN 只占预填 3.4%，本来也不值得移植。

**当前唯一还有量级收益的方向**：KV cache 从 FP8 换 NVFP4（KV 字节再砍一半 → 同显存下池子显著变大，
或同池子留出余量）。前置条件是打补丁绕过上游的 sm_100/103 门控；精度代价见 §8.8。

### 8.8 NVFP4 KV vs FP8 KV：可行性、精度、长上下文损失（2026-09-12）

#### 8.8.1 结论先行：**在本机跑不起来**

`--kv-cache-dtype nvfp4` 会被平台级检查直接拒（实测原文）：

```
ValueError: Selected backend AttentionBackendEnum.FLASHINFER is not valid for this configuration.
Reason: ['kv_cache_dtype not supported']
```

原因不是 vLLM 缺功能——`CacheDType` 里有 `nvfp4`/`nvfp4_4over6`、`KVQuantMode.NVFP4 = 5`、
`nvfp4_kv_cache_full_dim(head) = head//2 + head//16` 打包逻辑都在，FLASHINFER 的
`supported_kv_cache_dtypes` 也列了 nvfp4。**缺的是 FlashInfer 在 sm_120 上的 NVFP4 KV 内核**：
本机 jit-cache（`~/.cache/flashinfer/0.6.16.post3/120f/cached_ops/`）里所有 attention 内核都是
`dtype_kv_e4m3`（`batch_prefill_..._kv_e4m3_..._head_dim_256`、`xqa_..._kv_cache_e4m3_..._spec_q_seq_len_10`），
没有 nvfp4。社区项目正是靠 [FlashInfer PR #4346（SM120 NVFP4 paged-prefill backport）](https://github.com/flashinfer-ai/flashinfer/pull/4346) 补上的。

另外两条后端支持矩阵（决定了能测什么）：

| 后端 | 支持 KV 类型 |
|---|---|
| TRITON_ATTN | fp8 / fp8_e4m3 / fp8_e5m2 / int4_per_token_head / int8_per_token_head / **fp8_per_token_head**（**无 nvfp4**） |
| FLASHINFER（现生产） | fp8 / fp8_e4m3 / fp8_e5m2 / **nvfp4** / nvfp4_4over6（**无 per_token_head**） |

#### 8.8.2 容量账（若内核可用）

`nvfp4 = head/2 数据 + head/16 fp8 尺度 = 0.5625 字节/元素`，对 fp8 的 1.0 字节/元素：

**池子 ×1.78 → 157,824 × 1.78 ≈ 280,600 token**，同显存下 `max-model-len` 可从 150K 提到约 250K。
NVIDIA 官方口径是"KV 内存约减半、上下文预算翻倍"（[NVFP4 KV blog](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)）。

#### 8.8.3 精度评估（离线量化仿真，真实 KV 数据）

方法 `kv_quant_study.py`：用 **Qwen3-0.6B 的真实 KV**（8 个 KV head、1,835 token、中间层，
`outputs.past_key_values` 直接取）当样本，比较三种方案——fp8_e4m3/每张量（现生产、
`KVQuantMode.FP8_PER_TENSOR`）、fp8_e4m3/每token-head（`fp8_per_token_head`）、nvfp4
（E2M1 码表 + fp16/e4m3 block-16 尺度 + 全局 fp32 除数，即 vLLM 的打包布局）。指标：元素级往返 SNR
与 attention 输出 SNR（`softmax(qKᵀ/√d)V` 对 bf16 参考），后者按上下文深度 1K→128K 扫描
（真实 KV 有放回重采样拉长，保留真实通道统计）。

**元素级 SNR（dB，越高越好）**

| 方案 | K | V | 字节/元素 |
|---|---|---|---|
| fp8 / 每张量（现生产） | 31.47 | 31.54 | 1.0 |
| fp8 / 每token-head | **34.11** | **32.00** | 1.0 |
| **nvfp4** | 20.66 | 20.54 | 0.5625 |

**attention 输出 SNR vs 上下文深度（dB）**

| 深度 | 1K | 4K | 16K | 64K | 128K |
|---|---|---|---|---|---|
| fp8 / 每张量 | 32.55 | 34.43 | 35.82 | 35.69 | **36.18** |
| fp8 / 每token-head | **34.56** | **36.78** | **38.12** | **38.17** | **38.15** |
| nvfp4 | 22.00 | 24.05 | 23.63 | 24.96 | 25.09 |

**四条可复用的结论**：

1. **排序在所有分布下稳定**：`fp8/每token-head > fp8/每张量 > nvfp4`，NVFP4 恒定低 **约 11–13 dB**。
   这是 4 位码表的固有代价——E2M1 只有 {0,.5,1,1.5,2,3,4,6}，块内动态范围 12:1，小于块最大 1/24 的值直接归零；
   而 E4M3 有 3 位尾数 + 4 位指数。NVFP4 换来的是体积，不是每元素精度。
   而且 block-16 的块**横跨 16 个不同通道**，块尺度被最大通道决定 → 同块内的小通道被压扁，
   通道间离散度越大越吃亏（合成扫描 log_std 0.1→1.6 时，NVFP4 的 attention SNR 从 20.2 掉到 3.6 dB，
   fp8/每token-head 只从 31.6 掉到 14.8）。
2. **存储量化误差不随上下文深度累积**：1K→128K 平坦甚至略升。**"长任务越跑越糟"在存储层面不成立**——
   softmax 归一化会把每 key 的误差平均掉，而信号随 key 数增长。
3. **长上下文真正的坑是累加精度，与存储格式无关**：[vLLM 官方](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
   记录 Hopper 上 FP8 FA3 在 128K NIAH 从 91%（BF16）掉到 **13%**，根因是 FP8 Tensor Core 在大收缩维上的
   中间累加丢精度，靠 two-level FP32 accumulation 修回 89%。这条只能在具体 kernel 上实测，不能用存储 SNR 推断。
4. **NVIDIA 的"<1% 精度损失"有前提**：官方口径是 NVFP4 KV 省约 50% 内存、code/knowledge/long-context 基准
   精度损失 <1%，但配方是 **ModelOpt PTQ + 校准**（`mtq.NVFP4_KV_CFG` + calibration forward loop）。
   **本机 checkpoint 是 `kv_cache_quant_algo: null`（无 KV 标定）**，即现在换上去是"未标定 NVFP4"，
   拿不到那个 <1%。社区已有 KV 标定过的 NVFP4 模型（如
   [Laguna-S KVcal](https://huggingface.co/JasonW2025/Laguna-S-2.1-ModelOpt-NVFP4-W4A4-KVcal-vllm)、
   [Qwen3.8-27B-DFlash2-NVFP4-RTNcal](https://huggingface.co/maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal)）。

#### 8.8.4 今天能做的与不值得做的

- **不值得**：为了 `fp8_per_token_head` 换回 TRITON_ATTN。它只值约 +2 dB（元素 31.47→34.11、
  attention 34.4→36.8），却要放弃 FLASHINFER 的 **TTFT −22% / 解码 ×2**（§8.5）。现生产的 fp8/每张量
  已经是 32–38 dB，很健康。
- **值得做的（若要更长上下文）**：走社区那条路——FlashInfer PR #4346 backport + KV 校准的 NVFP4 权重，
  换 1.78× 池子（150K → 约 250K）。但注意 vLLM 官方文档同时提示 **`head_dim = 256`（正是本机）
  在 FP8 下预填性能仍略差于 BF16**，换成 NVFP4 存储后 attention 还要先反量化到 FP8 再算
  （NVIDIA 明确写了当前实现是 "dequantized from NVFP4 to FP8 before attention"），
  所以**容量收益确定、速度收益不确定**，需要实测。

工具：`kv_quant_study.py`（`--sensitivity` 扫通道离散度；真实 KV 需先在 `/tmp/kvsrc` 放一个小 Qwen 模型，
缺省自动回退合成数据）。脚本在 `~/Documents/ubuntu/ninfer-gdn-port/` 与机器 `/tmp/codebench/`。

> ⚠️ 仿真的 SNR 只能给**相对排序**，不能直接换算成任务准确率：attention 对输出误差相当宽容
> （softmax 峰值主导），而准确率对误差是非线性的。要拿准确率必须端到端跑（当前内核缺失，跑不了）。

### 8.9 DSH 的「缓存命中 0%」：根因与修复（2026-09-12）

**现象**：DSH 会话统计行显示 `缓存命中 0%`。

**链路**（三跳，缺任何一跳都是 0）：

1. `dsh-llm-pi-ai` 把 pi-ai 的 `usage.cacheRead` 映射成 `cacheReadTokens`
   （`dsh-llm-pi-ai/lib/index.js`：`...usage.cacheRead > 0 ? { cacheReadTokens: usage.cacheRead } : {}`）
2. pi-ai 只认两个字段（`@earendil-works/pi-ai/dist/api/openai-completions.js`）：
   ```js
   const cacheReadTokens = rawUsage.prompt_tokens_details?.cached_tokens
                        ?? rawUsage.prompt_cache_hit_tokens ?? 0;
   ```
3. DSH UI 的 `cacheHitPercent = cacheReadTokens / (未缓存输入 + 缓存写入 + 缓存读取)`
   （`dsh-client-ui-conversation/lib/client.js`；分母为 0 时整组省略，**只有分母 > 0 且读取为 0 才显示 0%**）

**根因**：vLLM 的 `enable_prompt_tokens_details` **默认 False**，所以响应里
`usage.prompt_tokens_details` 恒为 `null`（实测两个请求都是 `null`），pi-ai 只能取到 0。

**修复**：给启动脚本加一行 `--enable-prompt-tokens-details`（在 `vllm-merkyor-dflash-launch.sh` 里，
紧跟 `--enable-prefix-caching`）。实测效果：

```
request 1（冷）:  prompt_tokens_details = {cached_tokens: 0,    created_cache_tokens: 2736}
request 2（同）:  prompt_tokens_details = {cached_tokens: 1824, created_cache_tokens: 912}
```

即 1824/2861 = **63.8%**（不是 100% 因为 block_size=912，命中按 912 整数倍走）。
该 flag 只影响 usage 上报，不影响内核/显存/调度。

**顺带两条**：

- pi-ai 读缓存**写入**用的字段名是 `cache_write_tokens`，而 vLLM 报的是 `created_cache_tokens`
  —— 名字不一致，所以「缓存写入」那条永远是 0。对命中率没影响（分母仍是 prompt_tokens）。
- 别把仪表盘上的 `cache%` 和这个混为一谈：**仪表盘是累计** hits÷queries（卡片下标 `cumulative`），
  引擎日志里那个才是**每 10 秒窗口**的值、空闲窗口天然 0.0%；DSH 这个是全日志累计（见 §8.1d 读数规则）。

**⚠️ 同时踩到的坑（改启动脚本必看）**：往**反斜杠续行链中间**插 `#` 注释会**打断命令**——
续行合并后 `#` 起注释作用，吞掉该逻辑行剩余部分，于是后面每个 `--flag \` 都变成独立命令，
`set -euo pipefail` 直接让启动失败、systemd 进重启循环。**`bash -n` 查不出来**（语法合法）。
修法：注释写在 `exec` 之前（续行链之外），链里只放 flag。验证手段：
`VLLM_BIN=/bin/echo bash <launch>.sh` 干跑一遍，看输出是不是一条完整命令。
