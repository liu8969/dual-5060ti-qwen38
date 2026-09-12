# Qwen3.8-27B 双卡 5060 Ti 16G 部署总档案（实测 2026-09）

> **默认路线：llama.cpp**（launch.sh）。vLLM 为速度/工具调用备选。两栈切换 ~30 秒。

模型：pottokao/Qwen3.8-27B-NVFP4-MTP-2x16GB（vLLM 用，21GB）｜unsloth/Qwen3.8-27B-UD-Q6_K.gguf（llama.cpp 用，22GB，**Dynamic 3.0**，与官方仓库 sha256 一致）
架构：混合注意力+SSM（full_attention_interval=4，KV 天生压缩 ~3.7x，256K 原生上下文）
KV 实测：f16 ≈ 36.1 KB/token/卡；q8_0 ≈ 23.8 KB/token/卡（省 33%，非 50%）；vLLM FP8 ≈ 16 KB/token/卡

## 引擎终局对比（同机实测，全部闭环）

| 维度 | llama.cpp（Q6_K/q8/MTP2） | vLLM 宿主版（NVFP4/FP8/MTP3） |
|---|---|---|
| 解码·真实编码+思考（1.5K 提示） | 60.3 | **68.3（+13%）** |
| 解码·数数关思考（天花板） | 75.7 | 165（+118%，数数场景 MTP 接受率虚高） |
| 预填充 8K / 64K | 1069 / 971 | **1434 / 1089** |
| 最大上下文 | **168K（默认）/ 196K / 232K** | 128K（docker 版 150K） |
| 前缀缓存命中 | **默认开，实测 67×（37.8s→0.56s）** | 当前关（配方默认），可开 |
| 工具调用精确性 | 未验证 | 仓库验证过 |
| 启动/运维 | 极简、mmap 秒加载、抗折腾 | 坑多（见下），JIT 慢编译 |

**结论：默认 llama.cpp（上下文 + 缓存 + 稳定）；需要工具调用精度/预填充速度时切 vLLM。**

## llama.cpp 档位（全部深验证通过）

| 脚本 | 槽数 | 上下文 | MTP | 单流解码 | 并发总吞吐 |
|---|---|---|---|---|---|
| **launch.sh（默认）** | 4 | 168K | on | ~66–76 | ~184（4流×46） |
| launch-1slot-196k.sh | 1 | 196K | on | ~66–76 | — |
| launch-2slot-192k.sh | 2 | 192K | on | ~76 | ~114（2流×57） |
| launch-4slot-232k-nomtp.sh | 4 | 232K | off | ~34 | ~103（4流×26） |

- 稳定红线：每卡余量 ≥ ~750 MiB（满窗预填充稳定）；~470 MiB 会崩（208K 实测）
- 预填充并发矩阵与"串行长文档/并发短问答"分工见下节
- KV 池统一共享（--kv-unified）：单请求可吃满池，多请求动态瓜分，无需切换

## vLLM 宿主版档案（uv venv，v0.28.0 stable + cu130 + torch2.13）

启动：`~/deploy-5060ti/vllm-host-launch.sh`（最终参数已固化）
- util 0.977 / max-model-len 131072（128K 实测上限；132K 挂）/ MTP 3 / FP8 KV
- 16MB FlashInfer 工作区（64MB 会 OOM）、bf16 mamba 缓存、CUDA PATH（nvcc JIT 必需）
- 上下文阶梯（K=3）：122K ✅ → 128K ✅（池 134K）→ 132K ❌ → 136K ❌
- MTP 曲线（宿主版，代码题+思考开）：K2=63.7 K3=**68.3** K4=67.5 K5=64.5 K6=65.3（3~6 噪声带内）
- docker nightly 版（备选）：150K @ MTP6，KV 池 166K，内存账本优于宿主稳定版 ~0.7G

### vLLM 七坑（全部踩过，供后人）

1. 桌面环境占 ~0.45G/卡 → util 0.977 起不来；`systemctl stop gdm`
2. docker 容器（root）泄漏 /dev/shm psm_* 文件 → 需 sudo 清理
3. PyPI 无 vLLM 预发布版；flashinfer-cubin 最高 0.6.13 与 flashinfer 0.6.16 不匹配 → 卸 cubin 走 nvcc JIT
4. JIT 编译期（~5 分钟/新形状）日志循环 shm_broadcast 是**慢编译不是死锁**，用 py-spy 看栈判别；换 maxlen/MTP 都会触发新形状编译
5. KV 画像随机性：同一配置可用 KV 在 2.1–3.0 GiB 波动，卡线档位时过时不过
6. FlashInfer 工作区在预算外惰性分配 → 首请求 OOM；缩到 16MB 解决
7. 强杀进程要按 `VLLM::` 标题和解释器路径双杀，spawn 的 worker 会漏网占显存

## 缓存机制（默认路线实测）

- llama.cpp：LCP 前缀复用默认开，同文档第二次请求 TTFT 37.8s → 0.56s（67×）
- 用法：长文档多轮问答天然命中；Agent 把系统提示词放最前
- vLLM：`--no-enable-prefix-caching`（配方默认）；切 vLLM 跑 agent 时应改 `--enable-prefix-caching`

## 服务与集成

- 端点：http://192.168.0.119:8080/v1（两栈同名同端口，DSH 零改动）
- DSH：settings.yaml 注册 qwen-local provider（pi-ai 插件，热加载；占位 key 在 .credentials.yaml）
- 模型下载走 hf-mirror（HF_ENDPOINT=https://hf-mirror.com）；Docker 镜像走 daocloud 源
- 两栈切换：pkill 对应进程 + 跑对应脚本（llama.cpp 约 30s；vLLM 同形状约 2min）

## 未启用杠杆（按需）

- --cache-ram：KV 页溢出内存（32G 内存下可与页缓存抢空间，慎用）
- q4_0 KV（llama.cpp）：再省一半可开 256K，精度风险
- IQ3_XXS 权重：省 ~5G/卡，q8 KV 下可开 256K
- 32G 内存页缓存：22GB GGUF 可整文件驻留 → 重复加载 ~5s（"秒切"的免费实现）
- systemd Restart=always：压余量到 ~550 MiB 时配监管者

## P2P 解锁（2026-09-09 实测成功）

> **决策（2026-09-09）：不启用 P2P。** 生产继续走内存中转/SHM 路径（llama.cpp 22–28 GB/s、NCCL SHM 11.8–12.3 GB/s busbw），性能与开启 P2P 前一致。
> 现状保留：aikitoria `610.57.04-p2p-v3` 模块 + `RMForceStaticBar1=1`（零成本，内核升级后 DKMS 会自动回退原版模块）；
> 待定：GRUB 的 `iommu=pt pcie_acs_override=downstream,multifunction` 是为 P2P 加的，属安全权衡，彻底放弃 P2P 后可在下次重启时移除。
> 下面的内容保留为技术档案，供以后换平台时参考。

- **当前状态（2026-09-09 更新）：官方 610.57.04（`.run --dkms`）+ aikitoria `610.57.04-p2p-v3` 模块**（srcversion `F110D2947200966FBA4D579`）
- 关键：p2p 模块必须覆盖 `updates/dkms/`（优先级高于 kernel/），**删掉同名 `.ko.zst`**，再 `depmod -a`
- GRUB：`intel_iommu=on iommu=pt pcie_acs_override=downstream,multifunction`；BIOS 必须开 Above 4G + ReBAR（本机 BAR1=16GB/卡，已开）
- `nvidia-smi topo -p2p r` = OK；`cudaDeviceEnablePeerAccess` 16MB→1GB **全部成功**
  （旧分支 `610.57.04-p2p` v1 在 ≥16MB 报 `cudaErrorMapBufferObjectFailed`；v3 的"console 折叠进 BAR1 映射"修掉了这个窗口裁剪问题）
- 数据校验：`p2p_verify` 双向 0/16777216 不一致；torch 跨卡 `copy_` 现已正确（610.43.02 曾**静默无操作**）
- **必须加 regkey `RMForceStaticBar1=1`**（`/etc/modprobe.d/nvidia-p2p.conf` → `options nvidia NVreg_RegistryDwords="RMForceStaticBar1=1"`）
  - 默认策略是 AUTO；本机 BAR1(16GB) == 显存 且 `/proc/fb` 有 framebuffer console，AUTO 判定装不下 → **static BAR1 被关 → P2P 掉进未初始化的 mailbox 路径**
  - 症状：dmesg 反复刷 `kbusSetP2PMailboxBar1Area_GM200: P2P mailbox area expected from RM but no valid address is installed gpu=1` + `NV_ERR_INVALID_STATE @ p2p_api.c:445`
  - 这正是上游 PR #34「Fix BAR1 P2P on display-attached GPUs」描述的问题（上游默认已改 ENABLE，v3 仍是 AUTO）
  - 加 regkey 后：peer 测试全跑一遍，dmesg **0 条** p2p/mailbox 报错；VRAM 总量不变（16311 MiB/卡）
- **⚠️ 但 P2P 带宽仍比内存中转慢，且两条路径不叠加**（两卡都在 CPU 根复合体下、x8/x8、拓扑 PHB）
  - 256MB 同尺寸实测（regkey 生效后）：`cudaMemcpyPeer` 开 peer access **6.79 GB/s** / 不开 **22.4 GB/s** / SM 内核读 peer **11.4 GB/s**；D2H 26.7、H2D 26.9、**pipelined 中转 24.3 GB/s**
  - peer 双向同时 **9.88 GB/s**（只有单方向 1.45×，未做到全双工 2×）
  - **混合叠加实验（关键）**：P2P 一半 + 中转一半同方向并发 = **10.61 GB/s**；P2P 0→1 + 中转 1→0 并发 = **10.76 GB/s**
    → 两者都远低于纯中转 24.3，**不叠加反而互相拖累**（两条路径抢同一张卡的出方向链路）
  - 物理解释：单方向拷贝上限 = 一张卡的出方向链路 ≈ 27 GB/s；中转天然让两张卡各自出方向并行 → 24.3；peer 写入在本机被压到 6.8（CPU 根复合体 port-to-port 转发能力），所以没有收益空间
  - NCCL allreduce busbw：默认 **SHM 11.92 GB/s** vs `NCCL_P2P_LEVEL=SYS` 强制 `P2P/CUMEM` **4.94 GB/s**（慢 2.4×；加 `NCCL_MIN_NCHANNELS=8/16` 无效，非通道数问题）
  - 结论：**绝不设 `NCCL_P2P_LEVEL=SYS`**；NCCL 默认判 `isAllDirectP2p 0` 自动走 SHM，性能反而最好
  - 社区里"开 P2P 带宽更高/翻倍"的表格（5090 单向 55.6 vs 43.3、双向 111；3×5060 Ti 单向 14.09→双向 27.5）成立前提是 **peer 路径能跑满链路 + 全双工**，服务器平台/带 PCIe switch 的机器才有；本机两条都不满足
- **根因定位（2026-09-09 实测 + 社区交叉验证）：Intel 消费级 root complex 的跨设备 BAR 转发被掐**
  - 同一张卡、同一条链路（Gen5 x8）、同样满载 2850 MHz 的 SM：写本地显存 428 GB/s、**写主机内存 26.3 GB/s**、**写对端显存 6.79 GB/s**、读对端 11.5 GB/s
  - 6.8 与访问宽度（4B=16B）、并发流（1/2/4）、地址（0–7GiB 全切片）、时钟均无关 → 不是 SM/链路/窗口问题
  - 社区同款：r/LocalLLaMA「PSA 别用 Intel 消费平台做多卡」——Z890 + 2× **A6000**（数据中心卡）P2P 只有 **5.48 GB/s** 单向 / 10.96 双向；[NVIDIA issue #1253](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/1253) 证实 Arrow Lake root complex 跨 root port 路由 TLP 有缺陷，同栈在 Threadripper PRO 上正常；驱动 `chipset_info.c` 对 Z590/Z690/Z790 打了 `PDB_PROP_CL_HAS_RESIZABLE_BAR_ISSUE`（本机主板 Maxsun H770YTX D5，PCH 8086:7a05）
  - Linux 内核 p2pdma 白名单也只有 Xeon/Skylake-E（+ Grace/Google SoC），Intel 消费级与 AMD 均不在内
- **平台对比（P2P 单向 / 中转 / 结论）**
  | 平台 | 链路 | P2P 单向 | 内存中转 | P2P 是否值得 |
  |---|---|---|---|---|
  | **本机** i5-13600KF + H770 | gen5 x8 | **6.8** | **24.3** | ❌ 反向 |
  | Intel 消费级 Z890 + A6000 | gen5 x16 | 5.48 | — | ❌ |
  | Xeon Silver 4416+ 8×4090 | gen4 x16 | 22.7 | 16.8（4 卡 NCCL） | ✅ +20~35% |
  | **EPYC 7302 (Rome) 4×4090** | gen4 x16 | **26.3** | **4.1（I/O die 掐）** | ✅ **6.1×** |
  | EPYC 9354 / 9575F 8~9×5090 | gen5 x16 | 48.5 / 55.6 | 31.5 | ✅ +54% |
- **"P2P 上限 27" 是误读**：27 GB/s 只是本机 x8 gen5 链路。P2P 单向 ≈ min(两卡链路带宽) × 0.8~0.9，双向 ≈ 2×；gen4 x8≈14、gen4 x16≈26、gen5 x8≈24~27、gen5 x16≈48~55、NVLink(3090) 双向 112
- **P2P 的价值 = 它比中转快多少**：Rome 中转 4.1 → P2P 26 是 6×；本机中转 24.3 → P2P 上限 27 最多 +11%（实际 −72%）。换平台对我们这套 x8 gen5 卡也没收益（Rome 上会退成 x8 gen4 ≈14，仍低于现在的中转）
- llama.cpp TP2 解码无净收益（60.3 → 59.0）；vLLM/SGLang 维持 `--disable-custom-all-reduce`
- 补丁模块的价值在**正确性**（IPC/peer 映射不再静默出错），不在吞吐
- 历史数据存疑：610.43.02-p2p 下曾测到 65.1 GB/s 直连，在 610.57.04-v3 上无法复现（现为 6.8–22.4 GB/s），以本页新数据为准
- **重启后/内核升级后**：DKMS 会重建原版模块并覆盖 `updates/dkms/`，需重新拷贝 v3 模块 + `depmod`
- 安装脚本：`deploy-5060ti/install-p2p-v3.sh`（备份旧模块 → 停 gdm/coolercontrold → rmmod → modules_install → 覆盖 updates/dkms → depmod → modprobe → 校验）

## llama.cpp 速度优化结论（全部实测）

| 优化项 | 结果 |
|---|---|
| MTP 档位 | n=2 已最优（n=1 慢 15%；n≥3 在 168K 下 OOM） |
| NCCL 重建 | TP 通信 +6.7%，净持平 |
| DFlash2（新版 llama.cpp 0.4.0 + Q4_K_M 草稿） | 同配置比 MTP **+36%**（47.7 vs 35.2），但仅 layer 拆分可用，而 layer 比 tensor 慢 40% → 净亏 |
| DFlash2 + tensor 拆分 | ❌ 上游断言（ggml-backend-meta.cpp:543） |
| DFlash2 + row 拆分 | ❌ CUDA 不支持 split buffers |
| 结论 | 默认档已是本硬件实用最优；DFlash2 待上游修复 tensor 拆分后可期 +36% |

版本注记：DFlash2 支持自 llama.cpp 2026-08-27 起；旧版（1692f9e50，08-14）不认 dflash 架构。新版源码在 `~/llama.cpp-new`（0.4.0-dev），生产用 `~/llama.cpp`。

## 文档索引（2026-09-09）

- **`TEST-MATRIX.md`** —— 全部测试的参数与实测速度：P2P 11 项 + SGLang 上下文/推测解码 20 档（含全部失败原因）+ llama.cpp 256K 尝试 + 三条栈对照
- `SGLANG-DFLASH2.md`（服务器 `~/deploy-5060ti/`）—— SGLang + DFlash2 部署档案（65K 已验证档：117.9 / 68.4–74.9 tok/s）
- `trials/`（服务器）—— 原始日志 + `summary2.txt`–`summary10.txt`
- 关键结论：SGLang + DFlash2 + fp8 KV 的池子上限 ≈ **90K**（保 CUDA 图）；150K 需 fp4 KV（后端受限 + 质量未验证）或去掉草稿模型；长上下文继续走 llama.cpp（168K）
