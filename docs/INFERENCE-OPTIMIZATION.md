# 推理速度优化：实测结论与剩余方向

> 2026-09-13。全部数字来自**隔离引擎**（变体跑 8081，生产停）、**确定性提示词**（temperature=0 + 固定 nonce）、
> 每档**从 `/proc/<pid>/cmdline` 回读旋钮**核对过的测量。方法与五个坑见 §5。

## 0. 一句话结论

**调度层已经没有余量**（四个旋钮全部打平）；**预填的大头是 attention 且与内容无关**；
**解码的大头是接受率、且强依赖任务形态**。所以钱只剩在三处：
难任务的接受率、前缀缓存命中率、以及让上下文变短。其余方向（内核移植、更长的草稿、all-reduce）
实测上限都很低或根本无路。

## 1. 瓶颈分解（实测，非推断）

| 观察 | 数字 | 推论 |
|---|---|---|
| 预填随深度 | 8K→3,511 · 32K→2,783 · 64K→2,176 · 96K→1,784 · 128K→1,532 tok/s | 平滑下降；TTFT 2.5→84s。**预填是绝对成本** |
| 预填 vs 内容 | 同深度合成填充 1,806.8 vs 真实代码 1,786.2（差 1.2%） | **预填与内容无关** → 只能靠"少算"或"算得更快" |
| nsys（97K 预填） | attention 占 60%+；TP=2 all-reduce 16.5%（29,960 次 NCCL）；NVFP4 GEMM 14.2%；GDN **1.0%** | 大头是 attention；GDN 与结构通信都不值得动 |
| 解码的构成 | tok/s = tok/step × steps/s。steps/s 只随深度 26.5→20.5；**tok/step 实现类 10.31 vs 写补丁 4.6–6.7** | **解码的杠杆在接受率**，且它是任务形态的函数 |
| 解码功耗 | 75 W / 180 W | **访存延迟受限**，不是算力受限 → 提高算力/内核效率意义有限 |
| 前缀缓存 | TTFT 30K 档 11.9s→0.7s（16.9×）、90K 档 49.5s→1.2s（40.2×） | 这是**本项目最大的单点收益**，且还没调过参数 |

## 2. 已经关掉的门（别再试，附证据）

| 方向 | 结论 | 证据 |
|---|---|---|
| 调度类旋钮：`--async-scheduling`、`--max-num-batched-tokens 1536`、`draft_tensor_parallel_size=1` | **零收益** | 隔离引擎同工作负载四档：预填 2164.6/2167.1/2165.4/2168.2（±0.2%）、steps/s 23.15/23.23/23.20/23.14（±0.4%） |
| 更大草稿长度 K>10 | **断崖，且机理是草稿质量崩** | K=12 第 **0** 位接受率 99%→87%，总接受 93%→**44%**，acc/step 5.32 甚至低于 K=6 的 5.64 —— 不是"验证变贵" |
| K 从 6 加到 10 | 已经吃满 | steps/s 只 24.60→23.26（−5%），tok/step 6.64→10.31（+55%）→ 解码 +47% |
| `kv_sharing_fast_prefill` | **对本模型是 no-op** | 只有 `gemma3n.py` / `gemma4.py` 用 `enable_if` 声明"有资格的层"；`gpu_model_runner.py:7483` 的 eligible 集合恒为空 → 一层都不会跳过 token，反而白付 logits padding 开销；vLLM 自己把它列进 `unsupported`（待 PR #35045） |
| `--kv-offloading-size` | **是容量不是速度**；且有硬限制 | 需配 `--enable-cumem-allocator`（我们设了 `expandable_segments:True`）；≥16 GiB 撞 `/dev/shm`（裸机 = 内存/2 ≈ 15.9 GiB，实测 `Insufficient space in /dev/shm: 16379 MiB required`） |
| GDN 预填内核移植（hand-written for sm_120） | 上限 3.4% | T=1024×48 层占 97K 预填的 3.4%、nsys GPU 时间 1.0% |
| all-reduce 优化（SymmMem / FlashInfer AR / custom AR） | **无路** | sm_120 不在 SymmMem 白名单 `['9.0','10.0','10.3','10.7']`；FlashInfer AR 需 world_size>2；custom AR 必须禁用（否则 CUDA Graph 捕获期 `custom_all_reduce.cuh:164 'invalid argument'` 起不来） |
| `--max-num-batched-tokens 2048/4096` | 起不来 / 塌池 | 2048 在 65K 直接 HTTP 500；4096 把池打到 2.92 GiB，150K 上下文 ValueError |

## 3. 剩余方向（按 收益 × 可行性 排序）

### 3.1 难任务的接受率：prompt-lookup / suffix decoding ← 最高杠杆

现在最差的一档正是最重要的真实场景：**写 unified diff / patch 时 tok/step 只有 4.6–6.7**，而"实现给定函数"能到 10.31。
补丁类内容的特征是**大量 token 直接从上下文抄**（路径、标识符、既有代码块），而这正是
**检索式投机器**（vLLM 0.29 `SpeculativeConfig` 的 `prompt_lookup_min/max`、
`suffix_decoding_max_tree_depth`、`suffix_decoding_max_spec_factor`）的主场。

先验证两件事：① 它能否与 `method: dflash` 并存；② 若只能二选一，在 patch 类流量上谁赢。

### 3.2 前缀缓存命中率 ← 最便宜的大收益

- `block_size 912` 的对齐使每次命中最坏浪费 911 token（命中按整数块走）；
- `prefix_caching_hash_algo`、`prefix_cache_retention_interval` **从没测过**；
- agent 流量是"同一会话反复重发长上下文"的形状，命中率每涨一点都是线性 TTFT 收益。

### 3.3 让上下文变短（工作负载侧）

预填成本对 token 数线性，而 agent 每轮重发上下文。已经做的：把 DSH 的 `contextWindow`
从 196608 修正到 150000（压缩阈值 157K→120K）。接着可调：`retainRatio`（现 0.16）、
工具结果剪枝阈值。**每轮少几千 token ≈ 免费换硬件**，且与 3.1/3.2 叠加。

### 3.4 换更强的草稿模型（模型侧，上限高）

K 已在甜点、draft_tp 无收益 → 只剩"草稿本身更强"这条路。需要先确认有无可用权重。

## 4. 决定下一步的那一个测量

补一个**任务形态 × tok/step 矩阵**（实现类 / patch 类 / 散文类各 2–3 个固定任务，同深度、同提示词策略）。
理由：3.1 与 3.3 的收益都取决于"哪类流量最亏"，而目前只有一个 patch 点（4.6–6.7）和一个实现点（10.31），
不足以决定把力气压在哪。压测 MCP 一次 matrix 调用即可出（约半小时）。

## 5. 测得可信的前提：五个坑（都产出过"看起来合理"的错数据）

| 坑 | 症状 | 修法 |
|---|---|---|
| **内容不确定** | 各档接受率差 6 个点，像回归 | 提示词里的 nonce 由 `(题目, 深度, 轮次)` 派生 → 档间逐字相同 |
| **`ignore_eos=true`** | 接受率 82%→24%、解码 215→79 tok/s | 强制固定长度会把模型逼出分布外（车轱辘话）；让模型自然停，确定性靠固定 nonce |
| **实验层被应用两次** | 整条"K 曲线"四档全跑在 K=10 | `load_env_layer` 曾用 `set -a; . env` 覆盖调用方变量；profile 的 `environment_command` 又 `source` 一遍。改成"调用方优先"+ 清空 environment_command，并加单测守住 |
| **`pkill -f` 自匹配** | 启动命令静默不执行 | 命令行里含目标串，`pkill -f` 把自己那条 SSH 也杀掉。用 `[x]` 括号写法或不 pkill |
| **管道块缓冲** | 日志里看不到进度，误判卡住 | `tee｜grep` 的 grep 在非 TTY 下 4KB 缓冲；加 `--line-buffered`，数据本身在 tee 落盘文件里 |

另三条运维教训：**测量端点必须隔离到第二个端口**（`--port ${VLLM_PORT:-8080}`，变体跑 8081，否则别的会话
把请求打进被测引擎）；**harness 放持久目录**（`/tmp` 被重启清空会静默换掉工作负载）；
**旋钮必须回读核对**（每档启动后从 `/proc/<pid>/cmdline` 读回实际参数，和期望对账，不一致就作废该档）。

## 6. 工具与资产

- **压测 MCP（只测不调）**：`~/Documents/ubuntu/model-bench-mcp/`
  - `bin/bench.mjs doctor|live|mcp` —— 只读自检端点 / 面板实时快照 / MCP server
  - `bin/bench-run.mjs` —— pi-ai 做请求内核 + `/metrics` 差分算逐位接受率（**待补 model 的 `input` 等字段**）
  - 边界：不启停引擎、不换 profile、不改配置、不读 `/proc` —— 启停与调优属 `gpu-model` 侧（modelctl）
- **盒子上的测量资产**：`~/codebench/{measure_spec.py, ab_sweep3.sh, ab_depth.sh, longtasks.jsonl, repos/*}`
- **端到端数据**：`docs/TEST-MATRIX.md` §8.6（深度扫描）、§8.10（A/B 方法论 + K 曲线 + 深度曲线）

## 7. 历史数字的适用边界（避免把"变慢了吗"判错）

- §8.6（FLASHINFER + K=10）是**现行基线**；
- §8.1b（K=10 但仍 TRITON_ATTN）8K→149.9…128K→57.9，属历史档；
- 更早的「23K→68.6 … 92K→33.5」是 **DFlash2 之前**的数据。
三者相差 2–3 倍，**引用时必须带上下文长度与任务形态**，否则结论会反过来。
