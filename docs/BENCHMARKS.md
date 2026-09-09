# Benchmarks / 实测数据存档

All numbers were produced by `bench.sh` on the reference machine below. Do not compare across
different prompts, token limits, or cold/warm states.

所有数据来自下方参考机器的 `bench.sh`。不同 prompt、不同 max_tokens、冷热启动之间**不可直接比较**。

## Reference machine / 参考机器

| Item | Value |
|---|---|
| GPU | 2× RTX 5060 Ti 16 GB (GB206, sm_120a), no NVLink |
| CPU / RAM | i5-13600KF / 32 GB DDR5 |
| OS | Ubuntu 24.04, kernel 7.0.0-31, driver 610.57.04 |
| Stack | vLLM 0.28.0 + Merkyor W4A4 (NVFP4) + DFlash2-FP8 + FP8 KV |
| Kernels | `flashinfer-jit-cache==0.6.16.post3+cu130` (prebuilt, no JIT) |

## 1. Production config / 生产配置

```
--kv-cache-memory 3865470566        # pinned, -> 172,480 tokens
--max-num-batched-tokens 1024
--gpu-memory-utilization 0.977
--max-num-seqs 4
--kv-cache-dtype fp8
--attention-backend TRITON_ATTN
--max-model-len 150000
--speculative-config dflash, num_speculative_tokens=5
```

## 2. Headline / 总览

| Metric | Value |
|---|---|
| KV pool | **172,480 tokens** (1.15× a 150K request) |
| VRAM | 15,459–15,517 MiB / card of 16,384 |
| Single stream (thinking off) | **115.7 tok/s** (3 runs: 115.6 / 115.7 / 115.7) |
| Single stream (thinking on) | **79.1 tok/s** (3 runs: 79.2 / 79.1 / 79.1) |
| Aggregate C=4 | **255.5 tok/s** |
| 100K prefill | 1,393 tok/s |
| 100K needle recall | 5/5 |
| DFlash2 accept length | 3.46–4.77 (acceptance 49–75%) |
| Startup | 66–200 s (cudagraph capture) |
| Recovery after `kill -9` | 90 s |

## 3. Concurrency / 并发（共享 KV 池）

| C | Aggregate tok/s | Per-stream tok/s | Notes |
|---|---|---|---|
| 1 | 70.4 | 70.4 | |
| 2 | 149.4 | 76.3 / 74.7 | 2.1× scaling |
| **4** | **255.5** | 68.9 / 68.9 / 67.5 / 63.9 | **sweet spot** |
| 8 | 264.7 | half drop to 33.7 | only +3.6%, the rest queue |

> `bench_concurrency.py` uses `max_tokens=256` and a Chinese prompt, so C=1 here (70.4) is lower than
> the 115.7 from `bench_gsq.py` (`max_tokens=512`, English prompt). Compare **within** a table only.

## 4. Stack comparison / 三套推理栈对比

| Stack | Context | KV pool | Single stream (off/on) | 100K recall |
|---|---|---|---|---|
| **vLLM + W4A4 + DFlash2** (production) | 150K | **172,480** | **115.7 / 79.1** | 5/5 |
| SGLang + W4A4 + DFlash (mfs 0.92) | 106K | 106,357 | 100.5 / 72.9 | 5/5 |
| vLLM + FP8 KV + MTP3 | 150K | 247,150 | 73.2 / 58.3 | 5/5 |
| llama.cpp + GSQ-RCO IQ3_S + DFlash2 | 256K | — | 66.4 / 47.0 | 5/5 |

Notes:

- SGLang `--mem-fraction-static 0.94` gives a 121,509 pool but **crashes on 100K prefill**;
  0.92 is the safe setting.
- vLLM's MTP3 pool is larger (247,150) but decode is much slower than DFlash2 — DFlash2 wins for
  interactive use.
- llama.cpp is the only option for 256K (GGUF), but throughput is ~40% lower.

## 5. KV cost per token / 每 token 的 KV 成本

| Stack | KV dtype | Bytes/token/card | Consequence |
|---|---|---|---|
| SGLang | fp8_e4m3 | ~20.2 KB | 106K pool at mfs 0.92 |
| vLLM | fp8 | **16 KB** | 172K pool with the same VRAM |

This 20% difference is the main reason vLLM was chosen for long-context production.

## 6. Community comparison / 与社区配置对比

| Source | Config | KV pool |
|---|---|---|
| darksidewalker | single 5090 + DFlash2 + fp8 | 90.7–94.9K |
| 0xSero | 2×3090 + AWQ-INT4 + DSpark | 118,693 |
| club-5060ti | vLLM | 122,880 |
| **this project** | **2×5060Ti + W4A4 + DFlash2 + FP8 KV** | **172,480** |

## 7. Reproduction / 复现

```bash
bash bench.sh                 # full suite -> results/bench-<timestamp>.md
bash bench.sh 50000 100000    # custom needle sizes
SKIP_NEEDLE=1 bash bench.sh   # skip the long-context test
```

Always run the single-stream test **twice** and report the second run: the first request after
startup is 10–20% slower (cudagraph / autotune warm-up).

## 8. Decode speed vs context length / 解码速度与上下文长度

The single number everyone quotes (115.7 tok/s) is a **short-prompt** figure. Decode collapses as
context grows, because the draft model's acceptance drops and every step scans a longer KV cache:

| Context in prompt | Decode tok/s | How measured |
|---|---|---|
| 49 tokens | **115.7** | `bench_gsq.py`, 512-token generation |
| 92K tokens | **31.3 – 35.6** | club-5060ti protocol, 3,072-token budget, 1,260–1,782 generated |

Prefill also degrades mildly with context: 1,466 tok/s at 92K vs 1,257 tok/s at 115K.

**Always quote decode with its context length.** A 115.7 tok/s claim without "49-token prompt" is
misleading for long-document workloads.

## 9. club-5060ti protocol receipt — 131K tier / 官方协议 131K 档

Run with [`scripts/club_receipt_vllm.py`](../scripts/club_receipt_vllm.py) (a vLLM adapter that keeps
the upstream workload and pass/fail rules identical — same needle, same filler, same unique request
nonce, same calibration, same 2×512 retrieval and 2×3,072 sustained budgets).

| Case | Prompt tokens | TTFT | Prefill | Decode | Pass |
|---|---|---|---|---|---|
| retrieval ×2 | 115,406 / 115,409 | 91.84 s | 1,256.6 tok/s | — | ✅ |
| sustained ×2 | 91,828 | 62.63 s | 1,466.3 tok/s | 35.6 / 31.3 tok/s | ✅ |

Summary: `useful: true`, retrieval 2/2, sustained 2/2, prompt coverage ✅,
median decode **33.45 tok/s**, median prefill **1,361 tok/s**, median TTFT **77.2 s**.

Deviations from the upstream runner (both documented in the receipt):
`metrics` come from the streaming SSE stream (upstream reads llama.cpp's `response["timings"]`,
which vLLM does not emit), and the context ceiling is read from `/v1/models.max_model_len`
(upstream reads llama.cpp's `status.args --ctx-size/--parallel`).
