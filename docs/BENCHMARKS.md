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

| Prompt context | Decode tok/s | Source |
|---|---|---|
| 49 tokens | **115.7** | `bench_gsq.py`, 512-token generation |
| 23K | 68.6 | club protocol, sustained |
| 46K | 47.7 | club protocol, sustained |
| 69K | 39.5 | club protocol, sustained |
| 92K | **33.5** | club protocol, sustained |

Prefill degrades too: 2,619 tok/s @29K → 1,932 @58K → 1,522 @87K → 1,257 @115K.
TTFT is close to linear in prompt length: 11.0 s @29K → 29.9 s @58K → 56.9 s @87K → 91.8 s @115K.

**Always quote decode with its context length.** A 115.7 tok/s claim without "49-token prompt" is
misleading for long-document workloads — this is a 3.5× difference.

## 9. club-5060ti protocol — all four reachable tiers / 官方协议四档实测

Run with [`scripts/club_receipt_vllm.py`](../scripts/club_receipt_vllm.py): a vLLM adapter that keeps
the upstream workload and pass/fail rules identical (same needle, same filler, same unique request
nonce, same calibration, same 2×512 retrieval and 2×3,072 sustained budgets, same ≥1,076 generated /
≥800 visible-character floor).

| Tier | Retrieval | Sustained (gen / decode tok/s) | Verdict |
|---|---|---|---|
| 32K | ✅ 2/2 (28.9K) | 1,939 / 70.5 ✅ · 936 / 66.6 ❌ | NOT USEFUL |
| 65K | ✅ 2/2 (57.7K) | 1,287 / 50.2 ✅ · 1,024 / 45.1 ❌ | NOT USEFUL |
| 98K | 1/2 ❌ (86.6K) | 1,461 / 41.6 ✅ · 1,084 / 37.3 ✅ | NOT USEFUL |
| **131K** | ✅ 2/2 (115.4K) | 1,782 / 35.6 ✅ · 1,260 / 31.3 ✅ | **USEFUL** |

Why the lower tiers are "not useful" — both failure modes are honest findings, not context failures:

- **32K / 65K**: the model's answer stopped after 936 / 1,024 tokens, below the protocol's
  1,076-token sustained-generation floor. The first repeat of the same tier passed (1,939 / 1,287).
  Answer length is stochastic on this prompt; the tier verdict is a coin flip.
- **98K**: the second retrieval returned `CLUB-5060` — the correct needle **truncated after 8 tokens**
  (the full needle is `CLUB-5060TI-HIGH-CONTEXT-NEEDLE-48291`). An intermittent exact-match miss
  worth watching, possibly a speculative-decoding stop-condition artifact.

131K tier detail: retrieval 2/2 at 115,406 / 115,409 prompt tokens (TTFT 91.84 s, prefill 1,256.6
tok/s); sustained 2/2 at 91,828 prompt tokens (TTFT 62.63 s, prefill 1,466.3 tok/s, decode 35.6 /
31.3 tok/s). Median decode **33.45 tok/s**, median prefill **1,361 tok/s**, median TTFT **77.2 s**.

Running the **upstream** runner on vLLM yields the same pass/fail verdicts but
`decode_tok_s: null` — it reads llama.cpp's `response["timings"]`, which vLLM does not emit — so it
reports `NOT USEFUL` even when every check passes. The adapter's deviations (streaming SSE metrics;
context ceiling from `/v1/models.max_model_len`) are recorded in each receipt's `policy.deviations`.
