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
| Stack | vLLM **0.29.0** + Merkyor W4A4 (NVFP4) + DFlash2-FP8 + FP8 KV (0.28.0 for the §4/§9 runs) |
| Kernels | `flashinfer-jit-cache==0.6.18+cu130` (prebuilt, no JIT; 0.6.16.post3 before the 0.29 upgrade) |

## 1. Production config / 生产配置

```
--kv-cache-memory 3865470566        # pinned -> 170,280 tokens (172,480 with prefix caching off)
--enable-prefix-caching             # repeated long prompt: TTFT 49.5s -> 1.2s at 90K
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
| KV pool | **170,280 tokens** with prefix caching (172,480 without) |
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
| **vLLM + W4A4 + DFlash2** (production¹) | 150K | **170,280** | **122.5 / 83.6** | 5/5 |
| SGLang + W4A4 + DFlash (mfs 0.92) | 106K | 106,357 | 100.5 / 72.9 | 5/5 |
| vLLM + FP8 KV + MTP3 | 150K | 247,150 | 73.2 / 58.3 | 5/5 |
| llama.cpp + GSQ-RCO IQ3_S + DFlash2 | 256K | — | 66.4 / 47.0 | 5/5 |

Notes:

- SGLang `--mem-fraction-static 0.94` gives a 121,509 pool but **crashes on 100K prefill**;
  0.92 is the safe setting.
- vLLM's MTP3 pool is larger (247,150) but decode is much slower than DFlash2 — DFlash2 wins for
  interactive use.
- llama.cpp is the only option for 256K (GGUF), but throughput is ~40% lower.

¹ Current production values (vLLM 0.29.0, prefix caching on). The §4 stack comparison and §9 protocol
receipts were measured on vLLM 0.28.0 with prefix caching off, at a **172,480** pool and
**115.7 / 79.1** tok/s. Both are correct for their configuration — do not mix them.

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
| **this project** | **2×5060Ti + W4A4 + DFlash2 + FP8 KV** | **170,280** |

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

## 10. Prefix caching / 前缀缓存

Measured with [`scripts/prefix_cache_test.py`](../scripts/prefix_cache_test.py): the **same** long prompt
sent twice in a row (no nonce, so the second request can reuse the prefix).

| Prompt context | Cold TTFT | Cached TTFT | Speedup | Prefill cold → cached |
|---|---|---|---|---|
| 30K | 11.888 s | 0.703 s | **16.9×** | 2,524 → 42,679 tok/s |
| 90K | 49.512 s | 1.233 s | **40.2×** | 1,818 → 72,993 tok/s |

- Cost: the KV pool shrinks from 172,480 to **170,280** tokens (−1.3%) because prefix hashing needs
  its own bookkeeping. Server-reported `Prefix cache hit rate` reached 48% during the test window.
- The gain scales with context length — exactly the long-document / repeated-system-prompt case.
- All protocol receipts in §9 were taken **with prefix caching disabled** (pool 172,480). Re-running
  the 131K tier with caching on would only change the first (cold) request of each shape.

## 11. DFlash2 draft length sweep / 草稿长度扫描（vLLM 0.29）

`--speculative-config.num_speculative_tokens` (env `K` in `~/.modelctl.env`). Everything else fixed:
prefix caching on, `--max-num-batched-tokens 1024`, FP8 KV, pinned `--kv-cache-memory`.

| K | Decode off (tok/s) | Decode on | Mean accept length | Draft acceptance | KV pool |
|---|---|---|---|---|---|
| 5 | 121.6 | 83.6 | 5.70 | 94.1% | 170,280 |
| 7 | 144.5 | 85.4 | 7.33 | 90.5% | 164,807 |
| **10** | **162 – 176** | **87 – 90** | **9.63** | **86.3%** | **157,824** |
| 12 | 148.0 | 81.8 | 7.41 | **53.4%** | 153,589 |

**K=10 is the optimum, and K=12 is a cliff, not a slope.** Acceptance collapses 86% → 53% because the
DFlash2 draft model is trained to about ten tokens ahead; past that its predictions stop being useful
while every step still pays for drafting them. Per-request histograms show the shape clearly:

- K=10: `[0,0,1,1,0,0,1,1,0,0,12]` — **12 of 16 steps accepted all ten drafts**
- K=12: `[2,1,4,2,2,1,2,0,3,2,1,3,4]` — acceptance smeared across all positions, no mode

Cost of the win: the KV pool shrinks ~2,490 tokens per extra draft slot (170,280 → 157,824, −7.3%).
157,824 is still 1.05× a 150K request, so the 150K guarantee survives. K=14 would drop below 150K.

Measured with `--per-request-spec-decode-metrics summary`, which returns in every response body:

```json
"metrics": {
  "time_to_first_token_ms": ..., "generation_time_ms": ..., "queue_time_ms": ...,
  "mean_itl_ms": ..., "tokens_per_second": ...,
  "speculative_decoding": {
    "mean_acceptance_length": 9.63, "draft_acceptance_rate": 0.863,
    "acceptance_histogram": [0,0,1,1,0,0,1,1,0,0,12],
    "num_spec_steps": 16, "num_accepted_draft_tokens": 138, "num_draft_tokens": 160
  }
}
```

Aggregate (no flag needed) is on `/metrics`: `vllm:spec_decode_num_drafts_total`,
`..._draft_tokens_total`, `..._accepted_tokens_total`, and
`..._accepted_tokens_per_pos_total` (per draft position — this is what located the K=12 cliff).

## 12. Concurrency / 并发（vLLM 0.29, K=10）

| C | Aggregate tok/s | Per-stream tok/s | Notes |
|---|---|---|---|
| 1 | 74.6 | 74.7 | |
| 2 | 142.7 | 71.3 / 76.9 | 1.9× |
| **4** | **249.9** | 62.5 / 66.7 / 67.4 / 69.0 | **sweet spot, no queueing** |
| 6 | 193.8 | 2 streams drop to ~32 | queueing; aggregate *falls* |
| 8 | 243.1 | 4 streams drop to ~30 | queueing |

`--max-num-seqs 4` is confirmed optimal. Beyond 4, the extra requests do not run — they queue, and the
interference makes aggregate throughput **lower** than C=4. The binding constraint is the KV pool,
not the slot count: 4 streams already hold a large fraction of 157,824 tokens.

## 13. Platform limitations on sm_120 (source-verified) / 消费级 Blackwell 的引擎限制

Three 0.29 warnings are **engine platform gates, not misconfiguration**. Each was verified by reading
vLLM's own source (and, for the third, by importing vLLM's own constant):

| Message | Gate in the source | Verdict |
|---|---|---|
| `GDN prefill backend 'cutedsl' … Falling back to Triton/FLA` | `_resolve_gdn_prefill_backend()`: FlashInfer/CuteDSL require `is_device_capability(90)` or `is_device_capability_family(100)`. We are **12.0**. | Triton is the only GDN prefill backend for SM120. `auto` also resolves to Triton — nothing to tune. |
| `SymmMemCommunicator: Device capability 12.0 not supported` | `SYMM_MEM_ALL_REDUCE_MAX_SIZES` keys are `['9.0','10.0','10.3','10.7']`; `"12.0" in dict` → `False` | Symmetric-memory all-reduce is Hopper/datacenter-Blackwell only. |
| `FlashInfer All Reduce is disabled because it is not supported for world_size=2` | 0.29 enabled it by default for TP CUDA groups, but only for supported world sizes | Not available at TP=2 on this platform. |

Also seen and benign: `Add 2/4 padding layers, may waste at most 4.17% / 25.00% KV cache memory`,
`Padding mamba page size by 1.56%`, and `Using uncalibrated q_scale 1.0 … with fp8 attention`
(the last one is a genuine accuracy caveat — our 100K retrieval is still 5/5).

`--mamba-cache-mode all` starts fine and leaves the KV pool unchanged (157,824), but vLLM already
selects `align` automatically for this hybrid model when prefix caching is on
(`Mamba cache mode is set to 'align' for Qwen3_5ForConditionalGeneration by default when prefix
caching is enabled`). No measured benefit from overriding it, so the default is kept.
