# club-5060ti submission draft / 提交草稿

Target: <https://github.com/5p00kyy/club-5060ti/issues/new>
Template used: `docs/community-result-template.md`
Status: draft — not submitted yet.

---

## Title

`2x RTX 5060 Ti: Qwen3.8 27B NVFP4 vLLM with DFlash2 draft — 115.7 tok/s decode, 172,480-token KV pool, matched MTP3-vs-DFlash2 rows`

## Body

- Submission type: Raw issue report (maintainers can normalize)
- Verification status: `not verified`
- Suggested tier: `capable`

### Basic Result Metadata

- Contributor: (optional)
- Source/label: `dual-5060ti-qwen38`
- Submission date: `2026-09-10`
- Hardware lane: `2x RTX 5060 Ti`
- GPU count: `2`
- GPU model: `NVIDIA GeForce RTX 5060 Ti`
- VRAM per GPU: `16 GB` (16,311 MiB usable)
- Driver: `610.57.04`
- CPU: `Intel Core i5-13600KF`
- Host RAM: `31 GB DDR5`
- Inference/container RAM: `bare metal, no container`
- PCIe layout/link width: `both cards at PCIe Gen5 x8 (link width current 8 / max 16)`
- Motherboard/system: `Maxsun H770YTX D5 (ITX)`
- OS/container runtime: `Ubuntu 24.04.4 LTS, kernel 7.0.0-31-generic, bare metal`

### Runtime & Model

- Runtime/engine: `vLLM`
- Runtime version or commit: `0.28.0` (these receipts were measured on 0.28.0; production has since
  moved to 0.29.0, which measured **+5.9%** decode on the same config)
- Build flags: `stock uv wheel, no custom build`
- Model id: `Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2` (subdir `NVFP4/W4A4`)
- Family/variant: `Qwen3.8 27B hybrid (48 GDN + 16 full-attention layers, native 262,144 ctx)`
- Source: `Hugging Face`
- Quant: `NVFP4 W4A4 (compressed-tensors), 18.8 GB total, 9.3 GB/card, lm_head unquantized`
- Launch command or config:

```bash
vllm serve "$MODEL" \
  --served-model-name qwen3.8-27b-nvfp4 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --attention-backend TRITON_ATTN \
  --gpu-memory-utilization 0.977 \
  --kv-cache-memory 3865470566 \
  --max-model-len 150000 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 1024 \
  --enable-chunked-prefill \
  --speculative-config '{"model":"<...>/DFlash2-FP8","method":"dflash","num_speculative_tokens":5,"quantization":"compressed-tensors","draft_tensor_parallel_size":2}' \
  --no-enable-flashinfer-autotune \
  --no-enable-prefix-caching \
  --disable-custom-all-reduce \
  --language-model-only
```

### Serving Shape

- Route/server mode: `server (OpenAI-compatible /v1)`
- Context configured: `150,000`
- Actual prompt tokens: `96,588` (long-context check)
- Generated tokens: `512` (single-stream), `256` (concurrency sweep)
- KV cache dtype (K/V): `fp8`
- Tensor parallel: `2`
- Tensor split / split mode: `n/a (TP=2)`
- MTP/speculative settings: `DFlash2 draft, num_speculative_tokens=5, draft TP=2; measured accept length 3.46–4.77 (acceptance 49–75%)`
- Thinking/reasoning: `both measured (off / on)`
- Batch size / ubatch size: `max-num-batched-tokens 1024, max-num-seqs 4`
- Prompt set(s): `short code prompt (EN), 100K needle, concurrency sweep C=1/2/4/8, and the club high-context profile at the 131K tier`
- Runs / warmups: `3 warm runs; first request after startup discarded`
- Stream mode: `streamed for the protocol receipt (SSE), non-streamed for the earlier ad-hoc runs`

### Speed / Accuracy Details

- Prefill tok/s: `1,466` @92K · `1,257` @115K prompt tokens
- Decode tok/s: `115.7` short prompt (49 tokens, thinking off; runs 115.6/115.7/115.7), `79.1` (thinking on)
- **Decode tok/s at 92K context: `31.3 – 35.6`** ⚠️ see below
- TTFT: `62.6 s` @92K · `91.8 s` @115K (streaming client)
- Reported KV pool: `172,480 tokens` (`GPU KV cache size` from vLLM)
- VRAM in use: `~15,600 MiB / card`
- Retrieval: `5/5` needles at ~96.6K (ad-hoc); club protocol `2/2` at 115.4K
- Concurrency aggregate: `C=1 70.4 · C=2 149.4 · C=4 255.5 · C=8 264.7 tok/s` (shared paged KV pool, per-stream ~66–69 tok/s at C=4)

### club-5060ti high-context profile, 131K tier — `USEFUL`

Ran the upstream runner first, then a vLLM adapter of it (same workload, same checks; the upstream
runner reads llama.cpp's `response["timings"]`, so on vLLM it reports `decode_tok_s: null` and can
only conclude `NOT USEFUL` even when every check passes).

**All four reachable tiers** (the 204K+ tiers exceed this server's `max_model_len 150000`):

| Tier | Retrieval | Sustained (generated / decode tok/s) | Verdict |
|---|---|---|---|
| 32K | ✅ 2/2 (28.9K prompt) | 1,939 / 70.5 ✅ · 936 / 66.6 ❌ | NOT USEFUL |
| 65K | ✅ 2/2 (57.7K prompt) | 1,287 / 50.2 ✅ · 1,024 / 45.1 ❌ | NOT USEFUL |
| 98K | 1/2 ❌ (86.6K prompt) | 1,461 / 41.6 ✅ · 1,084 / 37.3 ✅ | NOT USEFUL |
| **131K** | ✅ 2/2 (115.4K prompt) | 1,782 / 35.6 ✅ · 1,260 / 31.3 ✅ | **USEFUL** |

131K tier detail: retrieval TTFT 91.84 s, prefill 1,256.6 tok/s; sustained TTFT 62.63 s, prefill
1,466.3 tok/s. Median decode **33.45 tok/s**, median prefill **1,361 tok/s**, median TTFT **77.2 s**.

Failure modes, reported as-is because they are findings:

- **32K / 65K**: the answer stopped after 936 / 1,024 tokens, below the protocol's 1,076-token
  sustained floor — while the first repeat of the same tier passed (1,939 / 1,287). Answer length is
  stochastic on this prompt, so the tier verdict is close to a coin flip.
- **98K**: the second retrieval returned `CLUB-5060` — the correct needle truncated after 8 tokens
  (full needle `CLUB-5060TI-HIGH-CONTEXT-NEEDLE-48291`). An intermittent exact-match miss worth
  watching; possibly a speculative-decoding stop-condition artifact.
- **Upstream runner on vLLM**: same pass/fail verdicts, `useful: false` only because
  `median_sustained_decode_tok_s` is `null`. Adapter deviations are recorded in each receipt's
  `policy.deviations`: streaming SSE metrics, and the context ceiling read from
  `/v1/models.max_model_len` instead of llama.cpp's `status.args`.

### The number worth publishing: decode scales with context, not just hardware

| Prompt context | Decode tok/s |
|---|---|
| 49 tokens | **115.7** |
| 23K | 68.6 |
| 46K | 47.7 |
| 69K | 39.5 |
| 92K | **33.5** |

A 3.5× collapse. Prefill degrades too (2,619 → 1,257 tok/s from 29K to 115K), and TTFT is close to
linear (11.0 s → 91.8 s). The short-prompt figure is what everyone quotes (including our own earlier
notes) and it is misleading for long-document work. This is exactly the failure mode the protocol's
92K sustained-generation check exists to expose — the most useful thing we can contribute.

### Matched MTP3 vs DFlash2 (the row the repo asks for)

Same machine, same model, same quant, same `--kv-cache-dtype fp8`, same `--kv-cache-memory`,
only the speculative method differs. Measured with the short prompt; a matched 92K-context row is
still pending:

| Speculative | KV pool | Decode off/on, short prompt (tok/s) | Accept |
|---|---|---|---|
| MTP n=3 | **247,150** | 73.2 / 58.3 | — |
| **DFlash2 n=5** | 172,480 | **115.7 / 79.1** | 3.46–4.77 (49–75%) |

Cross-reference: the published 2x NVFP4 preset measures **67.29 tok/s** decode with MTP n=3;
our MTP n=3 row on different silicon measures **73.2 tok/s**. The two MTP rows agree closely, which
suggests the published preset's decode is MTP-limited rather than hardware-limited, and that swapping
the draft to DFlash2 is the larger lever on this model.

Trade-off to state plainly: MTP3 keeps a **larger KV pool** (247,150 vs 172,480) because the DFlash2
draft costs ~2.4 GB across two cards. DFlash2 is the interactive-throughput pick; MTP3 is the
maximum-context pick.

### Config notes that moved the numbers

- `--kv-cache-memory` pinned at 3.6 GiB → deterministic 172,480 pool. (The club preset pins 2.5 GiB
  for 122,880; the same mechanism, just a larger pin.)
- `--max-num-batched-tokens 1024`: at 4096 the KV pool collapses 4.3 GiB → 2.92 GiB and 150K fails
  with `ValueError` (max 137,376).
- `--max-num-seqs 4`: C=4 reaches 255.5 tok/s aggregate with per-stream throughput unchanged;
  C=8 adds only 3.6% and queues half the requests.
- `--gpu-memory-utilization 0.977`: 0.985/0.99 fail with `Engine core init failed`.

### Caveats

- **Measurement harness differs from the club protocol**: non-streamed client, decode computed from
  `usage.completion_tokens / wall`, no TTFT. Numbers are therefore not protocol-comparable.
- **Clocks are stock/unlocked** (~2.5 GHz observed); the club seed baseline locks 2300 MHz. The
  cross-machine MTP-vs-DFlash2 comparison is confounded by that; the **within-machine** MTP3 vs
  DFlash2 row is clean.
- **Model source differs** from the published preset (`Merkyor W4A4` vs `unsloth/Qwen3.8-27B-NVFP4`).
- No matched 8K / 32K rows yet.
- **Prefix caching is now ON** (added after the receipts above were taken): KV pool 170,280
  instead of 172,480, and a repeated long prompt goes from 49.5 s TTFT to 1.2 s at 90K (40.2×).
  The receipts in this submission were measured with it disabled.
- FlashInfer kernels are prebuilt (`flashinfer-jit-cache`), so no JIT compilation at startup;
  startup is 66–200 s (cudagraph capture).

---

## 如果要升级成 PR

1. 用他们的 `scripts/run_high_context_profile.py` 在同一端点重测，产出 protocol-complete receipt。
2. 在 `unsloth/Qwen3.8-27B-NVFP4` 上复跑，隔离模型源变量。
3. 加 `examples/vllm-qwen38-27b-dflash2-dual-5060ti.sh` + `data/evidence/<name>.json`，
   过 `scripts/validate_evidence.py`。
4. 可选：把 `CUDA_HOME` 改动导致 FlashInfer 全套重编译 5–10 分钟这个坑写进 `docs/`。
