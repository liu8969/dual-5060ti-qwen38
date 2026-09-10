#!/usr/bin/env bash
set -euo pipefail
# vLLM + Merkyor W4A4 (NVFP4 compressed-tensors) + DFlash2 draft + FP8 KV.
# vLLM >= 0.28 has DFlash support (speculators v0.5.0); the method is also
# auto-detected from the draft model name containing "dflash".
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=16777216
# Pin the CUDA toolkit path: FlashInfer's JIT cache key includes the full nvcc path,
# so letting it drift between restarts costs a 5-10 minute recompile.
if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ]; then
  export PATH="$CUDA_HOME/bin:$PATH"
else
  _cuda=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1 || true)
  [ -n "$_cuda" ] && export PATH="$_cuda/bin:$PATH"
fi
export MAX_JOBS="${MAX_JOBS:-2}"

MODELS="${MODELS:-$HOME/Models}"
MODEL="${MODEL:-$MODELS/Merkyor-W4A4/NVFP4/W4A4}"
DRAFT="${DRAFT:-$MODEL/DFlash2-FP8}"
SERVED_MODEL="${SERVED_MODEL:-Qwen3.8-27B-Q6-dual-5060ti}"
# Prefer the switchable symlink (install.sh creates it) so upgrading vLLM is a
# one-command symlink swap instead of editing this script.
if [ -z "${VLLM_BIN:-}" ]; then
  if [ -x "$HOME/vllm-current/bin/vllm" ]; then
    VLLM_BIN="$HOME/vllm-current/bin/vllm"
  else
    VLLM_BIN="$HOME/vllm-venv/bin/vllm"
  fi
fi
K="${K:-5}"
MAXLEN="${MAXLEN:-150000}"
MAXNUM="${MAXNUM:-1024}"
UTIL="${UTIL:-0.977}"
ATTN="${ATTN:-TRITON_ATTN}"
EXTRA_ARGS=()
if [ -n "${KVBYTES:-}" ]; then EXTRA_ARGS+=(--kv-cache-memory "$KVBYTES"); fi

# --- vLLM 0.29 new knobs (all optional, env-driven) ---------------------------
# observability / diagnostics
[ -n "${SPEC_METRICS:-}" ] && EXTRA_ARGS+=(--per-request-spec-decode-metrics "$SPEC_METRICS")
[ -n "${SCHED_TOKENS:-}" ] && EXTRA_ARGS+=(--max-num-scheduled-tokens "$SCHED_TOKENS")
[ "${KV_METRICS:-0}" = "1" ] && EXTRA_ARGS+=(--kv-cache-metrics)
[ "${MFU_METRICS:-0}" = "1" ] && EXTRA_ARGS+=(--enable-mfu-metrics)
[ "${CG_METRICS:-0}" = "1" ] && EXTRA_ARGS+=(--cudagraph-metrics)
[ -n "${QUEUED_REQS:-}" ] && EXTRA_ARGS+=(--max-num-queued-reqs "$QUEUED_REQS")
# hybrid (GDN) prefix caching / kernels
[ -n "${MAMBA_MODE:-}" ] && EXTRA_ARGS+=(--mamba-cache-mode "$MAMBA_MODE")
[ -n "${PCRI:-}" ] && EXTRA_ARGS+=(--prefix-cache-retention-interval "$PCRI")
[ -n "${GDN_BACKEND:-}" ] && EXTRA_ARGS+=(--gdn-prefill-backend "$GDN_BACKEND")
# -----------------------------------------------------------------------------

SPEC=$(printf '{"model":"%s","method":"dflash","num_speculative_tokens":%s,"quantization":"compressed-tensors","draft_tensor_parallel_size":2}' "$DRAFT" "$K")

exec "$VLLM_BIN" serve "$MODEL" \
  --served-model-name "$SERVED_MODEL" \
  --host 0.0.0.0 --port 8080 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --attention-backend "$ATTN" \
  --mamba-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype bfloat16 \
  --gpu-memory-utilization "$UTIL" \
  "${EXTRA_ARGS[@]}" \
  --max-model-len "$MAXLEN" \
  --max-num-seqs "${SEQ:-4}" \
  --max-num-batched-tokens "$MAXNUM" \
  --enable-chunked-prefill \
  --speculative-config "$SPEC" \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching \
  --disable-custom-all-reduce \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --generation-config vllm
