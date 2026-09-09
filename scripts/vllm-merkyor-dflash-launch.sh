#!/usr/bin/env bash
set -euo pipefail
# vLLM + Merkyor W4A4 (NVFP4 compressed-tensors) + DFlash2 draft + FP8 KV.
# vLLM >= 0.28 has DFlash support (speculators v0.5.0); the method is also
# auto-detected from the draft model name containing "dflash".
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=16777216
export PATH=/usr/local/cuda-13.3/bin:$PATH
export MAX_JOBS="${MAX_JOBS:-2}"

MODEL="${MODEL:-/home/lcy/Models/Merkyor-W4A4/NVFP4/W4A4}"
DRAFT="${DRAFT:-/home/lcy/Models/Merkyor-W4A4/NVFP4/W4A4/DFlash2-FP8}"
SERVED_MODEL="${SERVED_MODEL:-Qwen3.8-27B-Q6-dual-5060ti}"
K="${K:-5}"
MAXLEN="${MAXLEN:-150000}"
MAXNUM="${MAXNUM:-4096}"
UTIL="${UTIL:-0.977}"
ATTN="${ATTN:-TRITON_ATTN}"
EXTRA_ARGS=()
if [ -n "${KVBYTES:-}" ]; then EXTRA_ARGS+=(--kv-cache-memory "$KVBYTES"); fi

SPEC=$(printf '{"model":"%s","method":"dflash","num_speculative_tokens":%s,"quantization":"compressed-tensors","draft_tensor_parallel_size":2}' "$DRAFT" "$K")

exec ~/vllm-venv/bin/vllm serve "$MODEL" \
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
  --no-enable-prefix-caching \
  --disable-custom-all-reduce \
  --language-model-only \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --generation-config vllm
