#!/usr/bin/env bash
# =============================================================================
#  dual-5060ti-qwen38  —  one-shot installer / 一键安装脚本
#
#  Target: Ubuntu 24.04+, 2x RTX 5060 Ti 16GB (or any 2x 16GB Blackwell),
#          ~32GB system RAM, driver >= 580 (CUDA 13.x runtime).
#
#  What it does / 做什么:
#    1. preflight  — verify GPUs, driver, CUDA, python, uv
#    2. venv       — create $VENV and install vLLM (pinned)
#    3. kernels    — install flashinfer-jit-cache (no JIT compile at startup)
#    4. models     — download Merkyor W4A4 + DFlash2-FP8 draft
#    5. scripts    — install modelctl + launch script + systemd unit
#    6. verify     — start, wait for health, print KV pool / VRAM / smoke test
#
#  Usage / 用法:
#    bash install.sh                 # normal install
#    DRY_RUN=1 bash install.sh       # print what would happen, change nothing
#    USE_SYSTEMD=0 bash install.sh   # skip systemd, run under modelctl supervisor only
#
#  Everything is overridable by env var; see the CONFIG block below.
# =============================================================================
set -euo pipefail

# ------------------------------- CONFIG --------------------------------------
VENV="${VENV:-$HOME/vllm-venv}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/deploy-5060ti}"
LOG_DIR="${LOG_DIR:-$HOME/modelctl-logs}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/Models/Merkyor-W4A4}"
HF_REPO="${HF_REPO:-Merkyor/Qwen3.8-27B-EfficientThink-K3-Opus5-Grok4.6-GPT5.6Sol-SFT-SimPO-DFlash2}"
MODEL_SUBDIR="${MODEL_SUBDIR:-NVFP4/W4A4}"
MODEL_DIR="$MODEL_ROOT/$MODEL_SUBDIR"
DRAFT_DIR="$MODEL_DIR/DFlash2-FP8"

VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
CUDA_TAG="${CUDA_TAG:-cu130}"                 # wheel tag for flashinfer-jit-cache
FLASHINFER_VER="${FLASHINFER_VER:-}"          # empty = detect from installed flashinfer
PORT="${PORT:-8080}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-27B-Q6-dual-5060ti}"
KV_BYTES="${KV_BYTES:-3865470566}"            # pinned KV pool -> 170,280 tokens
MAX_LEN="${MAX_LEN:-150000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-1024}"
GPU_UTIL="${GPU_UTIL:-0.977}"
MIN_VRAM_MIB="${MIN_VRAM_MIB:-15360}"         # per-card requirement
MIN_GPU_COUNT="${MIN_GPU_COUNT:-2}"

USE_SYSTEMD="${USE_SYSTEMD:-1}"
DRY_RUN="${DRY_RUN:-0}"
HF_MIRROR="${HF_MIRROR:-https://hf-mirror.com}"

# ------------------------------- helpers -------------------------------------
BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; DIM=$'\033[2m'; OFF=$'\033[0m'
step() { echo; echo "${BOLD}==> $*${OFF}"; }
info() { echo "    $*"; }
ok()   { echo "    ${GREEN}OK${OFF}   $*"; }
warn() { echo "    ${YELLOW}WARN${OFF} $*"; }
die()  { echo; echo "${RED}FAIL${OFF} $*" >&2; exit 1; }
run() {
  if [ "$DRY_RUN" = "1" ]; then
    echo "    ${DIM}[dry-run]${OFF} $*"
  else
    echo "    ${DIM}\$ $*${OFF}"
    "$@"
  fi
}

need() { command -v "$1" >/dev/null 2>&1; }

# uv / hf live in ~/.local/bin on many boxes and are missing from a non-login PATH
case ":$PATH:" in
  *":$HOME/.local/bin:"*) ;;
  *) PATH="$HOME/.local/bin:$PATH"; export PATH ;;
esac

echo "${BOLD}dual-5060ti-qwen38 installer${OFF}"
[ "$DRY_RUN" = "1" ] && warn "DRY_RUN=1 — nothing will be changed"

# ------------------------------- 1. preflight --------------------------------
step "1/6 preflight / 环境检查"

need nvidia-smi || die "nvidia-smi not found — install the NVIDIA driver first"
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
[ "$GPU_COUNT" -ge "$MIN_GPU_COUNT" ] || die "need >= $MIN_GPU_COUNT GPUs, found $GPU_COUNT"
ok "GPUs: $GPU_COUNT"

nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | while IFS= read -r l; do
  info "  $l"
done

while IFS=, read -r idx total; do
  total=$(echo "$total" | tr -dc '0-9')
  if [ "${total:-0}" -lt "$MIN_VRAM_MIB" ]; then
    die "GPU $idx has only ${total} MiB VRAM, need >= ${MIN_VRAM_MIB} MiB"
  fi
done < <(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader)
ok "per-card VRAM >= ${MIN_VRAM_MIB} MiB"

CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
case "$CAP" in
  12.*) ok "compute capability $CAP (Blackwell sm_120)" ;;
  *)    warn "compute capability $CAP — this recipe is tuned for sm_120 (Blackwell consumer)" ;;
esac

if pgrep -x gdm3 >/dev/null 2>&1 || pgrep -x gdm >/dev/null 2>&1; then
  warn "a display manager is running and will steal VRAM; consider: sudo systemctl stop gdm3"
fi

RAM_GB=$(awk '/^MemTotal:/{printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 0)
info "system RAM: ${RAM_GB} GB"
if [ "${RAM_GB:-0}" -lt 60 ]; then
  warn "with < 60 GB RAM keep MAX_JOBS=2 (each nvcc TU needs ~3 GB)"
fi

need curl || die "curl not found"
if need uv; then ok "uv $(uv --version | awk '{print $2}')"
elif need python3; then warn "uv not found — falling back to python3 -m venv + pip"
else die "need either uv (recommended) or python3"
fi

if [ "$DRY_RUN" = "0" ] && [ ! -d "$VENV" ]; then
  mkdir -p "$(dirname "$VENV")"
fi
mkdir -p "$INSTALL_DIR" "$LOG_DIR"

# ------------------------------- 2. venv + vLLM ------------------------------
step "2/6 python venv + vLLM $VLLM_VERSION"

if [ -x "$VENV/bin/python" ]; then
  ok "venv already exists: $VENV"
else
  if need uv; then
    run uv venv --python 3.13 "$VENV"
  else
    run python3 -m venv "$VENV"
  fi
fi

# A stable symlink so upgrading vLLM is "repoint the link and restart".
# Never clobber an existing link: on a machine that has deliberately switched to
# another venv, re-running this installer must not silently downgrade it.
if [ -L "$HOME/vllm-current" ]; then
  ok "keeping existing symlink: $HOME/vllm-current -> $(readlink "$HOME/vllm-current")"
else
  run ln -sfn "$VENV" "$HOME/vllm-current"
  ok "active interpreter symlink: $HOME/vllm-current -> $VENV"
fi

if need uv; then
  run uv pip install --python "$VENV/bin/python" "vllm==$VLLM_VERSION"
  run uv pip install --python "$VENV/bin/python" "huggingface_hub[hf_transfer]"
else
  run "$VENV/bin/python" -m pip install "vllm==$VLLM_VERSION" "huggingface_hub[hf_transfer]"
fi
ok "vLLM installed"

# ------------------------------- 3. prebuilt kernels -------------------------
step "3/6 flashinfer-jit-cache (prebuilt kernels / 预编译内核)"

FI_VER="$FLASHINFER_VER"
if [ -z "$FI_VER" ] && [ -x "$VENV/bin/python" ] && [ "$DRY_RUN" = "0" ]; then
  FI_VER=$("$VENV/bin/python" -c 'import flashinfer;print(flashinfer.__version__)' 2>/dev/null || true)
fi
if [ -z "$FI_VER" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    FI_VER="0.6.18"
    info "[dry-run] flashinfer not installed yet; assuming $FI_VER (override with FLASHINFER_VER=...)"
  else
    die "cannot detect flashinfer version — set FLASHINFER_VER=x.y.z"
  fi
fi
[ -n "$FI_VER" ] || die "cannot detect flashinfer version — set FLASHINFER_VER=x.y.z"
info "flashinfer version: $FI_VER  (cuda tag: $CUDA_TAG)"

WHEEL="flashinfer_jit_cache-${FI_VER}+${CUDA_TAG}-cp39-abi3-manylinux_2_28_x86_64.whl"
GH_BASE="https://github.com/flashinfer-ai/flashinfer/releases/download/v${FI_VER}/${WHEEL}"
PROXY_BASE="https://gh-proxy.com/${GH_BASE}"
TMP_WHEEL="/tmp/${WHEEL}"

# Prefer the mirror when GitHub is slow (typical in CN); probe both quickly.
pick_url() {
  local u
  for u in "$PROXY_BASE" "$GH_BASE"; do
    if curl -sIL -m 15 -o /dev/null "$u" 2>/dev/null; then echo "$u"; return 0; fi
  done
  return 1
}

if [ "$DRY_RUN" = "0" ] && "$VENV/bin/python" -c 'import flashinfer_jit_cache' >/dev/null 2>&1; then
  ok "flashinfer-jit-cache already installed — skipping"
else
  URL=$(pick_url || true)
  if [ -z "$URL" ]; then
    warn "prebuilt wheel not reachable for ${FI_VER}+${CUDA_TAG}; startup will JIT-compile (5-10 min once)"
  else
    info "downloading $WHEEL"
    run curl -L --retry 3 -o "$TMP_WHEEL" "$URL"
    if need uv; then
      run uv pip install --python "$VENV/bin/python" --no-deps "$TMP_WHEEL"
    else
      run "$VENV/bin/python" -m pip install --no-deps "$TMP_WHEEL"
    fi
    MODS=$("$VENV/bin/python" -c 'import flashinfer_jit_cache as m,pathlib;print(len(list(pathlib.Path(m.get_jit_cache_dir()).iterdir())))' 2>/dev/null || echo "?")
    ok "prebuilt kernels installed ($MODS modules) — no JIT at startup"
  fi
fi

# ------------------------------- 4. models -----------------------------------
step "4/6 models / 模型权重"

info "repo : $HF_REPO"
info "into : $MODEL_DIR"

if [ "$DRY_RUN" = "0" ] && [ -f "$MODEL_DIR/model-nvfp4-fast.safetensors" ] && [ -f "$DRAFT_DIR/model.safetensors" ]; then
  ok "weights already present ($(du -sh "$MODEL_DIR" | cut -f1))"
else
  mkdir -p "$MODEL_ROOT"
  export HF_ENDPOINT="$HF_MIRROR"
  export HF_HUB_ENABLE_HF_TRANSFER=1
  DL="$VENV/bin/hf"
  [ -x "$DL" ] || DL="$VENV/bin/huggingface-cli"
  run "$DL" download "$HF_REPO" --include "${MODEL_SUBDIR}/*" --local-dir "$MODEL_ROOT"
  [ "$DRY_RUN" = "1" ] || ok "weights downloaded: $(du -sh "$MODEL_DIR" | cut -f1)"
fi

# ------------------------------- 5. scripts ----------------------------------
step "5/6 scripts + service / 脚本与托管"

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run install -m 0755 "$SRC_DIR/scripts/modelctl" "$INSTALL_DIR/modelctl"
run install -m 0755 "$SRC_DIR/scripts/vllm-merkyor-dflash-launch.sh" "$INSTALL_DIR/vllm-merkyor-dflash-launch.sh"
run install -m 0644 "$SRC_DIR/scripts/bench_gsq.py" "$INSTALL_DIR/bench_gsq.py"
run install -m 0644 "$SRC_DIR/scripts/bench_concurrency.py" "$INSTALL_DIR/bench_concurrency.py"
ok "scripts installed into $INSTALL_DIR"

# point the launch script at this machine's paths
if [ "$DRY_RUN" = "0" ]; then
  sed -i \
    -e "s|^MODEL=.*|MODEL=\"\${MODEL:-$MODEL_DIR}\"|" \
    -e "s|^DRAFT=.*|DRAFT=\"\${DRAFT:-$DRAFT_DIR}\"|" \
    -e "s|^SERVED_MODEL=.*|SERVED_MODEL=\"\${SERVED_MODEL:-$SERVED_NAME}\"|" \
    "$INSTALL_DIR/vllm-merkyor-dflash-launch.sh"
  ok "launch script paths rewritten"
fi

if [ "$USE_SYSTEMD" = "1" ]; then
  UNIT=/etc/systemd/system/modelctl.service
  if [ "$DRY_RUN" = "1" ]; then
    info "[dry-run] would install $UNIT and enable it"
  elif sudo -n true 2>/dev/null; then
    sed -e "s|^User=.*|User=$(id -un)|" \
        -e "s|^Group=.*|Group=$(id -gn)|" \
        -e "s|^Environment=USER=.*|Environment=USER=$(id -un)|" \
        -e "s|/home/USER|$HOME|g" \
        "$SRC_DIR/systemd/modelctl.service" | sudo tee "$UNIT" >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable modelctl
    ok "systemd unit installed + enabled (survives reboot, auto-restarts on crash)"
  else
    warn "sudo needs a password — install the unit manually:"
    info "  sed -e 's|/home/USER|$HOME|g' $SRC_DIR/systemd/modelctl.service | sudo tee $UNIT"
    info "  sudo systemctl daemon-reload && sudo systemctl enable --now modelctl"
  fi
fi

# ------------------------------- 6. verify -----------------------------------
step "6/6 start + verify / 启动与验证"

if [ "$DRY_RUN" = "1" ]; then
  info "[dry-run] would start the server and wait for health"
  echo; ok "dry run complete — nothing was changed"
  exit 0
fi

if [ "$USE_SYSTEMD" = "1" ] && sudo -n true 2>/dev/null; then
  sudo systemctl restart modelctl
else
  setsid nohup bash "$INSTALL_DIR/modelctl" supervise vllm-dflash \
    >> "$LOG_DIR/supervisor.log" 2>&1 < /dev/null &
fi

info "waiting for health (60-200 s is normal: cudagraph capture, not compilation) ..."
T0=$(date +%s)
for _ in $(seq 1 60); do
  if curl -s -m 3 -o /dev/null "http://127.0.0.1:${PORT}/health"; then break; fi
  sleep 5
done

if ! curl -s -m 3 -o /dev/null "http://127.0.0.1:${PORT}/health"; then
  echo
  warn "server did not come up. Diagnostics:"
  info "  bash $INSTALL_DIR/modelctl logs 80"
  info "  ps -eo pid,etime,pcpu,args --sort=-pcpu | grep -E '[n]vcc|[c]icc'   # still compiling?"
  die "startup failed"
fi

ok "up in $(( $(date +%s) - T0 ))s"
echo
bash "$INSTALL_DIR/modelctl" status

echo
info "smoke test / 冒烟测试:"
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"reply with the single word: ready\"}],\"max_tokens\":8,\"temperature\":0}" \
  | "$VENV/bin/python" -c 'import json,sys; d=json.load(sys.stdin); print("   ->", d["choices"][0]["message"]["content"].strip())' \
  || warn "smoke test failed (see logs)"

echo
echo "${BOLD}${GREEN}done.${OFF} endpoint: http://0.0.0.0:${PORT}/v1  model: ${SERVED_NAME}"
echo "  status : bash $INSTALL_DIR/modelctl status"
echo "  logs   : bash $INSTALL_DIR/modelctl logs 80"
echo "  bench  : bash $SRC_DIR/bench.sh"
echo "  restart: sudo systemctl restart modelctl"
