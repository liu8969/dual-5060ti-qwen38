#!/usr/bin/env bash
# dashboard.sh — DEPRECATED 2026-09-12: the GPU model dashboard is now its own project.
#
#     ~/gpu-model-dashboard/      dashboard.py (monitor) + dashboard.sh (control) + README.md
#
# This file is kept only as a forwarding shim for old muscle memory and scripts.
# Lifecycle goes through systemd: the panel is installed as
# `gpu-model-dashboard.service` with Restart=always/RestartSec=3, so a `stop`
# implemented as pkill would have the panel back three seconds later — the same
# trap documented for `modelctl stop` under systemd.
#
#   status | logs | healthz   -> forwarded to the new project's dashboard.sh
#   start | stop | restart    -> systemctl on the unit (falls back to the new
#                                script when no unit is installed)
#
#   bash ~/deploy-5060ti/dashboard.sh status
set -uo pipefail

NEW="${GPU_MODEL_DASHBOARD:-$HOME/gpu-model-dashboard/dashboard.sh}"
UNIT="gpu-model-dashboard.service"
UNIT_PATH="/etc/systemd/system/$UNIT"
verb="${1:-status}"

warn() {
  echo "note: ~/deploy-5060ti/dashboard.sh is a forwarding shim; the project now lives in ~/gpu-model-dashboard" >&2
}

if [ ! -x "$NEW" ]; then
  cat >&2 <<EOF
dashboard.sh has moved: $NEW not found.

The GPU model dashboard is its own project now. Put it at \$HOME/gpu-model-dashboard
(or point GPU_MODEL_DASHBOARD at wherever it lives), then re-run.
EOF
  exit 1
fi

case "$verb" in
  status|logs|healthz)
    warn
    exec "$NEW" "$@"
    ;;
  start|stop|restart)
    warn
    if command -v systemctl >/dev/null 2>&1 && [ -f "$UNIT_PATH" ]; then
      echo "-> sudo systemctl $verb $UNIT" >&2
      exec sudo systemctl "$verb" "$UNIT"
    fi
    echo "-> no systemd unit installed; falling back to $NEW $verb" >&2
    exec "$NEW" "$@"
    ;;
  *)
    warn
    exec "$NEW" "$@"
    ;;
esac
