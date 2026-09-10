#!/usr/bin/env bash
# dashboard.sh start|stop|status — the modelwatch web UI as a managed background process.
#
#   bash ~/deploy-5060ti/dashboard.sh start [port]
#   bash ~/deploy-5060ti/dashboard.sh stop
#   bash ~/deploy-5060ti/dashboard.sh status
set -uo pipefail

D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${2:-8090}"
URL="${DASH_URL:-http://127.0.0.1:8080}"
PIDFILE="$HOME/.modelwatch.pid"
LOG="$HOME/modelwatch-dashboard.log"
PATTERN='[m]odelwatch.py --serve'          # bracketed so the pattern cannot match this script

is_up() { curl -s -m 3 -o /dev/null "http://127.0.0.1:${PORT}/" 2>/dev/null; }

case "${1:-status}" in
  start)
    if is_up; then echo "already running: http://$(hostname -I | awk '{print $1}'):${PORT}/"; exit 0; fi
    setsid nohup python3 "$D/modelwatch.py" --serve --bind 0.0.0.0 --port "$PORT" \
      --url "$URL" --interval 2 > "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 20); do
      if is_up; then
        IP=$(hostname -I | awk '{print $1}')
        echo "dashboard up: http://${IP}:${PORT}/   (pid $(cat "$PIDFILE"), log $LOG)"
        exit 0
      fi
      sleep 0.5
    done
    echo "failed to start — last log lines:"; tail -5 "$LOG"; exit 1
    ;;
  stop)
    pkill -f "$PATTERN" 2>/dev/null && echo "stopped" || echo "not running"
    rm -f "$PIDFILE"
    ;;
  status)
    if is_up; then
      IP=$(hostname -I | awk '{print $1}')
      echo "running: http://${IP}:${PORT}/  (pid $(pgrep -f "$PATTERN" | head -1), log $LOG)"
    else
      echo "not running (port $PORT)"
    fi
    ;;
  *)
    sed -n '2,8p' "$0"
    ;;
esac
