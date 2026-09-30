#!/usr/bin/env bash
# Thapar CFD Platform - one-shot launcher
#
# Builds (unless --skip-build) and starts the Spring Boot server (backend +
# server-rendered frontend, both served from the one process on :8090), then
# opens a Cloudflare Quick Tunnel to it and prints the public URL you can
# click. Safe to re-run: stops any previously launched instance first.
#
# Usage:
#   ./run_platform.sh              # build + launch + tunnel
#   ./run_platform.sh --skip-build # reuse the existing jar
#   ./run_platform.sh --no-tunnel  # backend/frontend only, no public URL
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$ROOT_DIR/server"
CUDA_LIB_DIR="$ROOT_DIR/src/cuda/build"
JAR_PATH="$SERVER_DIR/target/flowstudio-server-1.0.0-SNAPSHOT.jar"
PORT=8090

LOG_DIR="$ROOT_DIR/.run_logs"
mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"
TUNNEL_LOG="$LOG_DIR/cloudflared.log"

SKIP_BUILD=0
NO_TUNNEL=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --no-tunnel) NO_TUNNEL=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 1 ;;
    esac
done

echo "== Thapar CFD Platform launcher =="

# --- 1. Stop any previous instance of this app / tunnel we started -------
echo "[1/5] Stopping any previous instance..."
pkill -f "flowstudio-server-1.0.0-SNAPSHOT.jar" 2>/dev/null && sleep 2 || true
if [[ -f "$LOG_DIR/cloudflared.pid" ]]; then
    OLD_TUNNEL_PID="$(cat "$LOG_DIR/cloudflared.pid" 2>/dev/null || true)"
    if [[ -n "${OLD_TUNNEL_PID:-}" ]] && kill -0 "$OLD_TUNNEL_PID" 2>/dev/null; then
        kill "$OLD_TUNNEL_PID" 2>/dev/null || true
    fi
fi

# --- 2. Build (unless skipped) -------------------------------------------
if [[ "$SKIP_BUILD" -eq 0 ]]; then
    echo "[2/5] Building server jar (mvn package -DskipTests)..."
    if ! (cd "$SERVER_DIR" && mvn -q -o package -DskipTests 2>/dev/null); then
        if [[ -f "$JAR_PATH" ]]; then
            echo "      (offline build skipped; reusing pre-built jar at $JAR_PATH)"
        else
            echo "      (fetching dependencies and building jar...)"
            (cd "$SERVER_DIR" && mvn -q package -DskipTests)
        fi
    fi
else
    echo "[2/5] Skipping build (--skip-build)."
fi

if [[ ! -f "$JAR_PATH" ]]; then
    echo "ERROR: jar not found at $JAR_PATH -- run without --skip-build first." >&2
    exit 1
fi

# --- 3. Launch backend + frontend (one Spring Boot process) --------------
echo "[3/5] Starting server on port $PORT..."
: > "$SERVER_LOG"
setsid env LD_LIBRARY_PATH="$CUDA_LIB_DIR" \
    java --enable-preview --enable-native-access=ALL-UNNAMED -jar "$JAR_PATH" \
    > "$SERVER_LOG" 2>&1 < /dev/null &
SERVER_PID=$!
disown
echo "$SERVER_PID" > "$LOG_DIR/server.pid"

echo -n "      waiting for server to come up"
for i in $(seq 1 60); do
    if curl -s -o /dev/null --max-time 2 "http://localhost:$PORT/"; then
        echo " -> up (PID $SERVER_PID)"
        break
    fi
    echo -n "."
    sleep 1
    if [[ "$i" -eq 60 ]]; then
        echo ""
        echo "ERROR: server did not come up within 60s. Check $SERVER_LOG" >&2
        exit 1
    fi
done

# --- 4. Cloudflare Quick Tunnel -------------------------------------------
if [[ "$NO_TUNNEL" -eq 1 ]]; then
    echo "[4/5] Skipping tunnel (--no-tunnel)."
    echo "[5/5] Local URL: http://localhost:$PORT/"
    exit 0
fi

echo "[4/5] Opening Cloudflare quick tunnel to localhost:$PORT..."
: > "$TUNNEL_LOG"
setsid cloudflared tunnel --url "http://localhost:$PORT" \
    > "$TUNNEL_LOG" 2>&1 < /dev/null &
TUNNEL_PID=$!
disown
echo "$TUNNEL_PID" > "$LOG_DIR/cloudflared.pid"

echo -n "      waiting for tunnel URL"
TUNNEL_URL=""
for i in $(seq 1 30); do
    TUNNEL_URL="$(grep -a -oE 'https://[a-zA-Z0-9.-]+\.trycloudflare\.com' "$TUNNEL_LOG" | head -1 || true)"
    if [[ -n "$TUNNEL_URL" ]]; then
        echo " -> ready"
        break
    fi
    echo -n "."
    sleep 1
done

echo "[5/5] Done."
echo ""
echo "=================================================================="
echo " Local:  http://localhost:$PORT/"
if [[ -n "$TUNNEL_URL" ]]; then
    echo " Public: $TUNNEL_URL   <-- click this"
else
    echo " Public: tunnel URL not detected yet -- check $TUNNEL_LOG"
fi
echo "=================================================================="
echo ""
echo "Server log:    $SERVER_LOG"
echo "Tunnel log:    $TUNNEL_LOG"
echo "Stop with:     pkill -f flowstudio-server-1.0.0-SNAPSHOT.jar; kill \$(cat $LOG_DIR/cloudflared.pid)"
