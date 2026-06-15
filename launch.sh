#!/bin/bash

# Configuration Variables
PORT=6060
MODEL="mlx-community/Qwen3-14B-4bit"
PYTHON_BIN="lcenv/bin/python"
SERVER_URL="http://localhost:$PORT/v1/models"
APP_SCRIPT="app.py"
APP_ARGS="--web-port 9026 -H localhost -p 7900" # Change this to agent.py if needed

# FIX: Force Metal to recycle and combine buffer handles before hitting the 499k ceiling
export MLX_METAL_RECYCLE_FLUSH_THRESHOLD=1000

# Trap handler for graceful shutdown
cleanup() {
    echo -e "\n🛑 Catching exit signal... Shutting down background model processes..."
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null
    fi
    exit 0
}
trap cleanup SIGINT SIGTERM

echo "🚀 Starting Native Apple Silicon Stack..."

# 1. Spin up the mlx-lm server in the background
$PYTHON_BIN -m mlx_lm server --model "$MODEL" --port "$PORT" --log-level INFO --prefill-step-size 1024  > mlx_server.log 2>&1 &

SERVER_PID=$!

echo "⏳ Waiting for $MODEL to initialize on port $PORT..."
echo "📊 (You can tail logs via: tail -f mlx_server.log)"

# 2. Health check loop: wait for the server endpoint to become active
while ! curl -s "$SERVER_URL" > /dev/null; do
    # Check if the process died unexpectedly mid-boot
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Error: Server failed to start. Check mlx_server.log for details."
        exit 1
    fi
    sleep 2
done

echo "✅ Server is online and responsive!"

# 3. Launch your primary application code
echo "⚙️ Executing application pipeline: $APP_SCRIPT..."
$PYTHON_BIN "$APP_SCRIPT" $APP_ARGS

# 4. Clean up background services automatically upon application exit
echo "🛑 Shutting down background model processes..."
kill $SERVER_PID

