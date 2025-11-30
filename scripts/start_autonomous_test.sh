#!/bin/bash
# Start autonomous test in background with monitoring

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$PROJECT_DIR/data/autonomous_test.pid"
LOG_FILE="$PROJECT_DIR/data/logs/autonomous_stress_test.log"
TEST_SCRIPT="$SCRIPT_DIR/autonomous_stress_test.py"

cd "$PROJECT_DIR" || exit 1

# Check if already running
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null 2>&1; then
        echo "⚠️  Autonomous test is already running (PID: $PID)"
        echo "To stop it, run: kill $PID"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

# Create directories
mkdir -p "$PROJECT_DIR/data/logs"
mkdir -p "$PROJECT_DIR/data/autonomous_test"

# Start test in background
echo "Starting ICEBURG Autonomous Stress Test..."
echo "Log file: $LOG_FILE"
echo "PID file: $PID_FILE"
echo ""

nohup python3 "$TEST_SCRIPT" > "$LOG_FILE" 2>&1 &
TEST_PID=$!

# Save PID
echo $TEST_PID > "$PID_FILE"

echo "✅ Test started in background (PID: $TEST_PID)"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check status:"
echo "  $SCRIPT_DIR/monitor_autonomous_test.sh"
echo ""
echo "To stop:"
echo "  kill $TEST_PID"
echo ""

# Wait a moment to ensure it started
sleep 2

# Check if still running
if ps -p "$TEST_PID" > /dev/null 2>&1; then
    echo "✅ Test is running successfully"
    echo ""
    echo "Recent log output:"
    tail -n 10 "$LOG_FILE" 2>/dev/null || echo "  (Log file not created yet)"
else
    echo "❌ Test failed to start!"
    echo "Check the log file for errors: $LOG_FILE"
    rm -f "$PID_FILE"
    exit 1
fi

