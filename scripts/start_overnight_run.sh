#!/bin/bash
# Start Overnight Unbounded Research Run
# Usage: ./start_overnight_run.sh [query]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_FILE="$PROJECT_DIR/data/logs/overnight_run_$(date +%Y%m%d_%H%M%S).log"
RUN_SCRIPT="$SCRIPT_DIR/overnight_unbounded_run.py"

cd "$PROJECT_DIR" || exit 1

# Create logs directory
mkdir -p "$PROJECT_DIR/data/logs"
mkdir -p "$PROJECT_DIR/data/overnight_runs"

# Set query if provided
if [ -n "$1" ]; then
    export ICEBURG_OVERNIGHT_QUERY="$1"
fi

# Set environment for optimal overnight run
export PYTHONPATH="$PROJECT_DIR/src"
export ICEBURG_SURVEYOR_MODEL="${ICEBURG_SURVEYOR_MODEL:-qwen2.5:7b}"
export ICEBURG_DISSIDENT_MODEL="${ICEBURG_DISSIDENT_MODEL:-qwen2.5:7b}"
export ICEBURG_FAIL_FAST=0
export ICEBURG_DISABLE_HARDWARE_OPT=1

echo "======================================================================"
echo "OVERNIGHT UNBOUNDED RESEARCH RUN"
echo "======================================================================"
echo "Query: ${ICEBURG_OVERNIGHT_QUERY:-default}"
echo "Log file: $LOG_FILE"
echo "Started: $(date)"
echo "======================================================================"
echo ""

# Run in background with nohup
nohup python3 "$RUN_SCRIPT" > "$LOG_FILE" 2>&1 &
PID=$!

# Save PID
echo $PID > "$PROJECT_DIR/data/overnight_run.pid"

echo "✅ Overnight run started (PID: $PID)"
echo ""
echo "To monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "To check status:"
echo "  ps -p $PID"
echo ""
echo "To stop:"
echo "  kill $PID"
echo ""

# Show initial log output
sleep 2
if [ -f "$LOG_FILE" ]; then
    echo "Initial output:"
    tail -20 "$LOG_FILE"
fi

