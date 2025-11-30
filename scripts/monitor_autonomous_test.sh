#!/bin/bash
# Monitor script for autonomous test - checks frequently to ensure it's not stuck

TEST_PID_FILE="data/autonomous_test.pid"
LOG_FILE="data/logs/autonomous_stress_test.log"
CHECK_INTERVAL=30  # Check every 30 seconds
STUCK_THRESHOLD=300  # 5 minutes without activity

# Function to check if process is running
check_process() {
    if [ -f "$TEST_PID_FILE" ]; then
        PID=$(cat "$TEST_PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        else
            return 1
        fi
    else
        return 1
    fi
}

# Function to check log activity
check_log_activity() {
    if [ -f "$LOG_FILE" ]; then
        # Check if log file was modified in last STUCK_THRESHOLD seconds
        LAST_MOD=$(stat -f "%m" "$LOG_FILE" 2>/dev/null || stat -c "%Y" "$LOG_FILE" 2>/dev/null)
        CURRENT_TIME=$(date +%s)
        ELAPSED=$((CURRENT_TIME - LAST_MOD))
        
        if [ $ELAPSED -gt $STUCK_THRESHOLD ]; then
            return 1  # Stuck
        else
            return 0  # Active
        fi
    else
        return 1  # No log file
    fi
}

# Function to show status
show_status() {
    echo "=========================================="
    echo "ICEBURG Autonomous Test Monitor"
    echo "=========================================="
    echo "Time: $(date)"
    echo "PID File: $TEST_PID_FILE"
    echo "Log File: $LOG_FILE"
    
    if check_process; then
        PID=$(cat "$TEST_PID_FILE")
        echo "Status: ✅ RUNNING (PID: $PID)"
        
        # Show recent log entries
        echo ""
        echo "Recent Activity:"
        tail -n 5 "$LOG_FILE" 2>/dev/null || echo "  (No log entries yet)"
        
        if check_log_activity; then
            echo ""
            echo "Activity: ✅ ACTIVE (log updated recently)"
        else
            echo ""
            echo "Activity: ⚠️  STUCK (no log activity for >${STUCK_THRESHOLD}s)"
            echo "Warning: Process may be stuck!"
        fi
        
        # Show resource usage
        echo ""
        echo "Resource Usage:"
        ps -p "$PID" -o pid,pcpu,pmem,etime,command 2>/dev/null || echo "  (Unable to get process info)"
    else
        echo "Status: ❌ NOT RUNNING"
        echo ""
        echo "The autonomous test process is not running."
        echo "Check the log file for errors: $LOG_FILE"
    fi
    echo "=========================================="
}

# Main monitoring loop
echo "Starting ICEBURG Autonomous Test Monitor"
echo "Checking every $CHECK_INTERVAL seconds..."
echo ""

while true; do
    show_status
    
    # Check if process is stuck
    if check_process && ! check_log_activity; then
        echo ""
        echo "⚠️  WARNING: Process appears stuck!"
        echo "Attempting to restart..."
        
        # Kill stuck process
        PID=$(cat "$TEST_PID_FILE")
        kill "$PID" 2>/dev/null
        sleep 2
        kill -9 "$PID" 2>/dev/null
        
        # Remove PID file
        rm -f "$TEST_PID_FILE"
        
        echo "Stuck process terminated. Restart the test manually."
    fi
    
    sleep $CHECK_INTERVAL
done

