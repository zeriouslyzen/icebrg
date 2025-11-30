#!/bin/bash
# Continuous monitoring - refreshes every 5 seconds

while true; do
    clear
    echo "=== ICEBURG Research Test Monitor (Refreshing every 5 seconds) ==="
    echo "Press Ctrl+C to stop monitoring"
    echo ""
    
    # Check results
    if ls data/research_outputs/*.md 1> /dev/null 2>&1; then
        echo "✅ Results found:"
        ls -lht data/research_outputs/*.md 2>/dev/null | head -5 | while read line; do
            filename=$(echo "$line" | awk '{print $NF}')
            size=$(echo "$line" | awk '{print $5}')
            time=$(echo "$line" | awk '{print $6, $7, $8}')
            echo "  $filename ($size, $time)"
        done
        echo ""
        
        # Show progress
        count=$(ls data/research_outputs/*.md 2>/dev/null | wc -l | tr -d ' ')
        echo "Progress: $count / 3 queries completed"
    else
        echo "⏳ No results yet (queries take 3-7 minutes each)"
        echo "Progress: 0 / 3 queries completed"
    fi
    
    echo ""
    echo "Process status:"
    if ps aux | grep -E "python3.*breakthrough|ollama" | grep -v grep > /dev/null 2>&1; then
        echo "  ✅ Background process is running"
    else
        echo "  ⚠️  Background process not found (may have completed)"
    fi
    
    echo ""
    echo "Last updated: $(date '+%H:%M:%S')"
    echo "Next refresh in 5 seconds..."
    sleep 5
done
