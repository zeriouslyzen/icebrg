#!/bin/bash
# Monitor lab test progress

echo "=== ICEBURG Lab Test Monitor ==="
echo ""

if ls data/lab_tests/*.md 1>/dev/null 2>&1; then
    echo "✅ Lab test results found:"
    ls -lht data/lab_tests/*.md | head -5 | while read line; do
        filename=$(echo "$line" | awk '{print $NF}')
        size=$(echo "$line" | awk '{print $5}')
        time=$(echo "$line" | awk '{print $6, $7, $8}')
        echo "  $filename ($size, $time)"
    done
    echo ""
    
    count=$(ls data/lab_tests/*.md 2>/dev/null | wc -l | tr -d ' ')
    echo "Progress: $count / 3 tests completed"
    echo ""
    
    if [ $count -ge 1 ]; then
        latest=$(ls -t data/lab_tests/*.md | head -1)
        echo "Latest result: $(basename $latest)"
        echo ""
        echo "Summary (first 300 chars):"
        head -50 "$latest" | head -20
        echo "..."
    fi
else
    echo "⏳ No results yet (tests take 5-10 minutes each)"
fi

echo ""
echo "Process status:"
if ps aux | grep -E "test_lab_findings|python3.*lab" | grep -v grep > /dev/null 2>&1; then
    echo "  ✅ Lab testing process is running"
else
    echo "  ⚠️  Lab testing process not found (may have completed)"
fi

echo ""
echo "To view full latest result:"
echo "  cat \$(ls -t data/lab_tests/*.md | head -1)"
