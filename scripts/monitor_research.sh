#!/bin/bash
# Safe monitoring script - won't disrupt background processes

echo "=== ICEBURG Research Test Monitor ==="
echo ""

# Check if results exist
if ls data/research_outputs/*.md 1> /dev/null 2>&1; then
    echo "✅ Results found:"
    ls -lht data/research_outputs/*.md | head -5 | awk '{print "  " $9 " (" $5 ")"}'
    echo ""
    echo "Latest result:"
    latest=$(ls -t data/research_outputs/*.md 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "  File: $latest"
        echo "  Size: $(wc -c < "$latest" 2>/dev/null | awk '{print $1 " bytes"}')"
        echo "  Preview (first 200 chars):"
        head -c 200 "$latest" 2>/dev/null | sed 's/^/    /'
        echo "..."
    fi
else
    echo "⏳ No results yet (queries take 3-7 minutes each)"
fi

echo ""
echo "Process status:"
if ps aux | grep -E "python3.*breakthrough|ollama" | grep -v grep > /dev/null; then
    echo "  ✅ Background process is running"
else
    echo "  ⚠️  Background process not found (may have completed)"
fi

echo ""
echo "To see full latest result:"
echo "  tail -50 \$(ls -t data/research_outputs/*.md | head -1)"
