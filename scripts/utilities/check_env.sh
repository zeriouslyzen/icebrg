#!/bin/bash
# ICEBURG Environment Setup Verification

echo "=========================================="
echo "ICEBURG Environment Setup Check"
echo "=========================================="

# Check .env file
if [ -f .env ]; then
    echo "✅ .env file exists"
    source .env 2>/dev/null
    echo "   ENVIRONMENT=${ENVIRONMENT:-NOT SET}"
    echo "   ICEBURG_ENABLE_PRE_WARMED=${ICEBURG_ENABLE_PRE_WARMED:-NOT SET}"
    echo "   ICEBURG_ENABLE_MONITORING=${ICEBURG_ENABLE_MONITORING:-NOT SET}"
    echo "   ICEBURG_ENABLE_ALWAYS_ON=${ICEBURG_ENABLE_ALWAYS_ON:-NOT SET}"
else
    echo "❌ .env file not found"
fi

# Check virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment active"
    echo "   Run: source environments/venv2/bin/activate"
fi

# Check Python
echo "Python: $(python3 --version)"
echo "Python path: $(which python3)"

echo ""
echo "=========================================="
echo "Setup complete! Run this to verify:"
echo "  bash check_env.sh"
echo "=========================================="
