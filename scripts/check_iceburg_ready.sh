#!/bin/bash

# ICEBURG Readiness Check Script
# Verifies that all components are ready for use

echo "🔍 Checking ICEBURG 2.0 Readiness..."
echo ""

cd "$(dirname "$0")/.."

# Check API Server
echo "📡 API Server:"
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✅ Running on http://localhost:8000"
    HEALTH=$(curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null | grep -o '"status": "[^"]*"' | head -1)
    echo "  $HEALTH"
    API_READY=true
else
    echo "  ❌ Not running"
    echo "  Start with: ./scripts/start_iceburg.sh"
    API_READY=false
fi

# Check Frontend
echo ""
echo "🎨 Frontend:"
if curl -s http://localhost:3000 > /dev/null 2>&1; then
    echo "  ✅ Running on http://localhost:3000"
    FRONTEND_READY=true
else
    echo "  ⚠️  Not running (optional)"
    echo "  Start with: cd frontend && npm run dev"
    FRONTEND_READY=false
fi

# Check WebSocket
echo ""
echo "🔌 WebSocket:"
if [ "$API_READY" = true ]; then
    echo "  ✅ WebSocket endpoint available at ws://localhost:8000/ws"
    WS_READY=true
else
    echo "  ❌ WebSocket not available (API server not running)"
    WS_READY=false
fi

# Check Ollama
echo ""
echo "🤖 Ollama:"
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✅ Running on http://localhost:11434"
    OLLAMA_READY=true
else
    echo "  ⚠️  Not running (required for LLM functionality)"
    echo "  Start with: ollama serve"
    OLLAMA_READY=false
fi

# Check Models
echo ""
echo "📦 Ollama Models:"
if [ "$OLLAMA_READY" = true ]; then
    MODELS=$(curl -s http://localhost:11434/api/tags 2>/dev/null | python3 -c "import sys, json; data = json.load(sys.stdin); print('\n'.join([m['name'] for m in data.get('models', [])]))" 2>/dev/null | head -5)
    if [ -n "$MODELS" ]; then
        echo "$MODELS" | while read model; do
            echo "  ✅ $model"
        done
    else
        echo "  ⚠️  No models found"
        echo "  Install models with: ollama pull llama3.1:8b"
    fi
else
    echo "  ⚠️  Cannot check (Ollama not running)"
fi


if [ "$API_READY" = true ] && [ "$OLLAMA_READY" = true ]; then
    echo "✅ ICEBURG 2.0 is ready for prompting via UX!"
    echo ""
    echo "📡 API Server: http://localhost:8000"
    echo "🎨 Frontend: http://localhost:3000"
    echo "🔌 WebSocket: ws://localhost:8000/ws"
    echo ""
    echo "You can now:"
    echo "  - Open http://localhost:3000 in your browser"
    echo "  - Start prompting ICEBURG"
    echo "  - Select modes (chat, fast, research, etc.)"
    echo "  - Select agents (surveyor, dissident, synthesist, etc.)"
    echo "  - Use degradation mode for agent communication"
else
    echo "⚠️  Some components are not ready"
    echo ""
    if [ "$API_READY" = false ]; then
        echo "❌ API Server is required"
        echo "   Start with: ./scripts/start_iceburg.sh"
    fi
    if [ "$OLLAMA_READY" = false ]; then
        echo "⚠️  Ollama is required for LLM functionality"
        echo "   Start with: ollama serve"
    fi
fi
