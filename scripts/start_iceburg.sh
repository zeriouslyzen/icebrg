#!/bin/bash

# ICEBURG Complete Startup Script
# Starts both API server and frontend

echo "🚀 Starting ICEBURG 2.0..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

# Check if Node.js is installed (for frontend)
if ! command -v node &> /dev/null; then
    echo "⚠️  Node.js is not installed. Frontend will not start."
    echo "   Install Node.js 18+ to run the frontend."
    SKIP_FRONTEND=true
else
    SKIP_FRONTEND=false
fi

# Navigate to project root
cd "$(dirname "$0")/.."

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Ollama is not running. Starting Ollama..."
    echo "   Please ensure Ollama is installed and running."
    echo "   Start it with: ollama serve"
fi

# Start API server
echo "🌐 Starting API server on http://localhost:8000..."
python3 -m uvicorn src.iceburg.api.server:app --host 0.0.0.0 --port 8000 --reload > logs/api_server.log 2>&1 &
API_PID=$!

# Wait for API server to start
echo "⏳ Waiting for API server to start..."
sleep 5

# Check if API server is running
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API server is running on http://localhost:8000"
else
    echo "❌ API server failed to start. Check logs/api_server.log"
    kill $API_PID 2>/dev/null
    exit 1
fi

# Start frontend if Node.js is available
if [ "$SKIP_FRONTEND" = false ]; then
    echo "🎨 Starting frontend on http://localhost:3000..."
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "📦 Installing frontend dependencies..."
        npm install
    fi
    
    # Start Vite dev server
    npm run dev > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    cd ..
    
    echo "✅ Frontend is starting on http://localhost:3000"
    echo "   (Check logs/frontend.log for details)"
else
    FRONTEND_PID=""
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Save PIDs for later cleanup
echo $API_PID > logs/api_server.pid
if [ -n "$FRONTEND_PID" ]; then
    echo $FRONTEND_PID > logs/frontend.pid
fi

echo ""
echo "✅ ICEBURG 2.0 is ready!"
echo ""
echo "📡 API Server: http://localhost:8000"
echo "   Health Check: http://localhost:8000/health"
echo "   API Docs: http://localhost:8000/docs"
if [ "$SKIP_FRONTEND" = false ]; then
    echo "🎨 Frontend: http://localhost:3000"
fi
echo ""
echo "🔌 WebSocket: ws://localhost:8000/ws"
echo ""
echo "To stop ICEBURG:"
echo "  ./scripts/stop_iceburg.sh"
echo ""
echo "Or manually:"
echo "  kill $API_PID"
if [ -n "$FRONTEND_PID" ]; then
    echo "  kill $FRONTEND_PID"
fi
echo ""
echo "📝 Logs:"
echo "  API Server: logs/api_server.log"
if [ "$SKIP_FRONTEND" = false ]; then
    echo "  Frontend: logs/frontend.log"
fi

