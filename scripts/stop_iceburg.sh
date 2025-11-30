#!/bin/bash

# ICEBURG Stop Script
# Stops both API server and frontend

echo "🛑 Stopping ICEBURG 2.0..."

cd "$(dirname "$0")/.."

# Stop API server
if [ -f "logs/api_server.pid" ]; then
    API_PID=$(cat logs/api_server.pid)
    if ps -p $API_PID > /dev/null 2>&1; then
        echo "🛑 Stopping API server (PID: $API_PID)..."
        kill $API_PID 2>/dev/null
        rm logs/api_server.pid
        echo "✅ API server stopped"
    fi
fi

# Stop frontend
if [ -f "logs/frontend.pid" ]; then
    FRONTEND_PID=$(cat logs/frontend.pid)
    if ps -p $FRONTEND_PID > /dev/null 2>&1; then
        echo "🛑 Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null
        rm logs/frontend.pid
        echo "✅ Frontend stopped"
    fi
fi

# Kill any remaining processes
pkill -f "uvicorn.*api.server" 2>/dev/null
pkill -f "vite" 2>/dev/null

echo "✅ ICEBURG 2.0 stopped"

