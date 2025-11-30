#!/bin/bash

# ICEBURG Frontend Startup Script

echo "🚀 Starting ICEBURG Frontend..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

# Navigate to frontend directory
cd "$(dirname "$0")"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "✅ Created .env file. Please configure it if needed."
    else
        echo "⚠️  .env.example not found. Using defaults."
    fi
fi

# Check if backend is running (optional check)
echo "🔍 Checking backend connection..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ Backend is running on port 8000"
else
    echo "⚠️  Backend not detected on port 8000. Make sure the API server is running."
    echo "   Start it with: cd ../.. && python -m uvicorn src.iceburg.api.server:app --host 0.0.0.0 --port 8000 --reload"
fi

# Start Vite dev server
echo "🌐 Starting Vite dev server on http://localhost:3000..."
npm run dev

