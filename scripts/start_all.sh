#!/bin/bash
# Start ICEBURG 2.0 - API Server and Frontend

cd "$(dirname "$0")/.."

echo "Starting ICEBURG 2.0..."
echo "======================"

# Start API server in background
echo "Starting API server..."
python3 -m src.iceburg.api.run_server > /tmp/iceburg_api.log 2>&1 &
API_PID=$!
echo "API server started (PID: $API_PID)"

# Wait for API server to start
sleep 3

# Start frontend dev server
echo "Starting frontend..."
cd frontend
npm run dev > /tmp/iceburg_frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"

echo ""
echo "✅ ICEBURG 2.0 is running!"
echo "🌐 Frontend: http://localhost:3000"
echo "🔗 API: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Wait for user interrupt
trap "kill $API_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait

