#!/bin/bash
# Start ICEBURG API Server

cd "$(dirname "$0")/.."

echo "Starting ICEBURG 2.0 API Server..."
echo "===================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the API server
python3 -m src.iceburg.api.run_server

