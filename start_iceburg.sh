#!/bin/bash
# One-click startup for ICEBURG
# Ensures correct python path and module execution

# Kill any existing server on port 8000
lsof -t -i:8000 | xargs kill -9 2>/dev/null || true

# Set python path to src directory
export PYTHONPATH=src

# Run the server module
echo "🧊 Starting ICEBURG..."
python3 -m iceburg.api.run_server
