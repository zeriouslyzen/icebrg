#!/bin/bash
# ICEBURG Health Check Script

echo "ICEBURG Health Check"
echo "===================="

# Check Python
echo "Checking Python..."
python3 --version || echo "ERROR: Python not found"

# Check dependencies
echo "Checking dependencies..."
python3 -c "import iceburg" || echo "ERROR: ICEBURG not installed"

# Check services
echo "Checking services..."
curl -s http://localhost:8000/health || echo "WARNING: API server not running"

# Check Redis (optional)
echo "Checking Redis..."
redis-cli ping 2>/dev/null || echo "INFO: Redis not available (optional)"

# Check data directories
echo "Checking data directories..."
[ -d "data/models" ] && echo "✓ data/models exists" || echo "✗ data/models missing"
[ -d "data/lab" ] && echo "✓ data/lab exists" || echo "✗ data/lab missing"
[ -d "data/tenants" ] && echo "✓ data/tenants exists" || echo "✗ data/tenants missing"

# Check logs
echo "Checking logs..."
[ -d "logs" ] && echo "✓ logs directory exists" || echo "✗ logs directory missing"

echo "Health check complete!"

