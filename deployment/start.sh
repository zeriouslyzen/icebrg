#!/bin/bash

# Elite Financial AI Oracle - Production Startup Script
# This script starts all services in the correct order

set -e

echo "🚀 Starting Elite Financial AI Oracle..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose > /dev/null 2>&1; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/{models,cache,logs}
mkdir -p logs
mkdir -p models

# Set permissions
echo "🔐 Setting permissions..."
chmod -R 755 data/
chmod -R 755 logs/
chmod -R 755 models/

# Pull latest images
echo "📥 Pulling latest images..."
docker-compose pull

# Build custom images
echo "🔨 Building custom images..."
docker-compose build

# Start services in order
echo "🔄 Starting services..."

# Start Redis first
echo "  - Starting Redis..."
docker-compose up -d redis
sleep 5

# Start Ollama
echo "  - Starting Ollama..."
docker-compose up -d ollama
sleep 10

# Start Prometheus
echo "  - Starting Prometheus..."
docker-compose up -d prometheus
sleep 5

# Start ICEBURG Core
echo "  - Starting ICEBURG Core..."
docker-compose up -d iceburg
sleep 10

# Start Monitoring
echo "  - Starting Monitoring Dashboard..."
docker-compose up -d monitoring
sleep 5

# Start Jupyter
echo "  - Starting Jupyter Notebook..."
docker-compose up -d jupyter
sleep 5

# Check service health
echo "🏥 Checking service health..."

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo "  ✅ Redis is healthy"
else
    echo "  ❌ Redis is not responding"
fi

# Check Ollama
if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "  ✅ Ollama is healthy"
else
    echo "  ❌ Ollama is not responding"
fi

# Check ICEBURG Core
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✅ ICEBURG Core is healthy"
else
    echo "  ❌ ICEBURG Core is not responding"
fi

# Check Monitoring
if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "  ✅ Monitoring Dashboard is healthy"
else
    echo "  ❌ Monitoring Dashboard is not responding"
fi

# Check Jupyter
if curl -f http://localhost:8888 > /dev/null 2>&1; then
    echo "  ✅ Jupyter Notebook is healthy"
else
    echo "  ❌ Jupyter Notebook is not responding"
fi

echo ""
echo "🎉 Elite Financial AI Oracle is now running!"
echo ""
echo "📊 Service URLs:"
echo "  - ICEBURG Core: http://localhost:8000"
echo "  - Monitoring Dashboard: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Jupyter Notebook: http://localhost:8888 (token: iceburg)"
echo "  - Ollama API: http://localhost:11434"
echo ""
echo "📝 Useful Commands:"
echo "  - View logs: docker-compose logs -f [service_name]"
echo "  - Stop services: docker-compose down"
echo "  - Restart service: docker-compose restart [service_name]"
echo "  - Scale service: docker-compose up -d --scale [service_name]=N"
echo ""
echo "🔧 Configuration:"
echo "  - Environment: .env"
echo "  - Docker Compose: docker-compose.yml"
echo "  - ICEBURG Config: config/icberg_maximum_capability.yaml"
echo ""
echo "📚 Documentation:"
echo "  - User Guide: docs/USER_GUIDE.md"
echo "  - API Reference: docs/API_REFERENCE.md"
echo "  - Tutorials: tutorials/"
echo ""
echo "🚀 Ready to start trading with quantum-enhanced AI!"
