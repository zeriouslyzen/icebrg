# Elite Financial AI Oracle - Deployment Guide

This guide provides comprehensive instructions for deploying the Elite Financial AI Oracle system in production environments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Production Deployment](#production-deployment)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)
7. [Scaling](#scaling)

## Prerequisites

### System Requirements

- **CPU**: 8+ cores (16+ recommended)
- **RAM**: 16GB+ (32GB+ recommended)
- **Storage**: 100GB+ SSD
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **OS**: Linux (Ubuntu 20.04+ recommended) or macOS

### Software Requirements

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- Git

### Hardware Requirements for Quantum Computing

- **Quantum Simulator**: 8GB+ RAM for quantum circuit simulation
- **GPU**: NVIDIA GPU with CUDA 11.0+ for quantum acceleration
- **CPU**: AVX2 support for optimized quantum operations

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd iceburg
```

### 2. Set Up Environment

```bash
# Copy environment configuration
cp icberg_optimized.env .env

# Edit configuration if needed
nano .env
```

### 3. Start Services

```bash
# Make scripts executable
chmod +x deployment/start.sh

# Start all services
./deployment/start.sh
```

### 4. Verify Installation

```bash
# Check service status
docker-compose ps

# Check logs
docker-compose logs -f iceburg

# Test API endpoints
curl http://localhost:8000/health
```

## Production Deployment

### 1. Environment Configuration

Create production environment file:

```bash
# production.env
ICEBURG_DATA_DIR=/opt/iceburg/data
ICEBURG_USER=iceburg
ICEBURG_HOME=/opt/iceburg
ICEBURG_ENV=production

# Security settings
ICEBURG_SECRET_KEY=your-secret-key-here
ICEBURG_DEBUG=false
ICEBURG_LOG_LEVEL=INFO

# Database settings
REDIS_URL=redis://redis:6379
POSTGRES_URL=postgresql://iceburg:password@postgres:5432/iceburg

# API settings
ICEBURG_API_KEY=your-api-key-here
ICEBURG_RATE_LIMIT=1000

# Quantum settings
ICEBURG_QUANTUM_BACKEND=qiskit.aer
ICEBURG_QUANTUM_DEVICE=qasm_simulator
ICEBURG_QUANTUM_SHOTS=1000

# RL settings
ICEBURG_RL_ALGORITHM=PPO
ICEBURG_RL_LEARNING_RATE=0.0003
ICEBURG_RL_TOTAL_TIMESTEPS=100000

# Financial settings
ICEBURG_FINANCIAL_PROVIDER=polygon
ICEBURG_FINANCIAL_API_KEY=your-polygon-api-key
ICEBURG_FINANCIAL_SYMBOLS=AAPL,MSFT,GOOGL,TSLA
```

### 2. Production Docker Compose

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # PostgreSQL for persistent storage
  postgres:
    image: postgres:15-alpine
    container_name: iceburg_postgres
    environment:
      POSTGRES_DB: iceburg
      POSTGRES_USER: iceburg
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  # Redis for caching
  redis:
    image: redis:7-alpine
    container_name: iceburg_redis
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    restart: unless-stopped

  # ICEBURG Core with production settings
  iceburg:
    build:
      context: ..
      dockerfile: deployment/Dockerfile.prod
    container_name: iceburg_core
    environment:
      - ICEBURG_ENV=production
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - POSTGRES_URL=postgresql://iceburg:${POSTGRES_PASSWORD}@postgres:5432/iceburg
    volumes:
      - ../data:/app/data
      - ../models:/app/models
      - ../logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: iceburg_nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/nginx/ssl
    depends_on:
      - iceburg
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### 3. SSL Configuration

```bash
# Generate SSL certificates
mkdir -p deployment/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deployment/nginx/ssl/iceburg.key \
  -out deployment/nginx/ssl/iceburg.crt
```

### 4. Nginx Configuration

```nginx
# deployment/nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream iceburg_backend {
        server iceburg:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/iceburg.crt;
        ssl_certificate_key /etc/nginx/ssl/iceburg.key;

        location / {
            proxy_pass http://iceburg_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /ws {
            proxy_pass http://iceburg_backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
```

## Configuration

### 1. Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ICEBURG_DATA_DIR` | Data directory path | `/app/data` | Yes |
| `ICEBURG_USER` | System user | `iceburg` | Yes |
| `ICEBURG_HOME` | Home directory | `/app` | Yes |
| `ICEBURG_ENV` | Environment | `development` | Yes |
| `REDIS_URL` | Redis connection | `redis://localhost:6379` | Yes |
| `OLLAMA_URL` | Ollama API URL | `http://localhost:11434` | Yes |
| `ICEBURG_QUANTUM_BACKEND` | Quantum backend | `default.qubit` | No |
| `ICEBURG_RL_ALGORITHM` | RL algorithm | `PPO` | No |

### 2. Configuration Files

#### ICEBURG Configuration

```yaml
# config/icberg_production.yaml
quantum:
  backend: "qiskit.aer"
  device: "qasm_simulator"
  num_qubits: 8
  num_layers: 3
  shots: 1000

rl:
  algorithm: "PPO"
  learning_rate: 0.0003
  total_timesteps: 100000
  num_agents: 10
  batch_size: 64

financial:
  data_provider: "polygon"
  symbols: ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
  intervals: ["1m", "5m", "1h", "1d"]
  cache_ttl: 300

monitoring:
  prometheus_enabled: true
  grafana_enabled: true
  log_level: "INFO"
  metrics_interval: 5
```

#### Docker Configuration

```dockerfile
# deployment/Dockerfile.prod
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements*.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_quantum.txt
RUN pip install --no-cache-dir -r requirements_rl.txt

# Copy application
COPY src/ ./src/
COPY config/ ./config/

# Create non-root user
RUN useradd -m -u 1000 iceburg
USER iceburg

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "src.iceburg.main"]
```

## Monitoring

### 1. Prometheus Metrics

The system exposes metrics at `/metrics` endpoints:

- **ICEBURG Core**: `http://localhost:8000/metrics`
- **Quantum Backend**: `http://localhost:8001/metrics`
- **RL Agents**: `http://localhost:8002/metrics`
- **Financial Data**: `http://localhost:8003/metrics`

### 2. Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin):

- **System Overview**: CPU, memory, disk usage
- **Trading Performance**: Returns, Sharpe ratio, drawdown
- **Quantum Performance**: Circuit execution times, success rates
- **RL Agent Performance**: Rewards, episode lengths, convergence

### 3. Custom Monitoring

```python
# deployment/monitoring_dashboard.py
from src.iceburg.pipeline.monitoring import MonitoringSystem

# Initialize monitoring
monitoring = MonitoringSystem(config)

# Record custom metrics
monitoring.record_metric("trading_performance", 0.85)
monitoring.record_metric("quantum_circuit_time", 0.1)
monitoring.record_metric("rl_agent_reward", 100.0)

# Update health status
monitoring.update_health_status("quantum_backend", "healthy")
monitoring.update_health_status("rl_agents", "training")

# Trigger alerts
monitoring.trigger_alert("WARNING", "High memory usage", "system")
```

## Troubleshooting

### Common Issues

1. **Service Not Starting**
   ```bash
   # Check logs
   docker-compose logs -f [service_name]
   
   # Check resource usage
   docker stats
   
   # Restart service
   docker-compose restart [service_name]
   ```

2. **Quantum Backend Issues**
   ```bash
   # Check quantum dependencies
   pip list | grep pennylane
   
   # Test quantum circuit
   python -c "import pennylane as qml; print(qml.__version__)"
   ```

3. **RL Agent Issues**
   ```bash
   # Check RL dependencies
   pip list | grep stable-baselines3
   
   # Test RL environment
   python -c "import gymnasium as gym; print(gym.__version__)"
   ```

4. **Financial Data Issues**
   ```bash
   # Check API keys
   echo $ICEBURG_FINANCIAL_API_KEY
   
   # Test data connection
   curl -H "Authorization: Bearer $ICEBURG_FINANCIAL_API_KEY" \
        "https://api.polygon.io/v2/aggs/ticker/AAPL/prev"
   ```

### Performance Optimization

1. **GPU Acceleration**
   ```bash
   # Enable GPU for quantum computations
   export CUDA_VISIBLE_DEVICES=0
   export ICEBURG_QUANTUM_BACKEND=qiskit.aer
   ```

2. **Memory Optimization**
   ```bash
   # Increase memory limits
   docker-compose up -d --scale iceburg=2
   
   # Monitor memory usage
   docker stats iceburg_core
   ```

3. **CPU Optimization**
   ```bash
   # Set CPU limits
   docker-compose up -d --scale iceburg=4
   
   # Monitor CPU usage
   htop
   ```

## Scaling

### Horizontal Scaling

```bash
# Scale ICEBURG instances
docker-compose up -d --scale iceburg=3

# Scale with load balancer
docker-compose up -d nginx
```

### Vertical Scaling

```yaml
# docker-compose.override.yml
services:
  iceburg:
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8'
        reservations:
          memory: 8G
          cpus: '4'
```

### Database Scaling

```bash
# Use Redis Cluster
docker-compose up -d redis-cluster

# Use PostgreSQL with replication
docker-compose up -d postgres-master postgres-slave
```

## Security

### 1. API Security

```python
# Enable API authentication
ICEBURG_API_AUTH_ENABLED=true
ICEBURG_API_KEY=your-secure-api-key

# Rate limiting
ICEBURG_RATE_LIMIT=1000
ICEBURG_RATE_WINDOW=3600
```

### 2. Network Security

```bash
# Use HTTPS
docker-compose up -d nginx

# Firewall rules
ufw allow 80
ufw allow 443
ufw deny 8000
```

### 3. Data Security

```bash
# Encrypt data at rest
docker volume create --driver local \
  --opt type=none \
  --opt device=/encrypted/data \
  --opt o=bind iceburg_data

# Backup encryption
gpg --symmetric --cipher-algo AES256 backup.tar.gz
```

## Backup and Recovery

### 1. Data Backup

```bash
# Backup data directory
tar -czf iceburg_backup_$(date +%Y%m%d).tar.gz data/

# Backup database
docker-compose exec postgres pg_dump -U iceburg iceburg > backup.sql
```

### 2. Configuration Backup

```bash
# Backup configuration
cp -r config/ backup_config/
cp .env backup_config/
```

### 3. Recovery

```bash
# Restore data
tar -xzf iceburg_backup_20240101.tar.gz

# Restore database
docker-compose exec postgres psql -U iceburg iceburg < backup.sql
```

## Support

For technical support:

1. Check the troubleshooting section
2. Review logs: `docker-compose logs -f`
3. Check system resources: `docker stats`
4. Contact the development team

## License

This project is licensed under the MIT License. See LICENSE file for details.
