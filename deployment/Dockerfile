FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements_quantum.txt requirements_rl.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_quantum.txt
RUN pip install --no-cache-dir -r requirements_rl.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY data/ ./data/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p logs data/models data/cache

# Copy configuration files
COPY icberg_optimized.env .env
COPY config/icberg_maximum_capability.yaml ./config/

# Set environment variables
ENV ICEBURG_DATA_DIR=/app/data
ENV ICEBURG_USER=iceburg
ENV ICEBURG_HOME=/app
ENV PYTHONPATH=/app/src

# Expose ports
EXPOSE 8000 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application
CMD ["python", "-m", "src.iceburg.main"]
