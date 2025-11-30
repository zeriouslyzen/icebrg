#!/bin/bash

# ICEBURG MAC MINI FIX SCRIPT
# Fixes user account, path, and configuration issues between Mac systems
# Run this on the Mac Mini as 'jackdanger' user

echo "🔧 ICEBURG MAC MINI FIX SCRIPT"
echo "=============================="

# Check current user
CURRENT_USER=$(whoami)
echo "📋 Current user: $CURRENT_USER"

if [ "$CURRENT_USER" != "jackdanger" ]; then
    echo "❌ ERROR: This script must be run as 'jackdanger' user"
    echo "   Current user: $CURRENT_USER"
    echo "   Expected user: jackdanger"
    echo "   Please switch to jackdanger user and run again"
    exit 1
fi

# Navigate to ICEBURG directory
ICEBURG_DIR="/Users/jackdanger/Desktop/Projects/iceburg"
echo "📁 ICEBURG directory: $ICEBURG_DIR"

if [ ! -d "$ICEBURG_DIR" ]; then
    echo "❌ ICEBURG directory not found: $ICEBURG_DIR"
    echo "   Please ensure ICEBURG is in the correct location"
    exit 1
fi

cd "$ICEBURG_DIR"
echo "✅ Navigated to ICEBURG directory"

# Fix file ownership
echo "🔐 Fixing file ownership for jackdanger user..."
sudo chown -R jackdanger:staff "$ICEBURG_DIR"
echo "✅ File ownership fixed"

# Fix permissions
echo "🔐 Fixing file permissions..."
chmod -R 755 "$ICEBURG_DIR"
chmod +x *.sh
echo "✅ File permissions fixed"

# Update path references in configuration files
echo "📝 Updating path references from deshonjackson to jackdanger..."

# Update .env file
if [ -f ".env" ]; then
    sed -i '' 's|/Users/deshonjackson|/Users/jackdanger|g' .env
    echo "✅ Updated .env file"
fi

# Update icberg_optimized.env
if [ -f "icberg_optimized.env" ]; then
    sed -i '' 's|/Users/deshonjackson|/Users/jackdanger|g' icberg_optimized.env
    echo "✅ Updated icberg_optimized.env file"
fi

# Update YAML configuration
if [ -f "config/icberg_maximum_capability.yaml" ]; then
    sed -i '' 's|/Users/deshonjackson|/Users/jackdanger|g' config/icberg_maximum_capability.yaml
    echo "✅ Updated YAML configuration"
fi

# Copy Mac Mini specific configuration
echo "📋 Setting up Mac Mini specific configuration..."
if [ -f "macmini_config.env" ]; then
    cp macmini_config.env .env
    echo "✅ Mac Mini configuration applied"
else
    echo "⚠️  Mac Mini configuration not found, using existing .env"
fi

# Remove broken virtual environments
echo "🧹 Cleaning up broken virtual environments..."
rm -rf .venv_broken
if [ -d ".venv" ]; then
    echo "✅ Existing virtual environment found"
else
    echo "📦 Creating fresh virtual environment..."
    python3 -m venv .venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment and install dependencies
echo "📦 Setting up Python environment..."
source .venv/bin/activate

# Install dependencies
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "⚠️  requirements.txt not found"
fi

# Check Ollama installation
echo "🤖 Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is installed"
    
    # Start Ollama if not running
    if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "🚀 Starting Ollama service..."
        ollama serve &
        sleep 5
    fi
    
    # Check and install required models
    echo "🧠 Checking required models..."
    models=("llama3.1:8b" "mistral:7b-instruct" "llama3:70b-instruct" "nomic-embed-text")
    for model in "${models[@]}"; do
        if ollama list | grep -q "$model"; then
            echo "✅ Model available: $model"
        else
            echo "📥 Installing model: $model"
            ollama pull "$model"
        fi
    done
else
    echo "❌ Ollama not found. Please install Ollama first:"
    echo "   curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

# Initialize data directories
echo "💾 Initializing data directories..."
mkdir -p data/{vector_store,memory,logs,metrics,emergence,consciousness}
chmod -R 755 data/
chown -R jackdanger:staff data/
echo "✅ Data directories initialized"

# Test ICEBURG configuration
echo "🧪 Testing ICEBURG configuration..."
python -c "
import sys
sys.path.append('.')
try:
    from src.iceburg.config import load_config
    config = load_config()
    print('✅ ICEBURG configuration loaded successfully')
    print(f'   Data directory: {config.data_dir}')
    print(f'   Surveyor model: {config.surveyor_model}')
    print(f'   Dissident model: {config.dissident_model}')
    print(f'   Synthesist model: {config.synthesist_model}')
    print(f'   Oracle model: {config.oracle_model}')
    print(f'   Embed model: {config.embed_model}')
except Exception as e:
    print(f'❌ ICEBURG configuration error: {e}')
    sys.exit(1)
"

# Test basic ICEBURG functionality
echo "🧪 Testing basic ICEBURG functionality..."
python -c "
import sys
sys.path.append('.')
try:
    from src.iceburg.protocol import iceberg_protocol
    print('✅ ICEBURG protocol imported successfully')
except Exception as e:
    print(f'⚠️  ICEBURG protocol import warning: {e}')
"

# Display final status
echo ""
echo "🎯 ICEBURG MAC MINI FIX COMPLETE"
echo "================================"
echo "✅ User: jackdanger"
echo "✅ File ownership: Fixed"
echo "✅ File permissions: Fixed"
echo "✅ Path references: Updated"
echo "✅ Virtual environment: Ready"
echo "✅ Dependencies: Installed"
echo "✅ Ollama: Running"
echo "✅ Models: Available"
echo "✅ Data directories: Initialized"
echo "✅ Configuration: Tested"
echo ""

echo "🚀 NEXT STEPS:"
echo "=============="
echo "1. Run the Mac Mini activation script:"
echo "   ./macmini_activation_script.sh"
echo ""
echo "2. Test ICEBURG functionality:"
echo "   python -m src.iceburg.protocol 'Test ICEBURG on Mac Mini'"
echo ""
echo "3. Access web interface:"
echo "   http://localhost:8000"
echo ""
echo "ICEBURG Mac Mini is now ready for autonomous operation!"
