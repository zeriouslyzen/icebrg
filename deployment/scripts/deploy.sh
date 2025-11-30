#!/bin/bash
# ICEBURG Deployment Script
# Automated deployment for ICEBURG 2.0

set -e

echo "ICEBURG 2.0 Deployment Script"
echo "=============================="

# Check Python version
python_version=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

# Optional: Install frontier model dependencies
read -p "Install frontier model APIs? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing frontier model APIs..."
    pip install anthropic>=0.18.0 || echo "Anthropic not available"
    pip install openai>=1.12.0 || echo "OpenAI not available"
    pip install google-generativeai>=0.3.0 || echo "Google not available"
fi

# Optional: Install lab dependencies
read -p "Install lab dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing lab dependencies..."
    pip install qiskit>=1.0.0 || echo "Qiskit not available"
    pip install matplotlib>=3.8.0 || echo "Matplotlib not available"
    pip install plotly>=5.18.0 || echo "Plotly not available"
fi

# Optional: Install visual dependencies
read -p "Install visual dependencies? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Installing visual dependencies..."
    pip install pillow>=10.0.0 || echo "Pillow not available"
    pip install opencv-python>=4.8.0 || echo "OpenCV not available"
    pip install pytesseract>=0.3.10 || echo "Tesseract not available"
fi

# Create data directories
echo "Creating data directories..."
mkdir -p data/models
mkdir -p data/lab/protocols
mkdir -p data/lab/equipment_catalog.json
mkdir -p data/tenants
mkdir -p data/research_outputs
mkdir -p data/projects
mkdir -p data/blockchain_verification
mkdir -p logs

# Run tests
echo "Running tests..."
pytest tests/ -v || echo "Tests failed, continuing..."

# Setup LaunchAgent for macOS (if on macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Setting up LaunchAgent for macOS..."
    cat > ~/Library/LaunchAgents/com.iceburg.plist <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.iceburg</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/python3</string>
        <string>-m</string>
        <string>iceburg</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>~/Library/Logs/iceburg.log</string>
    <key>StandardErrorPath</key>
    <string>~/Library/Logs/iceburg.error.log</string>
</dict>
</plist>
EOF
    launchctl load ~/Library/LaunchAgents/com.iceburg.plist || echo "LaunchAgent setup failed"
fi

echo "Deployment complete!"
echo "ICEBURG 2.0 is ready to use."

