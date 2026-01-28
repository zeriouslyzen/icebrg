#!/bin/bash
# Setup Continuous Mining System for Strombeck Investigation

cd "$(dirname "$0")/.."

echo "=================================================================================="
echo "SETTING UP CONTINUOUS MINING SYSTEM"
echo "=================================================================================="

# Create mining directory
mkdir -p logs/mining
mkdir -p data/mining_results

# Check dependencies
echo ""
echo "Checking dependencies..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi
echo "✅ Python 3 found"

# Check Selenium
if ! python3 -c "import selenium" 2>/dev/null; then
    echo "⚠️  Selenium not installed. Installing..."
    pip3 install selenium
fi
echo "✅ Selenium available"

# Check ChromeDriver
if ! command -v chromedriver &> /dev/null; then
    echo "⚠️  ChromeDriver not found. Install with: brew install chromedriver"
    echo "   Or download from: https://chromedriver.chromium.org/"
else
    echo "✅ ChromeDriver found"
fi

# Create mining config
cat > config/mining_config.json << EOF
{
    "enabled": true,
    "interval_seconds": 3600,
    "sources": {
        "web_intelligence": true,
        "wayback_machine": true,
        "county_portals": true,
        "business_records": true
    },
    "targets": [
        "Strombeck Properties",
        "Steven Mark Strombeck",
        "Waltina Martha Strombeck",
        "Erik Strombeck",
        "STEATA LLC"
    ],
    "addresses": [
        "960 S G St Arcata",
        "Todd Court Arcata",
        "Western Avenue Arcata",
        "7th + P St Eureka",
        "965 W Harris Eureka"
    ]
}
EOF

echo "✅ Mining config created"

# Create systemd service file (for Linux) or launchd plist (for macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo ""
    echo "Creating macOS LaunchAgent..."
    cat > ~/Library/LaunchAgents/com.iceburg.mining.plist << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.iceburg.mining</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(which python3)</string>
        <string>$(pwd)/scripts/continuous_mining_daemon.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>$(pwd)</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>$(pwd)/logs/mining/daemon.log</string>
    <key>StandardErrorPath</key>
    <string>$(pwd)/logs/mining/daemon.error.log</string>
</dict>
</plist>
EOF
    echo "✅ LaunchAgent created at ~/Library/LaunchAgents/com.iceburg.mining.plist"
    echo "   Load with: launchctl load ~/Library/LaunchAgents/com.iceburg.mining.plist"
fi

echo ""
echo "=================================================================================="
echo "SETUP COMPLETE"
echo "=================================================================================="
echo ""
echo "To start continuous mining:"
echo "  python3 scripts/continuous_mining_daemon.py"
echo ""
echo "Or run individual miners:"
echo "  python3 scripts/mine_strombeck_intelligence.py"
echo "  python3 scripts/wayback_miner.py"
echo "  python3 scripts/selenium_county_scraper.py"
echo ""
