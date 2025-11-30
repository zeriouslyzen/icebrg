#!/bin/bash

# ICEBURG Real Money Trading System Launcher
# Military-grade secure launcher for real money trading

set -e

echo "🚀 ICEBURG Real Money Trading System"
echo "====================================="
echo ""

# Check if running as root (security risk)
if [ "$EUID" -eq 0 ]; then
    echo "❌ ERROR: Do not run as root for security reasons"
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ ERROR: Python 3.8+ required, found $python_version"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_quantum.txt
pip install -r requirements_rl.txt

# Install additional security packages
echo "🔒 Installing security packages..."
pip install cryptography
pip install python-binance
pip install alpaca-trade-api
pip install aiohttp
pip install sqlite3

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data
mkdir -p config
mkdir -p data/secure_wallets.db

# Set permissions
echo "🔐 Setting security permissions..."
chmod 700 config/
chmod 600 config/real_money_trading.json
chmod 700 data/
chmod 600 data/secure_wallets.db

# Check configuration
echo "⚙️ Checking configuration..."
if [ ! -f "config/real_money_trading.json" ]; then
    echo "📝 Creating default configuration..."
    python3 -c "
import json
import os
os.makedirs('config', exist_ok=True)
config = {
    'security': {
        'encryption_key': 'CHANGE_THIS_MILITARY_GRADE_PASSWORD_2025',
        'ip_whitelist': ['192.168.1.0/24', '10.0.0.0/8'],
        'max_daily_loss_percent': 5.0,
        'emergency_stop_threshold': 10.0,
        'rate_limit': 100,
        'session_timeout_minutes': 60
    },
    'broker': {
        'type': 'binance',
        'api_key': 'YOUR_BINANCE_API_KEY',
        'secret_key': 'YOUR_BINANCE_SECRET_KEY',
        'base_url': 'https://api.binance.com',
        'testnet': True
    },
    'wallet': {
        'wallet_id': 'trading_wallet_001',
        'wallet_type': 'hot',
        'max_balance': 10000.0,
        'daily_limit': 1000.0,
        'withdrawal_limit': 500.0
    },
    'trading': {
        'strategies': [
            {
                'name': 'BTC_Momentum',
                'symbols': ['BTCUSDT'],
                'max_position_size': 1000.0,
                'stop_loss_percent': 2.0,
                'take_profit_percent': 5.0,
                'risk_per_trade': 1.0,
                'enabled': True
            }
        ],
        'monitoring_interval': 1.0,
        'max_trades_per_day': 50
    }
}
with open('config/real_money_trading.json', 'w') as f:
    json.dump(config, f, indent=2)
print('✅ Default configuration created')
"
fi

# Security warnings
echo ""
echo "⚠️  SECURITY WARNINGS:"
echo "======================"
echo "1. Change the encryption key in config/real_money_trading.json"
echo "2. Update API keys with your actual broker credentials"
echo "3. Set appropriate IP whitelist for your network"
echo "4. Review and adjust risk limits before trading"
echo "5. Start with testnet=True for testing"
echo ""

# Check if configuration is secure
echo "🔍 Checking configuration security..."
if grep -q "CHANGE_THIS_MILITARY_GRADE_PASSWORD" config/real_money_trading.json; then
    echo "❌ WARNING: Default encryption key detected!"
    echo "   Please change the encryption key in config/real_money_trading.json"
    echo ""
fi

if grep -q "YOUR_BINANCE_API_KEY" config/real_money_trading.json; then
    echo "❌ WARNING: Default API keys detected!"
    echo "   Please update API keys in config/real_money_trading.json"
    echo ""
fi

# Display menu
echo "🎯 REAL MONEY TRADING SYSTEM"
echo "============================"
echo ""
echo "Select an option:"
echo "1. Start Trading (Real Money)"
echo "2. Start Trading (Testnet)"
echo "3. Show System Status"
echo "4. Emergency Stop"
echo "5. Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "🚀 Starting REAL MONEY trading..."
        echo "⚠️  WARNING: This will trade with REAL MONEY!"
        read -p "Are you sure? (yes/no): " confirm
        if [ "$confirm" = "yes" ]; then
            python3 src/iceburg/trading/real_money_launcher.py --config config/real_money_trading.json
        else
            echo "❌ Trading cancelled"
            exit 1
        fi
        ;;
    2)
        echo "🧪 Starting TESTNET trading..."
        python3 src/iceburg/trading/real_money_launcher.py --config config/real_money_trading.json
        ;;
    3)
        echo "📊 System Status:"
        python3 src/iceburg/trading/real_money_launcher.py --config config/real_money_trading.json --status
        ;;
    4)
        echo "🚨 Emergency Stop:"
        python3 src/iceburg/trading/real_money_launcher.py --config config/real_money_trading.json --emergency-stop
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
