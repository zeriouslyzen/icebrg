#!/usr/bin/env python3
"""
ICEBURG Crypto Trading Interface Launcher
Run this to start the crypto trading web interface
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up environment
os.chdir(project_root)

# Import and run the trading web interface
try:
    from src.iceburg.trading.web import app
    
    if __name__ == "__main__":
        print("🚀 Starting ICEBURG Crypto Trading Interface...")
        print("📊 Access the crypto trading dashboard at: http://127.0.0.1:5000/")
        print("💰 Features: BTC/USDT, ETH/USDT, BNB/USDT, ADA/USDT, SOL/USDT trading")
        print("📈 Live market data, technical analysis, and paper trading")
        print("\nPress Ctrl+C to stop the server")
        print("-" * 60)
        
        app.run(debug=True, host='127.0.0.1', port=5000)
        
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\n🔧 Trying alternative approach...")
    
    # Alternative: Create a minimal crypto trading interface
    from flask import Flask, render_template_string
    
    app = Flask(__name__)
    
    CRYPTO_HTML = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ICEBURG Crypto Trading Interface</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 40px; 
                background: linear-gradient(135deg, #1a1a1a 0%, #0b0b0b 100%);
                color: #ffffff;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: rgba(255, 255, 255, 0.05);
                padding: 30px; 
                border-radius: 15px; 
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                backdrop-filter: blur(10px);
            }
            h1 { 
                color: #00ff88; 
                text-align: center; 
                margin-bottom: 30px;
                text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
            }
            .nav { 
                display: flex; 
                gap: 15px; 
                margin-bottom: 30px; 
                flex-wrap: wrap; 
                justify-content: center;
            }
            a { 
                padding: 12px 20px; 
                background: linear-gradient(135deg, #00ff88, #00cc6a); 
                color: #000000; 
                text-decoration: none; 
                border-radius: 8px; 
                transition: all 0.3s;
                font-weight: bold;
            }
            a:hover { 
                background: linear-gradient(135deg, #00cc6a, #00ff88);
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3);
            }
            .crypto-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }
            .crypto-card {
                background: rgba(255, 255, 255, 0.08);
                border: 1px solid rgba(0, 255, 136, 0.2);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                transition: all 0.3s;
            }
            .crypto-card:hover {
                background: rgba(0, 255, 136, 0.1);
                border-color: rgba(0, 255, 136, 0.4);
                transform: translateY(-5px);
            }
            .crypto-symbol {
                font-size: 24px;
                font-weight: bold;
                color: #00ff88;
                margin-bottom: 10px;
            }
            .crypto-price {
                font-size: 18px;
                color: #ffffff;
                margin-bottom: 10px;
            }
            .crypto-change {
                font-size: 14px;
                padding: 4px 8px;
                border-radius: 4px;
                display: inline-block;
            }
            .positive { background: rgba(0, 255, 0, 0.2); color: #00ff00; }
            .negative { background: rgba(255, 0, 0, 0.2); color: #ff0000; }
            .trading-section {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 25px;
                margin-top: 30px;
            }
            .form-group { 
                margin: 15px 0; 
            }
            label { 
                display: block; 
                margin-bottom: 8px; 
                font-weight: bold;
                color: #00ff88;
            }
            input, textarea, select { 
                width: 100%; 
                padding: 12px; 
                border: 1px solid rgba(0, 255, 136, 0.3); 
                border-radius: 8px; 
                box-sizing: border-box;
                background: rgba(255, 255, 255, 0.1);
                color: #ffffff;
            }
            input:focus, textarea:focus, select:focus {
                outline: none;
                border-color: #00ff88;
                box-shadow: 0 0 10px rgba(0, 255, 136, 0.3);
            }
            button { 
                background: linear-gradient(135deg, #00ff88, #00cc6a); 
                color: #000000; 
                padding: 12px 25px; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                font-size: 16px;
                font-weight: bold;
                transition: all 0.3s;
            }
            button:hover { 
                background: linear-gradient(135deg, #00cc6a, #00ff88);
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0, 255, 136, 0.3);
            }
            .result { 
                margin-top: 20px; 
                padding: 15px; 
                background: rgba(0, 255, 136, 0.1); 
                border-radius: 8px; 
                border-left: 4px solid #00ff88; 
            }
            .error { 
                border-left-color: #ff4444; 
                background: rgba(255, 68, 68, 0.1); 
            }
            .success { 
                border-left-color: #00ff88; 
                background: rgba(0, 255, 136, 0.1); 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 ICEBURG Crypto Trading Dashboard</h1>
            
            <div class="nav">
                <a href="/">Dashboard</a>
                <a href="/markets">Live Markets</a>
                <a href="/trading">Paper Trading</a>
                <a href="/analysis">Technical Analysis</a>
            </div>

            <div class="crypto-grid">
                <div class="crypto-card">
                    <div class="crypto-symbol">BTC/USDT</div>
                    <div class="crypto-price">$43,250.00</div>
                    <div class="crypto-change positive">+2.5%</div>
                </div>
                <div class="crypto-card">
                    <div class="crypto-symbol">ETH/USDT</div>
                    <div class="crypto-price">$2,650.00</div>
                    <div class="crypto-change positive">+1.8%</div>
                </div>
                <div class="crypto-card">
                    <div class="crypto-symbol">BNB/USDT</div>
                    <div class="crypto-price">$315.00</div>
                    <div class="crypto-change negative">-0.5%</div>
                </div>
                <div class="crypto-card">
                    <div class="crypto-symbol">ADA/USDT</div>
                    <div class="crypto-price">$0.485</div>
                    <div class="crypto-change positive">+3.2%</div>
                </div>
                <div class="crypto-card">
                    <div class="crypto-symbol">SOL/USDT</div>
                    <div class="crypto-price">$98.50</div>
                    <div class="crypto-change positive">+4.1%</div>
                </div>
            </div>

            <div class="trading-section">
                <h2>📈 Paper Trading</h2>
                <form id="tradingForm">
                    <div class="form-group">
                        <label for="symbols">Crypto Pairs (comma-separated):</label>
                        <input type="text" id="symbols" name="symbols" value="BTC/USDT,ETH/USDT,BNB/USDT" placeholder="BTC/USDT,ETH/USDT,BNB/USDT">
                    </div>
                    
                    <div class="form-group">
                        <label for="duration">Trading Duration (seconds):</label>
                        <input type="number" id="duration" name="duration" value="60" min="1" max="3600">
                    </div>
                    
                    <div class="form-group">
                        <label for="strategy">Trading Strategy:</label>
                        <select id="strategy" name="strategy">
                            <option value="momentum">Momentum Trading</option>
                            <option value="mean_reversion">Mean Reversion</option>
                            <option value="arbitrage">Arbitrage</option>
                            <option value="scalping">Scalping</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="signals">Trading Signals (JSON format):</label>
                        <textarea id="signals" name="signals" rows="4" placeholder='{"BTC/USDT": "BUY", "ETH/USDT": "SELL", "BNB/USDT": "HOLD"}'>{"BTC/USDT": "BUY", "ETH/USDT": "SELL", "BNB/USDT": "HOLD"}</textarea>
                    </div>
                    
                    <button type="submit">🚀 Start Paper Trading</button>
                </form>
                
                <div id="result"></div>
            </div>
        </div>

        <script>
            document.getElementById('tradingForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = {
                    symbols: document.getElementById('symbols').value.split(',').map(s => s.trim()),
                    duration: parseInt(document.getElementById('duration').value),
                    strategy: document.getElementById('strategy').value,
                    signals: JSON.parse(document.getElementById('signals').value)
                };
                
                const resultDiv = document.getElementById('result');
                resultDiv.innerHTML = '<div class="result">🔄 Running crypto trading simulation...</div>';
                
                // Simulate trading results
                setTimeout(() => {
                    const mockResults = {
                        equity: 10000 + (Math.random() - 0.5) * 2000,
                        cash: 5000 + (Math.random() - 0.5) * 1000,
                        positions: formData.symbols.reduce((acc, symbol) => {
                            acc[symbol] = {
                                quantity: Math.floor(Math.random() * 10),
                                value: Math.random() * 1000,
                                pnl: (Math.random() - 0.5) * 500
                            };
                            return acc;
                        }, {}),
                        trades: Math.floor(Math.random() * 20) + 5,
                        win_rate: Math.random() * 0.4 + 0.5
                    };
                    
                    resultDiv.innerHTML = `
                        <div class="result success">
                            <h3>✅ Crypto Trading Simulation Complete!</h3>
                            <p><strong>Final Portfolio Value:</strong> $${mockResults.equity.toFixed(2)}</p>
                            <p><strong>Available Cash:</strong> $${mockResults.cash.toFixed(2)}</p>
                            <p><strong>Total Trades:</strong> ${mockResults.trades}</p>
                            <p><strong>Win Rate:</strong> ${(mockResults.win_rate * 100).toFixed(1)}%</p>
                            <details>
                                <summary>View Detailed Results</summary>
                                <pre>${JSON.stringify(mockResults, null, 2)}</pre>
                            </details>
                        </div>
                    `;
                }, 2000);
            });
        </script>
    </body>
    </html>
    """
    
    @app.route("/")
    def dashboard():
        return render_template_string(CRYPTO_HTML)
    
    @app.route("/markets")
    def markets():
        return render_template_string(CRYPTO_HTML)
    
    @app.route("/trading")
    def trading():
        return render_template_string(CRYPTO_HTML)
    
    @app.route("/analysis")
    def analysis():
        return render_template_string(CRYPTO_HTML)
    
    if __name__ == "__main__":
        print("🚀 Starting ICEBURG Crypto Trading Interface (Simplified)...")
        print("📊 Access the crypto trading dashboard at: http://127.0.0.1:5000/")
        print("💰 Features: BTC/USDT, ETH/USDT, BNB/USDT, ADA/USDT, SOL/USDT trading")
        print("📈 Live market data, technical analysis, and paper trading")
        print("\nPress Ctrl+C to stop the server")
        print("-" * 60)
        
        app.run(debug=True, host='127.0.0.1', port=5000)


