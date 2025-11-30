from flask import Flask, request, jsonify, render_template_string
import json
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.iceburg.trading.orchestrator import TradingOrchestrator
    from src.iceburg.trading.signal.oracle_bridge import parse_oracle_input
    from src.iceburg.trading.risk.risk_manager import RiskConfig, RiskManager
    from src.iceburg.trading.portfolio.portfolio_manager import PortfolioManager
    from src.iceburg.trading.execution.trade_executor import TradeExecutor
    from src.iceburg.trading.sim.paper_broker import PaperBroker
except ImportError as e:
    print(f"Import error: {e}")
    # Fallback to relative imports
    from .orchestrator import TradingOrchestrator
    from .signal.oracle_bridge import parse_oracle_input
    from .risk.risk_manager import RiskConfig, RiskManager
    from .portfolio.portfolio_manager import PortfolioManager
    from .execution.trade_executor import TradeExecutor
    from .sim.paper_broker import PaperBroker
from .market_data import MarketDataProvider, RealTimeBroker

app = Flask(__name__)
CONFIG_PATH = "config/trading_config.json"
REPORTS_DIR = "data/trading/reports"

HTML = """
<!doctype html>
<html>
<head>
    <title>ICEBURG Trading Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; text-align: center; margin-bottom: 30px; }
        .nav { display: flex; gap: 10px; margin-bottom: 30px; flex-wrap: wrap; }
        a { padding: 12px 20px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; transition: background 0.3s; }
        a:hover { background: #0056b3; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, textarea, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; box-sizing: border-box; }
        button { background: #28a745; color: white; padding: 12px 25px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
        button:hover { background: #218838; }
        .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; }
        .error { border-left-color: #dc3545; background: #f8d7da; }
        .success { border-left-color: #28a745; background: #d4edda; }
        pre { background: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 ICEBURG Trading Dashboard</h1>
        
        <div class="nav">
            <a href="/">Dashboard</a>
            <a href="/config">Config</a>
            <a href="/reports">Reports</a>
            <a href="/markets">Live Markets</a>
            <a href="/status">Status</a>
        </div>

        <h2>Paper Trading</h2>
        <form id="paperForm">
            <div class="form-group">
                <label for="symbols">Symbols (comma-separated):</label>
                <input type="text" id="symbols" name="symbols" value="BTC/USDT,ETH/USDT" placeholder="BTC/USDT,ETH/USDT">
            </div>
            
            <div class="form-group">
                <label for="duration">Duration (seconds):</label>
                <input type="number" id="duration" name="duration" value="10" min="1" max="300">
            </div>
            
            <div class="form-group">
                <label for="oracle_text">Oracle Signals (text or JSON):</label>
                <textarea id="oracle_text" name="oracle_text" rows="4" placeholder="BUY BTC/USDT (conf 0.7)&#10;SELL ETH/USDT (conf 0.6)">BUY BTC/USDT (conf 0.7)
SELL ETH/USDT (conf 0.6)</textarea>
    </div>
            
            <div class="form-group">
                <label>
                    <input type="checkbox" id="use_real_data" name="use_real_data"> 
                    Use Real Market Data (instead of simulation)
                </label>
            </div>
            
            <button type="submit">Run Paper Trading</button>
        </form>
        
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('paperForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = {
                symbols: document.getElementById('symbols').value.split(',').map(s => s.trim()),
                duration: parseInt(document.getElementById('duration').value),
                oracle_text: document.getElementById('oracle_text').value,
                use_real_data: document.getElementById('use_real_data').checked
            };
            
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<div class="result">Running simulation...</div>';
            
            try {
                const response = await fetch('/run_paper', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(formData)
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    resultDiv.innerHTML = `
                        <div class="result success">
                            <h3>✅ Simulation Complete!</h3>
                            <p><strong>Final Equity:</strong> $${data.equity.toFixed(2)}</p>
                            <p><strong>Cash:</strong> $${data.cash.toFixed(2)}</p>
                            <p><strong>Positions:</strong> ${Object.keys(data.positions).length}</p>
                            <details>
                                <summary>View Full Report</summary>
                                <pre>${JSON.stringify(data, null, 2)}</pre>
                            </details>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <div class="result error">
                            <h3>❌ Error</h3>
                            <p>${data.error || 'Unknown error occurred'}</p>
                        </div>
                    `;
                }
            } catch (error) {
                resultDiv.innerHTML = `
                    <div class="result error">
                        <h3>❌ Network Error</h3>
                        <p>${error.message}</p>
                    </div>
                `;
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def dashboard():
    return render_template_string(HTML)

@app.route("/status")
def status():
    return jsonify({
        "status": "running",
        "config_path": CONFIG_PATH,
        "reports_dir": REPORTS_DIR,
        "message": "ICEBURG Trading Server is active"
    })

@app.route("/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        data = request.json
        with open(CONFIG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return jsonify({"status": "updated"})
    
    try:
        with open(CONFIG_PATH, "r") as f:
            config_data = json.load(f)
        
        # Create a nice HTML view of the config
        html = f"""
        <html>
        <head>
            <title>ICEBURG Trading Config</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #333; }}
                pre {{ background: #f8f9fa; padding: 20px; border-radius: 5px; overflow-x: auto; border: 1px solid #dee2e6; }}
                .back-link {{ display: inline-block; margin-bottom: 20px; padding: 10px 20px; background: #6c757d; color: white; text-decoration: none; border-radius: 5px; }}
                .info {{ background: #d1ecf1; border: 1px solid #bee5eb; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/" class="back-link">← Back to Dashboard</a>
                <h1>⚙️ Trading Configuration</h1>
                
                <div class="info">
                    <strong>Note:</strong> This is the current configuration. To modify it, you can edit the file directly at <code>{CONFIG_PATH}</code> or use the API.
                </div>
                
                <h3>Current Configuration:</h3>
                <pre>{json.dumps(config_data, indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        return html
    except FileNotFoundError:
        return jsonify({"error": "Config file not found"}), 404

@app.route("/run_paper", methods=["POST"])
def run_paper():
    try:
        data = request.json or {}
        symbols = data.get("symbols", ["BTC/USDT", "ETH/USDT"])
        duration = data.get("duration", 10)
        oracle_text = data.get("oracle_text", "BUY BTC/USDT (conf 0.7)\nSELL ETH/USDT (conf 0.6)")
        use_real_data = data.get("use_real_data", False)

        cfg = {"symbols": symbols}
        
        # Choose broker based on real data option
        if use_real_data:
            broker = RealTimeBroker()
        else:
            broker = PaperBroker()
            
        risk = RiskManager(RiskConfig())
        portfolio = PortfolioManager()
        executor = TradeExecutor(broker)
        orch = TradingOrchestrator(cfg, broker, risk, portfolio, executor)
        
        # If using real data and no oracle text, generate signals from market data
        if use_real_data and not oracle_text.strip():
            all_signals = []
            for symbol in symbols:
                signals = broker.get_trading_signals(symbol)
                all_signals.extend(signals)
            signals = all_signals
        else:
            signals = parse_oracle_input(oracle_text)
            
        report = orch.run_paper(signals, duration)
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/reports")
def reports():
    try:
        if not os.path.exists(REPORTS_DIR):
            return render_template_string("""
            <html><body style="font-family: Arial; margin: 40px;">
                <h1>Reports</h1>
                <p>No reports directory found. Run a paper trading simulation first.</p>
                <a href="/">← Back to Dashboard</a>
            </body></html>
            """)
        
        files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".json")]
        files.sort(reverse=True)  # Most recent first
        
        if not files:
            return render_template_string("""
            <html><body style="font-family: Arial; margin: 40px;">
                <h1>Reports</h1>
                <p>No reports found. Run a paper trading simulation first.</p>
                <a href="/">← Back to Dashboard</a>
            </body></html>
            """)
        
        html = """
        <html>
        <head>
            <title>ICEBURG Trading Reports</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; }
                .report-item { padding: 15px; margin: 10px 0; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff; }
                .report-item:hover { background: #e9ecef; }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
                .back-link { display: inline-block; margin-bottom: 20px; padding: 10px 20px; background: #6c757d; color: white; text-decoration: none; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/" class="back-link">← Back to Dashboard</a>
                <h1>📊 Trading Reports</h1>
                <p>Found """ + str(len(files)) + """ report(s):</p>
        """
        
        for file in files:
            html += f"""
                <div class="report-item">
                    <h3><a href="/report/{file}">{file}</a></h3>
                    <p>Click to view full report details</p>
                </div>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route("/markets")
def markets():
    """Live market data and technical analysis"""
    try:
        market_data = MarketDataProvider()
        
        # Get data for popular symbols (including Binance.US USDT pairs)
        symbols = ["BTC-USD", "ETH-USD", "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT", "AAPL", "TSLA", "SPY"]
        market_info = {}
        
        for symbol in symbols:
            try:
                price = market_data.get_real_time_price(symbol)
                indicators = market_data.get_technical_indicators(symbol)
                signals = market_data.generate_trading_signals(symbol)
                
                market_info[symbol] = {
                    "price": price,
                    "indicators": indicators,
                    "signals": signals
                }
            except Exception as e:
                market_info[symbol] = {"error": str(e)}
        
        # Create HTML response
        html = """
        <html>
        <head>
            <title>ICEBURG Live Markets</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; }
                .market-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }
                .market-card { background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #007bff; }
                .price { font-size: 24px; font-weight: bold; color: #28a745; }
                .signal { padding: 5px 10px; border-radius: 15px; color: white; font-size: 12px; margin: 2px; display: inline-block; }
                .signal.buy { background: #28a745; }
                .signal.sell { background: #dc3545; }
                .indicator { margin: 5px 0; font-size: 14px; }
                .back-link { display: inline-block; margin-bottom: 20px; padding: 10px 20px; background: #6c757d; color: white; text-decoration: none; border-radius: 5px; }
                .refresh-btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin-bottom: 20px; }
                .trend { font-weight: bold; }
                .trend.bullish { color: #28a745; }
                .trend.bearish { color: #dc3545; }
                .trend.neutral { color: #6c757d; }
            </style>
        </head>
        <body>
            <div class="container">
                <a href="/" class="back-link">← Back to Dashboard</a>
                <h1>📈 Live Market Data & Technical Analysis</h1>
                
                <button class="refresh-btn" onclick="location.reload()">🔄 Refresh Data</button>
                
                <div class="market-grid">
        """
        
        for symbol, data in market_info.items():
            if "error" in data:
                html += f"""
                    <div class="market-card">
                        <h3>{symbol}</h3>
                        <p style="color: #dc3545;">Error: {data['error']}</p>
                    </div>
                """
                continue
            
            price = data.get('price', 0)
            indicators = data.get('indicators', {})
            signals = data.get('signals', [])
            
            trend = indicators.get('trend', {})
            trend_direction = trend.get('direction', 'unknown')
            trend_strength = trend.get('strength', 0)
            
            html += f"""
                <div class="market-card">
                    <h3>{symbol}</h3>
                    <div class="price">${price:,.2f}</div>
                    
                    <div class="indicator">
                        <strong>Trend:</strong> 
                        <span class="trend {trend_direction}">{trend_direction.upper()}</span>
                        ({trend_strength:.1f}%)
                    </div>
                    
                    <div class="indicator">
                        <strong>RSI:</strong> {indicators.get('rsi', 0):.1f}
                    </div>
                    
                    <div class="indicator">
                        <strong>MACD:</strong> {indicators.get('macd', 0):.4f}
                    </div>
                    
                    <div class="indicator">
                        <strong>Volatility:</strong> {indicators.get('volatility', 0):.2f}%
                    </div>
                    
                    <div style="margin-top: 15px;">
                        <strong>Signals:</strong><br>
            """
            
            if signals:
                for signal in signals:
                    signal_type = signal.get('type', '').lower()
                    confidence = signal.get('confidence', 0)
                    reason = signal.get('reason', '')
                    html += f'<span class="signal {signal_type}">{signal_type.upper()} ({confidence:.1%})</span><br>'
                    html += f'<small style="color: #666;">{reason}</small><br>'
            else:
                html += '<span style="color: #666;">No signals</span>'
            
            html += """
                </div>
                </div>
            """
        
        html += """
            </div>
            </div>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        return f"Error loading market data: {str(e)}", 500

@app.route("/report/<filename>")
def report(filename):
    try:
        path = os.path.join(REPORTS_DIR, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                return jsonify(json.load(f))
        return jsonify({"error": "Report not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting ICEBURG Trading Web Interface...")
    print("Access at: http://os.getenv("HOST", "127.0.0.1"):5000/")
    app.run(debug=True, host='os.getenv("HOST", "127.0.0.1")', port=5000)
