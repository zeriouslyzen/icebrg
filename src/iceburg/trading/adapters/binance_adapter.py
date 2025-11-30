"""
Binance.US exchange adapter for crypto trading
"""
import ccxt
import os
from typing import Dict, List, Optional, Any
from .base import ExchangeAdapter

class BinanceUSAdapter(ExchangeAdapter):
    """Binance.US exchange adapter using ccxt"""
    
    def __init__(self, paper: bool = True):
        self.paper = paper
        self.exchange = None
        
        if not paper:
            # Load API credentials from environment
            api_key = os.getenv('BINANCE_US_API_KEY')
            api_secret = os.getenv('BINANCE_US_API_SECRET')
            
            if not api_key or not api_secret:
                raise ValueError("Binance.US API credentials not found in environment variables")
            
            self.exchange = ccxt.binanceus({
                'apiKey': api_key,
                'secret': api_secret,
                'sandbox': False,  # Set to True for testnet
                'enableRateLimit': True,
            })
        else:
            # Paper trading mode - use public API only
            self.exchange = ccxt.binanceus({
                'sandbox': True,
                'enableRateLimit': True,
            })
    
    def get_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            if self.paper:
                # For paper trading, use public ticker
                ticker = self.exchange.fetch_ticker(symbol)
                return float(ticker['last'])
            else:
                # For live trading, get real price
                ticker = self.exchange.fetch_ticker(symbol)
                return float(ticker['last'])
        except Exception as e:
            return 0.0
    
    def get_balance(self) -> Dict[str, float]:
        """Get account balance"""
        if self.paper:
            # Return mock balance for paper trading
            return {
                'USDT': 10000.0,
                'BTC': 0.0,
                'ETH': 0.0,
                'BNB': 0.0
            }
        
        try:
            balance = self.exchange.fetch_balance()
            return {
                'USDT': balance.get('USDT', {}).get('free', 0.0),
                'BTC': balance.get('BTC', {}).get('free', 0.0),
                'ETH': balance.get('ETH', {}).get('free', 0.0),
                'BNB': balance.get('BNB', {}).get('free', 0.0)
            }
        except Exception as e:
            return {}
    
    def place_market_order(self, symbol: str, side: str, amount: float) -> Dict[str, Any]:
        """Place a market order"""
        if self.paper:
            # Simulate order for paper trading
            price = self.get_price(symbol)
            return {
                'id': f"paper_{int(time.time())}",
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'status': 'filled',
                'timestamp': int(time.time() * 1000)
            }
        
        try:
            order = self.exchange.create_market_order(symbol, side, amount)
            return order
        except Exception as e:
            return {}
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get order status"""
        if self.paper:
            return {'status': 'filled'}
        
        try:
            return self.exchange.fetch_order(order_id)
        except Exception as e:
            return {}
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available trading pairs"""
        try:
            markets = self.exchange.load_markets()
            # Filter for USDT pairs (most common on Binance.US)
            usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
            return usdt_pairs[:20]  # Return top 20 for performance
        except Exception as e:
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
    
    def get_24h_stats(self, symbol: str) -> Dict[str, Any]:
        """Get 24-hour trading statistics"""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'price': ticker['last'],
                'change_24h': ticker['change'],
                'change_percent_24h': ticker['percentage'],
                'volume_24h': ticker['baseVolume'],
                'high_24h': ticker['high'],
                'low_24h': ticker['low']
            }
        except Exception as e:
            return {}
