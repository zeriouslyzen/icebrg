"""
Real-time market data integration with technical analysis
"""
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import requests
import json
import ccxt
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import time

class MarketDataProvider:
    """Provides real-time market data and technical analysis"""
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 60  # 1 minute cache
        self.binance = ccxt.binanceus({'enableRateLimit': True})
        
    def get_real_time_price(self, symbol: str) -> float:
        """Get real-time price for a symbol"""
        try:
            # Check if it's a USDT pair (Binance.US)
            if '/USDT' in symbol:
                return self._get_binance_price(symbol)
            
            # Convert symbol format (BTC/USDC -> BTC-USD)
            yf_symbol = symbol.replace('/', '-').replace('USDC', 'USD')
            
            ticker = yf.Ticker(yf_symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                return float(data['Close'].iloc[-1])
            else:
                # Fallback to info
                info = ticker.info
                return float(info.get('regularMarketPrice', 0))
                
        except Exception as e:
            return 0.0
    
    def _get_binance_price(self, symbol: str) -> float:
        """Get price from Binance.US for USDT pairs"""
        try:
            ticker = self.binance.fetch_ticker(symbol)
            return float(ticker['last'])
        except Exception as e:
            return 0.0
    
    def get_technical_indicators(self, symbol: str, period: str = "1mo") -> Dict[str, Any]:
        """Get comprehensive technical analysis for a symbol"""
        try:
            # Handle USDT pairs differently
            if '/USDT' in symbol:
                return self._get_binance_technical_indicators(symbol)
            
            yf_symbol = symbol.replace('/', '-').replace('USDC', 'USD')
            ticker = yf.Ticker(yf_symbol)
            
            # Get historical data
            data = ticker.history(period=period)
            
            if data.empty:
                return {"error": "No data available"}
            
            # Calculate technical indicators
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = ta.sma(data['Close'], length=20).iloc[-1]
            indicators['sma_50'] = ta.sma(data['Close'], length=50).iloc[-1]
            indicators['ema_12'] = ta.ema(data['Close'], length=12).iloc[-1]
            indicators['ema_26'] = ta.ema(data['Close'], length=26).iloc[-1]
            
            # MACD
            macd = ta.macd(data['Close'])
            indicators['macd'] = macd['MACD_12_26_9'].iloc[-1]
            indicators['macd_signal'] = macd['MACDs_12_26_9'].iloc[-1]
            indicators['macd_histogram'] = macd['MACDh_12_26_9'].iloc[-1]
            
            # RSI
            indicators['rsi'] = ta.rsi(data['Close'], length=14).iloc[-1]
            
            # Bollinger Bands
            bb = ta.bbands(data['Close'], length=20)
            indicators['bb_upper'] = bb['BBU_20_2.0'].iloc[-1]
            indicators['bb_middle'] = bb['BBM_20_2.0'].iloc[-1]
            indicators['bb_lower'] = bb['BBL_20_2.0'].iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = ta.sma(data['Volume'], length=20).iloc[-1]
            indicators['current_volume'] = data['Volume'].iloc[-1]
            
            # Price action
            indicators['current_price'] = data['Close'].iloc[-1]
            indicators['high_52w'] = data['High'].max()
            indicators['low_52w'] = data['Low'].min()
            indicators['volatility'] = data['Close'].pct_change().std() * 100
            
            # Trend analysis
            indicators['trend'] = self._analyze_trend(data)
            indicators['support_resistance'] = self._find_support_resistance(data)
            
            return indicators
            
        except Exception as e:
            return {"error": f"Technical analysis failed: {str(e)}"}
    
    def _get_binance_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get technical indicators for Binance.US USDT pairs"""
        try:
            # Get OHLCV data from Binance
            ohlcv = self.binance.fetch_ohlcv(symbol, '1d', limit=100)
            
            if not ohlcv:
                return {"error": "No data available"}
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Calculate technical indicators
            indicators = {}
            
            # Moving Averages
            indicators['sma_20'] = ta.sma(df['close'], length=20).iloc[-1]
            indicators['sma_50'] = ta.sma(df['close'], length=50).iloc[-1]
            indicators['ema_12'] = ta.ema(df['close'], length=12).iloc[-1]
            indicators['ema_26'] = ta.ema(df['close'], length=26).iloc[-1]
            
            # MACD
            macd = ta.macd(df['close'])
            indicators['macd'] = macd['MACD_12_26_9'].iloc[-1]
            indicators['macd_signal'] = macd['MACDs_12_26_9'].iloc[-1]
            indicators['macd_histogram'] = macd['MACDh_12_26_9'].iloc[-1]
            
            # RSI
            indicators['rsi'] = ta.rsi(df['close'], length=14).iloc[-1]
            
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=20)
            indicators['bb_upper'] = bb['BBU_20_2.0'].iloc[-1]
            indicators['bb_middle'] = bb['BBM_20_2.0'].iloc[-1]
            indicators['bb_lower'] = bb['BBL_20_2.0'].iloc[-1]
            
            # Volume indicators
            indicators['volume_sma'] = ta.sma(df['volume'], length=20).iloc[-1]
            indicators['current_volume'] = df['volume'].iloc[-1]
            
            # Price action
            indicators['current_price'] = df['close'].iloc[-1]
            indicators['high_52w'] = df['high'].max()
            indicators['low_52w'] = df['low'].min()
            indicators['volatility'] = df['close'].pct_change().std() * 100
            
            # Trend analysis
            indicators['trend'] = self._analyze_trend(df)
            indicators['support_resistance'] = self._find_support_resistance(df)
            
            return indicators
            
        except Exception as e:
            return {"error": f"Binance technical analysis failed: {str(e)}"}
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend direction and strength"""
        try:
            # Simple trend analysis
            sma_20 = ta.sma(data['Close'], length=20)
            sma_50 = ta.sma(data['Close'], length=50)
            
            current_price = data['Close'].iloc[-1]
            sma_20_current = sma_20.iloc[-1]
            sma_50_current = sma_50.iloc[-1]
            
            trend = "neutral"
            strength = 0
            
            if current_price > sma_20_current > sma_50_current:
                trend = "bullish"
                strength = min(100, ((current_price - sma_50_current) / sma_50_current) * 100)
            elif current_price < sma_20_current < sma_50_current:
                trend = "bearish"
                strength = min(100, ((sma_50_current - current_price) / sma_50_current) * 100)
            
            return {
                "direction": trend,
                "strength": round(strength, 2),
                "price_vs_sma20": round(((current_price - sma_20_current) / sma_20_current) * 100, 2),
                "price_vs_sma50": round(((current_price - sma_50_current) / sma_50_current) * 100, 2)
            }
        except:
            return {"direction": "unknown", "strength": 0}
    
    def _find_support_resistance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Find key support and resistance levels"""
        try:
            # Simple pivot points
            high = data['High'].rolling(window=20).max()
            low = data['Low'].rolling(window=20).min()
            
            resistance = high.iloc[-1]
            support = low.iloc[-1]
            
            return {
                "resistance": float(resistance),
                "support": float(support),
                "current_price": float(data['Close'].iloc[-1])
            }
        except:
            return {"resistance": 0, "support": 0, "current_price": 0}
    
    def generate_trading_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate trading signals based on technical analysis"""
        try:
            indicators = self.get_technical_indicators(symbol)
            
            if "error" in indicators:
                return []
            
            signals = []
            current_price = indicators['current_price']
            rsi = indicators['rsi']
            macd = indicators['macd']
            macd_signal = indicators['macd_signal']
            trend = indicators['trend']
            
            # RSI signals
            if rsi < 30:  # Oversold
                signals.append({
                    "type": "BUY",
                    "symbol": symbol,
                    "confidence": min(0.9, (30 - rsi) / 30 * 0.8 + 0.1),
                    "reason": f"RSI oversold ({rsi:.1f})",
                    "indicator": "RSI"
                })
            elif rsi > 70:  # Overbought
                signals.append({
                    "type": "SELL",
                    "symbol": symbol,
                    "confidence": min(0.9, (rsi - 70) / 30 * 0.8 + 0.1),
                    "reason": f"RSI overbought ({rsi:.1f})",
                    "indicator": "RSI"
                })
            
            # MACD signals
            if macd > macd_signal and trend['direction'] == 'bullish':
                signals.append({
                    "type": "BUY",
                    "symbol": symbol,
                    "confidence": min(0.8, trend['strength'] / 100 * 0.7 + 0.1),
                    "reason": f"MACD bullish crossover, trend strength: {trend['strength']:.1f}%",
                    "indicator": "MACD"
                })
            elif macd < macd_signal and trend['direction'] == 'bearish':
                signals.append({
                    "type": "SELL",
                    "symbol": symbol,
                    "confidence": min(0.8, trend['strength'] / 100 * 0.7 + 0.1),
                    "reason": f"MACD bearish crossover, trend strength: {trend['strength']:.1f}%",
                    "indicator": "MACD"
                })
            
            return signals
            
        except Exception as e:
            return []

class RealTimeBroker:
    """Broker that uses real market data instead of simulation"""
    
    def __init__(self):
        self.market_data = MarketDataProvider()
        self.positions = {}
    
    def get_price(self, symbol: str) -> float:
        """Get real-time market price"""
        return self.market_data.get_real_time_price(symbol)
    
    def get_technical_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get technical analysis for symbol"""
        return self.market_data.get_technical_indicators(symbol)
    
    def get_trading_signals(self, symbol: str) -> List[Dict[str, Any]]:
        """Get AI-generated trading signals"""
        return self.market_data.generate_trading_signals(symbol)
