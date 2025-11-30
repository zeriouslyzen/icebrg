"""
Technical Indicators for Elite Financial AI

This module provides technical indicator calculations for financial analysis.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime


class TechnicalIndicators:
    """Technical indicators calculator for financial data."""
    
    def __init__(self):
        self.indicators_data = {}
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            "upper": upper_band,
            "middle": sma,
            "lower": lower_band
        }
    
    def calculate_all_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all technical indicators for given data."""
        indicators = {}
        
        if 'close' in data.columns:
            close_prices = data['close']
            indicators['rsi'] = self.calculate_rsi(close_prices)
            indicators['macd'] = self.calculate_macd(close_prices)
            indicators['bollinger_bands'] = self.calculate_bollinger_bands(close_prices)
        
        return indicators