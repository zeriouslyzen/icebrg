"""
Financial Data Pipeline for ICEBURG Elite Financial AI

This module provides comprehensive financial data ingestion, processing, and storage
capabilities for real-time and historical market data analysis.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import aiohttp
import yfinance as yf
from polygon import RESTClient
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MarketData:
    """Market data structure for financial information."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None


class DataPipeline:
    """
    Main data pipeline for financial data ingestion and processing.
    
    Supports multiple data sources:
    - Polygon API (real-time and historical)
    - Yahoo Finance (historical)
    - Alpha Vantage (historical)
    - Custom data sources
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize data pipeline with configuration."""
        self.config = config or {}
        self.data_sources = {}
        self.cache = {}
        self.data_dir = Path("data/financial")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data sources
        self._initialize_data_sources()
    
    def _initialize_data_sources(self):
        """Initialize available data sources."""
        # Polygon API
        polygon_api_key = os.getenv("POLYGON_API_KEY")
        if polygon_api_key:
            self.data_sources["polygon"] = PolygonAPI(polygon_api_key)
        
        # Yahoo Finance (always available)
        self.data_sources["yahoo"] = YahooFinanceAPI()
        
        # Alpha Vantage
        alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if alpha_vantage_key:
            self.data_sources["alpha_vantage"] = AlphaVantageAPI(alpha_vantage_key)
    
    async def get_real_time_data(self, symbols: List[str], source: str = "polygon") -> Dict[str, MarketData]:
        """Get real-time market data for given symbols."""
        if source not in self.data_sources:
            raise ValueError(f"Data source {source} not available")
        
        try:
            data_source = self.data_sources[source]
            real_time_data = await data_source.get_real_time_data(symbols)
            
            # Cache the data
            for symbol, data in real_time_data.items():
                self.cache[f"{symbol}_{source}"] = data
            
            return real_time_data
        
        except Exception as e:
            logger.error(f"Error getting real-time data: {e}")
            raise
    
    async def get_historical_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        source: str = "yahoo",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Get historical market data for given symbols."""
        if source not in self.data_sources:
            raise ValueError(f"Data source {source} not available")
        
        try:
            data_source = self.data_sources[source]
            historical_data = await data_source.get_historical_data(
                symbols, start_date, end_date, interval
            )
            
            # Cache the data
            for symbol, data in historical_data.items():
                cache_key = f"{symbol}_{source}_{start_date}_{end_date}"
                self.cache[cache_key] = data
            
            return historical_data
        
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            raise
    
    def get_cached_data(self, key: str) -> Optional[Any]:
        """Get cached data by key."""
        return self.cache.get(key)
    
    def save_data(self, data: Dict[str, pd.DataFrame], filename: str):
        """Save data to disk."""
        filepath = self.data_dir / filename
        try:
            with pd.HDFStore(filepath, mode='w') as store:
                for symbol, df in data.items():
                    store[f"/{symbol}"] = df
            logger.info(f"Data saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            raise
    
    def load_data(self, filename: str) -> Dict[str, pd.DataFrame]:
        """Load data from disk."""
        filepath = self.data_dir / filename
        try:
            data = {}
            with pd.HDFStore(filepath, mode='r') as store:
                for key in store.keys():
                    symbol = key.lstrip('/')
                    data[symbol] = store[key]
            logger.info(f"Data loaded from {filepath}")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise


class PolygonAPI:
    """Polygon API client for real-time and historical data."""
    
    def __init__(self, api_key: str):
        """Initialize Polygon API client."""
        self.api_key = api_key
        self.client = RESTClient(api_key)
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get real-time data for symbols."""
        real_time_data = {}
        
        for symbol in symbols:
            try:
                # Get real-time quote
                quote = self.client.get_last_trade_forex(symbol)
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=quote.p,
                    high=quote.p,
                    low=quote.p,
                    close=quote.p,
                    volume=1,
                    vwap=quote.p
                )
                
                real_time_data[symbol] = market_data
                
            except Exception as e:
                logger.warning(f"Error getting real-time data for {symbol}: {e}")
                continue
        
        return real_time_data
    
    async def get_historical_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data for symbols."""
        historical_data = {}
        
        for symbol in symbols:
            try:
                # Get historical data
                data = self.client.get_aggs(
                    symbol=symbol,
                    multiplier=1,
                    timespan="day" if interval == "1d" else "minute",
                    from_=start_date,
                    to=end_date
                )
                
                # Convert to DataFrame
                df = pd.DataFrame([
                    {
                        'timestamp': pd.to_datetime(agg.timestamp, unit='ms'),
                        'open': agg.open,
                        'high': agg.high,
                        'low': agg.low,
                        'close': agg.close,
                        'volume': agg.volume,
                        'vwap': agg.vwap
                    }
                    for agg in data
                ])
                
                df.set_index('timestamp', inplace=True)
                historical_data[symbol] = df
                
            except Exception as e:
                logger.warning(f"Error getting historical data for {symbol}: {e}")
                continue
        
        return historical_data


class YahooFinanceAPI:
    """Yahoo Finance API client for historical data."""
    
    def __init__(self):
        """Initialize Yahoo Finance client."""
        pass
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get real-time data for symbols (limited functionality)."""
        real_time_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                
                market_data = MarketData(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=info.get('open', 0),
                    high=info.get('dayHigh', 0),
                    low=info.get('dayLow', 0),
                    close=info.get('currentPrice', 0),
                    volume=info.get('volume', 0)
                )
                
                real_time_data[symbol] = market_data
                
            except Exception as e:
                logger.warning(f"Error getting real-time data for {symbol}: {e}")
                continue
        
        return real_time_data
    
    async def get_historical_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data for symbols."""
        historical_data = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=interval
                )
                
                # Clean column names
                df.columns = [col.lower() for col in df.columns]
                historical_data[symbol] = df
                
            except Exception as e:
                logger.warning(f"Error getting historical data for {symbol}: {e}")
                continue
        
        return historical_data


class AlphaVantageAPI:
    """Alpha Vantage API client for financial data."""
    
    def __init__(self, api_key: str):
        """Initialize Alpha Vantage client."""
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get real-time data for symbols."""
        # Implementation for Alpha Vantage real-time data
        # This is a placeholder - actual implementation would depend on Alpha Vantage API
        return {}
    
    async def get_historical_data(
        self, 
        symbols: List[str], 
        start_date: datetime, 
        end_date: datetime,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data for symbols."""
        # Implementation for Alpha Vantage historical data
        # This is a placeholder - actual implementation would depend on Alpha Vantage API
        return {}


# Example usage and testing
if __name__ == "__main__":
    async def test_data_pipeline():
        """Test the data pipeline functionality."""
        # Initialize pipeline
        pipeline = DataPipeline()
        
        # Test symbols
        symbols = ["AAPL", "GOOGL", "MSFT"]
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        historical_data = await pipeline.get_historical_data(
            symbols, start_date, end_date, source="yahoo"
        )
        
        print(f"Retrieved data for {len(historical_data)} symbols")
        for symbol, data in historical_data.items():
            print(f"{symbol}: {len(data)} records")
    
    # Run test
    asyncio.run(test_data_pipeline())
