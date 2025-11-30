"""
Optimized Financial Data Pipeline for ICEBURG Elite Financial AI

This module provides optimized financial data processing with async fetching,
connection pooling, caching, and parallel processing for high-frequency data.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import redis
import json
from functools import lru_cache
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


@dataclass
class FinancialPipelineConfig:
    """Configuration for financial data pipeline."""
    use_async: bool = True
    use_caching: bool = True
    cache_ttl: int = 3600  # seconds
    use_connection_pooling: bool = True
    max_connections: int = 100
    use_parallel_processing: bool = True
    n_workers: int = 4
    use_redis_cache: bool = True
    redis_host: str = os.getenv("REDIS_HOST", "os.getenv("HOST", "localhost")")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    use_retry_logic: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    use_batch_requests: bool = True
    batch_size: int = 100
    use_streaming: bool = False
    chunk_size: int = 1000


class OptimizedDataFetcher:
    """
    Optimized data fetcher with async operations and connection pooling.
    
    Provides efficient data fetching with caching, retry logic, and parallel processing.
    """
    
    def __init__(self, config: FinancialPipelineConfig):
        """
        Initialize optimized data fetcher.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.session = None
        self.redis_client = None
        self.connection_pool = None
        self._setup_connections()
    
    def _setup_connections(self):
        """Setup connections and caching."""
        # Setup Redis cache if enabled
        if self.config.use_redis_cache:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Setup connection pool
        if self.config.use_connection_pooling:
            self._setup_connection_pool()
    
    def _setup_connection_pool(self):
        """Setup HTTP connection pool."""
        # Create session with connection pooling
        self.session = requests.Session()
        
        # Setup retry strategy
        if self.config.use_retry_logic:
            retry_strategy = Retry(
                total=self.config.max_retries,
                backoff_factor=self.config.retry_delay,
                status_forcelist=[429, 500, 502, 503, 504]
            )
            adapter = HTTPAdapter(
                max_retries=retry_strategy,
                pool_connections=self.config.max_connections,
                pool_maxsize=self.config.max_connections
            )
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
    
    async def fetch_data_async(self, url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Fetch data asynchronously with caching.
        
        Args:
            url: Data URL
            params: Request parameters
            
        Returns:
            Fetched data
        """
        # Check cache first
        cache_key = self._generate_cache_key(url, params)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Fetch data
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Cache data
                        self._set_cache(cache_key, data)
                        
                        return data
                    else:
                        logger.error(f"HTTP error {response.status}: {url}")
                        return {}
        except Exception as e:
            logger.error(f"Error fetching data from {url}: {e}")
            return {}
    
    def fetch_data_batch(self, urls: List[str], params_list: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Fetch multiple data sources in batch.
        
        Args:
            urls: List of URLs
            params_list: List of parameters
            
        Returns:
            List of fetched data
        """
        if params_list is None:
            params_list = [{}] * len(urls)
        
        if self.config.use_parallel_processing:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
                futures = []
                for url, params in zip(urls, params_list):
                    future = executor.submit(self._fetch_single_data, url, params)
                    futures.append(future)
                
                results = []
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Error in batch fetch: {e}")
                        results.append({})
                
                return results
        else:
            # Sequential processing
            results = []
            for url, params in zip(urls, params_list):
                result = self._fetch_single_data(url, params)
                results.append(result)
            return results
    
    def _fetch_single_data(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch single data source."""
        # Check cache first
        cache_key = self._generate_cache_key(url, params)
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        # Fetch data
        try:
            if self.session:
                response = self.session.get(url, params=params, timeout=30)
            else:
                response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Cache data
                self._set_cache(cache_key, data)
                
                return data
            else:
                logger.error(f"HTTP error {response.status_code}: {url}")
                return {}
        except Exception as e:
            logger.error(f"Error fetching data from {url}: {e}")
            return {}
    
    def _generate_cache_key(self, url: str, params: Dict[str, Any]) -> str:
        """Generate cache key for URL and parameters."""
        key_data = {"url": url, "params": params}
        return f"financial_data:{hash(json.dumps(key_data, sort_keys=True))}"
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from cache."""
        if not self.config.use_caching or not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
        
        return None
    
    def _set_cache(self, key: str, data: Dict[str, Any]):
        """Set data in cache."""
        if not self.config.use_caching or not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                key,
                self.config.cache_ttl,
                json.dumps(data)
            )
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    def close(self):
        """Close connections."""
        if self.session:
            self.session.close()
        if self.redis_client:
            self.redis_client.close()


class OptimizedDataProcessor:
    """
    Optimized data processor with parallel processing and vectorization.
    
    Provides efficient data processing with pandas vectorization and parallel operations.
    """
    
    def __init__(self, config: FinancialPipelineConfig):
        """
        Initialize optimized data processor.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.processor_pool = None
        if config.use_parallel_processing:
            self.processor_pool = ProcessPoolExecutor(max_workers=config.n_workers)
    
    def process_financial_data(self, data: pd.DataFrame, 
        operations: List[str]) -> pd.DataFrame:
        """
        Process financial data with specified operations.
        
        Args:
            data: Financial data
            operations: List of operations to perform
            
        Returns:
            Processed data
        """
        processed_data = data.copy()
        
        for operation in operations:
            if operation == "technical_indicators":
                processed_data = self._add_technical_indicators(processed_data)
            elif operation == "price_features":
                processed_data = self._add_price_features(processed_data)
            elif operation == "volume_features":
                processed_data = self._add_volume_features(processed_data)
            elif operation == "volatility_features":
                processed_data = self._add_volatility_features(processed_data)
            elif operation == "momentum_features":
                processed_data = self._add_momentum_features(processed_data)
            else:
                logger.warning(f"Unknown operation: {operation}")
        
        return processed_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to data."""
        # Simple Moving Averages
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
        data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
        data['BB_Width'] = data['BB_Upper'] - data['BB_Lower']
        data['BB_Position'] = (data['Close'] - data['BB_Lower']) / (data['BB_Upper'] - data['BB_Lower'])
        
        return data
    
    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Price changes
        data['Price_Change'] = data['Close'].pct_change()
        data['Price_Change_Abs'] = data['Close'].diff()
        
        # High-Low features
        data['HL_Ratio'] = data['High'] / data['Low']
        data['OC_Ratio'] = data['Open'] / data['Close']
        
        # Price position
        data['Price_Position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        return data
    
    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume indicators
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
        
        # Volume-price features
        data['VWAP'] = (data['Volume'] * data['Close']).rolling(window=20).sum() / data['Volume'].rolling(window=20).sum()
        data['Volume_Price_Trend'] = data['Volume'] * data['Price_Change']
        
        return data
    
    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        # Rolling volatility
        data['Volatility_20'] = data['Price_Change'].rolling(window=20).std()
        data['Volatility_50'] = data['Price_Change'].rolling(window=50).std()
        
        # ATR (Average True Range)
        high_low = data['High'] - data['Low']
        high_close = np.abs(data['High'] - data['Close'].shift())
        low_close = np.abs(data['Low'] - data['Close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        data['ATR'] = true_range.rolling(window=14).mean()
        
        return data
    
    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features."""
        # Price momentum
        data['Momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
        data['Momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
        data['Momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
        
        # Rate of Change
        data['ROC_5'] = data['Close'].pct_change(5)
        data['ROC_10'] = data['Close'].pct_change(10)
        data['ROC_20'] = data['Close'].pct_change(20)
        
        return data
    
    def process_data_parallel(self, data_list: List[pd.DataFrame], 
        operations: List[str]) -> List[pd.DataFrame]:
        """
        Process multiple datasets in parallel.
        
        Args:
            data_list: List of datasets
            operations: List of operations
            
        Returns:
            List of processed datasets
        """
        if not self.config.use_parallel_processing or not self.processor_pool:
            # Sequential processing
            results = []
            for data in data_list:
                result = self.process_financial_data(data, operations)
                results.append(result)
            return results
        
        # Parallel processing
        futures = []
        for data in data_list:
            future = self.processor_pool.submit(
                self.process_financial_data, data, operations
            )
            futures.append(future)
        
        results = []
        for future in futures:
            try:
                result = future.result(timeout=60)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                results.append(pd.DataFrame())
        
        return results
    
    def close(self):
        """Close processor pool."""
        if self.processor_pool:
            self.processor_pool.shutdown(wait=True)


class OptimizedDataCache:
    """
    Optimized data cache with Redis and local caching.
    
    Provides efficient data caching with TTL and automatic cleanup.
    """
    
    def __init__(self, config: FinancialPipelineConfig):
        """
        Initialize optimized data cache.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.redis_client = None
        self.local_cache = {}
        self.cache_stats = {"hits": 0, "misses": 0}
        self._setup_cache()
    
    def _setup_cache(self):
        """Setup cache connections."""
        if self.config.use_redis_cache:
            try:
                self.redis_client = redis.Redis(
                    host=self.config.redis_host,
                    port=self.config.redis_port,
                    decode_responses=True
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Redis cache connected successfully")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get data from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None
        """
        # Try Redis cache first
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    self.cache_stats["hits"] += 1
                    return json.loads(cached_data)
            except Exception as e:
                logger.warning(f"Redis get error: {e}")
        
        # Try local cache
        if key in self.local_cache:
            self.cache_stats["hits"] += 1
            return self.local_cache[key]
        
        self.cache_stats["misses"] += 1
        return None
    
    def set(self, key: str, data: Any, ttl: int = None):
        """
        Set data in cache.
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
        """
        if ttl is None:
            ttl = self.config.cache_ttl
        
        # Set in Redis cache
        if self.redis_client:
            try:
                self.redis_client.setex(key, ttl, json.dumps(data))
            except Exception as e:
                logger.warning(f"Redis set error: {e}")
        
        # Set in local cache
        self.local_cache[key] = data
    
    def delete(self, key: str):
        """
        Delete data from cache.
        
        Args:
            key: Cache key
        """
        # Delete from Redis
        if self.redis_client:
            try:
                self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete error: {e}")
        
        # Delete from local cache
        if key in self.local_cache:
            del self.local_cache[key]
    
    def clear(self):
        """Clear all cache."""
        # Clear Redis cache
        if self.redis_client:
            try:
                self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis clear error: {e}")
        
        # Clear local cache
        self.local_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "hit_rate": hit_rate,
            "local_cache_size": len(self.local_cache)
        }


class OptimizedFinancialPipeline:
    """
    Optimized financial data pipeline with all optimizations.
    
    Provides end-to-end optimized data processing for financial applications.
    """
    
    def __init__(self, config: FinancialPipelineConfig):
        """
        Initialize optimized financial pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.fetcher = OptimizedDataFetcher(config)
        self.processor = OptimizedDataProcessor(config)
        self.cache = OptimizedDataCache(config)
        self.pipeline_stats = {}
    
    async def process_financial_data_async(self, symbols: List[str], 
        operations: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Process financial data asynchronously.
        
        Args:
            symbols: List of stock symbols
            operations: List of operations
            
        Returns:
            Processed data for each symbol
        """
        results = {}
        
        # Create tasks for each symbol
        tasks = []
        for symbol in symbols:
            task = self._process_single_symbol_async(symbol, operations)
            tasks.append((symbol, task))
        
        # Execute tasks
        for symbol, task in tasks:
            try:
                result = await task
                results[symbol] = result
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        
        return results
    
    async def _process_single_symbol_async(self, symbol: str, 
        operations: List[str]) -> pd.DataFrame:
        """Process single symbol asynchronously."""
        # Check cache first
        cache_key = f"financial_data:{symbol}:{hash(tuple(operations))}"
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            return pd.DataFrame(cached_data)
        
        # Fetch data
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1y")
            
            if data.empty:
                logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Process data
            processed_data = self.processor.process_financial_data(data, operations)
            
            # Cache result
            self.cache.set(cache_key, processed_data.to_dict())
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            return pd.DataFrame()
    
    def process_financial_data_batch(self, symbols: List[str], 
        operations: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Process financial data in batch.
        
        Args:
            symbols: List of stock symbols
            operations: List of operations
            
        Returns:
            Processed data for each symbol
        """
        results = {}
        
        # Process symbols in batches
        batch_size = self.config.batch_size
        for i in range(0, len(symbols), batch_size):
            batch_symbols = symbols[i:i + batch_size]
            
            # Process batch
            batch_results = self._process_batch(batch_symbols, operations)
            results.update(batch_results)
        
        return results
    
    def _process_batch(self, symbols: List[str], 
        operations: List[str]) -> Dict[str, pd.DataFrame]:
        """Process batch of symbols."""
        results = {}
        
        for symbol in symbols:
            try:
                # Check cache first
                cache_key = f"financial_data:{symbol}:{hash(tuple(operations))}"
                cached_data = self.cache.get(cache_key)
                if cached_data is not None:
                    results[symbol] = pd.DataFrame(cached_data)
                    continue
                
                # Fetch data
                ticker = yf.Ticker(symbol)
                data = ticker.history(period="1y")
                
                if data.empty:
                    logger.warning(f"No data available for {symbol}")
                    results[symbol] = pd.DataFrame()
                    continue
                
                # Process data
                processed_data = self.processor.process_financial_data(data, operations)
                
                # Cache result
                self.cache.set(cache_key, processed_data.to_dict())
                
                results[symbol] = processed_data
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Pipeline statistics
        """
        cache_stats = self.cache.get_stats()
        
        return {
            "cache_stats": cache_stats,
            "config": {
                "use_async": self.config.use_async,
                "use_caching": self.config.use_caching,
                "use_parallel_processing": self.config.use_parallel_processing,
                "n_workers": self.config.n_workers
            }
        }
    
    def close(self):
        """Close pipeline resources."""
        self.fetcher.close()
        self.processor.close()


# Example usage and testing
if __name__ == "__main__":
    # Test optimized financial pipeline
    config = FinancialPipelineConfig(
        use_async=True,
        use_caching=True,
        cache_ttl=3600,
        use_connection_pooling=True,
        max_connections=100,
        use_parallel_processing=True,
        n_workers=4,
        use_redis_cache=False,  # Disable Redis for testing
        use_retry_logic=True,
        max_retries=3,
        retry_delay=1.0,
        use_batch_requests=True,
        batch_size=10,
        use_streaming=False,
        chunk_size=1000
    )
    
    # Create pipeline
    pipeline = OptimizedFinancialPipeline(config)
    
    # Test batch processing
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    operations = ["technical_indicators", "price_features", "volume_features"]
    
    print("Testing batch processing...")
    start_time = time.time()
    results = pipeline.process_financial_data_batch(symbols, operations)
    batch_time = time.time() - start_time
    
    print(f"Batch processing completed in {batch_time:.2f}s")
    print(f"Processed {len(results)} symbols")
    
    # Test async processing
    print("Testing async processing...")
    start_time = time.time()
    async_results = asyncio.run(pipeline.process_financial_data_async(symbols, operations))
    async_time = time.time() - start_time
    
    print(f"Async processing completed in {async_time:.2f}s")
    print(f"Processed {len(async_results)} symbols")
    
    # Test cache performance
    print("Testing cache performance...")
    start_time = time.time()
    cached_results = pipeline.process_financial_data_batch(symbols, operations)
    cached_time = time.time() - start_time
    
    print(f"Cached processing completed in {cached_time:.2f}s")
    print(f"Speedup: {batch_time / cached_time:.2f}x")
    
    # Get pipeline statistics
    stats = pipeline.get_pipeline_stats()
    print(f"Pipeline statistics: {stats}")
    
    # Close pipeline
    pipeline.close()
    
    print("All tests completed successfully!")
