"""
Feature Engineering for ICEBURG Elite Financial AI

This module provides comprehensive feature engineering capabilities for financial data,
including technical indicators, quantum features, and RL-specific features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Feature set structure for financial data."""
    technical_features: pd.DataFrame
    quantum_features: Optional[pd.DataFrame] = None
    rl_features: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = None


class FeatureEngineer:
    """
    Main feature engineering class for financial data.
    
    Combines technical indicators, quantum features, and RL-specific features
    for comprehensive financial analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize feature engineer with configuration."""
        self.config = config or {}
        self.technical_indicators = TechnicalIndicators()
        self.quantum_features = QuantumFeatures()
        self.rl_features = RLFeatures()
        self.scaler = StandardScaler()
        self.feature_selector = None
    
    def engineer_features(
        self, 
        market_data: Dict[str, pd.DataFrame],
        feature_types: List[str] = ["technical", "quantum", "rl"]
    ) -> FeatureSet:
        """
        Engineer features from market data.
        
        Args:
            market_data: Dictionary of symbol -> DataFrame
            feature_types: List of feature types to engineer
            
        Returns:
            FeatureSet with engineered features
        """
        try:
            # Initialize feature sets
            technical_features = {}
            quantum_features = {}
            rl_features = {}
            
            for symbol, data in market_data.items():
                logger.info(f"Engineering features for {symbol}")
                
                # Technical indicators
                if "technical" in feature_types:
                    tech_features = self.technical_indicators.calculate_all(data)
                    technical_features[symbol] = tech_features
                
                # Quantum features
                if "quantum" in feature_types:
                    quantum_feat = self.quantum_features.extract(data)
                    quantum_features[symbol] = quantum_feat
                
                # RL features
                if "rl" in feature_types:
                    rl_feat = self.rl_features.extract(data)
                    rl_features[symbol] = rl_feat
            
            # Combine features
            combined_technical = self._combine_features(technical_features)
            combined_quantum = self._combine_features(quantum_features) if quantum_features else None
            combined_rl = self._combine_features(rl_features) if rl_features else None
            
            # Create feature set
            feature_set = FeatureSet(
                technical_features=combined_technical,
                quantum_features=combined_quantum,
                rl_features=combined_rl,
                metadata={
                    "feature_types": feature_types,
                    "n_symbols": len(market_data),
                    "n_technical": len(combined_technical.columns) if combined_technical is not None else 0,
                    "n_quantum": len(combined_quantum.columns) if combined_quantum is not None else 0,
                    "n_rl": len(combined_rl.columns) if combined_rl is not None else 0
                }
            )
            
            return feature_set
        
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
            raise
    
    def _combine_features(self, features_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine features from multiple symbols."""
        if not features_dict:
            return pd.DataFrame()
        
        # Combine all features
        combined_features = []
        for symbol, features in features_dict.items():
            features_with_symbol = features.copy()
            features_with_symbol['symbol'] = symbol
            combined_features.append(features_with_symbol)
        
        return pd.concat(combined_features, ignore_index=True)
    
    def scale_features(self, features: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
        """Scale features using specified method."""
        try:
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            # Scale numeric columns only
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            scaled_features = features.copy()
            scaled_features[numeric_columns] = scaler.fit_transform(features[numeric_columns])
            
            return scaled_features
        
        except Exception as e:
            logger.error(f"Error scaling features: {e}")
            raise
    
    def select_features(
        self, 
        features: pd.DataFrame, 
        target: pd.Series, 
        k: int = 20,
        method: str = "f_regression"
    ) -> pd.DataFrame:
        """Select top k features using specified method."""
        try:
            if method == "f_regression":
                selector = SelectKBest(score_func=f_regression, k=k)
            else:
                raise ValueError(f"Unknown selection method: {method}")
            
            # Select features
            numeric_columns = features.select_dtypes(include=[np.number]).columns
            selected_features = selector.fit_transform(features[numeric_columns], target)
            
            # Get selected column names
            selected_columns = numeric_columns[selector.get_support()]
            selected_df = pd.DataFrame(selected_features, columns=selected_columns)
            
            return selected_df
        
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            raise


class TechnicalIndicators:
    """Technical indicators for financial data."""
    
    def __init__(self):
        """Initialize technical indicators."""
        self.indicators = {
            "sma": self._simple_moving_average,
            "ema": self._exponential_moving_average,
            "rsi": self._rsi,
            "macd": self._macd,
            "bollinger": self._bollinger_bands,
            "stochastic": self._stochastic_oscillator,
            "williams_r": self._williams_r,
            "cci": self._commodity_channel_index,
            "atr": self._average_true_range,
            "obv": self._on_balance_volume
        }
    
    def calculate_all(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        features = data.copy()
        
        for name, indicator_func in self.indicators.items():
            try:
                indicator_data = indicator_func(data)
                if isinstance(indicator_data, pd.DataFrame):
                    features = pd.concat([features, indicator_data], axis=1)
                elif isinstance(indicator_data, pd.Series):
                    features[name] = indicator_data
            except Exception as e:
                logger.warning(f"Error calculating {name}: {e}")
                continue
        
        return features
    
    def _simple_moving_average(self, data: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Calculate simple moving averages."""
        sma_data = {}
        for period in periods:
            sma_data[f"sma_{period}"] = data['close'].rolling(window=period).mean()
        return pd.DataFrame(sma_data)
    
    def _exponential_moving_average(self, data: pd.DataFrame, periods: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Calculate exponential moving averages."""
        ema_data = {}
        for period in periods:
            ema_data[f"ema_{period}"] = data['close'].ewm(span=period).mean()
        return pd.DataFrame(ema_data)
    
    def _rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _macd(self, data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = data['close'].ewm(span=fast).mean()
        ema_slow = data['close'].ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram
        })
    
    def _bollinger_bands(self, data: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            'bb_upper': upper_band,
            'bb_middle': sma,
            'bb_lower': lower_band,
            'bb_width': upper_band - lower_band,
            'bb_position': (data['close'] - lower_band) / (upper_band - lower_band)
        })
    
    def _stochastic_oscillator(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            'stoch_k': k_percent,
            'stoch_d': d_percent
        })
    
    def _williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high_max = data['high'].rolling(window=period).max()
        low_min = data['low'].rolling(window=period).min()
        williams_r = -100 * ((high_max - data['close']) / (high_max - low_min))
        return williams_r
    
    def _commodity_channel_index(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        return cci
    
    def _average_true_range(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _on_balance_volume(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On Balance Volume."""
        obv = np.zeros(len(data))
        obv[0] = data['volume'].iloc[0]
        
        for i in range(1, len(data)):
            if data['close'].iloc[i] > data['close'].iloc[i-1]:
                obv[i] = obv[i-1] + data['volume'].iloc[i]
            elif data['close'].iloc[i] < data['close'].iloc[i-1]:
                obv[i] = obv[i-1] - data['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]
        
        return pd.Series(obv, index=data.index)


class QuantumFeatures:
    """Quantum-inspired features for financial data."""
    
    def __init__(self):
        """Initialize quantum features."""
        pass
    
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract quantum-inspired features."""
        features = pd.DataFrame(index=data.index)
        
        # Quantum-inspired features
        features['quantum_entanglement'] = self._quantum_entanglement(data)
        features['quantum_superposition'] = self._quantum_superposition(data)
        features['quantum_interference'] = self._quantum_interference(data)
        features['quantum_tunneling'] = self._quantum_tunneling(data)
        
        return features
    
    def _quantum_entanglement(self, data: pd.DataFrame) -> pd.Series:
        """Calculate quantum entanglement-inspired feature."""
        # Correlation between price and volume
        price_volume_corr = data['close'].rolling(window=20).corr(data['volume'])
        return price_volume_corr
    
    def _quantum_superposition(self, data: pd.DataFrame) -> pd.Series:
        """Calculate quantum superposition-inspired feature."""
        # Weighted average of multiple timeframes
        short_ma = data['close'].rolling(window=5).mean()
        medium_ma = data['close'].rolling(window=20).mean()
        long_ma = data['close'].rolling(window=50).mean()
        
        # Superposition as weighted combination
        superposition = 0.5 * short_ma + 0.3 * medium_ma + 0.2 * long_ma
        return superposition
    
    def _quantum_interference(self, data: pd.DataFrame) -> pd.Series:
        """Calculate quantum interference-inspired feature."""
        # Interference pattern between price and momentum
        price_momentum = data['close'].pct_change()
        volume_momentum = data['volume'].pct_change()
        
        # Interference as product of momenta
        interference = price_momentum * volume_momentum
        return interference
    
    def _quantum_tunneling(self, data: pd.DataFrame) -> pd.Series:
        """Calculate quantum tunneling-inspired feature."""
        # Probability of price breaking through resistance/support
        high_20 = data['high'].rolling(window=20).max()
        low_20 = data['low'].rolling(window=20).min()
        
        # Tunneling probability
        tunneling = (data['close'] - low_20) / (high_20 - low_20)
        return tunneling


class RLFeatures:
    """RL-specific features for financial data."""
    
    def __init__(self):
        """Initialize RL features."""
        pass
    
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract RL-specific features."""
        features = pd.DataFrame(index=data.index)
        
        # RL-specific features
        features['action_space'] = self._action_space_features(data)
        features['reward_signal'] = self._reward_signal(data)
        features['state_representation'] = self._state_representation(data)
        features['exploration_bonus'] = self._exploration_bonus(data)
        
        return features
    
    def _action_space_features(self, data: pd.DataFrame) -> pd.Series:
        """Calculate action space features."""
        # Volatility as action space size
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        return volatility
    
    def _reward_signal(self, data: pd.DataFrame) -> pd.Series:
        """Calculate reward signal."""
        # Sharpe ratio as reward signal
        returns = data['close'].pct_change()
        sharpe = returns.rolling(window=20).mean() / returns.rolling(window=20).std()
        return sharpe
    
    def _state_representation(self, data: pd.DataFrame) -> pd.Series:
        """Calculate state representation."""
        # Normalized price as state
        price_normalized = (data['close'] - data['close'].rolling(window=50).mean()) / data['close'].rolling(window=50).std()
        return price_normalized
    
    def _exploration_bonus(self, data: pd.DataFrame) -> pd.Series:
        """Calculate exploration bonus."""
        # Uncertainty as exploration bonus
        returns = data['close'].pct_change()
        uncertainty = returns.rolling(window=20).std()
        return uncertainty


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Engineer features
    market_data = {"AAPL": data}
    feature_set = feature_engineer.engineer_features(market_data)
    
    print(f"Technical features shape: {feature_set.technical_features.shape}")
    print(f"Feature metadata: {feature_set.metadata}")
