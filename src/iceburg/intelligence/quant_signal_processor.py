"""
Quantitative Signal Processor
Converts intelligence signals into tradeable alpha signals

This is the bridge between intelligence gathering and quantitative execution.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from ..intelligence.multi_source_aggregator import IntelligenceSignal, CorrelatedIntelligence

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Types of trading signals"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    STATISTICAL_ARBITRAGE = "stat_arb"
    EVENT_DRIVEN = "event_driven"
    SENTIMENT = "sentiment"
    REGIME_CHANGE = "regime_change"


@dataclass
class AlphaSignal:
    """Quantitative alpha signal ready for execution"""
    signal_id: str
    symbol: str
    signal_type: SignalType
    direction: int  # 1 = long, -1 = short, 0 = neutral
    strength: float  # 0.0 to 1.0
    confidence: float  # Statistical confidence
    expected_return: float  # Expected % return
    sharpe_estimate: float  # Expected Sharpe ratio
    time_horizon: str  # "1h", "1d", "1w", "1m"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size: Optional[float] = None  # % of portfolio
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StatArbPair:
    """Statistical arbitrage pair"""
    symbol_a: str
    symbol_b: str
    spread: float
    z_score: float
    half_life: float  # Mean reversion half-life
    correlation: float
    cointegration_p_value: float


class QuantitativeSignalProcessor:
    """
    Converts intelligence signals into quantitative alpha signals.
    
    Uses statistical models, machine learning, and quant techniques
    to generate tradeable signals from raw intelligence.
    """
    
    def __init__(self):
        self.alpha_signals: List[AlphaSignal] = []
        self.stat_arb_pairs: List[StatArbPair] = []
        
        # Historical data cache (symbol -> price history)
        self.price_history: Dict[str, np.ndarray] = {}
        
        logger.info("Quant Signal Processor initialized")
    
    def intelligence_to_alpha(
        self,
        intelligence: CorrelatedIntelligence,
        market_data: Optional[Dict[str, Any]] = None
    ) -> List[AlphaSignal]:
        """
        Convert correlated intelligence into tradeable alpha signals.
        
        This is the core quantitative conversion logic.
        
        Args:
            intelligence: Correlated intelligence narrative
            market_data: Optional market data for context
            
        Returns:
            List of alpha signals ready for execution
        """
        signals = []
        
        # Extract symbols from entities
        symbols = self._extract_symbols(intelligence.entities)
        
        for symbol in symbols:
            # Determine signal type based on intelligence content
            signal_type = self._classify_signal_type(intelligence)
            
            # Calculate quantitative metrics
            direction = self._calculate_direction(intelligence, symbol, signal_type)
            strength = self._calculate_signal_strength(intelligence, symbol)
            confidence = self._calculate_statistical_confidence(intelligence, symbol)
            
            # Risk/reward calculation
            expected_return, sharpe = self._estimate_return_and_sharpe(
                symbol, direction, strength, signal_type
            )
            
            # Position sizing via Kelly Criterion
            position_size = self._kelly_position_size(
                expected_return, confidence, sharpe
            )
            
            # Stop loss / take profit levels
            stop_loss, take_profit = self._calculate_risk_levels(
                symbol, direction, expected_return, sharpe
            )
            
            alpha_signal = AlphaSignal(
                signal_id=f"alpha_{intelligence.correlation_id}_{symbol}",
                symbol=symbol,
                signal_type=signal_type,
                direction=direction,
                strength=strength,
                confidence=confidence,
                expected_return=expected_return,
                sharpe_estimate=sharpe,
                time_horizon=intelligence.timeframe or "1d",
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "intelligence_id": intelligence.correlation_id,
                    "narrative": intelligence.narrative,
                    "signal_count": len(intelligence.signals)
                }
            )
            
            signals.append(alpha_signal)
            self.alpha_signals.append(alpha_signal)
        
        logger.info(f"Generated {len(signals)} alpha signals from intelligence")
        return signals
    
    def detect_statistical_arbitrage(
        self,
        symbols: List[str],
        lookback_days: int = 60
    ) -> List[StatArbPair]:
        """
        Detect statistical arbitrage opportunities via pairs trading.
        
        Uses cointegration, correlation, and mean reversion analysis.
        
        Args:
            symbols: List of symbols to analyze
            lookback_days: Historical data lookback period
            
        Returns:
            List of stat arb pairs
        """
        pairs = []
        
        # Pairwise cointegration testing
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol_a, symbol_b = symbols[i], symbols[j]
                
                # Get price history
                prices_a = self._get_price_history(symbol_a, lookback_days)
                prices_b = self._get_price_history(symbol_b, lookback_days)
                
                if prices_a is None or prices_b is None:
                    continue
                
                # Calculate metrics
                correlation = np.corrcoef(prices_a, prices_b)[0, 1]
                
                # Cointegration test (simplified - would use proper Engle-Granger)
                spread = prices_a - prices_b
                spread_mean = np.mean(spread)
                spread_std = np.std(spread)
                z_score = (spread[-1] - spread_mean) / spread_std if spread_std > 0 else 0
                
                # Mean reversion half-life (simplified)
                half_life = self._calculate_half_life(spread)
                
                # Mock p-value (would use actual cointegration test)
                coint_p_value = 0.05 if abs(correlation) > 0.7 else 0.5
                
                # Only add if statistically significant
                if coint_p_value < 0.05 and abs(z_score) > 2:
                    pair = StatArbPair(
                        symbol_a=symbol_a,
                        symbol_b=symbol_b,
                        spread=spread[-1],
                        z_score=z_score,
                        half_life=half_life,
                        correlation=correlation,
                        cointegration_p_value=coint_p_value
                    )
                    pairs.append(pair)
        
        self.stat_arb_pairs = pairs
        logger.info(f"Detected {len(pairs)} stat arb pairs")
        return pairs
    
    def calculate_portfolio_risk(
        self,
        signals: List[AlphaSignal],
        total_capital: float = 100000.0
    ) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics.
        
        Returns:
            Risk metrics including VaR, CVaR, max drawdown, etc.
        """
        if not signals:
            return {}
        
        # Calculate position-weighted metrics
        total_exposure = sum(s.position_size or 0 for s in signals)
        
        # Value at Risk (95% confidence, 1-day)
        returns = np.array([s.expected_return for s in signals])
        weights = np.array([s.position_size or 0 for s in signals])
        portfolio_return = np.dot(returns, weights)
        portfolio_std = np.std(returns) * np.sqrt(np.sum(weights**2))
        var_95 = -1.65 * portfolio_std * total_capital  # 95% VaR
        
        # Conditional VaR (Expected Shortfall)
        cvar_95 = var_95 * 1.3  # Approximation
        
        # Portfolio Sharpe estimate
        avg_sharpe = np.mean([s.sharpe_estimate for s in signals])
        
        return {
            "total_exposure": total_exposure,
            "portfolio_expected_return": portfolio_return,
            "portfolio_std": portfolio_std,
            "var_95_1d": var_95,
            "cvar_95_1d": cvar_95,
            "avg_sharpe": avg_sharpe,
            "signal_count": len(signals)
        }
    
    def _extract_symbols(self, entities: List[str]) -> List[str]:
        """Extract trading symbols from entity list."""
        # Simple heuristic - uppercase 1-5 char entities are likely tickers
        symbols = [e for e in entities if e.isupper() and 1 <= len(e) <= 5]
        return symbols if symbols else []
    
    def _classify_signal_type(self, intelligence: CorrelatedIntelligence) -> SignalType:
        """Classify intelligence into signal type."""
        content = intelligence.narrative.lower()
        
        if "earnings" in content or "announcement" in content:
            return SignalType.EVENT_DRIVEN
        elif "sentiment" in content or "fear" in content:
            return SignalType.SENTIMENT
        elif "regime" in content or "shift" in content:
            return SignalType.REGIME_CHANGE
        else:
            return SignalType.MOMENTUM  # Default
    
    def _calculate_direction(
        self,
        intelligence: CorrelatedIntelligence,
        symbol: str,
        signal_type: SignalType
    ) -> int:
        """Calculate trade direction from intelligence."""
        # Analyze narrative sentiment
        narrative = intelligence.narrative.lower()
        
        positive_words = ["bullish", "surge", "growth", "positive", "buy"]
        negative_words = ["bearish", "decline", "negative", "sell", "crash"]
        
        pos_score = sum(1 for word in positive_words if word in narrative)
        neg_score = sum(1 for word in negative_words if word in narrative)
        
        if pos_score > neg_score:
            return 1  # Long
        elif neg_score > pos_score:
            return -1  # Short
        else:
            return 0  # Neutral
    
    def _calculate_signal_strength(
        self,
        intelligence: CorrelatedIntelligence,
        symbol: str
    ) -> float:
        """Calculate signal strength 0-1."""
        # Based on intelligence confidence and signal count
        base_strength = intelligence.confidence
        signal_boost = min(len(intelligence.signals) / 10, 0.3)
        return min(base_strength + signal_boost, 1.0)
    
    def _calculate_statistical_confidence(
        self,
        intelligence: CorrelatedIntelligence,
        symbol: str
    ) -> float:
        """Calculate statistical confidence level."""
        # Multiple signal sources increase confidence
        source_diversity = len(set(s.source_type for s in intelligence.signals))
        confidence = 0.5 + (source_diversity * 0.1)
        return min(confidence, 0.95)
    
    def _estimate_return_and_sharpe(
        self,
        symbol: str,
        direction: int,
        strength: float,
        signal_type: SignalType
    ) -> Tuple[float, float]:
        """Estimate expected return and Sharpe ratio."""
        # Base return estimate on signal strength
        base_return = strength * 0.1 * direction  # 10% max expected return
        
        # Adjust by signal type
        type_multipliers = {
            SignalType.EVENT_DRIVEN: 1.5,
            SignalType.REGIME_CHANGE: 2.0,
            SignalType.STATISTICAL_ARBITRAGE: 0.8,
            SignalType.MOMENTUM: 1.0,
            SignalType.MEAN_REVERSION: 0.9,
            SignalType.SENTIMENT: 0.7
        }
        
        expected_return = base_return * type_multipliers.get(signal_type, 1.0)
        
        # Estimate Sharpe (assuming volatility)
        estimated_vol = 0.02  # 2% daily vol assumption
        sharpe = (expected_return / estimated_vol) if estimated_vol > 0 else 0
        
        return expected_return, sharpe
    
    def _kelly_position_size(
        self,
        expected_return: float,
        confidence: float,
        sharpe: float
    ) -> float:
        """Calculate position size using Kelly Criterion."""
        # Kelly = (p*b - q) / b, where p=win_prob, q=lose_prob, b=win/loss_ratio
        win_prob = 0.5 + (confidence * 0.4)  # 50-90% based on confidence
        win_ratio = abs(expected_return) * 1.5
        
        kelly_fraction = (win_prob * win_ratio - (1 - win_prob)) / win_ratio
        
        # Use fractional Kelly (0.25) for safety
        position_size = max(0, min(kelly_fraction * 0.25, 0.2))  # Cap at 20%
        
        return position_size
    
    def _calculate_risk_levels(
        self,
        symbol: str,
        direction: int,
        expected_return: float,
        sharpe: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        # Stop loss at 2x expected volatility
        stop_loss_pct = 0.02 * 2  # 4% stop loss
        
        # Take profit at expected return
        take_profit_pct = abs(expected_return)
        
        # Would need current price - returning percentages for now
        return stop_loss_pct, take_profit_pct
    
    def _get_price_history(
        self,
        symbol: str,
        lookback_days: int
    ) -> Optional[np.ndarray]:
        """Get price history for symbol."""
        # Placeholder - would integrate with market data provider
        # Return synthetic data for now
        if symbol not in self.price_history:
            # Generate synthetic mean-reverting prices
            prices = 100 + np.cumsum(np.random.randn(lookback_days) * 0.02)
            self.price_history[symbol] = prices
        
        return self.price_history[symbol]
    
    def _calculate_half_life(self, spread: np.ndarray) -> float:
        """Calculate mean reversion half-life."""
        # AR(1) regression: spread_t = alpha + beta*spread_{t-1} + epsilon
        lag_spread = spread[:-1]
        diff_spread = np.diff(spread)
        
        if len(lag_spread) > 0:
            beta = np.dot(diff_spread, lag_spread) / np.dot(lag_spread, lag_spread)
            half_life = -np.log(2) / np.log(abs(beta)) if beta != 0 else 1.0
            return max(half_life, 0.1)
        
        return 1.0


# Global processor instance
_processor: Optional[QuantitativeSignalProcessor] = None


def get_quant_processor() -> QuantitativeSignalProcessor:
    """Get or create global quant signal processor."""
    global _processor
    if _processor is None:
        _processor = QuantitativeSignalProcessor()
    return _processor
