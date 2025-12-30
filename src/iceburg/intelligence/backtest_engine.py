"""
Backtesting Framework for V2 Prediction System
Robust backtesting with walk-forward analysis and risk metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

from .quant_signal_processor import AlphaSignal

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""
    trade_id: str
    symbol: str
    direction: int  # 1=long, -1=short
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 1.0
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    
    def close(self, exit_price: float, exit_time: datetime):
        """Close the trade"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        price_change = (exit_price - self.entry_price) / self.entry_price
        self.pnl_pct = price_change * self.direction
        self.pnl = self.pnl_pct * (self.entry_price * self.quantity)


@dataclass
class BacktestResults:
    """Backtest performance metrics"""
    total_pnl: float
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    trades: List[Trade]
    equity_curve: np.ndarray


class BacktestEngine:
    """
    Rigorous backtesting engine for quant strategies.
    
    Features:
    - Walk-forward analysis
    - Transaction costs
    - Slippage modeling
    - Risk metrics (Sharpe, Sortino, Calmar)
    - Monte Carlo simulation
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_bps: float = 5.0,  # 5 basis points
        slippage_bps: float = 2.0
    ):
        self.initial_capital = initial_capital
        self.commission_bps = commission_bps / 10000
        self.slippage_bps = slippage_bps / 10000
        
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        
        logger.info(f"Backtest engine initialized with ${initial_capital:,.0f}")
    
    def run_backtest(
        self,
        signals: List[AlphaSignal],
        price_data: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResults:
        """
        Run backtest on historical data.
        
        Args:
            signals: List of alpha signals to backtest
            price_data: Historical price data (symbol -> DataFrame with OHLC)
            start_date: Backtest start date
            end_date: Backtest end date
            
        Returns:
            Backtest results with performance metrics
        """
        self.trades = []
        self.equity_curve = [self.initial_capital]
        
        current_capital = self.initial_capital
        open_positions: Dict[str, Trade] = {}
        
        # Sort signals by timestamp
        sorted_signals = sorted(signals, key=lambda s: s.timestamp)
        
        for signal in sorted_signals:
            if signal.timestamp < start_date or signal.timestamp > end_date:
                continue
            
            symbol = signal.symbol
            
            # Get price at signal time (with slippage)
            entry_price = self._get_price_with_slippage(
                symbol, signal.timestamp, signal.direction, price_data
            )
            
            if entry_price is None:
                continue
            
            # Calculate position size
            position_value = current_capital * (signal.position_size or 0.05)
            quantity = position_value / entry_price
            
            # Account for commission
            commission = position_value * self.commission_bps
            current_capital -= commission
            
            # Open trade
            trade = Trade(
                trade_id=f"trade_{len(self.trades)}",
                symbol=symbol,
                direction=signal.direction,
                entry_price=entry_price,
                quantity=quantity,
                entry_time=signal.timestamp,
                fees=commission
            )
            
            open_positions[symbol] = trade
            
            # Check for exit signals (stop loss / take profit)
            exit_time =  signal.timestamp + timedelta(days=1)  # 1-day holding
            exit_price = self._get_price_with_slippage(
                symbol, exit_time, -signal.direction, price_data
            )
            
            if exit_price:
                trade.close(exit_price, exit_time)
                current_capital += trade.pnl - (position_value * self.commission_bps)
                self.trades.append(trade)
                self.equity_curve.append(current_capital)
                del open_positions[symbol]
        
        # Close remaining positions
        for symbol, trade in open_positions.items():
            exit_price = self._get_latest_price(symbol, end_date, price_data)
            if exit_price:
                trade.close(exit_price, end_date)
                self.trades.append(trade)
        
        # Calculate metrics
        return self._calculate_metrics()
    
    def walk_forward_analysis(
        self,
        signals: List[AlphaSignal],
        price_data: Dict[str, pd.DataFrame],
        train_window_days: int = 180,
        test_window_days: int = 60
    ) -> List[BacktestResults]:
        """
        Perform walk-forward analysis to avoid overfitting.
        
        Returns:
            List of backtest results for each window
        """
        results = []
        
        # Get date range
        all_dates = []
        for df in price_data.values():
            all_dates.extend(df.index.tolist())
        
        if not all_dates:
            return results
        
        start_date = min(all_dates)
        end_date = max(all_dates)
        
        current_date = start_date
        while current_date < end_date:
            test_start = current_date + timedelta(days=train_window_days)
            test_end = test_start + timedelta(days=test_window_days)
            
            if test_end > end_date:
                break
            
            # Run backtest on test window
            window_result = self.run_backtest(
                signals, price_data, test_start, test_end
            )
            results.append(window_result)
            
            current_date = test_end
        
        logger.info(f"Walk-forward analysis complete: {len(results)} windows")
        return results
    
    def monte_carlo_simulation(
        self,
        base_results: BacktestResults,
        n_simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation to estimate confidence intervals.
        
        Returns:
            Statistics on simulated outcomes
        """
        if not base_results.trades:
            return {}
        
        returns = [t.pnl_pct for t in base_results.trades]
        
        # Bootstrap resampling
        simulated_sharpes = []
        simulated_returns = []
        
        for _ in range(n_simulations):
            # Resample returns with replacement
            sim_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Calculate metrics
            sim_mean = np.mean(sim_returns)
            sim_std = np.std(sim_returns)
            sim_sharpe = (sim_mean / sim_std) * np.sqrt(252) if sim_std > 0 else 0
            
            simulated_sharpes.append(sim_sharpe)
            simulated_returns.append(sim_mean * len(returns))
        
        return {
            "sharpe_mean": np.mean(simulated_sharpes),
            "sharpe_5th_percentile": np.percentile(simulated_sharpes, 5),
            "sharpe_95th_percentile": np.percentile(simulated_sharpes, 95),
            "return_mean": np.mean(simulated_returns),
            "return_5th_percentile": np.percentile(simulated_returns, 5),
            "return_95th_percentile": np.percentile(simulated_returns, 95)
        }
    
    def _calculate_metrics(self) -> BacktestResults:
        """Calculate performance metrics from trades."""
        if not self.trades:
            return BacktestResults(
                total_pnl=0, total_return=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, win_rate=0, profit_factor=0,
                total_trades=0, winning_trades=0, losing_trades=0,
                avg_win=0, avg_loss=0, trades=[], equity_curve=np.array([])
            )
        
        # Basic metrics
        total_pnl = sum(t.pnl for t in self.trades)
        total_return = total_pnl / self.initial_capital
        
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Sharpe ratio
        returns = np.array([t.pnl_pct for t in self.trades])
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        sortino = (np.mean(returns) / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        # Max drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return BacktestResults(
            total_pnl=total_pnl,
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades=self.trades,
            equity_curve=equity_array
        )
    
    def _get_price_with_slippage(
        self,
        symbol: str,
        timestamp: datetime,
        direction: int,
        price_data: Dict[str, pd.DataFrame]
    ) -> Optional[float]:
        """Get price with slippage applied."""
        price = self._get_latest_price(symbol, timestamp, price_data)
        if price:
            slippage = price * self.slippage_bps * direction
            return price + slippage
        return None
    
    def _get_latest_price(
        self,
        symbol: str,
        timestamp: datetime,
        price_data: Dict[str, pd.DataFrame]
    ) -> Optional[float]:
        """Get latest price for symbol."""
        if symbol not in price_data:
            return None
        
        df = price_data[symbol]
        
        # Find closest timestamp
        try:
            idx = df.index.get_indexer([timestamp], method='nearest')[0]
            return df.iloc[idx]['close']
        except:
            return None
