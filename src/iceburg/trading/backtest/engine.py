from typing import Callable, Dict, List
import numpy as np

from ...portfolio.portfolio_manager import PortfolioManager
from ...risk.risk_manager import RiskManager, RiskConfig
from ...execution.trade_executor import TradeExecutor
from ...signal.oracle_bridge import TradeSignal
from ...orchestrator import TradingOrchestrator
from ...sim.paper_broker import PaperBroker

class BacktestEngine:
    def __init__(self, config: Dict[str, any], ohlcv: Dict[str, List[Dict[str, float]]]) -> None:
        self.config = config
        self.ohlcv = ohlcv
        self.symbols = list(ohlcv.keys())
        initial_prices = {s: ohlcv[s][0]['close'] for s in self.symbols}
        self.broker = PaperBroker(initial_prices=initial_prices, seed=42)
        risk = RiskManager(RiskConfig(**config.get("risk", {})))
        portfolio = PortfolioManager(starting_cash_usd=config.get("starting_cash_usd", 100000.0))
        executor = TradeExecutor(self.broker)
        self.orch = TradingOrchestrator(config, self.broker, risk, portfolio, executor)

    def run(self, signal_generator: Callable[[Dict[str, Dict[str, float]]], List[TradeSignal]]) -> Dict[str, any]:
        equity_curve = []
        max_bars = max(len(self.ohlcv[s]) for s in self.symbols)
        for i in range(max_bars):
            current_bar = {s: self.ohlcv[s][min(i, len(self.ohlcv[s])-1)] for s in self.symbols}
            # Update broker prices to current close
            for s in self.symbols:
                self.broker.state[s].price = current_bar[s]['close']
            signals = signal_generator(current_bar)
            self.orch.process_signals(signals)
            prices = {s: self.broker.get_price(s) for s in self.symbols}
            equity = self.orch.portfolio_manager.get_equity(prices)
            equity_curve.append({"timestamp": current_bar[self.symbols[0]]['timestamp'], "equity": equity})

        returns = np.diff([e['equity'] for e in equity_curve]) / [e['equity'] for e in equity_curve[:-1]]
        total_return = (equity_curve[-1]['equity'] / equity_curve[0]['equity'] - 1) if equity_curve else 0.0
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0.0  # Annualized
        cum_returns = np.cumprod(1 + returns) - 1
        peak = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - peak) / (peak + 1e-9)
        max_dd = np.min(drawdown) if len(drawdown) > 0 else 0.0

        metrics = {"total_return": total_return, "sharpe": sharpe, "max_drawdown": max_dd}
        return {"equity_curve": equity_curve, "metrics": metrics}
