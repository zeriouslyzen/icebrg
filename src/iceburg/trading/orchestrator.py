import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .signal.oracle_bridge import TradeSignal, parse_oracle_input
try:
    from ...business.business_mode import BusinessMode
except ImportError:
    # Fallback for when running as module
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent.parent))
    from src.iceburg.business.business_mode import BusinessMode


@dataclass
class TradeDecision:
    symbol: str
    side: str
    qty: float
    price: float
    status: str
    reason: str
    timestamp: float


class TradingOrchestrator:
    """Coordinates signal intake, risk checks, execution, and reporting."""

    def __init__(
        self,
        config: Dict[str, Any],
        broker,
        risk_manager,
        portfolio_manager,
        executor,
    ) -> None:
        self.config = config
        self.broker = broker
        self.risk_manager = risk_manager
        self.portfolio_manager = portfolio_manager
        self.executor = executor
        self.decisions: List[TradeDecision] = []
        self.business = BusinessMode()

        reports_dir = Path("data/trading/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)

    def process_signals(self, signals: List[TradeSignal]) -> None:
        symbols = self.config.get("symbols", [])
        for signal in signals:
            if signal.symbol not in symbols:
                continue

            price = self.broker.get_price(signal.symbol)
            equity = self.portfolio_manager.get_equity({s: self.broker.get_price(s) for s in symbols})

            qty = self.risk_manager.propose_size(signal, self.portfolio_manager, price, equity)
            if qty <= 0:
                self.decisions.append(
                    TradeDecision(
                        symbol=signal.symbol,
                        side=signal.side,
                        qty=0.0,
                        price=price,
                        status="rejected",
                        reason="zero_size",
                        timestamp=time.time(),
                    )
                )
                continue

            fill = self.executor.place_market_order(signal.symbol, signal.side, qty)
            self.portfolio_manager.update_on_fill(fill)

            self.decisions.append(
                TradeDecision(
                    symbol=signal.symbol,
                    side=signal.side,
                    qty=fill.qty,
                    price=fill.price,
                    status="filled",
                    reason="risk_ok",
                    timestamp=fill.timestamp,
                )
            )

    def run_paper(self, signals: List[TradeSignal], duration_seconds: int = 15) -> Dict[str, Any]:
        start = time.time()
        symbols = self.config.get("symbols", [])

        # Single pass process
        self.process_signals(signals)

        # Simple time loop to simulate price drift and mark-to-market
        while time.time() - start < duration_seconds:
            for _ in range(1):
                self.broker.step(symbols)
            time.sleep(1)

        prices = {s: self.broker.get_price(s) for s in symbols}
        equity = self.portfolio_manager.get_equity(prices)
        report = {
            "timestamp": time.time(),
            "mode": "paper",
            "equity": equity,
            "cash": self.portfolio_manager.cash_usd,
            "positions": self.portfolio_manager.positions,
            "decisions": [asdict(d) for d in self.decisions],
            "prices": prices,
        }
        initial = self.config.get("starting_cash_usd", 100000.0)
        pnl = report["equity"] - initial
        self.business.update_trading_pnl(pnl)
        self._save_report(report)
        return report

    def _save_report(self, report: Dict[str, Any]) -> None:
        ts = int(report["timestamp"])
        out_path = Path(f"data/trading/reports/paper_report_{ts}.json")
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)

    def _get_prices(self) -> Dict[str, float]:
        return {s: self.broker.get_price(s) for s in self.config.get("symbols", [])}

    def run_live(self, oracle_text_path: str, interval: int) -> None:
        previous_equity = self.portfolio_manager.get_equity(self._get_prices())
        while True:
            if oracle_text_path:
                with open(oracle_text_path, "r") as f:
                    raw = f.read()
                signals = parse_oracle_input(raw)
                self.process_signals(signals)
            prices = self._get_prices()
            equity = self.portfolio_manager.get_equity(prices)
            pnl = equity - previous_equity
            self.business.update_trading_pnl(pnl)
            previous_equity = equity
            time.sleep(interval)


