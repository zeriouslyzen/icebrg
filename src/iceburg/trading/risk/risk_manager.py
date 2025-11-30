from dataclasses import dataclass
from typing import Any

from ..signal.oracle_bridge import TradeSignal


@dataclass
class RiskConfig:
    per_trade_risk_pct: float = 0.005  # 0.5% of equity
    daily_loss_cap_pct: float = 0.02   # 2% of equity
    max_leverage: float = 1.5


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.day_loss: float = 0.0

    def register_pnl(self, pnl: float) -> None:
        self.day_loss = min(0.0, self.day_loss + min(0.0, pnl))

    def propose_size(self, signal: TradeSignal, portfolio_manager: Any, price: float, equity: float) -> float:
        # Circuit breaker on daily loss
        if abs(self.day_loss) >= equity * self.cfg.daily_loss_cap_pct:
            return 0.0

        # Basic volatility proxy: 1% stop distance
        stop_distance = max(0.01 * price, 1e-6)
        risk_dollars = equity * self.cfg.per_trade_risk_pct
        qty = risk_dollars / stop_distance

        # Confidence and size hint scaling
        qty *= max(0.1, min(1.0, signal.confidence))
        qty *= max(0.1, min(1.0, signal.size_hint))

        # Ensure not exceeding leverage based on available cash
        max_notional = equity * self.cfg.max_leverage
        desired_notional = qty * price
        if desired_notional > max_notional:
            qty = max_notional / max(price, 1e-9)

        return max(0.0, qty)


