from dataclasses import dataclass
from typing import Dict


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    timestamp: float


class PortfolioManager:
    def __init__(self, starting_cash_usd: float = 100000.0) -> None:
        self.cash_usd: float = float(starting_cash_usd)
        self.positions: Dict[str, Dict[str, float]] = {}

    def update_on_fill(self, fill: Fill) -> None:
        pos = self.positions.setdefault(fill.symbol, {"qty": 0.0, "avg_price": 0.0})
        old_qty = pos["qty"]
        signed_qty = fill.qty if fill.side == "buy" else -fill.qty
        new_qty = old_qty + signed_qty

        # Update average price when position increases in direction of the trade
        if (old_qty >= 0 and signed_qty > 0) or (old_qty <= 0 and signed_qty < 0):
            notional = abs(fill.qty * fill.price)
            total_notional = abs(old_qty) * pos["avg_price"] + notional
            total_qty = abs(old_qty) + abs(signed_qty)
            pos["avg_price"] = total_notional / max(total_qty, 1e-9)

        pos["qty"] = new_qty

        # Cash update
        cash_change = -fill.qty * fill.price if fill.side == "buy" else fill.qty * fill.price
        self.cash_usd += cash_change - fill.fee

    def get_equity(self, prices: Dict[str, float]) -> float:
        equity = self.cash_usd
        for symbol, pos in self.positions.items():
            price = prices.get(symbol, pos["avg_price"])
            equity += pos["qty"] * price
        return float(equity)


