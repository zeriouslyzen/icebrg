from typing import Any

from ..portfolio.portfolio_manager import Fill


class TradeExecutor:
    def __init__(self, broker: Any, fee_bps: float = 5.0, slippage_bps: float = 5.0) -> None:
        self.broker = broker
        self.fee_bps = float(fee_bps)
        self.slippage_bps = float(slippage_bps)

    def place_market_order(self, symbol: str, side: str, qty: float) -> Fill:
        return self.broker.place_order(symbol=symbol, side=side, qty=qty, fee_bps=self.fee_bps, slippage_bps=self.slippage_bps)


