import os
from typing import TYPE_CHECKING
import time
from alpaca_trade_api import REST as Alpaca

if TYPE_CHECKING:
    from .base import ExchangeAdapter
    from ...portfolio.portfolio_manager import Fill

class AlpacaAdapter(ExchangeAdapter):
    def __init__(self, paper: bool = True) -> None:
        api_key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_API_SECRET")
        if not api_key or not secret:
            raise ValueError("Missing env vars for AlpacaAdapter")
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        self.api = Alpaca(api_key, secret, base_url=base_url)

    def get_price(self, symbol: str) -> float:
        trade = self.api.get_latest_trade(symbol)
        return float(trade.price)

    def place_order(self, symbol: str, side: str, qty: float, fee_bps: float, slippage_bps: float) -> "Fill":
        order = self.api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='market',
            time_in_force='gtc'
        )
        # Wait for fill (simplified, in production use websocket or poll)
        time.sleep(1)  # Naive wait
        filled_order = self.api.get_order(order.id)
        filled_qty = float(filled_order.filled_qty or 0)
        avg_price = float(filled_order.filled_avg_price or 0)
        fee = filled_qty * avg_price * (fee_bps / 10000.0)
        return Fill(
            symbol=symbol,
            side=side,
            qty=filled_qty,
            price=avg_price,
            fee=fee,
            timestamp=time.time()
        )
