import os
from typing import TYPE_CHECKING
import ccxt
import time

if TYPE_CHECKING:
    from .base import ExchangeAdapter
    from ...portfolio.portfolio_manager import Fill

class CCXTAdapter(ExchangeAdapter):
    def __init__(self) -> None:
        api_key = os.environ.get("COINBASE_API_KEY")
        secret = os.environ.get("COINBASE_API_SECRET")
        if not api_key or not secret:
            raise ValueError("Missing env vars for CCXTAdapter")
        self.exchange = ccxt.coinbasepro({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True
        })

    def get_price(self, symbol: str) -> float:
        ticker = self.exchange.fetch_ticker(symbol)
        return float(ticker['last'])

    def place_order(self, symbol: str, side: str, qty: float, fee_bps: float, slippage_bps: float) -> "Fill":
        if side == "buy":
            res = self.exchange.create_market_buy_order(symbol, qty)
        else:
            res = self.exchange.create_market_sell_order(symbol, qty)
        info = res['info']
        filled_qty = float(info.get('filled', qty))
        avg_price = float(info.get('price', 0.0))  # May need to calculate from trades
        fee = filled_qty * avg_price * (fee_bps / 10000.0)
        return Fill(
            symbol=symbol,
            side=side,
            qty=filled_qty,
            price=avg_price,
            fee=fee,
            timestamp=time.time()
        )
