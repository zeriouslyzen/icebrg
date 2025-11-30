import random
import time
from dataclasses import dataclass
from typing import Dict

from ..portfolio.portfolio_manager import Fill


@dataclass
class MarketState:
    price: float


class PaperBroker:
    def __init__(self, initial_prices: Dict[str, float] | None = None, seed: int = 42) -> None:
        random.seed(seed)
        self.state: Dict[str, MarketState] = {}
        initial_prices = initial_prices or {"BTC/USDC": 60000.0, "ETH/USDC": 3000.0}
        for sym, p in initial_prices.items():
            self.state[sym] = MarketState(price=float(p))

    def get_price(self, symbol: str) -> float:
        return float(self.state[symbol].price)

    def place_order(self, symbol: str, side: str, qty: float, fee_bps: float, slippage_bps: float) -> Fill:
        base_price = self.get_price(symbol)
        slip = base_price * (slippage_bps / 10000.0)
        price = base_price + slip if side == "buy" else base_price - slip
        fee = abs(qty * price) * (fee_bps / 10000.0)
        return Fill(symbol=symbol, side=side, qty=qty, price=price, fee=fee, timestamp=time.time())

    def step(self, symbols: list[str]) -> None:
        for s in symbols:
            p = self.state[s].price
            # Random walk with small drift
            change = random.gauss(mu=0.0, sigma=0.001) * p
            self.state[s].price = max(0.0001, p + change)


