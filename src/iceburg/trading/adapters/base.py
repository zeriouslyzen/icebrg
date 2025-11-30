from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...portfolio.portfolio_manager import Fill

class ExchangeAdapter(ABC):
    @abstractmethod
    def get_price(self, symbol: str) -> float:
        pass

    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float, fee_bps: float, slippage_bps: float) -> "Fill":
        pass
