import os
from typing import TYPE_CHECKING
from web3 import Web3
from uniswap import Uniswap
import time

if TYPE_CHECKING:
    from .base import ExchangeAdapter
    from ...portfolio.portfolio_manager import Fill

class UniswapAdapter(ExchangeAdapter):
    TOKEN_ADDRESSES = {
        "WETH": "0x4200000000000000000000000000000000000006",  # Base WETH
        "USDC": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",  # Base USDC
    }

    def __init__(self) -> None:
        address = os.environ.get("WALLET_ADDRESS")
        private_key = os.environ.get("PRIVATE_KEY")
        provider_url = os.environ.get("BASE_PROVIDER_URL")
        if not all([address, private_key, provider_url]):
            raise ValueError("Missing env vars for UniswapAdapter")
        self.w3 = Web3(Web3.HTTPProvider(provider_url))
        self.uni = Uniswap(
            address=address,
            private_key=private_key,
            provider=self.w3,
            version=3
        )

    def get_price(self, symbol: str) -> float:
        base, quote = symbol.split("/")
        token0 = self.TOKEN_ADDRESSES.get(base)
        token1 = self.TOKEN_ADDRESSES.get(quote)
        if not token0 or not token1:
            raise ValueError(f"Unsupported symbol {symbol}")
        # Price of 1 token0 in token1
        return self.uni.get_price_input(token0, token1, 10**18) / 10**6  # Assume 18/6 decimals

    def place_order(self, symbol: str, side: str, qty: float, fee_bps: float, slippage_bps: float) -> "Fill":
        base, quote = symbol.split("/")
        token0 = self.TOKEN_ADDRESSES.get(base)
        token1 = self.TOKEN_ADDRESSES.get(quote)
        if not token0 or not token1:
            raise ValueError(f"Unsupported symbol {symbol}")

        amount = int(qty * 10**18) if base == "WETH" else int(qty * 10**6)  # Adjust decimals
        if side == "buy":
            tx = self.uni.make_trade(token1, token0, amount)
        else:
            tx = self.uni.make_trade(token0, token1, amount)
        # Wait for tx
        receipt = self.w3.eth.wait_for_transaction_receipt(tx)
        # Mock fill for now
        price = self.get_price(symbol)
        fee = qty * price * (fee_bps / 10000.0)
        return Fill(symbol=symbol, side=side, qty=qty, price=price, fee=fee, timestamp=time.time())
