import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import time

from .orchestrator import TradingOrchestrator
from .signal.oracle_bridge import parse_oracle_input
from .risk.risk_manager import RiskConfig, RiskManager
from .portfolio.portfolio_manager import PortfolioManager
from .execution.trade_executor import TradeExecutor
from .sim.paper_broker import PaperBroker
from .adapters.uniswap_adapter import UniswapAdapter
from .adapters.ccxt_adapter import CCXTAdapter
from .adapters.alpaca_adapter import AlpacaAdapter
from .adapters.binance_adapter import BinanceUSAdapter


def load_config(path: str | None) -> Dict[str, Any]:
    default_path = Path("config/trading_config.json")
    cfg_path = Path(path) if path else default_path
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return json.load(f)
    # Default config
    return {
        "mode": "paper",
        "symbols": ["BTC/USDC", "ETH/USDC"],
        "starting_cash_usd": 100000.0,
        "risk": {"per_trade_risk_pct": 0.005, "daily_loss_cap_pct": 0.02, "max_leverage": 1.5},
    }


def cmd_paper(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    symbols: List[str] = args.symbols.split(",") if args.symbols else cfg.get("symbols", [])
    cfg["symbols"] = symbols

    broker = PaperBroker()
    risk = RiskManager(RiskConfig(**cfg.get("risk", {})))
    portfolio = PortfolioManager(starting_cash_usd=cfg.get("starting_cash_usd", 100000.0))
    executor = TradeExecutor(broker)

    orch = TradingOrchestrator(config=cfg, broker=broker, risk_manager=risk, portfolio_manager=portfolio, executor=executor)

    if args.oracle_text:
        with open(args.oracle_text, "r") as f:
            raw = f.read()
    else:
        # Minimal default signal for quick try
        raw = "BUY BTC/USDC (conf 0.7)\nSELL ETH/USDC (conf 0.6)"

    signals = parse_oracle_input(raw)
    report = orch.run_paper(signals, duration_seconds=args.duration)
    out = json.dumps(report, indent=2)
    print(out)


def cmd_live_dex(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    symbols: List[str] = args.symbols.split(",") if args.symbols else cfg.get("symbols", [])
    cfg["symbols"] = symbols
    broker = UniswapAdapter(paper=False)
    risk = RiskManager(RiskConfig(**cfg.get("risk", {})))
    portfolio = PortfolioManager(starting_cash_usd=cfg.get("starting_cash_usd", 100000.0))
    executor = TradeExecutor(broker)
    orch = TradingOrchestrator(config=cfg, broker=broker, risk_manager=risk, portfolio_manager=portfolio, executor=executor)
    orch.run_live(args.oracle_text, args.interval)


def cmd_live_cex(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    symbols: List[str] = args.symbols.split(",") if args.symbols else cfg.get("symbols", [])
    cfg["symbols"] = symbols
    broker = CCXTAdapter()
    risk = RiskManager(RiskConfig(**cfg.get("risk", {})))
    portfolio = PortfolioManager(starting_cash_usd=cfg.get("starting_cash_usd", 100000.0))
    executor = TradeExecutor(broker)
    orch = TradingOrchestrator(config=cfg, broker=broker, risk_manager=risk, portfolio_manager=portfolio, executor=executor)
    orch.run_live(args.oracle_text, args.interval)


def cmd_live_stocks(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    symbols: List[str] = args.symbols.split(",") if args.symbols else cfg.get("symbols", [])
    cfg["symbols"] = symbols
    broker = AlpacaAdapter(paper=False)
    risk = RiskManager(RiskConfig(**cfg.get("risk", {})))
    portfolio = PortfolioManager(starting_cash_usd=cfg.get("starting_cash_usd", 100000.0))
    executor = TradeExecutor(broker)
    orch = TradingOrchestrator(config=cfg, broker=broker, risk_manager=risk, portfolio_manager=portfolio, executor=executor)
    orch.run_live(args.oracle_text, args.interval)

def cmd_live_binance(args: argparse.Namespace) -> None:
    """Live crypto trading via Binance.US"""
    cfg = load_config(args.config)
    symbols: List[str] = args.symbols.split(",") if args.symbols else cfg.get("symbols", [])
    cfg["symbols"] = symbols
    
    broker = BinanceUSAdapter(paper=False)
    risk = RiskManager(RiskConfig(**cfg.get("risk", {})))
    portfolio = PortfolioManager(starting_cash_usd=cfg.get("starting_cash_usd", 100000.0))
    executor = TradeExecutor(broker)
    orch = TradingOrchestrator(config=cfg, broker=broker, risk_manager=risk, portfolio_manager=portfolio, executor=executor)
    orch.run_live(args.oracle_text, args.interval)


def main() -> None:
    parser = argparse.ArgumentParser(description="ICEBURG Trading CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_paper = sub.add_parser("paper", help="Run paper trading")
    p_paper.add_argument("--symbols", type=str, help="Comma-separated symbols, e.g., BTC/USDC,ETH/USDC")
    p_paper.add_argument("--config", type=str, help="Path to trading_config.json")
    p_paper.add_argument("--oracle-text", type=str, help="Path to oracle output (text or JSON)")
    p_paper.add_argument("--duration", type=int, default=10, help="Seconds to simulate after order placement")
    p_paper.add_argument("--interval", type=int, default=60, help="Interval seconds for live modes")
    p_paper.set_defaults(func=cmd_paper)

    p_live_dex = sub.add_parser("live-dex", help="Run live trading on DEX")
    p_live_dex.add_argument("--symbols", type=str, help="Comma-separated symbols, e.g., BTC/USDC,ETH/USDC")
    p_live_dex.add_argument("--config", type=str, help="Path to trading_config.json")
    p_live_dex.add_argument("--oracle-text", type=str, help="Path to oracle output (text or JSON)")
    p_live_dex.add_argument("--interval", type=int, default=60, help="Interval seconds for live modes")
    p_live_dex.set_defaults(func=cmd_live_dex)

    p_live_cex = sub.add_parser("live-cex", help="Run live trading on CEX")
    p_live_cex.add_argument("--symbols", type=str, help="Comma-separated symbols, e.g., BTC/USDC,ETH/USDC")
    p_live_cex.add_argument("--config", type=str, help="Path to trading_config.json")
    p_live_cex.add_argument("--oracle-text", type=str, help="Path to oracle output (text or JSON)")
    p_live_cex.add_argument("--interval", type=int, default=60, help="Interval seconds for live modes")
    p_live_cex.set_defaults(func=cmd_live_cex)

    p_live_stocks = sub.add_parser("live-stocks", help="Run live trading on Stocks")
    p_live_stocks.add_argument("--symbols", type=str, help="Comma-separated symbols, e.g., BTC/USDC,ETH/USDC")
    p_live_stocks.add_argument("--config", type=str, help="Path to trading_config.json")
    p_live_stocks.add_argument("--oracle-text", type=str, help="Path to oracle output (text or JSON)")
    p_live_stocks.add_argument("--interval", type=int, default=60, help="Interval seconds for live modes")
    p_live_stocks.set_defaults(func=cmd_live_stocks)

    p_live_binance = sub.add_parser("live-binance", help="Run live trading on Binance.US")
    p_live_binance.add_argument("--symbols", type=str, help="Comma-separated symbols, e.g., BTC/USDT,ETH/USDT")
    p_live_binance.add_argument("--config", type=str, help="Path to trading_config.json")
    p_live_binance.add_argument("--oracle-text", type=str, help="Path to oracle output (text or JSON)")
    p_live_binance.add_argument("--interval", type=int, default=60, help="Interval seconds for live modes")
    p_live_binance.set_defaults(func=cmd_live_binance)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


