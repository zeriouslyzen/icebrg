import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TradeSignal:
    symbol: str
    side: str  # "buy" or "sell"
    size_hint: float  # 0..1
    confidence: float  # 0..1
    horizon: str  # e.g., "intraday", "swing"


def parse_from_oracle_text(text: str) -> List[TradeSignal]:
    signals: List[TradeSignal] = []
    # Simple heuristic: lines like "BUY BTC/USDC (conf 0.7)"
    pattern = re.compile(r"\b(BUY|SELL)\s+([A-Z0-9\/\-]+)(?:.*?conf\s*(0\.?\d*))?", re.IGNORECASE)
    for line in text.splitlines():
        m = pattern.search(line)
        if not m:
            continue
        side = "buy" if m.group(1).upper() == "BUY" else "sell"
        symbol = m.group(2).upper()
        conf = float(m.group(3)) if m.group(3) else 0.6
        signals.append(TradeSignal(symbol=symbol, side=side, size_hint=0.5, confidence=conf, horizon="intraday"))
    return signals


def parse_from_oracle_json(obj: Dict[str, Any]) -> List[TradeSignal]:
    signals: List[TradeSignal] = []
    arr = obj.get("trade_signals") or []
    for s in arr:
        symbol = str(s.get("symbol", "")).upper()
        side = str(s.get("side", "")).lower()
        size_hint = float(s.get("size_hint", 0.5))
        confidence = max(0.0, min(1.0, float(s.get("confidence", 0.6))))
        horizon = str(s.get("horizon", "intraday"))
        if symbol and side in {"buy", "sell"}:
            signals.append(TradeSignal(symbol=symbol, side=side, size_hint=size_hint, confidence=confidence, horizon=horizon))
    return signals


def parse_oracle_input(raw: str) -> List[TradeSignal]:
    """Try JSON first; fallback to text parsing."""
    try:
        obj = json.loads(raw)
        return parse_from_oracle_json(obj)
    except Exception:
        return parse_from_oracle_text(raw)


