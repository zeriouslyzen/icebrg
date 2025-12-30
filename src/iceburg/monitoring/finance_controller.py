"""
Finance & Prediction Dashboard Controller

Exposes ICEBURG's trading and financial AI capabilities via API endpoints
for the V10 admin dashboard.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Import ICEBURG's existing trading infrastructure
from ..trading.market_data import MarketDataProvider, RealTimeBroker
from ..integration.financial_ai_integration import ICEBURGFinancialAIIntegration
from ..config import IceburgConfig

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/finance", tags=["finance"])

# Initialize components (singleton pattern)
_market_data_provider = None
_financial_ai = None


def get_market_data_provider() -> MarketDataProvider:
    """Get or create market data provider singleton."""
    global _market_data_provider
    if _market_data_provider is None:
        _market_data_provider = MarketDataProvider()
    return _market_data_provider


def get_financial_ai() -> ICEBURGFinancialAIIntegration:
    """Get or create financial AI singleton."""
    global _financial_ai
    if _financial_ai is None:
        config = IceburgConfig()
        _financial_ai = ICEBURGFinancialAIIntegration(config)
    return _financial_ai


@router.get("/market-data")
async def get_market_data(symbols: str = "BTC-USD,ETH-USD,AAPL,GOOGL,TSLA"):
    """
    Get real-time market data for multiple symbols.
    
    Args:
        symbols: Comma-separated list of symbols
        
    Returns:
        {
            "timestamp": "2025-12-29T15:30:00",
            "data": {
                "BTC-USD": {"price": 45000.0, "change_24h": 2.5, ...},
                ...
            }
        }
    """
    try:
        provider = get_market_data_provider()
        symbol_list = [s.strip() for s in symbols.split(",")]
        
        market_data = {}
        for symbol in symbol_list:
            try:
                price = provider.get_real_time_price(symbol)
                market_data[symbol] = {
                    "price": price,
                    "symbol": symbol,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.warning(f"Failed to get price for {symbol}: {e}")
                market_data[symbol] = {
                    "error": str(e),
                    "symbol": symbol
                }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data": market_data
        }
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/technical-analysis/{symbol}")
async def get_technical_analysis(symbol: str, period: str = "1mo"):
    """
    Get comprehensive technical analysis for a symbol.
    
    Args:
        symbol: Stock or crypto symbol (e.g., "AAPL", "BTC-USD")
        period: Time period for analysis (default: "1mo")
        
    Returns:
        {
            "symbol": "AAPL",
            "indicators": {
                "RSI": 65.3,
                "MACD": {...},
                "BB": {...},
                ...
            },
            "trend": {...},
            "support_resistance": {...}
        }
    """
    try:
        provider = get_market_data_provider()
        analysis = provider.get_technical_indicators(symbol, period)
        
        return {
            "symbol": symbol,
            "period": period,
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Technical analysis error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai-signals")
async def get_ai_signals(symbols: str = "AAPL,GOOGL,TSLA,BTC-USD,ETH-USD"):
    """
    Get AI-generated trading signals for multiple symbols.
    
    Args:
        symbols: Comma-separated list of symbols
        
    Returns:
        {
            "timestamp": "2025-12-29T15:30:00",
            "signals": [
                {
                    "symbol": "AAPL",
                    "signal": "BUY",
                    "confidence": 0.85,
                    "reasoning": "Strong technical setup with RSI oversold...",
                    "price": 175.50,
                    ...
                },
                ...
            ]
        }
    """
    try:
        provider = get_market_data_provider()
        symbol_list = [s.strip() for s in symbols.split(",")]
        
        all_signals = []
        for symbol in symbol_list:
            try:
                signals = provider.generate_trading_signals(symbol)
                if signals and "signal" in signals:
                    all_signals.append({
                        "symbol": symbol,
                        "timestamp": datetime.now().isoformat(),
                        **signals
                    })
            except Exception as e:
                logger.warning(f"Failed to get signals for {symbol}: {e}")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "signals": all_signals
        }
    except Exception as e:
        logger.error(f"AI signals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/predictions/{symbol}")
async def get_predictions(symbol: str):
    """
    Get AI predictions and event forecasts for a symbol.
    
    Args:
        symbol: Stock or crypto symbol
        
    Returns:
        {
            "symbol": "AAPL",
            "predictions": {
                "price_target_1d": 180.0,
                "price_target_7d": 185.0,
                "price_target_30d": 195.0,
                "confidence": 0.75,
                "events": [...],
                ...
            }
        }
    """
    try:
        financial_ai = get_financial_ai()
        
        # Use financial AI to generate predictions
        query = f"Predict future price movements and key events for {symbol}"
        context = {"symbol": symbol, "analysis_type": "prediction"}
        
        response = financial_ai.activate_financial_ai(query, context)
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "predictions": response.get("predictions", {}),
            "ai_analysis": response.get("analysis", "")
        }
    except Exception as e:
        logger.error(f"Predictions error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio")
async def get_portfolio():
    """
    Get current portfolio status.
    
    Returns:
        {
            "total_value": 125000.0,
            "cash": 25000.0,
            "positions": [
                {"symbol": "AAPL", "qty": 100, "avg_cost": 150.0, "current_price": 175.0, ...},
                ...
            ],
            "pnl_today": 2500.0,
            "pnl_total": 15000.0
        }
    """
    try:
        # TODO: Integrate with actual portfolio manager when available
        # For now, return placeholder
        return {
            "timestamp": datetime.now().isoformat(),
            "total_value": 0.0,
            "cash": 0.0,
            "positions": [],
            "pnl_today": 0.0,
            "pnl_total": 0.0,
            "note": "Portfolio tracking not yet configured. Connect a broker to see positions."
        }
    except Exception as e:
        logger.error(f"Portfolio error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/wallet-status")
async def get_wallet_status():
    """
    Get cryptocurrency wallet balances.
    
    Returns:
        {
            "wallets": [
                {"currency": "BTC", "balance": 0.5, "value_usd": 22500.0, ...},
                {"currency": "ETH", "balance": 5.0, "value_usd": 10000.0, ...},
                ...
            ],
            "total_value_usd": 32500.0
        }
    """
    try:
        # TODO: Integrate with secure_wallet_manager when configured
        # For now, return placeholder
        return {
            "timestamp": datetime.now().isoformat(),
            "wallets": [],
            "total_value_usd": 0.0,
            "note": "Wallet integration not yet configured. Set up wallet credentials to see balances."
        }
    except Exception as e:
        logger.error(f"Wallet status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_finance_status():
    """
    Get overall finance system status.
    
    Returns:
        {
            "market_data": "operational",
            "ai_signals": "operational",
            "portfolio": "not_configured",
            "wallets": "not_configured"
        }
    """
    status = {
        "timestamp": datetime.now().isoformat(),
        "market_data": "operational",
        "ai_signals": "operational",
        "portfolio": "not_configured",
        "wallets": "not_configured",
        "version": "1.0.0"
    }
    
    return status
