"""
ICEBURG Financial Data and Analysis Module

This module provides financial data integration and analysis capabilities for elite financial AI,
including data pipelines, feature engineering, and market simulation.

Key Components:
- data_pipeline: Financial data ingestion and processing
- feature_engineering: Feature extraction and engineering
- market_simulator: Market simulation and backtesting
- technical_indicators: Technical analysis indicators
- risk_metrics: Risk assessment and metrics
- config: Financial system configuration
- utils: Financial utilities and helpers
"""

from .data_pipeline import DataPipeline, PolygonAPI, YahooFinanceAPI
from .feature_engineering import FeatureEngineer, TechnicalIndicators, QuantumFeatures
from .market_simulator import MarketSimulator, Backtester, PortfolioSimulator
from .technical_indicators import TechnicalIndicators, MovingAverages, Oscillators
from .risk_metrics import RiskMetrics, VaR, CVaR, SharpeRatio
from .config import FinancialConfig
from .utils import FinancialUtils, DataValidator

__all__ = [
    "DataPipeline",
    "PolygonAPI",
    "YahooFinanceAPI", 
    "FeatureEngineer",
    "TechnicalIndicators",
    "QuantumFeatures",
    "MarketSimulator",
    "Backtester",
    "PortfolioSimulator",
    "RiskMetrics",
    "VaR",
    "CVaR", 
    "SharpeRatio",
    "FinancialConfig",
    "FinancialUtils",
    "DataValidator"
]

__version__ = "1.0.0"
__author__ = "ICEBURG Protocol"
__description__ = "Financial data and analysis module for elite financial AI"
