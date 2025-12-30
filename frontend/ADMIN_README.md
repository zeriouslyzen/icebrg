# ICEBURG Admin Dashboard

This dashboard provides a unified view of the ICEBURG system, integrating **V2 Prediction Markets** and **V10 Quantitative Finance** capabilities.

## Features

### 1. Intelligence Feed (V2)
- **Live Signals**: Real-time intelligence signals from the OSINT/CORPINT aggregator.
- **Priority Filtering**: Signals are automatically ranked by urgency and impact.
- **Alpha Conversion**: Intelligence signals are processed into actionable trading alpha.

### 2. Event Prediction Matrix
- **Geopolitical Forecasting**: Predictions on global conflicts, regime changes, and sanctions.
- **Economic Regime Detection**: Recession probability, interest rate shifts, and currency crises.
- **Black Swan Alerts**: Detection of paradigm-shifting events and extreme outliers.

### 3. Network Analysis
- **Influence Graphs**: Visualization of power centers and community structures.
- **Cascade Prediction**: Modeling the propagation of influence across networks.
- **Hidden Coalitions**: Identification of structural equivalence in adversarial networks.

### 4. Finance Integration (V10)
- **Portfolio Tracking**: Real-time performance metrics and exposure monitoring.
- **Risk Metrics**: VaR, CVaR, and Sharpe ratio analysis.
- **Monte Carlo Simulations**: 100k-1M path simulations for portfolio stress testing.

## Technical Architecture

The dashboard communicates with several backend routers:
- `intelligence_bridge_router`: Bridges V2 intelligence data to the finance module.
- `finance_controller`: Handles core trading logic and data retrieval.
- `network_analyzer`: Computes graph-based metrics.
- `simulation_engine`: Runs parallel Monte Carlo scenarios.

## Development

- **Frontend**: Vanilla JS (ES6+), CSS Grid/Flexbox, dynamic DOM rendering.
- **Backend**: FastAPI, Python 3.9+.
- **Data Access**: `admin.js` handles multi-endpoint data fetching and synchronization.

## Access
Accessible locally at: `http://localhost:8000/admin.html`
