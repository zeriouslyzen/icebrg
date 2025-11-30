"""
Real Broker Integration for ICEBURG Financial Trading System
Military-grade secure integration with real brokers (Binance, Alpaca)
"""

import os
import time
import logging
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hmac
import hashlib
import base64
from urllib.parse import urlencode

from .military_security import MilitarySecurityManager, SecurityConfig

logger = logging.getLogger(__name__)


@dataclass
class BrokerConfig:
    """Broker configuration with military-grade security"""
    broker_type: str  # 'binance' or 'alpaca'
    api_key: str
    secret_key: str
    base_url: str
    testnet: bool = True  # Start with testnet
    max_retries: int = 3
    timeout_seconds: int = 30
    rate_limit_delay: float = 0.1  # 100ms between requests


@dataclass
class TradingLimits:
    """Military-grade trading limits"""
    max_daily_loss: float = 1000.0  # $1000 max daily loss
    max_position_size: float = 5000.0  # $5000 max position
    max_trades_per_day: int = 50
    emergency_stop_loss: float = 2000.0  # $2000 emergency stop
    max_leverage: float = 1.0  # No leverage for safety


class RealBrokerIntegration:
    """
    Military-grade secure broker integration for real money trading.
    
    Supports:
    - Binance (crypto trading)
    - Alpaca (stock trading)
    - Military-grade security
    - Real-time risk management
    - Emergency stop capabilities
    """
    
    def __init__(self, broker_config: BrokerConfig, security_config: SecurityConfig):
        self.config = broker_config
        self.security = MilitarySecurityManager(security_config)
        self.limits = TradingLimits()
        
        # Initialize broker-specific client
        if broker_config.broker_type == 'binance':
            self._init_binance_client()
        elif broker_config.broker_type == 'alpaca':
            self._init_alpaca_client()
        else:
            raise ValueError(f"Unsupported broker: {broker_config.broker_type}")
        
        # Trading state
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.positions = {}
        self.emergency_stop = False
        
        logger.info(f"🔒 Real broker integration initialized: {broker_config.broker_type}")
    
    def _init_binance_client(self):
        """Initialize Binance client with security"""
        try:
            # Import Binance client
            from binance.client import Client
            from binance.exceptions import BinanceAPIException
            
            # Create client with encrypted credentials
            api_key = self.security.decrypt_sensitive_data(self.config.api_key)
            secret_key = self.security.decrypt_sensitive_data(self.config.secret_key)
            
            self.client = Client(api_key, secret_key, testnet=self.config.testnet)
            self.broker_type = 'binance'
            
            logger.info("✅ Binance client initialized with military-grade security")
            
        except Exception as e:
            logger.error(f"❌ Binance initialization failed: {e}")
            raise
    
    def _init_alpaca_client(self):
        """Initialize Alpaca client with security"""
        try:
            # Import Alpaca client
            import alpaca_trade_api as tradeapi
            
            # Create client with encrypted credentials
            api_key = self.security.decrypt_sensitive_data(self.config.api_key)
            secret_key = self.security.decrypt_sensitive_data(self.config.secret_key)
            
            self.client = tradeapi.REST(
                key_id=api_key,
                secret_key=secret_key,
                base_url=self.config.base_url,
                api_version='v2'
            )
            self.broker_type = 'alpaca'
            
            logger.info("✅ Alpaca client initialized with military-grade security")
            
        except Exception as e:
            logger.error(f"❌ Alpaca initialization failed: {e}")
            raise
    
    async def get_account_balance(self) -> Dict[str, float]:
        """Get account balance with security validation"""
        try:
            if self.broker_type == 'binance':
                account = self.client.get_account()
                balance = {}
                for asset in account['balances']:
                    free = float(asset['free'])
                    locked = float(asset['locked'])
                    if free > 0 or locked > 0:
                        balance[asset['asset']] = free + locked
                return balance
            
            elif self.broker_type == 'alpaca':
                account = self.client.get_account()
                return {
                    'USD': float(account.cash),
                    'buying_power': float(account.buying_power),
                    'portfolio_value': float(account.portfolio_value)
                }
            
        except Exception as e:
            logger.error(f"❌ Failed to get account balance: {e}")
            self.security._log_security_event("BALANCE_ERROR", str(e))
            return {}
    
    async def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get current positions with security validation"""
        try:
            if self.broker_type == 'binance':
                # Get spot positions
                account = self.client.get_account()
                positions = {}
                for asset in account['balances']:
                    free = float(asset['free'])
                    locked = float(asset['locked'])
                    if free > 0 or locked > 0:
                        positions[asset['asset']] = {
                            'quantity': free + locked,
                            'free': free,
                            'locked': locked
                        }
                return positions
            
            elif self.broker_type == 'alpaca':
                positions = self.client.list_positions()
                return {
                    pos.symbol: {
                        'quantity': float(pos.qty),
                        'market_value': float(pos.market_value),
                        'unrealized_pl': float(pos.unrealized_pl),
                        'side': pos.side
                    }
                    for pos in positions
                }
            
        except Exception as e:
            logger.error(f"❌ Failed to get positions: {e}")
            self.security._log_security_event("POSITIONS_ERROR", str(e))
            return {}
    
    async def place_order(self, symbol: str, side: str, quantity: float, 
                         order_type: str = 'market', price: float = None) -> Dict[str, Any]:
        """
        Place order with military-grade security checks
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT', 'AAPL')
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            order_type: 'market', 'limit', 'stop'
            price: Price for limit orders
        """
        try:
            # Security checks
            if self.emergency_stop:
                raise Exception("EMERGENCY STOP ACTIVE - Trading disabled")
            
            # Get current balance
            balance = await self.get_account_balance()
            
            # Check trading limits
            limits_check = self.security.check_trading_limits(
                trade_amount=quantity * (price or 0),
                account_balance=sum(balance.values()),
                daily_pnl=self.daily_pnl
            )
            
            if not limits_check["allowed"]:
                raise Exception(f"Trading limits exceeded: {limits_check['reasons']}")
            
            # Check daily trade limit
            if self.daily_trades >= self.limits.max_trades_per_day:
                raise Exception("Daily trade limit exceeded")
            
            # Place order based on broker
            if self.broker_type == 'binance':
                order = await self._place_binance_order(symbol, side, quantity, order_type, price)
            elif self.broker_type == 'alpaca':
                order = await self._place_alpaca_order(symbol, side, quantity, order_type, price)
            
            # Update trading state
            self.daily_trades += 1
            self.security._log_security_event("ORDER_PLACED", f"{side} {quantity} {symbol}")
            
            return order
            
        except Exception as e:
            logger.error(f"❌ Order placement failed: {e}")
            self.security._log_security_event("ORDER_FAILED", str(e))
            raise
    
    async def _place_binance_order(self, symbol: str, side: str, quantity: float, 
                                  order_type: str, price: float = None) -> Dict[str, Any]:
        """Place Binance order"""
        try:
            if order_type == 'market':
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=quantity
                )
            elif order_type == 'limit':
                order = self.client.create_order(
                    symbol=symbol,
                    side=side,
                    type='LIMIT',
                    timeInForce='GTC',
                    quantity=quantity,
                    price=price
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            return order
            
        except Exception as e:
            logger.error(f"❌ Binance order failed: {e}")
            raise
    
    async def _place_alpaca_order(self, symbol: str, side: str, quantity: float, 
                                 order_type: str, price: float = None) -> Dict[str, Any]:
        """Place Alpaca order"""
        try:
            if order_type == 'market':
                order = self.client.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side.lower(),
                    type='market',
                    time_in_force='day'
                )
            elif order_type == 'limit':
                order = self.client.submit_order(
                    symbol=symbol,
                    qty=quantity,
                    side=side.lower(),
                    type='limit',
                    time_in_force='day',
                    limit_price=price
                )
            else:
                raise ValueError(f"Unsupported order type: {order_type}")
            
            return order
            
        except Exception as e:
            logger.error(f"❌ Alpaca order failed: {e}")
            raise
    
    async def close_all_positions(self) -> Dict[str, Any]:
        """Close all positions immediately (emergency stop)"""
        try:
            self.emergency_stop = True
            self.security._log_security_event("EMERGENCY_STOP", "Closing all positions")
            
            positions = await self.get_positions()
            closed_positions = {}
            
            for symbol, position in positions.items():
                if self.broker_type == 'binance':
                    # Close Binance positions
                    if position['quantity'] > 0:
                        # Convert to USDT and sell
                        if symbol != 'USDT':
                            order = await self.place_order(
                                symbol=f"{symbol}USDT",
                                side='SELL',
                                quantity=position['quantity'],
                                order_type='market'
                            )
                            closed_positions[symbol] = order
                
                elif self.broker_type == 'alpaca':
                    # Close Alpaca positions
                    if position['quantity'] > 0:
                        side = 'SELL' if position['side'] == 'long' else 'BUY'
                        order = await self.place_order(
                            symbol=symbol,
                            side=side,
                            quantity=abs(position['quantity']),
                            order_type='market'
                        )
                        closed_positions[symbol] = order
            
            self.security._log_security_event("POSITIONS_CLOSED", f"Closed {len(closed_positions)} positions")
            return closed_positions
            
        except Exception as e:
            logger.error(f"❌ Failed to close positions: {e}")
            self.security._log_security_event("CLOSE_FAILED", str(e))
            raise
    
    def trigger_emergency_stop(self):
        """Trigger emergency stop"""
        self.emergency_stop = True
        self.security._trigger_emergency_stop()
        logger.critical("🚨 EMERGENCY STOP TRIGGERED")
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status"""
        return {
            "broker_type": self.broker_type,
            "emergency_stop": self.emergency_stop,
            "daily_trades": self.daily_trades,
            "daily_pnl": self.daily_pnl,
            "security_status": self.security.get_security_status(),
            "limits": {
                "max_daily_loss": self.limits.max_daily_loss,
                "max_position_size": self.limits.max_position_size,
                "max_trades_per_day": self.limits.max_trades_per_day
            }
        }


# Example usage
if __name__ == "__main__":
    # Initialize military-grade security
    security_config = SecurityConfig(
        encryption_key="your_military_password_2025",
        ip_whitelist=["192.168.1.0/24"],
        max_daily_loss_percent=5.0,
        emergency_stop_threshold=10.0
    )
    
    # Initialize broker (example with Binance)
    broker_config = BrokerConfig(
        broker_type='binance',
        api_key=security_config.encryption_key,  # Will be encrypted
        secret_key=security_config.encryption_key,  # Will be encrypted
        base_url='https://api.binance.com',
        testnet=True  # Start with testnet
    )
    
    # Create integration
    integration = RealBrokerIntegration(broker_config, security_config)
    
    # Test functions
    async def test_integration():
        # Get account balance
        balance = await integration.get_account_balance()
        print(f"Account balance: {balance}")
        
        # Get positions
        positions = await integration.get_positions()
        print(f"Positions: {positions}")
        
        # Get trading status
        status = integration.get_trading_status()
        print(f"Trading status: {status}")
    
    # Run test
    asyncio.run(test_integration())
