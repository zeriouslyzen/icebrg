"""
Real Money Trading Launcher for ICEBURG
Military-grade secure launcher for real money trading system
"""

import asyncio
import logging
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.iceburg.trading.military_security import MilitarySecurityManager, SecurityConfig
from src.iceburg.trading.real_broker_integration import RealBrokerIntegration, BrokerConfig
from src.iceburg.trading.real_money_trading import RealMoneyTradingSystem, TradingStrategy
from src.iceburg.trading.secure_wallet_manager import SecureWalletManager, WalletConfig

logger = logging.getLogger(__name__)


class RealMoneyTradingLauncher:
    """
    Military-grade secure launcher for real money trading system.
    
    Features:
    - Secure configuration loading
    - Multi-broker support (Binance, Alpaca)
    - Wallet management
    - Risk management
    - Real-time monitoring
    - Emergency controls
    """
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config/real_money_trading.json"
        self.config = self._load_config()
        self.security = None
        self.broker = None
        self.wallet_manager = None
        self.trading_system = None
        
        # Initialize logging
        self._setup_logging()
        
        logger.info("🚀 Real Money Trading Launcher initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration with security validation"""
        try:
            if not os.path.exists(self.config_path):
                # Create default config
                self._create_default_config()
            
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Validate required fields
            required_fields = ['security', 'broker', 'wallet', 'trading']
            for field in required_fields:
                if field not in config:
                    raise Exception(f"Missing required config field: {field}")
            
            logger.info("✅ Configuration loaded successfully")
            return config
            
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            raise
    
    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            "security": {
                "encryption_key": "CHANGE_THIS_MILITARY_GRADE_PASSWORD",
                "ip_whitelist": ["192.168.1.0/24", "10.0.0.0/8"],
                "max_daily_loss_percent": 5.0,
                "emergency_stop_threshold": 10.0,
                "rate_limit": 100,
                "session_timeout_minutes": 60
            },
            "broker": {
                "type": "binance",  # or "alpaca"
                "api_key": "YOUR_API_KEY",
                "secret_key": "YOUR_SECRET_KEY",
                "base_url": "https://api.binance.com",
                "testnet": True
            },
            "wallet": {
                "wallet_id": "trading_wallet_001",
                "wallet_type": "hot",
                "max_balance": 10000.0,
                "daily_limit": 1000.0,
                "withdrawal_limit": 500.0
            },
            "trading": {
                "strategies": [
                    {
                        "name": "BTC_Momentum",
                        "symbols": ["BTCUSDT"],
                        "max_position_size": 1000.0,
                        "stop_loss_percent": 2.0,
                        "take_profit_percent": 5.0,
                        "risk_per_trade": 1.0,
                        "enabled": True
                    }
                ],
                "monitoring_interval": 1.0,
                "max_trades_per_day": 50
            }
        }
        
        # Create config directory
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        
        # Write config
        with open(self.config_path, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        logger.info(f"✅ Default configuration created: {self.config_path}")
        logger.warning("⚠️ Please update configuration with your actual API keys and settings")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/real_money_trading.log'),
                logging.StreamHandler()
            ]
        )
        
        # Create logs directory
        os.makedirs('logs', exist_ok=True)
    
    async def initialize_system(self):
        """Initialize the complete trading system"""
        try:
            logger.info("🔧 Initializing real money trading system...")
            
            # Initialize security
            security_config = SecurityConfig(
                encryption_key=self.config['security']['encryption_key'],
                ip_whitelist=self.config['security']['ip_whitelist'],
                max_daily_loss_percent=self.config['security']['max_daily_loss_percent'],
                emergency_stop_threshold=self.config['security']['emergency_stop_threshold'],
                api_rate_limit=self.config['security']['rate_limit'],
                session_timeout_minutes=self.config['security']['session_timeout_minutes']
            )
            self.security = MilitarySecurityManager(security_config)
            
            # Initialize broker
            broker_config = BrokerConfig(
                broker_type=self.config['broker']['type'],
                api_key=self.security.encrypt_sensitive_data(self.config['broker']['api_key']),
                secret_key=self.security.encrypt_sensitive_data(self.config['broker']['secret_key']),
                base_url=self.config['broker']['base_url'],
                testnet=self.config['broker']['testnet']
            )
            self.broker = RealBrokerIntegration(broker_config, security_config)
            
            # Initialize wallet manager
            self.wallet_manager = SecureWalletManager(security_config)
            
            # Create wallet if not exists
            wallet_config = WalletConfig(
                wallet_id=self.config['wallet']['wallet_id'],
                wallet_type=self.config['wallet']['wallet_type'],
                max_balance=self.config['wallet']['max_balance'],
                daily_limit=self.config['wallet']['daily_limit'],
                withdrawal_limit=self.config['wallet']['withdrawal_limit'],
                encryption_key=security_config.encryption_key
            )
            
            try:
                self.wallet_manager.create_wallet(wallet_config)
            except Exception:
                # Wallet already exists
                pass
            
            # Initialize trading system
            self.trading_system = RealMoneyTradingSystem(broker_config, security_config)
            
            # Add trading strategies
            for strategy_config in self.config['trading']['strategies']:
                strategy = TradingStrategy(
                    name=strategy_config['name'],
                    symbols=strategy_config['symbols'],
                    max_position_size=strategy_config['max_position_size'],
                    stop_loss_percent=strategy_config['stop_loss_percent'],
                    take_profit_percent=strategy_config['take_profit_percent'],
                    risk_per_trade=strategy_config['risk_per_trade'],
                    enabled=strategy_config['enabled']
                )
                self.trading_system.add_trading_strategy(strategy)
            
            # Set callbacks
            self.trading_system.set_callbacks(
                on_trade=self._on_trade_callback,
                on_alert=self._on_alert_callback,
                on_emergency=self._on_emergency_callback
            )
            
            logger.info("✅ Real money trading system initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    async def _on_trade_callback(self, trade_info: Dict[str, Any]):
        """Handle trade execution callback"""
        logger.info(f"📈 Trade executed: {trade_info}")
        
        # Log to database
        # Update wallet balance
        # Send notifications
    
    async def _on_alert_callback(self, alert_message: str):
        """Handle risk alert callback"""
        logger.warning(f"⚠️ Risk alert: {alert_message}")
        
        # Send email/SMS alerts
        # Log to database
        # Update monitoring dashboard
    
    async def _on_emergency_callback(self, emergency_message: str):
        """Handle emergency callback"""
        logger.critical(f"🚨 EMERGENCY: {emergency_message}")
        
        # Send emergency notifications
        # Log to database
        # Update monitoring dashboard
        # Close all positions
    
    async def start_trading(self, session_name: str = None):
        """Start real money trading"""
        try:
            if not self.trading_system:
                raise Exception("Trading system not initialized")
            
            # Start trading session
            session_id = await self.trading_system.start_trading_session(session_name)
            logger.info(f"🚀 Trading session started: {session_id}")
            
            # Get initial status
            status = self.trading_system.get_trading_status()
            logger.info(f"📊 Trading status: {json.dumps(status, indent=2, default=str)}")
            
            # Keep running until stopped
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("🛑 Trading stopped by user")
            await self.stop_trading()
        except Exception as e:
            logger.error(f"❌ Trading error: {e}")
            await self.stop_trading()
    
    async def stop_trading(self, close_positions: bool = True):
        """Stop trading and close positions"""
        try:
            if self.trading_system:
                await self.trading_system.stop_trading_session(close_positions)
                logger.info("🛑 Trading session stopped")
            
        except Exception as e:
            logger.error(f"❌ Stop trading error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "system_status": "ACTIVE" if self.trading_system else "INACTIVE"
            }
            
            if self.trading_system:
                status["trading_status"] = self.trading_system.get_trading_status()
            
            if self.wallet_manager:
                wallet_status = self.wallet_manager.get_wallet_status(
                    self.config['wallet']['wallet_id']
                )
                status["wallet_status"] = wallet_status
            
            if self.security:
                status["security_status"] = self.security.get_security_status()
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Failed to get system status: {e}")
            return {"error": str(e)}
    
    def emergency_stop(self):
        """Trigger emergency stop"""
        try:
            if self.trading_system:
                self.trading_system.trigger_emergency_stop()
                logger.critical("🚨 EMERGENCY STOP TRIGGERED")
            
        except Exception as e:
            logger.error(f"❌ Emergency stop failed: {e}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='ICEBURG Real Money Trading System')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--session', help='Trading session name')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--emergency-stop', action='store_true', help='Trigger emergency stop')
    
    args = parser.parse_args()
    
    # Create launcher
    launcher = RealMoneyTradingLauncher(args.config)
    
    try:
        # Initialize system
        await launcher.initialize_system()
        
        if args.status:
            # Show status and exit
            status = launcher.get_system_status()
            print(json.dumps(status, indent=2, default=str))
            return
        
        if args.emergency_stop:
            # Emergency stop
            launcher.emergency_stop()
            print("🚨 Emergency stop triggered")
            return
        
        # Start trading
        await launcher.start_trading(args.session)
        
    except KeyboardInterrupt:
        logger.info("🛑 Shutdown requested")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
