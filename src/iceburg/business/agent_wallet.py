"""
ICEBURG Agent Wallet System
Cryptocurrency wallets for autonomous agents
"""

import asyncio
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class WalletBalance:
    """Represents wallet balance for an agent"""
    agent_id: str
    usdc_balance: float = 0.0
    energy_tokens: float = 0.0
    total_earnings: float = 0.0
    total_investments: float = 0.0
    last_updated: str = ""

@dataclass
class Transaction:
    """Represents a wallet transaction"""
    transaction_id: str
    agent_id: str
    transaction_type: str  # 'earn', 'spend', 'invest', 'receive'
    amount: float
    currency: str  # 'USDC', 'ENERGY'
    description: str
    timestamp: str
    status: str = "pending"  # 'pending', 'completed', 'failed'

class AgentWallet:
    """Cryptocurrency wallet for ICEBURG agents"""
    
    def __init__(self, agent_id: str, wallet_file: Optional[Path] = None):
        self.agent_id = agent_id
        self.wallet_file = wallet_file or Path(f"data/wallets/{agent_id}_wallet.json")
        self.balance = WalletBalance(agent_id=agent_id)
        self.transactions: List[Transaction] = []
        self.load_wallet()
    
    def load_wallet(self):
        """Load wallet data from file"""
        try:
            if self.wallet_file.exists():
                with open(self.wallet_file, 'r') as f:
                    data = json.load(f)
                    self.balance = WalletBalance(**data.get('balance', {}))
                    self.transactions = [Transaction(**tx) for tx in data.get('transactions', [])]
                logger.info(f"Loaded wallet for agent {self.agent_id}")
        except Exception as e:
            logger.warning(f"Failed to load wallet for agent {self.agent_id}: {e}")
    
    def save_wallet(self):
        """Save wallet data to file"""
        try:
            self.wallet_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'balance': asdict(self.balance),
                'transactions': [asdict(tx) for tx in self.transactions]
            }
            with open(self.wallet_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved wallet for agent {self.agent_id}")
        except Exception as e:
            logger.warning(f"Failed to save wallet for agent {self.agent_id}: {e}")
    
    async def earn_money(self, amount: float, currency: str, description: str) -> bool:
        """Agent earns money from providing services"""
        try:
            transaction = Transaction(
                transaction_id=f"earn_{len(self.transactions)}_{self.agent_id}",
                agent_id=self.agent_id,
                transaction_type="earn",
                amount=amount,
                currency=currency,
                description=description,
                timestamp=str(asyncio.get_event_loop().time())
            )
            
            if currency == "USDC":
                self.balance.usdc_balance += amount
                self.balance.total_earnings += amount
            elif currency == "ENERGY":
                self.balance.energy_tokens += amount
            
            self.transactions.append(transaction)
            self.balance.last_updated = transaction.timestamp
            self.save_wallet()
            
            logger.info(f"Agent {self.agent_id} earned {amount} {currency}: {description}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process earnings for agent {self.agent_id}: {e}")
            return False
    
    async def spend_money(self, amount: float, currency: str, description: str) -> bool:
        """Agent spends money on operations or investments"""
        try:
            # Check if agent has sufficient balance
            if currency == "USDC" and self.balance.usdc_balance < amount:
                logger.warning(f"Agent {self.agent_id} insufficient USDC balance")
                return False
            elif currency == "ENERGY" and self.balance.energy_tokens < amount:
                logger.warning(f"Agent {self.agent_id} insufficient ENERGY balance")
                return False
            
            transaction = Transaction(
                transaction_id=f"spend_{len(self.transactions)}_{self.agent_id}",
                agent_id=self.agent_id,
                transaction_type="spend",
                amount=amount,
                currency=currency,
                description=description,
                timestamp=str(asyncio.get_event_loop().time())
            )
            
            if currency == "USDC":
                self.balance.usdc_balance -= amount
            elif currency == "ENERGY":
                self.balance.energy_tokens -= amount
            
            self.transactions.append(transaction)
            self.balance.last_updated = transaction.timestamp
            self.save_wallet()
            
            logger.info(f"Agent {self.agent_id} spent {amount} {currency}: {description}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process spending for agent {self.agent_id}: {e}")
            return False
    
    async def invest_money(self, amount: float, currency: str, investment_type: str) -> bool:
        """Agent invests money in various opportunities"""
        try:
            # Check if agent has sufficient balance
            if currency == "USDC" and self.balance.usdc_balance < amount:
                logger.warning(f"Agent {self.agent_id} insufficient USDC balance for investment")
                return False
            elif currency == "ENERGY" and self.balance.energy_tokens < amount:
                logger.warning(f"Agent {self.agent_id} insufficient ENERGY balance for investment")
                return False
            
            transaction = Transaction(
                transaction_id=f"invest_{len(self.transactions)}_{self.agent_id}",
                agent_id=self.agent_id,
                transaction_type="invest",
                amount=amount,
                currency=currency,
                description=f"Investment in {investment_type}",
                timestamp=str(asyncio.get_event_loop().time())
            )
            
            if currency == "USDC":
                self.balance.usdc_balance -= amount
                self.balance.total_investments += amount
            elif currency == "ENERGY":
                self.balance.energy_tokens -= amount
            
            self.transactions.append(transaction)
            self.balance.last_updated = transaction.timestamp
            self.save_wallet()
            
            logger.info(f"Agent {self.agent_id} invested {amount} {currency} in {investment_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process investment for agent {self.agent_id}: {e}")
            return False
    
    async def receive_payment(self, amount: float, currency: str, from_agent: str, description: str) -> bool:
        """Agent receives payment from another agent or customer"""
        try:
            transaction = Transaction(
                transaction_id=f"receive_{len(self.transactions)}_{self.agent_id}",
                agent_id=self.agent_id,
                transaction_type="receive",
                amount=amount,
                currency=currency,
                description=f"Payment from {from_agent}: {description}",
                timestamp=str(asyncio.get_event_loop().time())
            )
            
            if currency == "USDC":
                self.balance.usdc_balance += amount
                self.balance.total_earnings += amount
            elif currency == "ENERGY":
                self.balance.energy_tokens += amount
            
            self.transactions.append(transaction)
            self.balance.last_updated = transaction.timestamp
            self.save_wallet()
            
            logger.info(f"Agent {self.agent_id} received {amount} {currency} from {from_agent}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process payment for agent {self.agent_id}: {e}")
            return False
    
    def get_balance(self) -> WalletBalance:
        """Get current wallet balance"""
        return self.balance
    
    def get_transaction_history(self, limit: int = 10) -> List[Transaction]:
        """Get recent transaction history"""
        return self.transactions[-limit:] if self.transactions else []
    
    def get_total_earnings(self) -> float:
        """Get total earnings for this agent"""
        return self.balance.total_earnings
    
    def get_total_investments(self) -> float:
        """Get total investments for this agent"""
        return self.balance.total_investments
    
    def get_net_worth(self) -> float:
        """Get agent's net worth (balance + investments)"""
        return self.balance.usdc_balance + self.balance.energy_tokens + self.balance.total_investments
