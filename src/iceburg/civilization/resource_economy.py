"""
Resource Economy System for AGI Civilization

Implements trading, market dynamics, and resource distribution.
"""

import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    """Status of a trade."""
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class TradeOffer:
    """Represents a trade offer between agents."""
    trade_id: str
    seller_id: str
    buyer_id: str
    resource_offered: str
    amount_offered: float
    resource_requested: str
    amount_requested: float
    status: TradeStatus
    created_time: float
    completed_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketPrice:
    """Represents the market price for a resource."""
    resource_name: str
    current_price: float
    supply: float
    demand: float
    price_history: List[Tuple[float, float]] = field(default_factory=list)  # (timestamp, price)
    last_updated: float = 0.0


@dataclass
class AgentWealth:
    """Tracks an agent's wealth and resources."""
    agent_id: str
    resources: Dict[str, float] = field(default_factory=dict)
    total_value: float = 0.0
    trade_count: int = 0
    reputation: float = 0.5


class ResourceEconomy:
    """
    Complete resource trading and economy system.
    
    Features:
    - Trade system between agents (offers, acceptance, completion)
    - Dynamic market prices based on supply/demand
    - Wealth tracking and distribution
    - Economic statistics and Gini coefficient
    - Trade history and analytics
    """
    
    def __init__(self, max_trades: int = 1000):
        """
        Initialize the resource economy.
        
        Args:
            max_trades: Maximum trade history to keep
        """
        self.max_trades = max_trades
        
        # Trade tracking
        self.trades: List[TradeOffer] = []
        self.pending_trades: Dict[str, TradeOffer] = {}
        self.trade_counter = 0
        
        # Market prices
        self.market_prices: Dict[str, MarketPrice] = {}
        
        # Agent wealth tracking
        self.agent_wealth: Dict[str, AgentWealth] = {}
        
        # Supply and demand tracking
        self.resource_supply: Dict[str, float] = defaultdict(float)
        self.resource_demand: Dict[str, float] = defaultdict(float)
        
        # Price adjustment parameters
        self.price_sensitivity = 0.1  # How quickly prices respond
        self.base_prices = {
            "energy": 1.0,
            "knowledge": 2.0,
            "materials": 1.5,
            "compute": 3.0,
            "reputation": 5.0
        }
        
        # Statistics
        self.stats = {
            "total_trades": 0,
            "successful_trades": 0,
            "total_volume": 0.0,
            "gini_coefficient": 0.0,
            "avg_trade_value": 0.0
        }
    
    def initialize(self, initial_resources: List[Dict[str, Any]] = None):
        """
        Initialize the economy with resource types.
        
        Args:
            initial_resources: List of initial resource definitions
        """
        # Set up market prices for known resources
        for resource_name, base_price in self.base_prices.items():
            self.market_prices[resource_name] = MarketPrice(
                resource_name=resource_name,
                current_price=base_price,
                supply=100.0,  # Initial balanced supply
                demand=100.0,  # Initial balanced demand
                last_updated=time.time()
            )
        
        # Add any custom resources
        if initial_resources:
            for resource_def in initial_resources:
                name = resource_def.get("name", "")
                if name and name not in self.market_prices:
                    self.market_prices[name] = MarketPrice(
                        resource_name=name,
                        current_price=resource_def.get("base_price", 1.0),
                        supply=resource_def.get("supply", 100.0),
                        demand=resource_def.get("demand", 100.0),
                        last_updated=time.time()
                    )
        
        logger.info(f"Resource economy initialized with {len(self.market_prices)} resource types")
    
    def register_agent(self, agent_id: str, initial_resources: Dict[str, float] = None):
        """
        Register an agent in the economy.
        
        Args:
            agent_id: Agent identifier
            initial_resources: Initial resource holdings
        """
        if agent_id not in self.agent_wealth:
            self.agent_wealth[agent_id] = AgentWealth(
                agent_id=agent_id,
                resources=initial_resources or {},
                total_value=0.0,
                trade_count=0,
                reputation=0.5
            )
            self._update_agent_value(agent_id)
    
    def create_trade_offer(self,
                           seller_id: str,
                           buyer_id: str,
                           resource_offered: str,
                           amount_offered: float,
                           resource_requested: str,
                           amount_requested: float) -> Optional[TradeOffer]:
        """
        Create a new trade offer.
        
        Args:
            seller_id: Agent offering resources
            buyer_id: Agent to receive offer
            resource_offered: Resource being offered
            amount_offered: Amount being offered
            resource_requested: Resource being requested
            amount_requested: Amount being requested
            
        Returns:
            TradeOffer or None if invalid
        """
        # Validate seller has resources
        if seller_id not in self.agent_wealth:
            self.register_agent(seller_id)
        
        seller = self.agent_wealth[seller_id]
        if seller.resources.get(resource_offered, 0) < amount_offered:
            logger.warning(f"Trade rejected: {seller_id} lacks sufficient {resource_offered}")
            return None
        
        trade = TradeOffer(
            trade_id=f"trade_{self.trade_counter}",
            seller_id=seller_id,
            buyer_id=buyer_id,
            resource_offered=resource_offered,
            amount_offered=amount_offered,
            resource_requested=resource_requested,
            amount_requested=amount_requested,
            status=TradeStatus.PENDING,
            created_time=time.time()
        )
        
        self.pending_trades[trade.trade_id] = trade
        self.trade_counter += 1
        self.stats["total_trades"] += 1
        
        # Update demand tracking
        self.resource_demand[resource_requested] += amount_requested
        
        logger.debug(f"Trade offer created: {trade.trade_id}")
        return trade
    
    def accept_trade(self, trade_id: str) -> bool:
        """
        Accept a pending trade.
        
        Args:
            trade_id: ID of trade to accept
            
        Returns:
            True if trade was completed successfully
        """
        if trade_id not in self.pending_trades:
            return False
        
        trade = self.pending_trades[trade_id]
        
        # Validate buyer has resources
        if trade.buyer_id not in self.agent_wealth:
            self.register_agent(trade.buyer_id)
        
        buyer = self.agent_wealth[trade.buyer_id]
        if buyer.resources.get(trade.resource_requested, 0) < trade.amount_requested:
            trade.status = TradeStatus.REJECTED
            logger.warning(f"Trade rejected: {trade.buyer_id} lacks sufficient {trade.resource_requested}")
            return False
        
        # Execute trade
        seller = self.agent_wealth[trade.seller_id]
        
        # Transfer resources
        seller.resources[trade.resource_offered] = seller.resources.get(trade.resource_offered, 0) - trade.amount_offered
        seller.resources[trade.resource_requested] = seller.resources.get(trade.resource_requested, 0) + trade.amount_requested
        
        buyer.resources[trade.resource_requested] = buyer.resources.get(trade.resource_requested, 0) - trade.amount_requested
        buyer.resources[trade.resource_offered] = buyer.resources.get(trade.resource_offered, 0) + trade.amount_offered
        
        # Update trade counts
        seller.trade_count += 1
        buyer.trade_count += 1
        
        # Update trade status
        trade.status = TradeStatus.COMPLETED
        trade.completed_time = time.time()
        
        # Move to completed trades
        del self.pending_trades[trade_id]
        self.trades.append(trade)
        
        # Trim old trades
        if len(self.trades) > self.max_trades:
            self.trades = self.trades[-self.max_trades:]
        
        # Update statistics
        self.stats["successful_trades"] += 1
        trade_value = self._calculate_trade_value(trade)
        self.stats["total_volume"] += trade_value
        
        # Update supply/demand
        self.resource_supply[trade.resource_offered] += trade.amount_offered
        self.resource_supply[trade.resource_requested] -= trade.amount_requested
        
        # Update market prices
        self._update_prices()
        
        # Update agent values
        self._update_agent_value(trade.seller_id)
        self._update_agent_value(trade.buyer_id)
        
        logger.debug(f"Trade completed: {trade.trade_id}")
        return True
    
    def reject_trade(self, trade_id: str) -> bool:
        """
        Reject a pending trade.
        
        Args:
            trade_id: ID of trade to reject
            
        Returns:
            True if trade was rejected
        """
        if trade_id not in self.pending_trades:
            return False
        
        trade = self.pending_trades[trade_id]
        trade.status = TradeStatus.REJECTED
        
        del self.pending_trades[trade_id]
        self.trades.append(trade)
        
        return True
    
    def get_price(self, resource_name: str) -> float:
        """
        Get current market price for a resource.
        
        Args:
            resource_name: Name of resource
            
        Returns:
            Current price
        """
        if resource_name in self.market_prices:
            return self.market_prices[resource_name].current_price
        return self.base_prices.get(resource_name, 1.0)
    
    def _calculate_trade_value(self, trade: TradeOffer) -> float:
        """Calculate the total value of a trade."""
        offered_value = trade.amount_offered * self.get_price(trade.resource_offered)
        requested_value = trade.amount_requested * self.get_price(trade.resource_requested)
        return (offered_value + requested_value) / 2
    
    def _update_prices(self):
        """Update market prices based on supply and demand."""
        current_time = time.time()
        
        for resource_name, market in self.market_prices.items():
            supply = self.resource_supply.get(resource_name, 100.0)
            demand = self.resource_demand.get(resource_name, 100.0)
            
            # Simple supply/demand price adjustment
            if supply > 0:
                ratio = demand / max(supply, 0.01)
                base_price = self.base_prices.get(resource_name, 1.0)
                
                # Price moves toward equilibrium
                target_price = base_price * (0.5 + ratio * 0.5)
                price_change = (target_price - market.current_price) * self.price_sensitivity
                
                new_price = max(0.01, market.current_price + price_change)
                
                # Record price history
                market.price_history.append((current_time, market.current_price))
                if len(market.price_history) > 100:
                    market.price_history = market.price_history[-100:]
                
                market.current_price = new_price
                market.supply = supply
                market.demand = demand
                market.last_updated = current_time
    
    def _update_agent_value(self, agent_id: str):
        """Update an agent's total value."""
        if agent_id not in self.agent_wealth:
            return
        
        agent = self.agent_wealth[agent_id]
        total = 0.0
        
        for resource, amount in agent.resources.items():
            price = self.get_price(resource)
            total += amount * price
        
        agent.total_value = total
    
    def update(self, actions: List[Dict[str, Any]]):
        """
        Update economy based on agent actions.
        
        Args:
            actions: List of agent actions
        """
        for action in actions:
            action_type = action.get("type", "")
            
            if action_type == "produce":
                self._handle_production(action)
            elif action_type == "consume_resource":
                self._handle_consumption(action)
            elif action_type == "create_resource":
                self._handle_creation(action)
        
        # Update aggregate statistics
        self._update_stats()
    
    def _handle_production(self, action: Dict[str, Any]):
        """Handle resource production."""
        agent_id = action.get("agent_id", "")
        resource = action.get("resource_name", "")
        amount = action.get("amount", 0)
        
        if agent_id and resource and amount > 0:
            if agent_id not in self.agent_wealth:
                self.register_agent(agent_id)
            
            self.agent_wealth[agent_id].resources[resource] = \
                self.agent_wealth[agent_id].resources.get(resource, 0) + amount
            
            self.resource_supply[resource] += amount
            self._update_agent_value(agent_id)
    
    def _handle_consumption(self, action: Dict[str, Any]):
        """Handle resource consumption."""
        agent_id = action.get("agent_id", "")
        resource = action.get("resource_name", "")
        amount = action.get("amount", 0)
        
        if agent_id and resource and amount > 0:
            if agent_id in self.agent_wealth:
                current = self.agent_wealth[agent_id].resources.get(resource, 0)
                self.agent_wealth[agent_id].resources[resource] = max(0, current - amount)
                self._update_agent_value(agent_id)
            
            self.resource_demand[resource] += amount
    
    def _handle_creation(self, action: Dict[str, Any]):
        """Handle new resource creation."""
        resource = action.get("name", "")
        amount = action.get("amount", 0)
        
        if resource and amount > 0:
            self.resource_supply[resource] += amount
            
            # Create market if doesn't exist
            if resource not in self.market_prices:
                self.market_prices[resource] = MarketPrice(
                    resource_name=resource,
                    current_price=1.0,
                    supply=amount,
                    demand=0,
                    last_updated=time.time()
                )
    
    def _update_stats(self):
        """Update economic statistics."""
        # Calculate Gini coefficient
        if self.agent_wealth:
            values = sorted([a.total_value for a in self.agent_wealth.values()])
            n = len(values)
            if n > 0 and sum(values) > 0:
                # Gini formula
                index_sum = sum((2 * i - n - 1) * v for i, v in enumerate(values, 1))
                self.stats["gini_coefficient"] = index_sum / (n * sum(values))
            else:
                self.stats["gini_coefficient"] = 0.0
        
        # Average trade value
        if self.stats["successful_trades"] > 0:
            self.stats["avg_trade_value"] = self.stats["total_volume"] / self.stats["successful_trades"]
    
    def get_market_prices(self) -> Dict[str, float]:
        """Get all current market prices."""
        return {name: market.current_price for name, market in self.market_prices.items()}
    
    def get_agent_wealth(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get an agent's wealth information."""
        if agent_id not in self.agent_wealth:
            return None
        
        agent = self.agent_wealth[agent_id]
        return {
            "agent_id": agent.agent_id,
            "resources": agent.resources.copy(),
            "total_value": agent.total_value,
            "trade_count": agent.trade_count,
            "reputation": agent.reputation
        }
    
    def get_wealth_distribution(self) -> List[Dict[str, Any]]:
        """Get wealth distribution across all agents."""
        return sorted(
            [
                {"agent_id": a.agent_id, "total_value": a.total_value, "trade_count": a.trade_count}
                for a in self.agent_wealth.values()
            ],
            key=lambda x: x["total_value"],
            reverse=True
        )
    
    def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trade history."""
        return [
            {
                "trade_id": t.trade_id,
                "seller_id": t.seller_id,
                "buyer_id": t.buyer_id,
                "resource_offered": t.resource_offered,
                "amount_offered": t.amount_offered,
                "resource_requested": t.resource_requested,
                "amount_requested": t.amount_requested,
                "status": t.status.value,
                "created_time": t.created_time
            }
            for t in self.trades[-limit:]
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get economy statistics."""
        return self.stats.copy()
