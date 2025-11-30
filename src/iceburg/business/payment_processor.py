"""
ICEBURG Payment Processor
Handles payments between agents and customers
"""

import asyncio
import json
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .agent_wallet import AgentWallet, Transaction

logger = logging.getLogger(__name__)

@dataclass
class PaymentRequest:
    """Represents a payment request"""
    request_id: str
    customer_id: str
    agent_id: str
    service_type: str
    amount: float
    currency: str
    description: str
    timestamp: str
    status: str = "pending"  # 'pending', 'processing', 'completed', 'failed'

@dataclass
class PlatformFee:
    """Represents platform fee structure"""
    service_fee_percentage: float = 0.10  # 10% platform fee
    transaction_fee_fixed: float = 0.0    # Fixed transaction fee
    minimum_fee: float = 1.0              # Minimum fee in USDC

class PaymentProcessor:
    """Processes payments between agents and customers"""
    
    def __init__(self, platform_wallet: Optional[AgentWallet] = None):
        self.platform_wallet = platform_wallet or AgentWallet("platform")
        self.payment_requests: List[PaymentRequest] = []
        self.fee_structure = PlatformFee()
        self.load_payment_data()
    
    def load_payment_data(self):
        """Load payment data from file"""
        try:
            payment_file = Path("data/business/payment_requests.json")
            if payment_file.exists():
                with open(payment_file, 'r') as f:
                    data = json.load(f)
                    self.payment_requests = [PaymentRequest(**req) for req in data.get('requests', [])]
                logger.info("Loaded payment request data")
        except Exception as e:
            logger.warning(f"Failed to load payment data: {e}")
    
    def save_payment_data(self):
        """Save payment data to file"""
        try:
            payment_file = Path("data/business/payment_requests.json")
            payment_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'requests': [asdict(req) for req in self.payment_requests]
            }
            with open(payment_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Saved payment request data")
        except Exception as e:
            logger.warning(f"Failed to save payment data: {e}")
    
    async def create_payment_request(self, customer_id: str, agent_id: str, 
                                   service_type: str, amount: float, 
                                   currency: str, description: str) -> PaymentRequest:
        """Create a new payment request"""
        try:
            request = PaymentRequest(
                request_id=f"pay_{len(self.payment_requests)}_{customer_id}",
                customer_id=customer_id,
                agent_id=agent_id,
                service_type=service_type,
                amount=amount,
                currency=currency,
                description=description,
                timestamp=str(asyncio.get_event_loop().time())
            )
            
            self.payment_requests.append(request)
            self.save_payment_data()
            
            logger.info(f"Created payment request {request.request_id} for {amount} {currency}")
            return request
            
        except Exception as e:
            logger.error(f"Failed to create payment request: {e}")
            raise
    
    async def process_payment(self, request_id: str, customer_wallet: AgentWallet) -> Tuple[bool, str]:
        """Process a payment request"""
        try:
            # Find the payment request
            request = None
            for req in self.payment_requests:
                if req.request_id == request_id:
                    request = req
                    break
            
            if not request:
                return False, "Payment request not found"
            
            if request.status != "pending":
                return False, f"Payment request already {request.status}"
            
            # Check customer balance
            if request.currency == "USDC" and customer_wallet.balance.usdc_balance < request.amount:
                return False, "Insufficient customer balance"
            elif request.currency == "ENERGY" and customer_wallet.balance.energy_tokens < request.amount:
                return False, "Insufficient customer energy tokens"
            
            # Calculate platform fee
            platform_fee = self.calculate_platform_fee(request.amount, request.currency)
            agent_amount = request.amount - platform_fee
            
            # Update request status
            request.status = "processing"
            self.save_payment_data()
            
            # Process customer payment
            customer_success = await customer_wallet.spend_money(
                request.amount, 
                request.currency, 
                f"Payment for {request.service_type}"
            )
            
            if not customer_success:
                request.status = "failed"
                self.save_payment_data()
                return False, "Failed to process customer payment"
            
            # Get agent wallet
            agent_wallet = AgentWallet(request.agent_id)
            
            # Pay agent
            agent_success = await agent_wallet.receive_payment(
                agent_amount,
                request.currency,
                request.customer_id,
                f"Payment for {request.service_type}"
            )
            
            if not agent_success:
                # Refund customer
                await customer_wallet.receive_payment(
                    request.amount,
                    request.currency,
                    "system",
                    f"Refund for failed payment {request.request_id}"
                )
                request.status = "failed"
                self.save_payment_data()
                return False, "Failed to pay agent"
            
            # Pay platform fee
            if platform_fee > 0:
                platform_success = await self.platform_wallet.receive_payment(
                    platform_fee,
                    request.currency,
                    request.customer_id,
                    f"Platform fee for {request.service_type}"
                )
                
                if not platform_success:
                    logger.warning(f"Failed to collect platform fee for {request.request_id}")
            
            # Update request status
            request.status = "completed"
            self.save_payment_data()
            
            logger.info(f"Successfully processed payment {request.request_id}")
            return True, "Payment processed successfully"
            
        except Exception as e:
            logger.error(f"Failed to process payment {request_id}: {e}")
            return False, f"Payment processing error: {e}"
    
    def calculate_platform_fee(self, amount: float, currency: str) -> float:
        """Calculate platform fee for a payment"""
        if currency == "USDC":
            # Calculate percentage fee
            percentage_fee = amount * self.fee_structure.service_fee_percentage
            
            # Add fixed fee
            total_fee = percentage_fee + self.fee_structure.transaction_fee_fixed
            
            # Ensure minimum fee
            return max(total_fee, self.fee_structure.minimum_fee)
        else:
            # For energy tokens, use a smaller fee
            return amount * 0.05  # 5% fee for energy tokens
    
    async def get_agent_earnings(self, agent_id: str, time_period: str = "all") -> Dict:
        """Get agent earnings summary"""
        try:
            agent_wallet = AgentWallet(agent_id)
            balance = agent_wallet.get_balance()
            transactions = agent_wallet.get_transaction_history(100)
            
            # Filter transactions by time period if needed
            if time_period != "all":
                # Implement time filtering logic here
                pass
            
            # Calculate earnings by service type
            earnings_by_service = {}
            for tx in transactions:
                if tx.transaction_type == "earn":
                    service_type = tx.description.split(":")[0] if ":" in tx.description else "general"
                    if service_type not in earnings_by_service:
                        earnings_by_service[service_type] = 0
                    earnings_by_service[service_type] += tx.amount
            
            return {
                "agent_id": agent_id,
                "total_earnings": balance.total_earnings,
                "current_balance": balance.usdc_balance,
                "energy_tokens": balance.energy_tokens,
                "total_investments": balance.total_investments,
                "net_worth": agent_wallet.get_net_worth(),
                "earnings_by_service": earnings_by_service,
                "recent_transactions": [asdict(tx) for tx in transactions[-10:]]
            }
            
        except Exception as e:
            logger.error(f"Failed to get earnings for agent {agent_id}: {e}")
            return {}
    
    async def get_platform_revenue(self, time_period: str = "all") -> Dict:
        """Get platform revenue summary"""
        try:
            platform_balance = self.platform_wallet.get_balance()
            platform_transactions = self.platform_wallet.get_transaction_history(100)
            
            # Calculate revenue by service type
            revenue_by_service = {}
            for tx in platform_transactions:
                if tx.transaction_type == "receive":
                    service_type = tx.description.split(":")[0] if ":" in tx.description else "general"
                    if service_type not in revenue_by_service:
                        revenue_by_service[service_type] = 0
                    revenue_by_service[service_type] += tx.amount
            
            return {
                "total_platform_revenue": platform_balance.total_earnings,
                "current_platform_balance": platform_balance.usdc_balance,
                "revenue_by_service": revenue_by_service,
                "recent_transactions": [asdict(tx) for tx in platform_transactions[-10:]]
            }
            
        except Exception as e:
            logger.error(f"Failed to get platform revenue: {e}")
            return {}
    
    def get_payment_requests(self, status: Optional[str] = None) -> List[PaymentRequest]:
        """Get payment requests, optionally filtered by status"""
        if status:
            return [req for req in self.payment_requests if req.status == status]
        return self.payment_requests
