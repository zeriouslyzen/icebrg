"""
ICEBURG Customer Interface
Interface for customers to interact with agents
"""

import asyncio
import json
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

from .business_mode import BusinessMode
from .character_system import CharacterSystem
from .payment_processor import PaymentProcessor
from .revenue_tracker import RevenueTracker

logger = logging.getLogger(__name__)

@dataclass
class Customer:
    """Represents a customer"""
    customer_id: str
    name: str
    email: str
    total_spent: float = 0.0
    total_services: int = 0
    average_rating: float = 5.0
    preferred_agents: List[str] = None
    created_at: str = ""
    
    def __post_init__(self):
        if self.preferred_agents is None:
            self.preferred_agents = []

@dataclass
class ServiceRequest:
    """Represents a service request from a customer"""
    request_id: str
    customer_id: str
    agent_type: str
    service_description: str
    budget: float
    urgency: str = "normal"  # 'low', 'normal', 'high', 'urgent'
    status: str = "pending"  # 'pending', 'accepted', 'in_progress', 'completed', 'cancelled'
    created_at: str = ""
    completed_at: str = ""

class CustomerInterface:
    """Interface for customers to interact with ICEBURG agents"""
    
    def __init__(self):
        self.business_mode = BusinessMode()
        self.character_system = CharacterSystem()
        self.payment_processor = PaymentProcessor()
        self.revenue_tracker = RevenueTracker()
        self.customers: Dict[str, Customer] = {}
        self.service_requests: List[ServiceRequest] = []
        self.load_customer_data()
    
    def load_customer_data(self):
        """Load customer data from file"""
        try:
            customer_file = Path("data/business/customers.json")
            if customer_file.exists():
                with open(customer_file, 'r') as f:
                    data = json.load(f)
                    self.customers = {
                        customer_id: Customer(**customer_data)
                        for customer_id, customer_data in data.get('customers', {}).items()
                    }
                    self.service_requests = [
                        ServiceRequest(**request_data)
                        for request_data in data.get('service_requests', [])
                    ]
                logger.info(f"Loaded {len(self.customers)} customers and {len(self.service_requests)} service requests")
        except Exception as e:
            logger.warning(f"Failed to load customer data: {e}")
    
    def save_customer_data(self):
        """Save customer data to file"""
        try:
            customer_file = Path("data/business/customers.json")
            customer_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'customers': {
                    customer_id: asdict(customer)
                    for customer_id, customer in self.customers.items()
                },
                'service_requests': [asdict(request) for request in self.service_requests]
            }
            with open(customer_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.customers)} customers and {len(self.service_requests)} service requests")
        except Exception as e:
            logger.warning(f"Failed to save customer data: {e}")
    
    async def register_customer(self, name: str, email: str) -> str:
        """Register a new customer"""
        try:
            customer_id = f"customer_{len(self.customers)}_{int(asyncio.get_event_loop().time())}"
            
            customer = Customer(
                customer_id=customer_id,
                name=name,
                email=email,
                created_at=str(asyncio.get_event_loop().time())
            )
            
            self.customers[customer_id] = customer
            self.save_customer_data()
            
            logger.info(f"Registered new customer: {name} ({email})")
            return customer_id
            
        except Exception as e:
            logger.error(f"Failed to register customer: {e}")
            raise
    
    async def browse_agents(self, customer_id: str) -> Dict[str, Any]:
        """Browse available agents and their services"""
        try:
            if not self.business_mode.is_business_mode_active():
                return {"error": "Business mode is not active"}
            
            agents_info = {}
            characters = self.character_system.get_all_characters()
            
            for agent_id, character in characters.items():
                agents_info[agent_id] = {
                    "name": character.name,
                    "personality": character.personality,
                    "signature_phrase": character.signature_phrase,
                    "service_description": character.service_description,
                    "base_price": character.base_price,
                    "specializations": character.specializations,
                    "reputation_score": character.reputation_score,
                    "customer_rating": character.customer_rating,
                    "total_services": character.total_services
                }
            
            return {
                "customer_id": customer_id,
                "available_agents": agents_info,
                "business_mode_active": True
            }
            
        except Exception as e:
            logger.error(f"Failed to browse agents: {e}")
            return {"error": str(e)}
    
    async def interact_with_agent(self, customer_id: str, agent_id: str, 
                                interaction_type: str, message: str) -> str:
        """Interact with a specific agent"""
        try:
            if not self.business_mode.is_business_mode_active():
                return "Business mode is not active. Please enable business mode to interact with agents."
            
            # Generate character-appropriate response
            response = await self.character_system.generate_character_interaction(
                agent_id, customer_id, interaction_type, message
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to interact with agent: {e}")
            return "I apologize, but I'm having trouble processing your request right now."
    
    async def request_service(self, customer_id: str, agent_type: str, 
                            service_description: str, budget: float, 
                            urgency: str = "normal") -> Dict[str, Any]:
        """Request a service from an agent"""
        try:
            if not self.business_mode.is_business_mode_active():
                return {"error": "Business mode is not active"}
            
            # Create service request
            request = ServiceRequest(
                request_id=f"service_{len(self.service_requests)}_{customer_id}",
                customer_id=customer_id,
                agent_type=agent_type,
                service_description=service_description,
                budget=budget,
                urgency=urgency,
                created_at=str(asyncio.get_event_loop().time())
            )
            
            self.service_requests.append(request)
            self.save_customer_data()
            
            # Get agent character info
            character = self.character_system.get_character(agent_type)
            if not character:
                return {"error": f"Agent {agent_type} not found"}
            
            # Calculate estimated price
            estimated_price = await self.business_mode._calculate_service_price(
                {"query": service_description}, agent_type
            )
            
            # Check if budget is sufficient
            budget_sufficient = budget >= estimated_price
            
            return {
                "request_id": request.request_id,
                "agent_type": agent_type,
                "agent_name": character.name,
                "service_description": service_description,
                "estimated_price": estimated_price,
                "budget": budget,
                "budget_sufficient": budget_sufficient,
                "urgency": urgency,
                "status": "pending",
                "agent_response": await self.character_system.generate_character_interaction(
                    agent_type, customer_id, "service_request", service_description
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to request service: {e}")
            return {"error": str(e)}
    
    async def process_payment(self, request_id: str, customer_id: str) -> Dict[str, Any]:
        """Process payment for a service request"""
        try:
            # Find the service request
            request = None
            for req in self.service_requests:
                if req.request_id == request_id and req.customer_id == customer_id:
                    request = req
                    break
            
            if not request:
                return {"error": "Service request not found"}
            
            if request.status != "pending":
                return {"error": f"Service request is already {request.status}"}
            
            # Get customer wallet (create if doesn't exist)
            customer_wallet = self.payment_processor.platform_wallet  # Simplified for now
            
            # Calculate price
            estimated_price = await self.business_mode._calculate_service_price(
                {"query": request.service_description}, request.agent_type
            )
            
            # Create payment request
            payment_request = await self.payment_processor.create_payment_request(
                customer_id=customer_id,
                agent_id=request.agent_type,
                service_type=f"{request.agent_type}_service",
                amount=estimated_price,
                currency="USDC",
                description=request.service_description
            )
            
            # Process payment
            success, message = await self.payment_processor.process_payment(
                payment_request.request_id, customer_wallet
            )
            
            if success:
                # Update service request status
                request.status = "accepted"
                request.completed_at = str(asyncio.get_event_loop().time())
                
                # Update customer spending
                if customer_id in self.customers:
                    self.customers[customer_id].total_spent += estimated_price
                    self.customers[customer_id].total_services += 1
                
                # Record transaction
                await self.revenue_tracker.record_transaction(
                    request.agent_type, estimated_price, f"{request.agent_type}_service",
                    self.payment_processor.calculate_platform_fee(estimated_price, "USDC")
                )
                
                self.save_customer_data()
                
                return {
                    "success": True,
                    "message": "Payment processed successfully",
                    "amount": estimated_price,
                    "currency": "USDC",
                    "service_status": "accepted"
                }
            else:
                return {
                    "success": False,
                    "message": message,
                    "error": "Payment processing failed"
                }
            
        except Exception as e:
            logger.error(f"Failed to process payment: {e}")
            return {"error": str(e)}
    
    async def get_customer_dashboard(self, customer_id: str) -> Dict[str, Any]:
        """Get customer dashboard information"""
        try:
            customer = self.customers.get(customer_id)
            if not customer:
                return {"error": "Customer not found"}
            
            # Get customer's service requests
            customer_requests = [
                req for req in self.service_requests 
                if req.customer_id == customer_id
            ]
            
            # Get recent transactions
            recent_transactions = []
            for req in customer_requests[-5:]:  # Last 5 requests
                if req.status == "completed":
                    recent_transactions.append({
                        "request_id": req.request_id,
                        "agent_type": req.agent_type,
                        "service_description": req.service_description,
                        "completed_at": req.completed_at
                    })
            
            # Get available agents
            available_agents = await self.browse_agents(customer_id)
            
            return {
                "customer_info": asdict(customer),
                "recent_requests": [asdict(req) for req in customer_requests[-10:]],
                "recent_transactions": recent_transactions,
                "available_agents": available_agents.get("available_agents", {}),
                "business_mode_active": self.business_mode.is_business_mode_active()
            }
            
        except Exception as e:
            logger.error(f"Failed to get customer dashboard: {e}")
            return {"error": str(e)}
    
    async def get_platform_status(self) -> Dict[str, Any]:
        """Get platform status for customers"""
        try:
            platform_summary = await self.business_mode.get_platform_summary()
            revenue_summary = self.revenue_tracker.get_revenue_summary(30)
            health_metrics = self.revenue_tracker.get_platform_health_metrics()
            
            return {
                "platform_status": "active" if self.business_mode.is_business_mode_active() else "inactive",
                "business_mode": self.business_mode.get_mode(),
                "total_customers": len(self.customers),
                "total_agents": len(self.character_system.get_all_characters()),
                "platform_summary": platform_summary,
                "revenue_summary": revenue_summary,
                "health_metrics": health_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get platform status: {e}")
            return {"error": str(e)}
