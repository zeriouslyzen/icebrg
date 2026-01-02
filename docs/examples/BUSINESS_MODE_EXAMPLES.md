# Business Mode Integration Examples

## Overview

This document provides practical examples for integrating ICEBURG's Business Mode System, including agent wallets, payment processing, revenue tracking, and customer interface management.

## Basic Setup

### 1. Initialize Business Mode

```python
from iceburg.business import BusinessModeManager, AgentWallet, PaymentProcessor, RevenueTracker

# Initialize business mode manager
business_manager = BusinessModeManager({
    "operation_mode": "hybrid",
    "platform_fee": 5.0,
    "usdc_integration": {
        "enabled": True,
        "network": "ethereum"
    }
})

# Initialize payment processor
payment_processor = PaymentProcessor()

# Initialize revenue tracker
revenue_tracker = RevenueTracker()
```

### 2. Basic Agent Wallet Operations

```python
import asyncio
from iceburg.business import AgentWallet

async def basic_wallet_operations():
    """Basic agent wallet operations example"""
    
    # Create wallet for surveyor agent
    surveyor_wallet = AgentWallet("surveyor_agent", initial_balance=100.0)
    
    try:
        # Get current balance
        balance = await surveyor_wallet.get_balance()
        print(f"Surveyor wallet balance: {balance} USDC")
        
        # Add funds
        success = await surveyor_wallet.add_funds(50.0, "platform_reward")
        if success:
            print("Added 50 USDC to surveyor wallet")
            balance = await surveyor_wallet.get_balance()
            print(f"New balance: {balance} USDC")
        
        # Deduct funds for service
        success = await surveyor_wallet.deduct_funds(25.0, "research_analysis")
        if success:
            print("Deducted 25 USDC for research analysis")
            balance = await surveyor_wallet.get_balance()
            print(f"New balance: {balance} USDC")
        
        # Get transaction history
        history = await surveyor_wallet.get_transaction_history(limit=10)
        print(f"Transaction history: {history}")
        
    except Exception as e:
        print(f"Error in wallet operations: {e}")

# Run the example
asyncio.run(basic_wallet_operations())
```

### 3. Payment Processing

```python
import asyncio
from iceburg.business import PaymentProcessor

async def payment_processing_example():
    """Payment processing example"""
    
    payment_processor = PaymentProcessor()
    
    try:
        # Process payment request
        payment_request = {
            "customer_id": "customer_123",
            "service_type": "research_analysis",
            "amount": 100.0,
            "currency": "USDC",
            "description": "Market analysis for renewable energy sector"
        }
        
        result = await payment_processor.process_payment(payment_request)
        print(f"Payment processing result: {result}")
        
        # Validate payment
        if result["status"] == "success":
            payment_id = result["payment_id"]
            validation = await payment_processor.validate_payment(payment_id)
            print(f"Payment validation: {validation}")
            
            # Calculate platform fee
            platform_fee = await payment_processor.calculate_platform_fee(100.0)
            print(f"Platform fee: {platform_fee} USDC")
            
            # Distribute payment to agents
            distribution = await payment_processor.distribute_payment(result)
            print(f"Payment distribution: {distribution}")
        
    except Exception as e:
        print(f"Error in payment processing: {e}")

# Run the example
asyncio.run(payment_processing_example())
```

## Advanced Examples

### 4. Revenue Tracking and Analytics

```python
import asyncio
from iceburg.business import RevenueTracker

async def revenue_tracking_example():
    """Revenue tracking and analytics example"""
    
    revenue_tracker = RevenueTracker()
    
    try:
        # Track multiple transactions
        transactions = [
            {
                "transaction_id": "tx_001",
                "amount": 100.0,
                "agent_id": "surveyor_agent",
                "service_type": "research_analysis",
                "timestamp": "2025-10-23T10:00:00Z"
            },
            {
                "transaction_id": "tx_002",
                "amount": 150.0,
                "agent_id": "dissident_agent",
                "service_type": "critical_analysis",
                "timestamp": "2025-10-23T11:00:00Z"
            },
            {
                "transaction_id": "tx_003",
                "amount": 200.0,
                "agent_id": "synthesist_agent",
                "service_type": "synthesis_analysis",
                "timestamp": "2025-10-23T12:00:00Z"
            }
        ]
        
        # Track each transaction
        for transaction in transactions:
            success = await revenue_tracker.track_transaction(transaction)
            if success:
                print(f"Tracked transaction {transaction['transaction_id']}")
        
        # Get revenue metrics
        daily_metrics = await revenue_tracker.get_revenue_metrics("daily")
        print(f"Daily revenue metrics: {daily_metrics}")
        
        # Get agent performance
        surveyor_performance = await revenue_tracker.get_agent_performance("surveyor_agent")
        print(f"Surveyor agent performance: {surveyor_performance}")
        
        # Generate revenue report
        report = await revenue_tracker.generate_revenue_report(
            "2025-10-23", "2025-10-23"
        )
        print(f"Revenue report: {report}")
        
    except Exception as e:
        print(f"Error in revenue tracking: {e}")

# Run the example
asyncio.run(revenue_tracking_example())
```

### 5. Customer Interface Management

```python
import asyncio
from iceburg.business import CustomerInterface, BusinessModeManager

async def customer_interface_example():
    """Customer interface management example"""
    
    customer_interface = CustomerInterface()
    business_manager = BusinessModeManager()
    
    try:
        # Register new customer
        customer_data = {
            "name": "Acme Corporation",
            "email": "contact@acme.com",
            "company": "Acme Corporation",
            "industry": "Technology",
            "budget": 10000.0
        }
        
        customer_id = await customer_interface.register_customer(customer_data)
        print(f"Registered customer: {customer_id}")
        
        # Process service request
        service_request = {
            "service_type": "comprehensive_analysis",
            "description": "Market analysis for AI adoption in healthcare",
            "urgency": "high",
            "budget": 5000.0
        }
        
        service_result = await customer_interface.process_service_request(
            customer_id, service_request
        )
        print(f"Service request result: {service_result}")
        
        # Calculate service cost
        service_cost = await customer_interface.calculate_service_cost(
            "comprehensive_analysis",
            {
                "complexity": 0.8,
                "urgency": "high",
                "budget": 5000.0
            }
        )
        print(f"Calculated service cost: {service_cost} USDC")
        
        # Deliver service
        service_id = service_result["service_id"]
        delivery_success = await customer_interface.deliver_service(
            service_id, customer_id
        )
        print(f"Service delivery: {delivery_success}")
        
    except Exception as e:
        print(f"Error in customer interface: {e}")

# Run the example
asyncio.run(customer_interface_example())
```

### 6. Dynamic Service Pricing

```python
import asyncio
from iceburg.business import BusinessModeManager

async def dynamic_pricing_example():
    """Dynamic service pricing example"""
    
    business_manager = BusinessModeManager()
    
    try:
        # Calculate prices for different service types and complexities
        service_types = [
            "research_analysis",
            "critical_analysis", 
            "synthesis_analysis",
            "comprehensive_analysis"
        ]
        
        complexities = [0.3, 0.5, 0.7, 0.9]
        
        for service_type in service_types:
            print(f"\nService type: {service_type}")
            for complexity in complexities:
                price = await business_manager.calculate_service_price(
                    service_type, complexity
                )
                print(f"  Complexity {complexity}: {price:.2f} USDC")
        
        # Get platform fees
        platform_fees = await business_manager.get_platform_fees()
        print(f"\nPlatform fees: {platform_fees}%")
        
    except Exception as e:
        print(f"Error in dynamic pricing: {e}")

# Run the example
asyncio.run(dynamic_pricing_example())
```

## Multi-Agent Business Examples

### 7. Agent-to-Agent Transactions

```python
import asyncio
from iceburg.business import AgentWallet

async def agent_to_agent_transactions():
    """Agent-to-agent transaction example"""
    
    # Create wallets for multiple agents
    surveyor_wallet = AgentWallet("surveyor_agent", initial_balance=200.0)
    dissident_wallet = AgentWallet("dissident_agent", initial_balance=150.0)
    synthesist_wallet = AgentWallet("synthesist_agent", initial_balance=100.0)
    
    try:
        print("Initial balances:")
        print(f"Surveyor: {await surveyor_wallet.get_balance()} USDC")
        print(f"Dissident: {await dissident_wallet.get_balance()} USDC")
        print(f"Synthesist: {await synthesist_wallet.get_balance()} USDC")
        
        # Surveyor pays Dissident for critical analysis
        print("\nSurveyor paying Dissident for critical analysis...")
        success = await surveyor_wallet.transfer_to_agent("dissident_agent", 50.0)
        if success:
            print("Transfer successful")
            print(f"Surveyor balance: {await surveyor_wallet.get_balance()} USDC")
            print(f"Dissident balance: {await dissident_wallet.get_balance()} USDC")
        
        # Dissident pays Synthesist for synthesis work
        print("\nDissident paying Synthesist for synthesis work...")
        success = await dissident_wallet.transfer_to_agent("synthesist_agent", 30.0)
        if success:
            print("Transfer successful")
            print(f"Dissident balance: {await dissident_wallet.get_balance()} USDC")
            print(f"Synthesist balance: {await synthesist_wallet.get_balance()} USDC")
        
        # Synthesist pays Surveyor for research data
        print("\nSynthesist paying Surveyor for research data...")
        success = await synthesist_wallet.transfer_to_agent("surveyor_agent", 25.0)
        if success:
            print("Transfer successful")
            print(f"Synthesist balance: {await synthesist_wallet.get_balance()} USDC")
            print(f"Surveyor balance: {await surveyor_wallet.get_balance()} USDC")
        
        print("\nFinal balances:")
        print(f"Surveyor: {await surveyor_wallet.get_balance()} USDC")
        print(f"Dissident: {await dissident_wallet.get_balance()} USDC")
        print(f"Synthesist: {await synthesist_wallet.get_balance()} USDC")
        
    except Exception as e:
        print(f"Error in agent-to-agent transactions: {e}")

# Run the example
asyncio.run(agent_to_agent_transactions())
```

### 8. Business Mode Operation Modes

```python
import asyncio
from iceburg.business import BusinessModeManager

async def business_mode_operations():
    """Business mode operation example"""
    
    business_manager = BusinessModeManager()
    
    try:
        # Test different operation modes
        modes = ["research", "business", "hybrid"]
        
        for mode in modes:
            print(f"\nSetting operation mode to: {mode}")
            success = await business_manager.set_operation_mode(mode)
            
            if success:
                print(f"Successfully set mode to {mode}")
                
                # Test service pricing in this mode
                price = await business_manager.calculate_service_price(
                    "research_analysis", 0.7
                )
                print(f"Service price in {mode} mode: {price} USDC")
                
                # Test payment processing
                payment_request = {
                    "customer_id": "test_customer",
                    "service_type": "research_analysis",
                    "amount": price
                }
                
                result = await business_manager.process_payment_request(
                    "test_customer", "research_analysis", price
                )
                print(f"Payment processing result: {result}")
                
            else:
                print(f"Failed to set mode to {mode}")
        
    except Exception as e:
        print(f"Error in business mode operations: {e}")

# Run the example
asyncio.run(business_mode_operations())
```

## Application-Specific Examples

### 9. E-commerce Integration

```python
import asyncio
from iceburg.business import BusinessModeManager, CustomerInterface, PaymentProcessor

class EcommerceIntegration:
    """E-commerce integration with ICEBURG business mode"""
    
    def __init__(self):
        self.business_manager = BusinessModeManager()
        self.customer_interface = CustomerInterface()
        self.payment_processor = PaymentProcessor()
    
    async def process_order(self, customer_id: str, order_items: list):
        """Process e-commerce order"""
        
        try:
            total_amount = 0.0
            
            # Calculate total for each item
            for item in order_items:
                service_type = item["service_type"]
                complexity = item["complexity"]
                
                price = await self.business_manager.calculate_service_price(
                    service_type, complexity
                )
                total_amount += price
                print(f"Item: {service_type}, Price: {price} USDC")
            
            print(f"Total order amount: {total_amount} USDC")
            
            # Process payment
            payment_result = await self.payment_processor.process_payment({
                "customer_id": customer_id,
                "service_type": "ecommerce_order",
                "amount": total_amount,
                "currency": "USDC",
                "items": order_items
            })
            
            if payment_result["status"] == "success":
                print("Order processed successfully")
                
                # Deliver services
                for item in order_items:
                    service_id = f"service_{item['service_type']}_{customer_id}"
                    await self.customer_interface.deliver_service(service_id, customer_id)
                    print(f"Delivered service: {item['service_type']}")
                
                return {"status": "success", "order_id": payment_result["payment_id"]}
            else:
                return {"status": "failed", "error": payment_result["error"]}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}

# Usage example
async def ecommerce_example():
    ecommerce = EcommerceIntegration()
    
    # Sample order
    order_items = [
        {"service_type": "research_analysis", "complexity": 0.6},
        {"service_type": "critical_analysis", "complexity": 0.8},
        {"service_type": "synthesis_analysis", "complexity": 0.7}
    ]
    
    result = await ecommerce.process_order("customer_123", order_items)
    print(f"Order result: {result}")

# Run the example
asyncio.run(ecommerce_example())
```

### 10. Subscription Service Management

```python
import asyncio
from iceburg.business import BusinessModeManager, CustomerInterface, RevenueTracker

class SubscriptionService:
    """Subscription service management"""
    
    def __init__(self):
        self.business_manager = BusinessModeManager()
        self.customer_interface = CustomerInterface()
        self.revenue_tracker = RevenueTracker()
        self.subscriptions = {}
    
    async def create_subscription(self, customer_id: str, plan: str, duration_months: int):
        """Create subscription for customer"""
        
        try:
            # Calculate subscription cost
            base_price = await self.business_manager.calculate_service_price(
                f"subscription_{plan}", 0.5
            )
            total_cost = base_price * duration_months
            
            print(f"Creating {plan} subscription for {duration_months} months")
            print(f"Total cost: {total_cost} USDC")
            
            # Process subscription payment
            payment_result = await self.business_manager.process_payment_request(
                customer_id, f"subscription_{plan}", total_cost
            )
            
            if payment_result["status"] == "success":
                # Create subscription record
                subscription_id = f"sub_{customer_id}_{plan}"
                self.subscriptions[subscription_id] = {
                    "customer_id": customer_id,
                    "plan": plan,
                    "duration_months": duration_months,
                    "start_date": "2025-10-23",
                    "status": "active"
                }
                
                # Track revenue
                await self.revenue_tracker.track_transaction({
                    "transaction_id": payment_result["payment_id"],
                    "amount": total_cost,
                    "agent_id": "subscription_system",
                    "service_type": f"subscription_{plan}"
                })
                
                print(f"Subscription created: {subscription_id}")
                return {"status": "success", "subscription_id": subscription_id}
            else:
                return {"status": "failed", "error": payment_result["error"]}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def get_subscription_status(self, subscription_id: str):
        """Get subscription status"""
        return self.subscriptions.get(subscription_id, {"status": "not_found"})

# Usage example
async def subscription_example():
    subscription_service = SubscriptionService()
    
    # Create subscription
    result = await subscription_service.create_subscription(
        "customer_123", "premium", 12
    )
    print(f"Subscription creation result: {result}")
    
    # Check status
    if result["status"] == "success":
        status = await subscription_service.get_subscription_status(
            result["subscription_id"]
        )
        print(f"Subscription status: {status}")

# Run the example
asyncio.run(subscription_example())
```

## Error Handling and Best Practices

### 11. Robust Error Handling

```python
import asyncio
import logging
from iceburg.business import BusinessModeManager, AgentWallet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def robust_business_operations():
    """Robust business operations with error handling"""
    
    business_manager = BusinessModeManager()
    wallet = AgentWallet("test_agent", initial_balance=100.0)
    
    try:
        # Test business operations with error handling
        operations = [
            ("set_operation_mode", "hybrid"),
            ("calculate_service_price", ("research_analysis", 0.7)),
            ("get_platform_fees", None)
        ]
        
        for operation, args in operations:
            try:
                if args is None:
                    result = await getattr(business_manager, operation)()
                else:
                    result = await getattr(business_manager, operation)(*args)
                
                logger.info(f"Operation {operation} successful: {result}")
                
            except Exception as e:
                logger.error(f"Operation {operation} failed: {e}")
                
                # Implement fallback logic
                if operation == "set_operation_mode":
                    logger.info("Falling back to research mode")
                    await business_manager.set_operation_mode("research")
                elif operation == "calculate_service_price":
                    logger.info("Using default pricing")
                    result = 50.0  # Default price
                    logger.info(f"Default price: {result}")
        
        # Test wallet operations with error handling
        wallet_operations = [
            ("get_balance", None),
            ("add_funds", (50.0, "test_source")),
            ("deduct_funds", (25.0, "test_purpose"))
        ]
        
        for operation, args in wallet_operations:
            try:
                if args is None:
                    result = await getattr(wallet, operation)()
                else:
                    result = await getattr(wallet, operation)(*args)
                
                logger.info(f"Wallet operation {operation} successful: {result}")
                
            except Exception as e:
                logger.error(f"Wallet operation {operation} failed: {e}")
        
    except Exception as e:
        logger.error(f"Critical error in business operations: {e}")

# Run the example
asyncio.run(robust_business_operations())
```

### 12. Configuration Management

```python
import asyncio
import json
from iceburg.business import BusinessModeManager

class BusinessConfigManager:
    """Configuration manager for business mode"""
    
    def __init__(self, config_file: str = "business_config.json"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            # Return default configuration
            return {
                "operation_mode": "hybrid",
                "platform_fee": 5.0,
                "usdc_integration": {
                    "enabled": True,
                    "network": "ethereum"
                },
                "payment_processing": {
                    "enabled": True,
                    "auto_pricing": True
                }
            }
    
    def save_config(self):
        """Save configuration to file"""
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=2)
    
    async def run_with_config(self):
        """Run business operations with loaded configuration"""
        
        business_manager = BusinessModeManager(self.config)
        
        try:
            # Set operation mode from config
            await business_manager.set_operation_mode(self.config["operation_mode"])
            print(f"Operation mode set to: {self.config['operation_mode']}")
            
            # Test service pricing
            price = await business_manager.calculate_service_price(
                "research_analysis", 0.7
            )
            print(f"Service price: {price} USDC")
            
            # Test platform fees
            fees = await business_manager.get_platform_fees()
            print(f"Platform fees: {fees}%")
            
        except Exception as e:
            print(f"Error in configured business operations: {e}")

# Usage example
async def config_manager_example():
    config_manager = BusinessConfigManager()
    await config_manager.run_with_config()

# Run the example
asyncio.run(config_manager_example())
```

## Summary

These examples demonstrate how to integrate ICEBURG's Business Mode System with various applications, from basic wallet operations to advanced e-commerce integration. The system provides:

- **Agent wallet management** (balance tracking, transactions, transfers)
- **Payment processing** (USDC integration, validation, distribution)
- **Revenue tracking** (analytics, reporting, performance metrics)
- **Customer interface** (registration, service requests, delivery)
- **Dynamic pricing** (service cost calculation, platform fees)
- **Multi-agent transactions** (agent-to-agent payments)
- **Business mode operations** (research, business, hybrid modes)

The examples show both basic usage and advanced integration patterns, with proper error handling and configuration management for production use.
