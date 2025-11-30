"""
ICEBURG Revenue Tracker
Tracks revenue and performance metrics
"""

import asyncio
import json
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class RevenueMetrics:
    """Represents revenue metrics for a time period"""
    period: str
    total_revenue: float
    platform_fees: float
    agent_earnings: float
    transaction_count: int
    average_transaction_value: float
    top_performing_agent: str
    revenue_by_service: Dict[str, float]
    timestamp: str

@dataclass
class AgentPerformance:
    """Represents agent performance metrics"""
    agent_id: str
    total_earnings: float
    service_count: int
    average_service_value: float
    customer_rating: float
    success_rate: float
    revenue_growth: float
    timestamp: str

class RevenueTracker:
    """Tracks revenue and performance metrics for ICEBURG business operations"""
    
    def __init__(self, metrics_file: Optional[Path] = None):
        self.metrics_file = metrics_file or Path("data/business/revenue_metrics.json")
        self.revenue_metrics: List[RevenueMetrics] = []
        self.agent_performance: List[AgentPerformance] = []
        self.load_metrics()
    
    def load_metrics(self):
        """Load revenue metrics from file"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.revenue_metrics = [RevenueMetrics(**metric) for metric in data.get('revenue_metrics', [])]
                    self.agent_performance = [AgentPerformance(**perf) for perf in data.get('agent_performance', [])]
                logger.info(f"Loaded {len(self.revenue_metrics)} revenue metrics and {len(self.agent_performance)} agent performance records")
        except Exception as e:
            logger.warning(f"Failed to load revenue metrics: {e}")
    
    def save_metrics(self):
        """Save revenue metrics to file"""
        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                'revenue_metrics': [asdict(metric) for metric in self.revenue_metrics],
                'agent_performance': [asdict(perf) for perf in self.agent_performance]
            }
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved {len(self.revenue_metrics)} revenue metrics and {len(self.agent_performance)} agent performance records")
        except Exception as e:
            logger.warning(f"Failed to save revenue metrics: {e}")
    
    async def record_transaction(self, agent_id: str, amount: float, service_type: str, 
                               platform_fee: float, customer_rating: Optional[float] = None):
        """Record a completed transaction"""
        try:
            timestamp = datetime.now().isoformat()
            
            # Update revenue metrics
            await self._update_revenue_metrics(amount, platform_fee, service_type, timestamp)
            
            # Update agent performance
            await self._update_agent_performance(agent_id, amount, service_type, customer_rating, timestamp)
            
            logger.info(f"Recorded transaction: {agent_id} earned {amount} USDC from {service_type}")
            
        except Exception as e:
            logger.error(f"Failed to record transaction: {e}")
    
    async def _update_revenue_metrics(self, amount: float, platform_fee: float, 
                                    service_type: str, timestamp: str):
        """Update revenue metrics"""
        try:
            # Get current period (daily for now)
            current_period = datetime.now().strftime("%Y-%m-%d")
            
            # Find existing metrics for current period
            existing_metrics = None
            for metric in self.revenue_metrics:
                if metric.period == current_period:
                    existing_metrics = metric
                    break
            
            if existing_metrics:
                # Update existing metrics
                existing_metrics.total_revenue += amount
                existing_metrics.platform_fees += platform_fee
                existing_metrics.agent_earnings += (amount - platform_fee)
                existing_metrics.transaction_count += 1
                existing_metrics.average_transaction_value = existing_metrics.total_revenue / existing_metrics.transaction_count
                
                # Update service revenue
                if service_type not in existing_metrics.revenue_by_service:
                    existing_metrics.revenue_by_service[service_type] = 0
                existing_metrics.revenue_by_service[service_type] += amount
                
                existing_metrics.timestamp = timestamp
            else:
                # Create new metrics for current period
                new_metrics = RevenueMetrics(
                    period=current_period,
                    total_revenue=amount,
                    platform_fees=platform_fee,
                    agent_earnings=amount - platform_fee,
                    transaction_count=1,
                    average_transaction_value=amount,
                    top_performing_agent="",  # Will be calculated later
                    revenue_by_service={service_type: amount},
                    timestamp=timestamp
                )
                self.revenue_metrics.append(new_metrics)
            
            self.save_metrics()
            
        except Exception as e:
            logger.error(f"Failed to update revenue metrics: {e}")
    
    async def _update_agent_performance(self, agent_id: str, amount: float, 
                                      service_type: str, customer_rating: Optional[float], 
                                      timestamp: str):
        """Update agent performance metrics"""
        try:
            # Find existing performance record for agent
            existing_performance = None
            for perf in self.agent_performance:
                if perf.agent_id == agent_id:
                    existing_performance = perf
                    break
            
            if existing_performance:
                # Update existing performance
                existing_performance.total_earnings += amount
                existing_performance.service_count += 1
                existing_performance.average_service_value = existing_performance.total_earnings / existing_performance.service_count
                
                if customer_rating is not None:
                    # Update customer rating (weighted average)
                    existing_performance.customer_rating = (existing_performance.customer_rating * 0.8) + (customer_rating * 0.2)
                
                existing_performance.timestamp = timestamp
            else:
                # Create new performance record
                new_performance = AgentPerformance(
                    agent_id=agent_id,
                    total_earnings=amount,
                    service_count=1,
                    average_service_value=amount,
                    customer_rating=customer_rating or 5.0,
                    success_rate=1.0,  # Assume success for now
                    revenue_growth=0.0,  # Will be calculated later
                    timestamp=timestamp
                )
                self.agent_performance.append(new_performance)
            
            self.save_metrics()
            
        except Exception as e:
            logger.error(f"Failed to update agent performance: {e}")
    
    def get_revenue_summary(self, period_days: int = 30) -> Dict[str, Any]:
        """Get revenue summary for specified period"""
        try:
            cutoff_date = datetime.now() - timedelta(days=period_days)
            
            # Filter metrics by period
            recent_metrics = [
                metric for metric in self.revenue_metrics
                if datetime.fromisoformat(metric.timestamp) >= cutoff_date
            ]
            
            if not recent_metrics:
                return {
                    "period_days": period_days,
                    "total_revenue": 0.0,
                    "platform_fees": 0.0,
                    "agent_earnings": 0.0,
                    "transaction_count": 0,
                    "average_transaction_value": 0.0,
                    "top_services": {},
                    "revenue_trend": "No data available"
                }
            
            # Calculate totals
            total_revenue = sum(metric.total_revenue for metric in recent_metrics)
            total_platform_fees = sum(metric.platform_fees for metric in recent_metrics)
            total_agent_earnings = sum(metric.agent_earnings for metric in recent_metrics)
            total_transactions = sum(metric.transaction_count for metric in recent_metrics)
            
            # Calculate top services
            service_revenue = {}
            for metric in recent_metrics:
                for service, revenue in metric.revenue_by_service.items():
                    if service not in service_revenue:
                        service_revenue[service] = 0
                    service_revenue[service] += revenue
            
            top_services = dict(sorted(service_revenue.items(), key=lambda x: x[1], reverse=True)[:5])
            
            # Calculate revenue trend
            if len(recent_metrics) >= 2:
                recent_revenue = recent_metrics[-1].total_revenue
                previous_revenue = recent_metrics[-2].total_revenue
                if previous_revenue > 0:
                    trend_percentage = ((recent_revenue - previous_revenue) / previous_revenue) * 100
                    revenue_trend = f"{trend_percentage:+.1f}%"
                else:
                    revenue_trend = "New"
            else:
                revenue_trend = "Insufficient data"
            
            return {
                "period_days": period_days,
                "total_revenue": total_revenue,
                "platform_fees": total_platform_fees,
                "agent_earnings": total_agent_earnings,
                "transaction_count": total_transactions,
                "average_transaction_value": total_revenue / total_transactions if total_transactions > 0 else 0,
                "top_services": top_services,
                "revenue_trend": revenue_trend,
                "metrics_count": len(recent_metrics)
            }
            
        except Exception as e:
            logger.error(f"Failed to get revenue summary: {e}")
            return {"error": str(e)}
    
    def get_agent_performance_summary(self) -> Dict[str, Any]:
        """Get agent performance summary"""
        try:
            if not self.agent_performance:
                return {"error": "No agent performance data available"}
            
            # Calculate performance metrics
            total_agents = len(self.agent_performance)
            total_earnings = sum(perf.total_earnings for perf in self.agent_performance)
            total_services = sum(perf.service_count for perf in self.agent_performance)
            average_rating = sum(perf.customer_rating for perf in self.agent_performance) / total_agents
            
            # Get top performing agents
            top_agents = sorted(self.agent_performance, key=lambda x: x.total_earnings, reverse=True)[:5]
            
            # Get performance by agent
            agent_performance = {}
            for perf in self.agent_performance:
                agent_performance[perf.agent_id] = {
                    "total_earnings": perf.total_earnings,
                    "service_count": perf.service_count,
                    "average_service_value": perf.average_service_value,
                    "customer_rating": perf.customer_rating,
                    "success_rate": perf.success_rate
                }
            
            return {
                "total_agents": total_agents,
                "total_earnings": total_earnings,
                "total_services": total_services,
                "average_rating": average_rating,
                "top_agents": [asdict(agent) for agent in top_agents],
                "agent_performance": agent_performance
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent performance summary: {e}")
            return {"error": str(e)}
    
    def get_platform_health_metrics(self) -> Dict[str, Any]:
        """Get platform health metrics"""
        try:
            revenue_summary = self.get_revenue_summary(30)
            agent_performance = self.get_agent_performance_summary()
            
            # Calculate health score
            health_score = 0
            
            # Revenue health (40% weight)
            if revenue_summary.get("total_revenue", 0) > 0:
                health_score += 40
            
            # Agent performance health (30% weight)
            if agent_performance.get("total_agents", 0) > 0:
                health_score += 30
            
            # Transaction volume health (20% weight)
            if revenue_summary.get("transaction_count", 0) > 10:
                health_score += 20
            
            # Customer satisfaction health (10% weight)
            if agent_performance.get("average_rating", 0) > 4.0:
                health_score += 10
            
            # Determine health status
            if health_score >= 80:
                health_status = "Excellent"
            elif health_score >= 60:
                health_status = "Good"
            elif health_score >= 40:
                health_status = "Fair"
            else:
                health_status = "Needs Improvement"
            
            return {
                "health_score": health_score,
                "health_status": health_status,
                "revenue_summary": revenue_summary,
                "agent_performance": agent_performance,
                "recommendations": self._generate_recommendations(health_score, revenue_summary, agent_performance)
            }
            
        except Exception as e:
            logger.error(f"Failed to get platform health metrics: {e}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, health_score: int, revenue_summary: Dict, 
                                agent_performance: Dict) -> List[str]:
        """Generate recommendations based on health metrics"""
        recommendations = []
        
        if health_score < 40:
            recommendations.append("Consider launching marketing campaigns to attract more customers")
            recommendations.append("Review agent pricing to ensure competitive rates")
            recommendations.append("Improve agent training to enhance service quality")
        
        if revenue_summary.get("transaction_count", 0) < 10:
            recommendations.append("Focus on increasing transaction volume")
            recommendations.append("Consider offering promotional pricing for new customers")
        
        if agent_performance.get("average_rating", 0) < 4.0:
            recommendations.append("Implement customer feedback system to improve service quality")
            recommendations.append("Provide additional training for agents with low ratings")
        
        if revenue_summary.get("total_revenue", 0) < 1000:
            recommendations.append("Consider expanding service offerings")
            recommendations.append("Target enterprise customers for higher-value transactions")
        
        if not recommendations:
            recommendations.append("Platform is performing well - continue current strategies")
        
        return recommendations
