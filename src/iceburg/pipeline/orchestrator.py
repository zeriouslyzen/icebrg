"""
ICEBURG Pipeline Orchestrator

Main orchestrator for the financial analysis pipeline,
coordinating all components and providing a unified interface.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import asyncio
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ..config import IceburgConfig
from .financial_pipeline import FinancialAnalysisPipeline, PipelineConfig
from .monitoring import PipelineMonitor, MonitoringConfig

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the pipeline orchestrator."""
    enable_auto_scaling: bool = True
    enable_load_balancing: bool = True
    enable_fault_tolerance: bool = True
    max_concurrent_analyses: int = 50
    analysis_timeout: int = 300  # 5 minutes
    retry_attempts: int = 3
    retry_delay: int = 5  # seconds
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    enable_analytics: bool = True
    analytics_retention_days: int = 30


class PipelineOrchestrator:
    """
    Main orchestrator for the financial analysis pipeline.
    
    Coordinates all pipeline components, provides load balancing,
    fault tolerance, and unified management capabilities.
    """
    
    def __init__(self, config: IceburgConfig, orchestrator_config: OrchestratorConfig = None):
        """Initialize pipeline orchestrator."""
        self.config = config
        self.orchestrator_config = orchestrator_config or OrchestratorConfig()
        
        # Pipeline components
        self.pipeline = FinancialAnalysisPipeline(config)
        self.monitor = PipelineMonitor(config)
        
        # Orchestrator state
        self.orchestrator_active = False
        self.analysis_queue = []
        self.active_analyses = {}
        self.completed_analyses = {}
        self.failed_analyses = {}
        self.performance_analytics = {}
        
        # Load balancing
        self.load_balancer = None
        self.worker_pools = {}
        
        # Fault tolerance
        self.health_checker = None
        self.fault_detector = None
        
        # Analytics
        self.analytics_engine = None
    
    async def start_orchestrator(self):
        """Start the pipeline orchestrator."""
        try:
            logger.info("Starting pipeline orchestrator...")
            
            # Start pipeline
            await self.pipeline.start_pipeline()
            
            # Start monitoring
            await self.monitor.start_monitoring()
            
            # Initialize load balancer
            if self.orchestrator_config.enable_load_balancing:
                await self._initialize_load_balancer()
            
            # Initialize fault tolerance
            if self.orchestrator_config.enable_fault_tolerance:
                await self._initialize_fault_tolerance()
            
            # Initialize analytics
            if self.orchestrator_config.enable_analytics:
                await self._initialize_analytics()
            
            # Start orchestrator tasks
            await self._start_orchestrator_tasks()
            
            # Activate orchestrator
            self.orchestrator_active = True
            
            logger.info("Pipeline orchestrator started successfully")
        
        except Exception as e:
            logger.error(f"Error starting orchestrator: {e}")
            raise
    
    async def stop_orchestrator(self):
        """Stop the pipeline orchestrator."""
        try:
            logger.info("Stopping pipeline orchestrator...")
            
            # Stop orchestrator tasks
            await self._stop_orchestrator_tasks()
            
            # Stop analytics
            if self.analytics_engine:
                await self._stop_analytics()
            
            # Stop fault tolerance
            if self.fault_detector:
                await self._stop_fault_tolerance()
            
            # Stop load balancer
            if self.load_balancer:
                await self._stop_load_balancer()
            
            # Stop monitoring
            await self.monitor.stop_monitoring()
            
            # Stop pipeline
            await self.pipeline.stop_pipeline()
            
            # Deactivate orchestrator
            self.orchestrator_active = False
            
            logger.info("Pipeline orchestrator stopped successfully")
        
        except Exception as e:
            logger.error(f"Error stopping orchestrator: {e}")
            raise
    
    async def analyze_query(self, query: str, context: Dict[str, Any] = None, 
                          priority: str = "normal") -> Dict[str, Any]:
        """
        Analyze a financial query using the orchestrator.
        
        Args:
            query: Financial query to analyze
            context: Additional context
            priority: Analysis priority (low, normal, high, critical)
            
        Returns:
            Analysis results
        """
        try:
            logger.info(f"Analyzing query with priority {priority}: {query[:100]}...")
            
            # Check if orchestrator is active
            if not self.orchestrator_active:
                await self.start_orchestrator()
            
            # Create analysis request
            analysis_request = {
                "id": f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                "query": query,
                "context": context or {},
                "priority": priority,
                "timestamp": datetime.now().isoformat(),
                "status": "queued"
            }
            
            # Add to analysis queue
            self.analysis_queue.append(analysis_request)
            
            # Process analysis
            if self.orchestrator_config.enable_load_balancing:
                result = await self._process_with_load_balancing(analysis_request)
            else:
                result = await self._process_analysis(analysis_request)
            
            return result
        
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return await self._fallback_analysis(query, context)
    
    async def _process_with_load_balancing(self, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process analysis with load balancing."""
        try:
            # Select worker based on load balancing strategy
            worker = await self._select_worker(analysis_request)
            
            # Process analysis with selected worker
            result = await self._process_analysis_with_worker(analysis_request, worker)
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing with load balancing: {e}")
            return await self._process_analysis(analysis_request)
    
    async def _select_worker(self, analysis_request: Dict[str, Any]) -> str:
        """Select worker for analysis based on load balancing strategy."""
        try:
            # Simple round-robin load balancing
            # In practice, this would be more sophisticated
            available_workers = list(self.worker_pools.keys())
            if not available_workers:
                return "default"
            
            # Select worker based on priority and load
            priority = analysis_request.get("priority", "normal")
            
            if priority == "critical":
                # Use dedicated high-performance worker
                return "high_performance"
            elif priority == "high":
                # Use dedicated worker
                return "dedicated"
            else:
                # Use round-robin
                worker_index = hash(analysis_request["id"]) % len(available_workers)
                return available_workers[worker_index]
        
        except Exception as e:
            logger.error(f"Error selecting worker: {e}")
            return "default"
    
    async def _process_analysis_with_worker(self, analysis_request: Dict[str, Any], worker: str) -> Dict[str, Any]:
        """Process analysis with specific worker."""
        try:
            # Update analysis status
            analysis_request["status"] = "processing"
            analysis_request["worker"] = worker
            
            # Add to active analyses
            self.active_analyses[analysis_request["id"]] = analysis_request
            
            # Process analysis
            result = await self._process_analysis(analysis_request)
            
            # Update analysis status
            analysis_request["status"] = "completed"
            analysis_request["result"] = result
            
            # Move to completed analyses
            self.completed_analyses[analysis_request["id"]] = analysis_request
            del self.active_analyses[analysis_request["id"]]
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing analysis with worker {worker}: {e}")
            
            # Update analysis status
            analysis_request["status"] = "failed"
            analysis_request["error"] = str(e)
            
            # Move to failed analyses
            self.failed_analyses[analysis_request["id"]] = analysis_request
            if analysis_request["id"] in self.active_analyses:
                del self.active_analyses[analysis_request["id"]]
            
            return await self._fallback_analysis(analysis_request["query"], analysis_request["context"])
    
    async def _process_analysis(self, analysis_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process analysis request."""
        try:
            # Update analysis status
            analysis_request["status"] = "processing"
            
            # Add to active analyses
            self.active_analyses[analysis_request["id"]] = analysis_request
            
            # Process with pipeline
            result = await self.pipeline.analyze_query(
                analysis_request["query"],
                analysis_request["context"]
            )
            
            # Update analysis status
            analysis_request["status"] = "completed"
            analysis_request["result"] = result
            
            # Move to completed analyses
            self.completed_analyses[analysis_request["id"]] = analysis_request
            del self.active_analyses[analysis_request["id"]]
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing analysis: {e}")
            
            # Update analysis status
            analysis_request["status"] = "failed"
            analysis_request["error"] = str(e)
            
            # Move to failed analyses
            self.failed_analyses[analysis_request["id"]] = analysis_request
            if analysis_request["id"] in self.active_analyses:
                del self.active_analyses[analysis_request["id"]]
            
            return await self._fallback_analysis(analysis_request["query"], analysis_request["context"])
    
    async def _fallback_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when processing fails."""
        return {
            "query": query,
            "response": "I encountered an error processing your request. Please try again.",
            "error": True,
            "orchestrator_metadata": {
                "timestamp": datetime.now().isoformat(),
                "orchestrator_active": self.orchestrator_active
            }
        }
    
    async def _initialize_load_balancer(self):
        """Initialize load balancer."""
        try:
            # Create worker pools
            self.worker_pools = {
                "default": {"capacity": 10, "active": 0},
                "dedicated": {"capacity": 5, "active": 0},
                "high_performance": {"capacity": 3, "active": 0}
            }
            
            # Initialize load balancer
            self.load_balancer = {
                "strategy": "round_robin",
                "health_checks": True,
                "auto_scaling": self.orchestrator_config.enable_auto_scaling
            }
            
            logger.info("Load balancer initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing load balancer: {e}")
            raise
    
    async def _initialize_fault_tolerance(self):
        """Initialize fault tolerance systems."""
        try:
            # Initialize health checker
            self.health_checker = {
                "enabled": True,
                "interval": 30,  # seconds
                "timeout": 10,  # seconds
                "retry_attempts": 3
            }
            
            # Initialize fault detector
            self.fault_detector = {
                "enabled": True,
                "thresholds": {
                    "error_rate": 0.1,
                    "response_time": 10.0,
                    "availability": 0.95
                }
            }
            
            logger.info("Fault tolerance systems initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing fault tolerance: {e}")
            raise
    
    async def _initialize_analytics(self):
        """Initialize analytics engine."""
        try:
            # Initialize analytics engine
            self.analytics_engine = {
                "enabled": True,
                "retention_days": self.orchestrator_config.analytics_retention_days,
                "metrics": [
                    "response_time",
                    "throughput",
                    "error_rate",
                    "queue_size",
                    "worker_utilization"
                ]
            }
            
            logger.info("Analytics engine initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing analytics: {e}")
            raise
    
    async def _start_orchestrator_tasks(self):
        """Start orchestrator background tasks."""
        try:
            # Start queue processor
            queue_task = asyncio.create_task(self._queue_processor_loop())
            
            # Start health checker
            if self.health_checker:
                health_task = asyncio.create_task(self._health_checker_loop())
            
            # Start analytics collector
            if self.analytics_engine:
                analytics_task = asyncio.create_task(self._analytics_collector_loop())
            
            logger.info("Orchestrator tasks started successfully")
        
        except Exception as e:
            logger.error(f"Error starting orchestrator tasks: {e}")
            raise
    
    async def _stop_orchestrator_tasks(self):
        """Stop orchestrator background tasks."""
        try:
            # Stop all background tasks
            # This would be implemented with proper task cancellation
            logger.info("Orchestrator tasks stopped successfully")
        
        except Exception as e:
            logger.error(f"Error stopping orchestrator tasks: {e}")
            raise
    
    async def _queue_processor_loop(self):
        """Main queue processor loop."""
        while self.orchestrator_active:
            try:
                # Process queued analyses
                if self.analysis_queue:
                    analysis_request = self.analysis_queue.pop(0)
                    await self._process_analysis(analysis_request)
                
                # Wait for next cycle
                await asyncio.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in queue processor loop: {e}")
                await asyncio.sleep(1)
    
    async def _health_checker_loop(self):
        """Main health checker loop."""
        while self.orchestrator_active:
            try:
                # Perform health checks
                health_status = await self._perform_health_checks()
                
                # Update health status
                # This would update the orchestrator's health status
                
                # Wait for next health check
                await asyncio.sleep(self.health_checker["interval"])
            
            except Exception as e:
                logger.error(f"Error in health checker loop: {e}")
                await asyncio.sleep(self.health_checker["interval"])
    
    async def _analytics_collector_loop(self):
        """Main analytics collector loop."""
        while self.orchestrator_active:
            try:
                # Collect analytics data
                analytics_data = await self._collect_analytics_data()
                
                # Store analytics data
                self.performance_analytics[datetime.now().isoformat()] = analytics_data
                
                # Clean old analytics data
                await self._clean_old_analytics()
                
                # Wait for next collection
                await asyncio.sleep(60)  # 1 minute
            
            except Exception as e:
                logger.error(f"Error in analytics collector loop: {e}")
                await asyncio.sleep(60)
    
    async def _perform_health_checks(self) -> Dict[str, Any]:
        """Perform orchestrator health checks."""
        try:
            health_status = {
                "orchestrator_active": self.orchestrator_active,
                "pipeline_active": self.pipeline.pipeline_active,
                "monitoring_active": self.monitor.monitoring_active,
                "queue_size": len(self.analysis_queue),
                "active_analyses": len(self.active_analyses),
                "completed_analyses": len(self.completed_analyses),
                "failed_analyses": len(self.failed_analyses)
            }
            
            return health_status
        
        except Exception as e:
            logger.error(f"Error performing health checks: {e}")
            return {"error": str(e)}
    
    async def _collect_analytics_data(self) -> Dict[str, Any]:
        """Collect analytics data."""
        try:
            analytics_data = {
                "timestamp": datetime.now().isoformat(),
                "queue_size": len(self.analysis_queue),
                "active_analyses": len(self.active_analyses),
                "completed_analyses": len(self.completed_analyses),
                "failed_analyses": len(self.failed_analyses),
                "success_rate": self._calculate_success_rate(),
                "average_response_time": self._calculate_average_response_time(),
                "worker_utilization": self._calculate_worker_utilization()
            }
            
            return analytics_data
        
        except Exception as e:
            logger.error(f"Error collecting analytics data: {e}")
            return {"error": str(e)}
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate."""
        try:
            total_analyses = len(self.completed_analyses) + len(self.failed_analyses)
            if total_analyses == 0:
                return 0.0
            
            successful_analyses = len(self.completed_analyses)
            return successful_analyses / total_analyses
        
        except Exception as e:
            logger.error(f"Error calculating success rate: {e}")
            return 0.0
    
    def _calculate_average_response_time(self) -> float:
        """Calculate average response time."""
        try:
            # Mock calculation
            return 1.5  # seconds
        
        except Exception as e:
            logger.error(f"Error calculating average response time: {e}")
            return 0.0
    
    def _calculate_worker_utilization(self) -> Dict[str, float]:
        """Calculate worker utilization."""
        try:
            utilization = {}
            for worker, pool in self.worker_pools.items():
                if pool["capacity"] > 0:
                    utilization[worker] = pool["active"] / pool["capacity"]
                else:
                    utilization[worker] = 0.0
            
            return utilization
        
        except Exception as e:
            logger.error(f"Error calculating worker utilization: {e}")
            return {}
    
    async def _clean_old_analytics(self):
        """Clean old analytics data."""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.orchestrator_config.analytics_retention_days)
            
            # Remove old analytics data
            keys_to_remove = []
            for timestamp, data in self.performance_analytics.items():
                if datetime.fromisoformat(timestamp) < cutoff_date:
                    keys_to_remove.append(timestamp)
            
            for key in keys_to_remove:
                del self.performance_analytics[key]
            
            logger.info(f"Cleaned {len(keys_to_remove)} old analytics entries")
        
        except Exception as e:
            logger.error(f"Error cleaning old analytics: {e}")
    
    async def _stop_load_balancer(self):
        """Stop load balancer."""
        try:
            self.load_balancer = None
            self.worker_pools = {}
            logger.info("Load balancer stopped")
        
        except Exception as e:
            logger.error(f"Error stopping load balancer: {e}")
    
    async def _stop_fault_tolerance(self):
        """Stop fault tolerance systems."""
        try:
            self.health_checker = None
            self.fault_detector = None
            logger.info("Fault tolerance systems stopped")
        
        except Exception as e:
            logger.error(f"Error stopping fault tolerance: {e}")
    
    async def _stop_analytics(self):
        """Stop analytics engine."""
        try:
            self.analytics_engine = None
            logger.info("Analytics engine stopped")
        
        except Exception as e:
            logger.error(f"Error stopping analytics: {e}")
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get orchestrator status and metrics."""
        return {
            "orchestrator_active": self.orchestrator_active,
            "pipeline_status": self.pipeline.get_pipeline_status(),
            "monitoring_status": self.monitor.get_monitoring_status(),
            "queue_size": len(self.analysis_queue),
            "active_analyses": len(self.active_analyses),
            "completed_analyses": len(self.completed_analyses),
            "failed_analyses": len(self.failed_analyses),
            "success_rate": self._calculate_success_rate(),
            "worker_utilization": self._calculate_worker_utilization()
        }
    
    async def get_analysis_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get analysis history."""
        try:
            # Combine completed and failed analyses
            all_analyses = list(self.completed_analyses.values()) + list(self.failed_analyses.values())
            
            # Sort by timestamp
            all_analyses.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Return limited results
            return all_analyses[:limit]
        
        except Exception as e:
            logger.error(f"Error getting analysis history: {e}")
            return []
    
    async def clear_analysis_cache(self):
        """Clear analysis cache."""
        try:
            self.completed_analyses.clear()
            self.failed_analyses.clear()
            logger.info("Analysis cache cleared")
        
        except Exception as e:
            logger.error(f"Error clearing analysis cache: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Test pipeline orchestrator
    config = IceburgConfig()
    orchestrator_config = OrchestratorConfig(
        enable_auto_scaling=True,
        enable_load_balancing=True,
        enable_fault_tolerance=True,
        enable_analytics=True
    )
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(config, orchestrator_config)
    
    # Test queries
    queries = [
        "What are the best quantum trading strategies for AAPL?",
        "Analyze the risk profile of a tech portfolio",
        "What are the best HFT strategies for market making?"
    ]
    
    # Test orchestrator
    import asyncio
    
    async def test_orchestrator():
        # Start orchestrator
        await orchestrator.start_orchestrator()
        
        # Test queries
        for query in queries:
            response = await orchestrator.analyze_query(query, priority="normal")
            print(f"Query: {query}")
            print(f"Response: {response}")
            print("---")
        
        # Get orchestrator status
        status = orchestrator.get_orchestrator_status()
        print(f"Orchestrator status: {status}")
        
        # Get analysis history
        history = await orchestrator.get_analysis_history()
        print(f"Analysis history: {len(history)} analyses")
        
        # Stop orchestrator
        await orchestrator.stop_orchestrator()
    
    # Run test
    asyncio.run(test_orchestrator())
