"""
ICEBURG Quantum-RL Integration

Integrates quantum-RL systems with ICEBURG's core architecture,
providing seamless access to quantum-enhanced financial AI capabilities.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import json

from ..config import IceburgConfig
from ..protocol import iceberg_protocol
from ..agents.surveyor import Surveyor
from ..memory.unified_memory import UnifiedMemory
from ..reasoning.hybrid_reasoning_engine import HybridReasoningEngine
from ..emergence.quantum_emergence_detector import QuantumEmergenceDetector
from ..hybrid.quantum_rl import QuantumRLIntegration, QuantumRLConfig
from ..quantum.circuits import VQC, QuantumCircuit
from ..rl.agents.ppo_trader import PPOTrader
from ..rl.agents.sac_trader import SACTrader
from ..financial.data_pipeline import FinancialDataPipeline
from ..financial.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class ICEBURGQuantumRLIntegration:
    """
    Main integration class for quantum-RL systems with ICEBURG.
    
    Provides seamless integration between ICEBURG's core capabilities
    and the new quantum-RL financial AI systems.
    """
    
    def __init__(self, config: IceburgConfig):
        """Initialize ICEBURG quantum-RL integration."""
        self.config = config
        self.quantum_rl_config = QuantumRLConfig()
        self.quantum_rl_integration = QuantumRLIntegration(self.quantum_rl_config)
        
        # ICEBURG components
        self.memory = UnifiedMemory(config)
        self.reasoning_engine = HybridReasoningEngine(config)
        self.quantum_emergence_detector = QuantumEmergenceDetector()
        self.surveyor = Surveyor(config)
        
        # Financial components
        self.data_pipeline = FinancialDataPipeline(config)
        self.feature_engineer = FeatureEngineer(config)
        
        # Integration state
        self.integration_active = False
        self.quantum_agents = {}
        self.performance_metrics = {}
        self.integration_history = []
    
    async def activate_quantum_rl(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Activate quantum-RL capabilities for a query.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            Response with quantum-RL enhancements
        """
        try:
            logger.info(f"Activating quantum-RL for query: {query[:100]}...")
            
            # Check if quantum-RL is appropriate for this query
            if not self._should_use_quantum_rl(query, context):
                return await self._standard_icberg_response(query, context)
            
            # Initialize quantum-RL components
            await self._initialize_quantum_rl_components()
            
            # Process query with quantum-RL enhancement
            response = await self._process_quantum_rl_query(query, context)
            
            # Update integration metrics
            self._update_integration_metrics(response)
            
            return response
        
        except Exception as e:
            logger.error(f"Error in quantum-RL activation: {e}")
            return await self._fallback_response(query, context)
    
    def _should_use_quantum_rl(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if quantum-RL should be used for this query."""
        quantum_rl_keywords = [
            "quantum", "financial", "trading", "market", "portfolio", "risk",
            "optimization", "prediction", "alpha", "beta", "volatility",
            "arbitrage", "hedge", "derivatives", "options", "futures",
            "high frequency", "hft", "algorithmic", "quantitative"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in quantum_rl_keywords)
    
    async def _initialize_quantum_rl_components(self):
        """Initialize quantum-RL components."""
        if self.integration_active:
            return
        
        try:
            # Initialize financial data pipeline
            await self.data_pipeline.initialize()
            
            # Initialize feature engineering
            await self.feature_engineer.initialize()
            
            # Create quantum-RL agents
            await self._create_quantum_rl_agents()
            
            # Activate integration
            self.integration_active = True
            
            logger.info("Quantum-RL components initialized successfully")
        
        except Exception as e:
            logger.error(f"Error initializing quantum-RL components: {e}")
            raise
    
    async def _create_quantum_rl_agents(self):
        """Create quantum-RL agents."""
        try:
            # Create PPO trader with quantum enhancement
            ppo_trader = PPOTrader("quantum_ppo_trader", None, {})
            quantum_ppo_trader = self.quantum_rl_integration.integrate_with_agent(ppo_trader)
            self.quantum_agents["quantum_ppo_trader"] = quantum_ppo_trader
            
            # Create SAC trader with quantum enhancement
            sac_trader = SACTrader("quantum_sac_trader", None, {})
            quantum_sac_trader = self.quantum_rl_integration.integrate_with_agent(sac_trader)
            self.quantum_agents["quantum_sac_trader"] = quantum_sac_trader
            
            logger.info(f"Created {len(self.quantum_agents)} quantum-RL agents")
        
        except Exception as e:
            logger.error(f"Error creating quantum-RL agents: {e}")
            raise
    
    async def _process_quantum_rl_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process query with quantum-RL enhancement."""
        try:
            # Get financial data if needed
            financial_data = await self._get_financial_data(query, context)
            
            # Process with quantum-RL agents
            quantum_insights = await self._get_quantum_insights(query, financial_data)
            
            # Integrate with ICEBURG reasoning
            icberg_analysis = await self._get_icberg_analysis(query, context)
            
            # Combine quantum-RL and ICEBURG insights
            combined_response = await self._combine_insights(
                query, quantum_insights, icberg_analysis, context
            )
            
            return combined_response
        
        except Exception as e:
            logger.error(f"Error processing quantum-RL query: {e}")
            return await self._fallback_response(query, context)
    
    async def _get_financial_data(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get relevant financial data for the query."""
        try:
            # Extract symbols from query
            symbols = self._extract_symbols_from_query(query)
            
            # Get market data
            market_data = await self.data_pipeline.get_market_data(symbols)
            
            # Get additional financial data
            financial_data = {
                "market_data": market_data,
                "query": query,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            
            return financial_data
        
        except Exception as e:
            logger.error(f"Error getting financial data: {e}")
            return {}
    
    def _extract_symbols_from_query(self, query: str) -> List[str]:
        """Extract stock symbols from query."""
        # Simple symbol extraction - in practice, this would be more sophisticated
        symbols = []
        query_upper = query.upper()
        
        # Common stock symbols
        common_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "NFLX"]
        
        for symbol in common_symbols:
            if symbol in query_upper:
                symbols.append(symbol)
        
        return symbols
    
    async def _get_quantum_insights(self, query: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get quantum insights for the query."""
        try:
            insights = {
                "quantum_advantage": 0.0,
                "quantum_confidence": 0.5,
                "quantum_features": [],
                "recommendations": [],
                "risk_assessment": {},
                "market_insights": []
            }
            
            # Process with quantum-RL agents
            for agent_name, agent in self.quantum_agents.items():
                try:
                    # Get agent insights
                    agent_insights = await self._get_agent_insights(agent, query, financial_data)
                    insights["recommendations"].append({
                        "agent": agent_name,
                        "insights": agent_insights
                    })
                    
                    # Update quantum advantage
                    if "quantum_advantage" in agent_insights:
                        insights["quantum_advantage"] = max(
                            insights["quantum_advantage"],
                            agent_insights["quantum_advantage"]
                        )
                    
                    # Update quantum confidence
                    if "quantum_confidence" in agent_insights:
                        insights["quantum_confidence"] = max(
                            insights["quantum_confidence"],
                            agent_insights["quantum_confidence"]
                        )
                
                except Exception as e:
                    logger.error(f"Error getting insights from {agent_name}: {e}")
                    continue
            
            return insights
        
        except Exception as e:
            logger.error(f"Error getting quantum insights: {e}")
            return {"quantum_advantage": 0.0, "quantum_confidence": 0.5}
    
    async def _get_agent_insights(self, agent, query: str, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get insights from a quantum-RL agent."""
        try:
            # This would be implemented based on the specific agent type
            # For now, return mock insights
            return {
                "quantum_advantage": 0.1,
                "quantum_confidence": 0.7,
                "recommendation": "buy",
                "risk_level": "medium"
            }
        
        except Exception as e:
            logger.error(f"Error getting agent insights: {e}")
            return {}
    
    async def _get_icberg_analysis(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get ICEBURG analysis for the query."""
        try:
            # Use ICEBURG's reasoning engine
            reasoning_result = await self.reasoning_engine.process_query(query, context)
            
            # Use ICEBURG's memory system
            memory_context = await self.memory.get_relevant_context(query)
            
            # Use ICEBURG's surveyor for technical analysis
            surveyor_analysis = await self.surveyor.analyze(query, context)
            
            return {
                "reasoning_result": reasoning_result,
                "memory_context": memory_context,
                "surveyor_analysis": surveyor_analysis
            }
        
        except Exception as e:
            logger.error(f"Error getting ICEBURG analysis: {e}")
            return {}
    
    async def _combine_insights(self, query: str, quantum_insights: Dict[str, Any], 
                              icberg_analysis: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Combine quantum-RL and ICEBURG insights."""
        try:
            # Combine insights
            combined_response = {
                "query": query,
                "quantum_insights": quantum_insights,
                "icberg_analysis": icberg_analysis,
                "combined_analysis": {
                    "quantum_advantage": quantum_insights.get("quantum_advantage", 0.0),
                    "quantum_confidence": quantum_insights.get("quantum_confidence", 0.5),
                    "icberg_confidence": icberg_analysis.get("reasoning_result", {}).get("confidence", 0.5),
                    "overall_confidence": 0.0,
                    "recommendations": [],
                    "risk_assessment": {},
                    "market_insights": []
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Calculate overall confidence
            quantum_conf = quantum_insights.get("quantum_confidence", 0.5)
            icberg_conf = icberg_analysis.get("reasoning_result", {}).get("confidence", 0.5)
            combined_response["combined_analysis"]["overall_confidence"] = (quantum_conf + icberg_conf) / 2
            
            # Combine recommendations
            if "recommendations" in quantum_insights:
                combined_response["combined_analysis"]["recommendations"].extend(
                    quantum_insights["recommendations"]
                )
            
            # Combine risk assessments
            if "risk_assessment" in quantum_insights:
                combined_response["combined_analysis"]["risk_assessment"].update(
                    quantum_insights["risk_assessment"]
                )
            
            # Add ICEBURG insights
            if "surveyor_analysis" in icberg_analysis:
                combined_response["combined_analysis"]["market_insights"].append(
                    icberg_analysis["surveyor_analysis"]
                )
            
            return combined_response
        
        except Exception as e:
            logger.error(f"Error combining insights: {e}")
            return await self._fallback_response(query, context)
    
    async def _standard_icberg_response(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get standard ICEBURG response without quantum-RL."""
        try:
            # Use standard ICEBURG protocol
            response = await iceberg_protocol(
                query=query,
                context=context,
                mode="research",
                enhanced_capabilities=True
            )
            
            return response
        
        except Exception as e:
            logger.error(f"Error getting standard ICEBURG response: {e}")
            return await self._fallback_response(query, context)
    
    async def _fallback_response(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback response when quantum-RL fails."""
        return {
            "query": query,
            "response": "I encountered an error processing your request with quantum-RL capabilities. Please try again.",
            "quantum_advantage": 0.0,
            "quantum_confidence": 0.0,
            "error": True,
            "timestamp": datetime.now().isoformat()
        }
    
    def _update_integration_metrics(self, response: Dict[str, Any]):
        """Update integration performance metrics."""
        try:
            # Update metrics
            self.performance_metrics.update({
                "total_queries": self.performance_metrics.get("total_queries", 0) + 1,
                "quantum_advantage": response.get("quantum_insights", {}).get("quantum_advantage", 0.0),
                "quantum_confidence": response.get("quantum_insights", {}).get("quantum_confidence", 0.5),
                "overall_confidence": response.get("combined_analysis", {}).get("overall_confidence", 0.5)
            })
            
            # Update integration history
            self.integration_history.append({
                "timestamp": datetime.now().isoformat(),
                "response": response
            })
            
            # Keep only recent history
            if len(self.integration_history) > 1000:
                self.integration_history.pop(0)
        
        except Exception as e:
            logger.error(f"Error updating integration metrics: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status and metrics."""
        return {
            "integration_active": self.integration_active,
            "quantum_agents_count": len(self.quantum_agents),
            "performance_metrics": self.performance_metrics,
            "integration_history_count": len(self.integration_history),
            "quantum_rl_status": self.quantum_rl_integration.get_integration_status()
        }
    
    async def reset_integration(self):
        """Reset integration state."""
        try:
            self.integration_active = False
            self.quantum_agents = {}
            self.performance_metrics = {}
            self.integration_history = []
            
            # Reset quantum-RL integration
            self.quantum_rl_integration.reset_integration()
            
            logger.info("ICEBURG quantum-RL integration reset successfully")
        
        except Exception as e:
            logger.error(f"Error resetting integration: {e}")
            raise


# Example usage and testing
if __name__ == "__main__":
    # Test ICEBURG quantum-RL integration
    config = IceburgConfig()
    
    # Create integration
    integration = ICEBURGQuantumRLIntegration(config)
    
    # Test query
    query = "What are the best quantum trading strategies for AAPL?"
    context = {"symbols": ["AAPL"], "timeframe": "1d"}
    
    # Test activation
    import asyncio
    response = asyncio.run(integration.activate_quantum_rl(query, context))
    print(f"Response: {response}")
    
    # Test status
    status = integration.get_integration_status()
    print(f"Status: {status}")
