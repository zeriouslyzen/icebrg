"""
Parallel Healing Actions
Test multiple healing actions simultaneously for faster recovery
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from .bottleneck_detector import BottleneckAlert, AutoHealer
from ..protocol.execution.runner import ParallelExecutionEngine

logger = logging.getLogger(__name__)


class ParallelHealing:
    """Parallel healing actions for faster recovery."""
    
    def __init__(self, config: Dict[str, Any] = None, cfg=None):
        """
        Initialize parallel healing.
        
        Args:
            config: Configuration for parallel healing
            cfg: ICEBURG configuration
        """
        self.config = config or {}
        self.cfg = cfg
        self.auto_healer = AutoHealer(config or {})
        self.parallel_engine = ParallelExecutionEngine(cfg) if cfg else None
        
        # Parallel healing configuration
        self.max_parallel_actions = self.config.get("max_parallel_actions", 3)
        self.test_timeout = self.config.get("test_timeout", 10.0)  # 10 seconds per action
    
    async def test_healing_actions_parallel(
        self,
        alert: BottleneckAlert,
        available_actions: List[str]
    ) -> Dict[str, Any]:
        """
        Test multiple healing actions in parallel.
        
        Args:
            alert: Bottleneck alert
            available_actions: List of available healing actions
            
        Returns:
            Best action with results
        """
        logger.info(f"Testing {len(available_actions)} healing actions in parallel")
        
        # Limit to max parallel actions
        actions_to_test = available_actions[:self.max_parallel_actions]
        
        # Test actions in parallel
        test_tasks = []
        for action in actions_to_test:
            task = self._test_healing_action(alert, action)
            test_tasks.append(task)
        
        # Execute tests in parallel
        try:
            if self.parallel_engine:
                results = await self.parallel_engine.execute_parallel(test_tasks, timeout=self.test_timeout)
            else:
                # Fallback to asyncio.gather
                results = await asyncio.gather(*test_tasks, return_exceptions=True)
                # Filter out exceptions
                results = [r for r in results if not isinstance(r, Exception)]
        except Exception as e:
            logger.error(f"Parallel testing failed: {e}")
            # Fallback to sequential testing
            results = []
            for action in actions_to_test:
                try:
                    result = await self._test_healing_action(alert, action)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Action test failed: {e}")
                    results.append({"action": action, "success": False, "error": str(e)})
        
        # Select best action
        best_action = self._select_best_action(results)
        
        logger.info(f"Selected best action: {best_action.get('action', 'unknown')}")
        
        return {
            "success": True,
            "best_action": best_action,
            "all_results": results
        }
    
    async def _test_healing_action(
        self,
        alert: BottleneckAlert,
        action: str
    ) -> Dict[str, Any]:
        """Test a single healing action."""
        try:
            # Execute action
            start_time = asyncio.get_event_loop().time()
            success = await self.auto_healer._execute_healing_action(action, alert)
            end_time = asyncio.get_event_loop().time()
            
            recovery_time = end_time - start_time
            
            # Calculate expected reward (simulated - in real implementation, would measure metrics)
            expected_reward = 1.0 if success else 0.0
            
            return {
                "action": action,
                "success": success,
                "recovery_time": recovery_time,
                "expected_reward": expected_reward
            }
        except Exception as e:
            logger.error(f"Action test error: {e}")
            return {
                "action": action,
                "success": False,
                "error": str(e),
                "recovery_time": 0.0,
                "expected_reward": 0.0
            }
    
    def _select_best_action(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select best action based on test results."""
        # Find best action (highest expected reward, fastest recovery)
        best_action = None
        best_score = -1.0
        
        for result in results:
            if result.get("success", False):
                # Score = expected_reward / recovery_time (higher is better)
                expected_reward = result.get("expected_reward", 0.0)
                recovery_time = max(result.get("recovery_time", 1.0), 0.1)  # Avoid division by zero
                score = expected_reward / recovery_time
                
                if score > best_score:
                    best_score = score
                    best_action = result
        
        if best_action is None:
            # Fallback to first result
            best_action = results[0] if results else {"action": "unknown", "success": False}
        
        return best_action
    
    async def apply_best_healing_action(
        self,
        alert: BottleneckAlert,
        available_actions: List[str]
    ) -> bool:
        """
        Test actions in parallel and apply best one.
        
        Args:
            alert: Bottleneck alert
            available_actions: List of available healing actions
            
        Returns:
            True if healing was successful
        """
        # Test actions in parallel
        test_results = await self.test_healing_actions_parallel(alert, available_actions)
        
        if not test_results.get("success", False):
            logger.error("Failed to test healing actions")
            return False
        
        # Get best action
        best_action = test_results.get("best_action", {})
        action_name = best_action.get("action")
        
        if not action_name:
            logger.error("No best action found")
            return False
        
        # Apply best action
        try:
            success = await self.auto_healer._execute_healing_action(action_name, alert)
            if success:
                logger.info(f"Applied best healing action: {action_name}")
            return success
        except Exception as e:
            logger.error(f"Failed to apply best action: {e}")
            return False

