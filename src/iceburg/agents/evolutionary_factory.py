"""
Evolutionary Agent Factory
Generates multiple code variants and tests in parallel (like AlphaEvolve)
"""

import logging
from typing import Dict, Any, List, Optional
from .dynamic_agent_factory import DynamicAgentFactory, AgentTemplate
from ..protocol.execution.runner import ParallelExecutionEngine

logger = logging.getLogger(__name__)


class EvolutionaryFactory:
    """Evolutionary code generation for agents."""
    
    def __init__(self, cfg):
        """
        Initialize evolutionary factory.
        
        Args:
            cfg: ICEBURG configuration
        """
        self.cfg = cfg
        self.factory = DynamicAgentFactory(cfg)
        self.parallel_engine = ParallelExecutionEngine(cfg)
        
        # Evolution configuration
        self.n_variants = 3  # Generate 3 variants
        self.test_timeout = 30.0  # 30 seconds per test
    
    async def generate_and_test_variants(
        self,
        template: AgentTemplate,
        emergence_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate multiple code variants and test in parallel.
        
        Args:
            template: Agent template
            emergence_data: Emergence data
            
        Returns:
            Best variant with performance metrics
        """
        logger.info(f"Generating {self.n_variants} code variants for agent")
        
        # Generate multiple variants
        variants = []
        for i in range(self.n_variants):
            try:
                # Generate variant (each call to _generate_agent_code may produce different results)
                agent_name = f"{template.specialization}_{i}"
                code = self.factory._generate_agent_code(agent_name, template, emergence_data)
                
                variants.append({
                    "variant_id": i,
                    "agent_name": agent_name,
                    "code": code,
                    "template": template,
                    "emergence_data": emergence_data
                })
            except Exception as e:
                logger.error(f"Failed to generate variant {i}: {e}")
        
        if not variants:
            logger.error("No variants generated")
            return {"success": False, "error": "No variants generated"}
        
        # Test variants in parallel
        test_results = await self._test_variants_parallel(variants)
        
        # Select best variant
        best_variant = self._select_best_variant(variants, test_results)
        
        logger.info(f"Selected best variant: {best_variant.get('variant_id', 'unknown')}")
        
        return {
            "success": True,
            "best_variant": best_variant,
            "all_variants": variants,
            "test_results": test_results
        }
    
    async def _test_variants_parallel(self, variants: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Test variants in parallel using ParallelExecutionEngine."""
        test_tasks = []
        
        for variant in variants:
            task = self._test_variant(variant)
            test_tasks.append(task)
        
        # Execute tests in parallel
        try:
            results = await self.parallel_engine.execute_parallel(test_tasks, timeout=self.test_timeout)
            return results
        except Exception as e:
            logger.error(f"Parallel testing failed: {e}")
            # Fallback to sequential testing
            results = []
            for variant in variants:
                try:
                    result = await self._test_variant(variant)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Variant test failed: {e}")
                    results.append({"variant_id": variant["variant_id"], "success": False, "error": str(e)})
            return results
    
    async def _test_variant(self, variant: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single variant."""
        try:
            # Validate code
            from .code_validator import CodeValidator
            validator = CodeValidator()
            
            validation_result = validator.validate_code(variant["code"])
            
            if not validation_result:
                return {
                    "variant_id": variant["variant_id"],
                    "success": False,
                    "error": "Code validation failed",
                    "performance_score": 0.0
                }
            
            # Test code execution (simulated - in real implementation, would execute in sandbox)
            # For now, use code quality metrics
            performance_score = self._calculate_code_quality(variant["code"])
            
            return {
                "variant_id": variant["variant_id"],
                "success": True,
                "performance_score": performance_score,
                "validation_passed": True
            }
        except Exception as e:
            logger.error(f"Variant test error: {e}")
            return {
                "variant_id": variant["variant_id"],
                "success": False,
                "error": str(e),
                "performance_score": 0.0
            }
    
    def _calculate_code_quality(self, code: str) -> float:
        """Calculate code quality score (0.0 to 1.0)."""
        score = 0.0
        
        # Check for required components
        if "class" in code:
            score += 0.2
        if "def __init__" in code:
            score += 0.2
        if "def run" in code:
            score += 0.2
        if "chat_complete" in code:
            score += 0.2
        if "logger" in code:
            score += 0.1
        if "try:" in code and "except" in code:
            score += 0.1  # Error handling
        
        return min(1.0, score)
    
    def _select_best_variant(
        self,
        variants: List[Dict[str, Any]],
        test_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Select best variant based on test results."""
        # Create mapping of variant_id to test result
        result_map = {r["variant_id"]: r for r in test_results}
        
        # Find best variant (highest performance score)
        best_variant = None
        best_score = -1.0
        
        for variant in variants:
            variant_id = variant["variant_id"]
            result = result_map.get(variant_id, {})
            
            if result.get("success", False):
                score = result.get("performance_score", 0.0)
                if score > best_score:
                    best_score = score
                    best_variant = variant
        
        if best_variant is None:
            # Fallback to first variant
            best_variant = variants[0]
        
        return best_variant

