"""
Query Optimizer
Optimizes query processing with complexity pre-analysis and parallel execution
"""

from typing import Any, Dict, Optional, List
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import re


class QueryOptimizer:
    """Optimizes query processing"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.complexity_cache: Dict[str, Dict[str, Any]] = {}
    
    def analyze_complexity(self, query: str) -> Dict[str, Any]:
        """Analyze query complexity"""
        if query in self.complexity_cache:
            return self.complexity_cache[query]
        
        analysis = {
            "query": query,
            "complexity_level": "medium",
            "estimated_time": 5.0,
            "requires_parallel": False,
            "can_early_terminate": False,
            "indicators": []
        }
        
        query_lower = query.lower()
        
        # Simple queries (factual, direct)
        simple_indicators = ["what is", "define", "explain", "how does", "when did"]
        if any(indicator in query_lower for indicator in simple_indicators):
            analysis["complexity_level"] = "simple"
            analysis["estimated_time"] = 2.0
            analysis["can_early_terminate"] = True
            analysis["indicators"].extend(simple_indicators)
        
        # Complex queries (analysis, synthesis, discovery)
        complex_indicators = ["analyze", "synthesize", "discover", "investigate", "find patterns", "compare"]
        if any(indicator in query_lower for indicator in complex_indicators):
            analysis["complexity_level"] = "complex"
            analysis["estimated_time"] = 10.0
            analysis["requires_parallel"] = True
            analysis["indicators"].extend(complex_indicators)
        
        # Check for multiple sub-queries
        sub_queries = re.split(r'[?;]', query)
        if len(sub_queries) > 2:
            analysis["complexity_level"] = "complex"
            analysis["estimated_time"] = len(sub_queries) * 3.0
            analysis["requires_parallel"] = True
            analysis["indicators"].append("multiple_subqueries")
        
        # Check query length
        if len(query) > 500:
            analysis["complexity_level"] = "complex"
            analysis["estimated_time"] = 8.0
            analysis["indicators"].append("long_query")
        
        # Cache analysis
        self.complexity_cache[query] = analysis
        
        return analysis
    
    def optimize_query(self, query: str) -> Dict[str, Any]:
        """Optimize query processing"""
        complexity = self.analyze_complexity(query)
        
        optimization = {
            "query": query,
            "complexity": complexity,
            "optimization_strategy": "standard",
            "parallel_execution": complexity["requires_parallel"],
            "early_termination": complexity["can_early_terminate"],
            "estimated_time": complexity["estimated_time"]
        }
        
        # Determine optimization strategy
        if complexity["complexity_level"] == "simple":
            optimization["optimization_strategy"] = "fast_path"
        elif complexity["complexity_level"] == "complex":
            optimization["optimization_strategy"] = "parallel"
        else:
            optimization["optimization_strategy"] = "standard"
        
        return optimization
    
    async def process_parallel(
        self,
        tasks: List[callable],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """Process tasks in parallel"""
        if not tasks:
            return []
        
        try:
            results = await asyncio.gather(
                *[asyncio.to_thread(task) for task in tasks],
                return_exceptions=True,
                timeout=timeout
            )
            
            # Filter out exceptions
            return [r for r in results if not isinstance(r, Exception)]
        except asyncio.TimeoutError:
            return []
        except Exception:
            return []
    
    def process_threaded(
        self,
        tasks: List[callable],
        timeout: Optional[float] = None
    ) -> List[Any]:
        """Process tasks in thread pool"""
        if not tasks:
            return []
        
        try:
            futures = [self.thread_pool.submit(task) for task in tasks]
            results = []
            
            for future in futures:
                try:
                    result = future.result(timeout=timeout)
                    results.append(result)
                except Exception:
                    pass
            
            return results
        except Exception:
            return []
    
    def should_use_parallel(self, query: str) -> bool:
        """Determine if query should use parallel processing"""
        complexity = self.analyze_complexity(query)
        return complexity["requires_parallel"]
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            "complexity_cache_size": len(self.complexity_cache),
            "max_workers": self.max_workers,
            "thread_pool_active": self.thread_pool._threads,
            "process_pool_active": self.process_pool._processes if hasattr(self.process_pool, '_processes') else 0
        }

