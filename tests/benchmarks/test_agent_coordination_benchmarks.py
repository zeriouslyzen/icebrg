"""
Performance benchmarks for agent coordination
"""

import unittest
import time
import statistics
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.capability_registry import get_registry
from iceburg.protocol.planner import plan, optimize_plan, get_parallelizable_groups
from iceburg.protocol.models import Query, Mode
from iceburg.protocol.config import ProtocolConfig
from iceburg.infrastructure.dynamic_resource_allocator import get_resource_allocator
from iceburg.distributed.load_balancer import IntelligentLoadBalancer, LoadBalancingStrategy


class AgentCoordinationBenchmarks(unittest.TestCase):
    """Performance benchmarks for agent coordination"""
    
    def setUp(self):
        """Set up benchmark fixtures"""
        self.registry = get_registry()
        self.query = Query(
            text="What are the implications of AI development for society?",
            metadata={}
        )
        self.config = ProtocolConfig(fast=False, verbose=False)
        self.iterations = 10
    
    def benchmark_dependency_resolution(self):
        """Benchmark dependency resolution performance"""
        agent_ids = ["surveyor", "dissident", "archaeologist", "synthesist", "oracle"]
        
        times = []
        for _ in range(self.iterations):
            start = time.time()
            ordered = self.registry.resolve_dependencies(agent_ids)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"\nDependency Resolution Benchmark:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  Min: {min_time*1000:.2f}ms")
        print(f"  Max: {max_time*1000:.2f}ms")
        
        # Should be fast (< 10ms)
        self.assertLess(avg_time, 0.01)
    
    def benchmark_parallel_grouping(self):
        """Benchmark parallel execution grouping performance"""
        agent_ids = ["surveyor", "dissident", "archaeologist", "synthesist", "oracle", "scrutineer"]
        
        times = []
        for _ in range(self.iterations):
            start = time.time()
            groups = self.registry.get_parallelizable_groups(agent_ids)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        
        print(f"\nParallel Grouping Benchmark:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  Groups created: {len(groups)}")
        
        # Should be fast (< 20ms)
        self.assertLess(avg_time, 0.02)
    
    def benchmark_planner_optimization(self):
        """Benchmark planner optimization performance"""
        tasks = plan(self.query, Mode.RESEARCH, self.config)
        
        times = []
        for _ in range(self.iterations):
            start = time.time()
            optimized = optimize_plan(tasks, self.config)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        
        print(f"\nPlanner Optimization Benchmark:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  Tasks optimized: {len(optimized)}")
        
        # Should be fast (< 50ms)
        self.assertLess(avg_time, 0.05)
    
    def benchmark_agent_selection(self):
        """Benchmark agent selection performance"""
        times = []
        for _ in range(self.iterations):
            start = time.time()
            agent = self.registry.find_best_agent(
                required_capabilities=["information_gathering", "research_synthesis"]
            )
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        
        print(f"\nAgent Selection Benchmark:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  Selected agent: {agent.agent_id if agent else None}")
        
        # Should be fast (< 20ms)
        self.assertLess(avg_time, 0.02)
    
    def benchmark_resource_allocation(self):
        """Benchmark resource allocation performance"""
        allocator = get_resource_allocator()
        
        times = []
        for _ in range(self.iterations):
            start = time.time()
            allocation = allocator.allocate_resources("surveyor", priority=5)
            elapsed = time.time() - start
            times.append(elapsed)
            
            if allocation:
                allocator.release_resources(f"surveyor_{int(time.time() * 1000)}")
        
        avg_time = statistics.mean(times)
        
        print(f"\nResource Allocation Benchmark:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        
        # Should be fast (< 10ms)
        self.assertLess(avg_time, 0.01)
    
    def benchmark_load_balancer_selection(self):
        """Benchmark load balancer worker selection"""
        import asyncio
        
        load_balancer = IntelligentLoadBalancer(
            strategy=LoadBalancingStrategy.ADAPTIVE
        )
        
        # Add workers
        load_balancer.add_worker("surveyor", weight=1.0, capabilities=["information_gathering"])
        load_balancer.add_worker("dissident", weight=1.0, capabilities=["alternative_perspectives"])
        load_balancer.add_worker("synthesist", weight=1.0, capabilities=["synthesis"])
        
        async def benchmark_selection():
            times = []
            for _ in range(self.iterations):
                start = time.time()
                worker_id = await load_balancer.select_worker(
                    query="test query",
                    required_capabilities=["information_gathering"]
                )
                elapsed = time.time() - start
                times.append(elapsed)
            return statistics.mean(times)
        
        avg_time = asyncio.run(benchmark_selection())
        
        print(f"\nLoad Balancer Selection Benchmark:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        
        # Should be fast (< 20ms)
        self.assertLess(avg_time, 0.02)
    
    def benchmark_linguistic_enhancement(self):
        """Benchmark linguistic enhancement performance"""
        from iceburg.agents.linguistic_intelligence import get_linguistic_engine
        
        linguistic_engine = get_linguistic_engine()
        text = "It is important to note that this is a very good solution that we should definitely consider."
        
        times = []
        for _ in range(self.iterations):
            start = time.time()
            enhanced = linguistic_engine.enhance_text(
                text,
                verbosity_reduction=0.3,
                power_enhancement=0.5
            )
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = statistics.mean(times)
        
        print(f"\nLinguistic Enhancement Benchmark:")
        print(f"  Average: {avg_time*1000:.2f}ms")
        print(f"  Original length: {len(text)}")
        print(f"  Enhanced length: {len(enhanced.enhanced_text)}")
        
        # Should be fast (< 50ms)
        self.assertLess(avg_time, 0.05)
    
    def run_all_benchmarks(self):
        """Run all benchmarks"""
        print("\n" + "="*70)
        print("AGENT COORDINATION BENCHMARKS")
        print("="*70)
        
        self.benchmark_dependency_resolution()
        self.benchmark_parallel_grouping()
        self.benchmark_planner_optimization()
        self.benchmark_agent_selection()
        self.benchmark_resource_allocation()
        self.benchmark_load_balancer_selection()
        self.benchmark_linguistic_enhancement()
        
        print("\n" + "="*70)
        print("BENCHMARKS COMPLETE")
        print("="*70 + "\n")


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(AgentCoordinationBenchmarks)
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run benchmarks
    benchmarks = AgentCoordinationBenchmarks()
    benchmarks.setUp()
    benchmarks.run_all_benchmarks()
    
    # Run unit tests
    result = runner.run(suite)

