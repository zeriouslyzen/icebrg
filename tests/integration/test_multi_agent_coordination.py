"""
Integration tests for multi-agent coordination workflows
"""

import unittest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from iceburg.agents.capability_registry import get_registry
from iceburg.agents.interaction_protocol import (
    get_protocol,
    create_request,
    create_response,
    AgentRequest,
    AgentResponse
)
from iceburg.protocol.planner import plan, optimize_plan, get_parallelizable_groups
from iceburg.protocol.execution.runner import run_agent_tasks
from iceburg.protocol.models import AgentTask, Query, Mode
from iceburg.protocol.config import ProtocolConfig


class TestMultiAgentCoordination(unittest.TestCase):
    """Integration tests for multi-agent coordination"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.registry = get_registry()
        self.protocol = get_protocol()
        
        self.query = Query(
            text="What are the implications of AI development?",
            metadata={}
        )
        
        self.config = ProtocolConfig(
            fast=False,
            verbose=False
        )
    
    def test_agent_registry_integration(self):
        """Test agent registry integration"""
        # Verify registry has agents
        all_agents = self.registry.get_all_agents()
        self.assertGreater(len(all_agents), 0)
        
        # Verify core agents exist
        self.assertIn("surveyor", all_agents)
        self.assertIn("dissident", all_agents)
        self.assertIn("synthesist", all_agents)
        self.assertIn("oracle", all_agents)
    
    def test_agent_dependency_resolution(self):
        """Test agent dependency resolution"""
        agent_ids = ["surveyor", "dissident", "synthesist", "oracle"]
        
        # Resolve dependencies
        ordered = self.registry.resolve_dependencies(agent_ids)
        
        # Verify surveyor comes before dissident
        self.assertLess(
            ordered.index("surveyor"),
            ordered.index("dissident")
        )
        
        # Verify synthesist comes after surveyor and dissident
        self.assertLess(
            ordered.index("surveyor"),
            ordered.index("synthesist")
        )
        self.assertLess(
            ordered.index("dissident"),
            ordered.index("synthesist")
        )
    
    def test_parallel_execution_groups(self):
        """Test parallel execution grouping"""
        agent_ids = ["surveyor", "archaeologist", "dissident", "synthesist"]
        
        # Get parallelizable groups
        groups = self.registry.get_parallelizable_groups(agent_ids)
        
        # Verify groups are created
        self.assertGreater(len(groups), 0)
        
        # Verify first group contains independent agents
        first_group = groups[0]
        self.assertIn("surveyor", first_group or "archaeologist" in first_group)
    
    def test_interaction_protocol(self):
        """Test agent interaction protocol"""
        # Create request
        request = create_request(
            sender_id="surveyor",
            receiver_id="dissident",
            action="process_alternative",
            payload={"query": "test query"},
            priority=AgentRequest.priority.__class__.HIGH
        )
        
        # Send request
        request_id = self.protocol.send_request(request)
        
        self.assertIsNotNone(request_id)
        self.assertEqual(request.message_id, request_id)
    
    def test_planner_optimization(self):
        """Test planner optimization with registry"""
        # Create tasks
        tasks = plan(self.query, Mode.RESEARCH, self.config)
        
        # Optimize plan
        optimized = optimize_plan(tasks, self.config)
        
        # Verify optimization occurred
        self.assertIsInstance(optimized, list)
        self.assertGreaterEqual(len(optimized), len(tasks))
    
    def test_parallelizable_groups_from_planner(self):
        """Test getting parallelizable groups from planner"""
        # Create tasks
        tasks = plan(self.query, Mode.RESEARCH, self.config)
        
        # Get parallelizable groups
        groups = get_parallelizable_groups(tasks, self.config)
        
        # Verify groups are created
        self.assertGreater(len(groups), 0)
        
        # Verify each group contains tasks
        for group in groups:
            self.assertGreater(len(group), 0)
            for task in group:
                self.assertIsInstance(task, AgentTask)
    
    def test_circuit_breaker_integration(self):
        """Test circuit breaker integration in agent execution"""
        from iceburg.infrastructure.retry_manager import RetryManager, RetryConfig
        
        retry_config = RetryConfig(
            max_retries=2,
            circuit_breaker_threshold=3,
            circuit_breaker_timeout=60.0
        )
        retry_manager = RetryManager(retry_config)
        
        # Get circuit breaker for agent
        circuit_breaker = retry_manager.get_circuit_breaker("agent_surveyor")
        
        # Verify circuit breaker exists
        self.assertIsNotNone(circuit_breaker)
        self.assertTrue(circuit_breaker.can_execute())
    
    def test_resource_allocation_integration(self):
        """Test dynamic resource allocation integration"""
        from iceburg.infrastructure.dynamic_resource_allocator import get_resource_allocator
        
        allocator = get_resource_allocator()
        
        # Get resource status
        status = allocator.get_resource_status()
        
        # Verify status structure
        self.assertIn("system", status)
        self.assertIn("allocated", status)
        self.assertIn("available", status)
    
    def test_load_balancer_integration(self):
        """Test load balancer integration"""
        from iceburg.distributed.load_balancer import IntelligentLoadBalancer, LoadBalancingStrategy
        
        load_balancer = IntelligentLoadBalancer(
            strategy=LoadBalancingStrategy.ADAPTIVE
        )
        
        # Add workers (agents)
        load_balancer.add_worker("surveyor", weight=1.0, capabilities=["information_gathering"])
        load_balancer.add_worker("dissident", weight=1.0, capabilities=["alternative_perspectives"])
        
        # Select worker
        async def test_selection():
            worker_id = await load_balancer.select_worker(
                query="test query",
                required_capabilities=["information_gathering"]
            )
            return worker_id
        
        worker_id = asyncio.run(test_selection())
        
        # Verify worker was selected
        self.assertIsNotNone(worker_id)
        self.assertIn(worker_id, ["surveyor", "dissident"])


class TestAgentWorkflowIntegration(unittest.TestCase):
    """Integration tests for complete agent workflows"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.query = Query(
            text="What are the implications of AI development?",
            metadata={}
        )
        
        self.config = ProtocolConfig(
            fast=False,
            verbose=False
        )
    
    def test_full_protocol_execution(self):
        """Test full protocol execution with all enhancements"""
        # Create plan
        tasks = plan(self.query, Mode.RESEARCH, self.config)
        
        # Optimize plan
        optimized_tasks = optimize_plan(tasks, self.config)
        
        # Get parallelizable groups
        groups = get_parallelizable_groups(optimized_tasks, self.config)
        
        # Verify workflow structure
        self.assertGreater(len(optimized_tasks), 0)
        self.assertGreater(len(groups), 0)
        
        # Verify dependencies are respected
        task_map = {task.agent: task for task in optimized_tasks}
        
        # Check that synthesist comes after surveyor and dissident
        if "synthesist" in task_map and "surveyor" in task_map:
            surveyor_idx = optimized_tasks.index(task_map["surveyor"])
            synthesist_idx = optimized_tasks.index(task_map["synthesist"])
            self.assertLess(surveyor_idx, synthesist_idx)
    
    def test_linguistic_enhancement_integration(self):
        """Test linguistic enhancement in agent communication"""
        from iceburg.agents.linguistic_intelligence import get_linguistic_engine
        
        linguistic_engine = get_linguistic_engine()
        
        # Test text enhancement
        original_text = "It is important to note that this is a very good solution."
        enhanced = linguistic_engine.enhance_text(
            original_text,
            verbosity_reduction=0.3,
            power_enhancement=0.5
        )
        
        # Verify enhancement occurred
        self.assertIsInstance(enhanced, type(enhanced).__class__)
        self.assertIsNotNone(enhanced.enhanced_text)
        self.assertLessEqual(len(enhanced.enhanced_text), len(original_text) * 1.2)  # Should be similar or shorter
    
    def test_security_hardening_integration(self):
        """Test security hardening integration"""
        from iceburg.security.security_hardening import get_security_hardening
        
        security_hardening = get_security_hardening()
        
        # Test input validation
        is_valid = security_hardening.validate_input(
            "SELECT * FROM users",
            schema=None
        )
        
        # Should detect SQL injection attempt
        self.assertFalse(is_valid)
        
        # Test output sanitization
        sanitized = security_hardening.sanitize_output("<script>alert('xss')</script>")
        self.assertNotIn("<script>", sanitized)
    
    def test_performance_optimization_integration(self):
        """Test performance optimization integration"""
        from iceburg.optimization.performance_optimizer import get_performance_optimizer
        
        optimizer = get_performance_optimizer()
        
        # Test caching
        @optimizer.cached(ttl=3600)
        def expensive_function(x):
            return x * 2
        
        result1 = expensive_function(5)
        result2 = expensive_function(5)  # Should use cache
        
        self.assertEqual(result1, result2)
        self.assertEqual(result1, 10)
        
        # Test graph optimization
        graph = {
            "surveyor": set(),
            "dissident": {"surveyor"},
            "synthesist": {"surveyor", "dissident"}
        }
        
        optimized = optimizer.optimize_graph(graph)
        
        # Verify topological order
        self.assertLess(optimized.index("surveyor"), optimized.index("dissident"))
        self.assertLess(optimized.index("dissident"), optimized.index("synthesist"))


if __name__ == '__main__':
    unittest.main()

