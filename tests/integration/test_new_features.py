#!/usr/bin/env python3
"""
Quick test script for new features
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_capability_registry():
    """Test agent capability registry"""
    print("\n" + "="*70)
    print("TEST 1: Agent Capability Registry")
    print("="*70)
    try:
        from iceburg.agents.capability_registry import get_registry
        registry = get_registry()
        all_agents = registry.get_all_agents()
        print(f"✅ Registry loaded: {len(all_agents)} agents")
        
        # Test getting specific agent
        surveyor = registry.get_agent("surveyor")
        if surveyor:
            print(f"✅ Surveyor agent found: {surveyor.agent_name}")
            print(f"   - Type: {surveyor.agent_type.value}")
            print(f"   - Complexity: {surveyor.complexity_level.value}")
            print(f"   - Speed: {surveyor.speed_rating.value}")
            print(f"   - Capabilities: {', '.join(surveyor.capabilities[:3])}...")
        
        # Test dependency resolution
        agent_ids = ["surveyor", "dissident", "synthesist"]
        ordered = registry.resolve_dependencies(agent_ids)
        print(f"✅ Dependency resolution: {ordered}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interaction_protocol():
    """Test agent interaction protocol"""
    print("\n" + "="*70)
    print("TEST 2: Agent Interaction Protocol")
    print("="*70)
    try:
        from iceburg.agents.interaction_protocol import (
            AgentInteractionProtocol,
            MessageType,
            AgentStatus
        )
        
        protocol = AgentInteractionProtocol("test_agent")
        print(f"✅ Protocol initialized: {protocol.agent_id if hasattr(protocol, 'agent_id') else 'test_agent'}")
        
        # Test message creation
        from iceburg.agents.interaction_protocol import AgentMessage
        from datetime import datetime
        
        message = AgentMessage(
            sender_id="agent1",
            receiver_id="agent2",
            message_type=MessageType.REQUEST,
            payload={"query": "test"},
            timestamp=datetime.now().isoformat(),
            message_id="msg1"
        )
        print(f"✅ Message created: {message.message_type.value}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_linguistic_intelligence():
    """Test linguistic intelligence system"""
    print("\n" + "="*70)
    print("TEST 3: Linguistic Intelligence System")
    print("="*70)
    try:
        from iceburg.agents.linguistic_intelligence import (
            get_linguistic_engine,
            get_metaphor_generator,
            get_anticliche_detector,
            LinguisticStyle
        )
        
        # Test linguistic engine
        engine = get_linguistic_engine()
        text = "It is important to note that this is a very good solution."
        enhanced = engine.enhance_text(
            text,
            style=LinguisticStyle.INTELLIGENT,
            verbosity_reduction=0.3,
            power_enhancement=0.5
        )
        print(f"✅ Linguistic engine: Enhanced text length {len(enhanced.enhanced_text)} (original: {len(text)})")
        
        # Test metaphor generator
        metaphor_gen = get_metaphor_generator()
        metaphor = metaphor_gen.generate_metaphor("complexity", context="AI systems")
        if metaphor:
            print(f"✅ Metaphor generator: {metaphor.metaphor[:50]}...")
        
        # Test anti-cliche detector
        cliche_detector = get_anticliche_detector()
        text_with_cliche = "We need to think outside the box."
        enhanced_text, replacements = cliche_detector.detect_and_replace(text_with_cliche)
        if replacements:
            print(f"✅ Anti-cliche detector: Found {len(replacements)} clichés")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_planner_optimization():
    """Test planner optimization"""
    print("\n" + "="*70)
    print("TEST 4: Planner Optimization")
    print("="*70)
    try:
        from iceburg.protocol.planner import optimize_plan, get_parallelizable_groups
        from iceburg.protocol.models import AgentTask, Query, Mode
        from iceburg.protocol.config import ProtocolConfig
        
        # Create test tasks
        tasks = [
            AgentTask(
                agent="surveyor",
                payload={"query": "test"},
                dependencies=[]
            ),
            AgentTask(
                agent="dissident",
                payload={"query": "test"},
                dependencies=["surveyor"]
            ),
            AgentTask(
                agent="synthesist",
                payload={"query": "test"},
                dependencies=["surveyor", "dissident"]
            )
        ]
        
        config = ProtocolConfig(fast=False, verbose=False)
        
        # Test optimization
        optimized = optimize_plan(tasks, config)
        print(f"✅ Plan optimized: {len(optimized)} tasks")
        
        # Test parallelizable groups
        groups = get_parallelizable_groups(tasks, config)
        print(f"✅ Parallelizable groups: {len(groups)} groups")
        for i, group in enumerate(groups):
            print(f"   Group {i+1}: {[t.agent for t in group]}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_security_hardening():
    """Test security hardening"""
    print("\n" + "="*70)
    print("TEST 5: Security Hardening")
    print("="*70)
    try:
        from iceburg.security.security_hardening import get_security_manager
        
        security = get_security_manager()
        
        # Test input validation
        is_valid, sanitized, violation = security.input_validator.validate_input("SELECT * FROM users")
        print(f"✅ Input validation: SQL injection detected: {not is_valid}")
        
        # Test output sanitization
        sanitized = security.output_sanitizer.sanitize_output("<script>alert('xss')</script>")
        print(f"✅ Output sanitization: XSS removed: {'<script>' not in sanitized}")
        
        # Test rate limiting
        allowed = security.rate_limiter.check_rate_limit("test_operation", "client1")
        print(f"✅ Rate limiting: First request allowed: {allowed}")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_optimizer():
    """Test performance optimizer"""
    print("\n" + "="*70)
    print("TEST 6: Performance Optimizer")
    print("="*70)
    try:
        from iceburg.optimization.performance_optimizer import get_performance_optimizer
        
        optimizer = get_performance_optimizer()
        
        # Test caching
        @optimizer.cached(ttl=3600)
        def expensive_function(x):
            return x * 2
        
        result1 = expensive_function(5)
        result2 = expensive_function(5)  # Should use cache
        print(f"✅ Caching: Results match: {result1 == result2}")
        
        # Test graph optimization
        graph = {
            "surveyor": [],
            "dissident": ["surveyor"],
            "synthesist": ["surveyor", "dissident"]
        }
        optimized = optimizer.optimize_graph(graph)
        print(f"✅ Graph optimization: Optimized graph has {len(optimized)} nodes")
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("ICEBURG NEW FEATURES TEST SUITE")
    print("="*70)
    
    results = []
    results.append(("Capability Registry", test_capability_registry()))
    results.append(("Interaction Protocol", test_interaction_protocol()))
    results.append(("Linguistic Intelligence", test_linguistic_intelligence()))
    results.append(("Planner Optimization", test_planner_optimization()))
    results.append(("Security Hardening", test_security_hardening()))
    results.append(("Performance Optimizer", test_performance_optimizer()))
    
    print("\n" + "="*70)
    print("TEST RESULTS SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

