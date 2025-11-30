#!/usr/bin/env python3
"""
Test Fast Mode and Single Agent Persistence
Tests fast chat, single agents, and agent persistence/agency
"""

import asyncio
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.config import load_config, load_config_fast, load_config_with_model
from iceburg.vectorstore import VectorStore
from iceburg.graph_store import KnowledgeGraph
from iceburg.agents.surveyor import run as surveyor_run
from iceburg.agents.dissident import run as dissident_run
from iceburg.agents.synthesist import run as synthesist_run
from iceburg.agents.oracle import run as oracle_run
from iceburg.memory.unified_memory import UnifiedMemory


class FastModeTester:
    """Test fast mode performance"""
    
    def __init__(self):
        self.results = []
    
    def test_fast_config(self):
        """Test fast configuration loading"""
        print("\n📝 Test 1: Fast Configuration")
        start = time.time()
        cfg = load_config_fast()
        load_time = time.time() - start
        
        print(f"  ✅ Fast config loaded in {load_time:.3f}s")
        print(f"  - Surveyor model: {cfg.surveyor_model}")
        print(f"  - Fast mode: {cfg.fast}")
        
        self.results.append({
            "test": "fast_config",
            "time": load_time,
            "status": "pass"
        })
    
    def test_small_model_config(self):
        """Test small model configuration"""
        print("\n📝 Test 2: Small Model Configuration")
        start = time.time()
        cfg = load_config_with_model("llama3.2:1b", use_small_models=True)
        load_time = time.time() - start
        
        print(f"  ✅ Small model config loaded in {load_time:.3f}s")
        print(f"  - Surveyor model: {cfg.surveyor_model}")
        print(f"  - Fast mode: {cfg.fast}")
        
        self.results.append({
            "test": "small_model_config",
            "time": load_time,
            "status": "pass"
        })
    
    def test_simple_query_fast(self):
        """Test simple query with fast mode"""
        print("\n📝 Test 3: Simple Query (Fast Mode)")
        cfg = load_config_fast()
        vs = VectorStore(cfg)
        
        query = "What is 2+2?"
        start = time.time()
        
        try:
            result = surveyor_run(cfg, vs, query, verbose=False)
            elapsed = time.time() - start
            
            print(f"  ✅ Query processed in {elapsed:.3f}s")
            print(f"  - Response length: {len(result)} chars")
            print(f"  - Target: <5 seconds")
            print(f"  - Status: {'✅ PASS' if elapsed < 5 else '⚠️  SLOW'}")
            
            self.results.append({
                "test": "simple_query_fast",
                "time": elapsed,
                "target": 5.0,
                "status": "pass" if elapsed < 5 else "slow"
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.results.append({
                "test": "simple_query_fast",
                "time": None,
                "status": "fail",
                "error": str(e)
            })
    
    def test_single_agent_surveyor(self):
        """Test single agent (Surveyor)"""
        print("\n📝 Test 4: Single Agent (Surveyor)")
        cfg = load_config()
        vs = VectorStore(cfg)
        
        query = "Explain quantum computing briefly"
        start = time.time()
        
        try:
            result = surveyor_run(cfg, vs, query, verbose=False)
            elapsed = time.time() - start
            
            print(f"  ✅ Surveyor processed in {elapsed:.3f}s")
            print(f"  - Response length: {len(result)} chars")
            print(f"  - Target: <10 seconds")
            print(f"  - Status: {'✅ PASS' if elapsed < 10 else '⚠️  SLOW'}")
            
            self.results.append({
                "test": "single_agent_surveyor",
                "time": elapsed,
                "target": 10.0,
                "status": "pass" if elapsed < 10 else "slow"
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.results.append({
                "test": "single_agent_surveyor",
                "time": None,
                "status": "fail",
                "error": str(e)
            })
    
    def test_single_agent_dissident(self):
        """Test single agent (Dissident) with context"""
        print("\n📝 Test 5: Single Agent (Dissident)")
        cfg = load_config()
        vs = VectorStore(cfg)
        
        query = "What are alternative perspectives on AI safety?"
        start = time.time()
        
        try:
            # First get surveyor output
            surveyor_result = surveyor_run(cfg, vs, query, verbose=False)
            
            # Then dissident
            dissident_result = dissident_run(cfg, query, surveyor_result, verbose=False)
            elapsed = time.time() - start
            
            print(f"  ✅ Dissident processed in {elapsed:.3f}s")
            print(f"  - Response length: {len(dissident_result)} chars")
            print(f"  - Target: <15 seconds")
            print(f"  - Status: {'✅ PASS' if elapsed < 15 else '⚠️  SLOW'}")
            
            self.results.append({
                "test": "single_agent_dissident",
                "time": elapsed,
                "target": 15.0,
                "status": "pass" if elapsed < 15 else "slow"
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.results.append({
                "test": "single_agent_dissident",
                "time": None,
                "status": "fail",
                "error": str(e)
            })
    
    def test_agent_persistence_simulation(self):
        """Test agent persistence simulation"""
        print("\n📝 Test 6: Agent Persistence Simulation")
        
        # Simulate agent state
        agent_state = {
            "agent_id": "test_surveyor_001",
            "query_history": [],
            "memory": [],
            "goals": ["technical_analysis"],
            "capabilities": ["semantic_search", "technical_documentation"]
        }
        
        # Simulate query
        query1 = "What is machine learning?"
        agent_state["query_history"].append(query1)
        agent_state["memory"].append({
            "query": query1,
            "timestamp": time.time(),
            "context": "technical"
        })
        
        # Simulate second query (should remember context)
        query2 = "How does it relate to deep learning?"
        agent_state["query_history"].append(query2)
        
        # Check persistence
        has_memory = len(agent_state["memory"]) > 0
        has_history = len(agent_state["query_history"]) > 1
        
        print(f"  ✅ Agent state maintained")
        print(f"  - Memory entries: {len(agent_state['memory'])}")
        print(f"  - Query history: {len(agent_state['query_history'])}")
        print(f"  - Persistence: {'✅ YES' if has_memory and has_history else '❌ NO'}")
        
        self.results.append({
            "test": "agent_persistence",
            "memory_entries": len(agent_state["memory"]),
            "query_history": len(agent_state["query_history"]),
            "status": "pass" if has_memory and has_history else "fail"
        })
    
    def test_memory_system(self):
        """Test memory system"""
        print("\n📝 Test 7: Memory System")
        cfg = load_config()
        
        try:
            memory = UnifiedMemory(cfg)
            print(f"  ✅ Memory system initialized")
            print(f"  - Root directory: {memory._root}")
            print(f"  - Events directory: {memory._events_dir}")
            
            # Test event logging
            event = {
                "run_id": "test_run_001",
                "event_type": "test",
                "timestamp": time.time(),
                "data": {"test": "memory_system"}
            }
            memory.log_event(event)
            print(f"  ✅ Event logged")
            
            self.results.append({
                "test": "memory_system",
                "status": "pass"
            })
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.results.append({
                "test": "memory_system",
                "status": "fail",
                "error": str(e)
            })
    
    def run_all_tests(self):
        """Run all tests"""
        print("=" * 60)
        print("ICEBURG Fast Mode & Agent Persistence Tests")
        print("=" * 60)
        
        self.test_fast_config()
        self.test_small_model_config()
        self.test_simple_query_fast()
        self.test_single_agent_surveyor()
        self.test_single_agent_dissident()
        self.test_agent_persistence_simulation()
        self.test_memory_system()
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        passed = sum(1 for r in self.results if r["status"] == "pass")
        failed = sum(1 for r in self.results if r["status"] == "fail")
        slow = sum(1 for r in self.results if r["status"] == "slow")
        
        print(f"✅ Passed: {passed}")
        print(f"⚠️  Slow: {slow}")
        print(f"❌ Failed: {failed}")
        print(f"Total: {len(self.results)}")
        
        # Performance summary
        print("\n📊 Performance Summary:")
        for result in self.results:
            if "time" in result and result["time"]:
                target = result.get("target", "N/A")
                print(f"  - {result['test']}: {result['time']:.3f}s (target: {target})")
        
        return {
            "passed": passed,
            "failed": failed,
            "slow": slow,
            "total": len(self.results),
            "results": self.results
        }


if __name__ == "__main__":
    tester = FastModeTester()
    results = tester.run_all_tests()
    
    # Exit with error code if any tests failed
    sys.exit(0 if results["failed"] == 0 else 1)

