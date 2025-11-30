#!/usr/bin/env python3
"""
ICEBURG Autonomous Stress Test & Emergence Forcing Script

This script tests ICEBURG's autonomous capabilities including:
- Forced emergence detection and data generation
- Self-modification testing
- Stress testing to find intelligence gaps
- Autonomous schematic studies without direction
- Background execution with health monitoring
"""

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import signal
import psutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.config import load_config
from iceburg.protocol import iceberg_protocol
from iceburg.autonomous.research_orchestrator import AutonomousResearchOrchestrator
from iceburg.evolution.evolution_pipeline import EvolutionPipeline
from iceburg.emergence_engine import EmergenceEngine
from iceburg.learning.autonomous_improvement import AutonomousLearner
try:
    from iceburg.protocol.execution.agents.autonomous_goal_formation import run as autonomous_goals
except ImportError:
    # Fallback if not available
    def autonomous_goals(cfg, query, verbose=False):
        return {"error": "autonomous_goal_formation not available"}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/logs/autonomous_stress_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global state for monitoring
class TestState:
    def __init__(self):
        self.active = True
        self.cycle_count = 0
        self.emergence_count = 0
        self.breakthrough_count = 0
        self.self_modification_count = 0
        self.errors = []
        self.last_activity = time.time()
        self.stuck_threshold = 300  # 5 minutes
        self.results = []
        
    def check_stuck(self) -> bool:
        """Check if system is stuck (no activity for threshold time)"""
        elapsed = time.time() - self.last_activity
        return elapsed > self.stuck_threshold
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()

test_state = TestState()

# Signal handlers
def signal_handler(sig, frame):
    logger.info("Received interrupt signal, shutting down gracefully...")
    test_state.active = False
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def force_emergence_detection(cfg, query: str) -> Dict[str, Any]:
    """Force emergence detection and generate new data"""
    logger.info(f"[EMERGENCE] Forcing emergence detection for: {query[:50]}...")
    test_state.update_activity()
    
    try:
        # Run protocol to generate data
        result = iceberg_protocol(
            query,
            verbose=True,
            fast=False,
            evidence_strict=False
        )
        
        # Analyze for emergence
        emergence_engine = EmergenceEngine({"data_dir": cfg.data_dir} if cfg else {})
        emergence_result = await emergence_engine.detect_emergence(query)
        
        if emergence_result.get("emergence_detected"):
            test_state.emergence_count += 1
            logger.info(f"[EMERGENCE] ✅ Emergence detected! Confidence: {emergence_result.get('confidence', 0):.2f}")
        
        test_state.update_activity()
        return {
            "query": query,
            "result": result,
            "emergence": emergence_result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"[EMERGENCE] Error: {e}")
        test_state.errors.append({"type": "emergence", "error": str(e)})
        test_state.update_activity()
        return {"error": str(e)}


async def test_self_modification(cfg) -> Dict[str, Any]:
    """Test ICEBURG's self-modification capabilities"""
    logger.info("[SELF-MOD] Testing self-modification capabilities...")
    test_state.update_activity()
    
    try:
        # Test autonomous goal formation
        goal_query = "Formulate autonomous research goals and self-improvement objectives"
        goal_result = autonomous_goals(cfg, goal_query, verbose=True)
        
        # Test evolution pipeline
        evolution_pipeline = EvolutionPipeline({"max_concurrent_jobs": 1})
        evolution_result = await evolution_pipeline.evolve_system("autonomous_test")
        
        # Test autonomous learner
        learner = AutonomousLearner({"learning_interval_hours": 0.1, "min_interactions": 10})
        await learner.start_learning()
        await asyncio.sleep(5)  # Let it run briefly
        improvement_result = learner.get_learning_status()
        await learner.stop_learning()
        
        test_state.self_modification_count += 1
        test_state.update_activity()
        
        logger.info(f"[SELF-MOD] ✅ Self-modification test completed")
        
        return {
            "goals": goal_result,
            "evolution": evolution_result,
            "improvements": improvement_result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"[SELF-MOD] Error: {e}")
        test_state.errors.append({"type": "self_modification", "error": str(e)})
        test_state.update_activity()
        return {"error": str(e)}


async def stress_test_intelligence(cfg) -> Dict[str, Any]:
    """Stress test to find gaps in intelligence"""
    logger.info("[STRESS] Starting intelligence gap stress test...")
    test_state.update_activity()
    
    stress_queries = [
        "What are the fundamental limitations of current AI systems?",
        "How can consciousness emerge from non-conscious components?",
        "What patterns exist across quantum physics, biology, and consciousness?",
        "What knowledge has been systematically suppressed in scientific research?",
        "How can we verify the existence of phenomena that cannot be directly observed?",
        "What are the boundary conditions for emergent intelligence?",
        "How does information flow in systems that transcend classical computation?",
        "What are the unknown unknowns in our understanding of reality?",
    ]
    
    results = []
    for query in stress_queries:
        try:
            logger.info(f"[STRESS] Testing: {query[:50]}...")
            result = await force_emergence_detection(cfg, query)
            results.append(result)
            
            # Check for gaps
            if "error" in result:
                logger.warning(f"[STRESS] ⚠️ Gap detected in query: {query[:50]}")
            
            await asyncio.sleep(2)  # Rate limiting
            test_state.update_activity()
        except Exception as e:
            logger.error(f"[STRESS] Error with query '{query[:50]}...': {e}")
            test_state.errors.append({"type": "stress_test", "query": query, "error": str(e)})
            test_state.update_activity()
    
    return {
        "stress_tests": results,
        "total_tests": len(stress_queries),
        "errors": len([r for r in results if "error" in r]),
        "timestamp": datetime.now().isoformat()
    }


async def autonomous_schematic_studies(cfg) -> Dict[str, Any]:
    """Let ICEBURG do its own thing - generate autonomous schematic studies"""
    logger.info("[AUTONOMOUS] Starting autonomous schematic studies...")
    test_state.update_activity()
    
    try:
        # Initialize autonomous research orchestrator
        orchestrator = AutonomousResearchOrchestrator({"max_concurrent_queries": 3})
        
        # Start autonomous research
        await orchestrator.start_autonomous_research()
        
        # Let it run for a bit
        await asyncio.sleep(60)  # 1 minute autonomous operation
        
        # Get results (with error handling for missing methods)
        try:
            # Get results from history
            results = list(orchestrator.results_history)[-10:] if hasattr(orchestrator, 'results_history') else []
        except:
            results = []
        
        try:
            # Get emergence patterns
            patterns = orchestrator.emergence_patterns if hasattr(orchestrator, 'emergence_patterns') else []
        except:
            patterns = []
        
        # Stop autonomous research
        await orchestrator.stop_autonomous_research()
        
        test_state.update_activity()
        logger.info(f"[AUTONOMOUS] ✅ Generated {len(results)} autonomous results, {len(patterns)} emergence patterns")
        
        return {
            "results": results,
            "patterns": patterns,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"[AUTONOMOUS] Error: {e}")
        test_state.errors.append({"type": "autonomous_studies", "error": str(e)})
        test_state.update_activity()
        return {"error": str(e)}


async def health_monitor():
    """Monitor system health and check for stuck states"""
    while test_state.active:
        await asyncio.sleep(30)  # Check every 30 seconds
        
        if test_state.check_stuck():
            logger.warning(f"[HEALTH] ⚠️ System appears stuck (no activity for {test_state.stuck_threshold}s)")
            logger.warning(f"[HEALTH] Last activity: {time.time() - test_state.last_activity:.0f}s ago")
            logger.warning(f"[HEALTH] Cycle count: {test_state.cycle_count}")
            logger.warning(f"[HEALTH] Errors: {len(test_state.errors)}")
            
            # Try to recover
            test_state.update_activity()
            logger.info("[HEALTH] Attempting recovery...")
        
        # Log status
        logger.info(f"[HEALTH] Status - Cycles: {test_state.cycle_count}, "
                   f"Emergence: {test_state.emergence_count}, "
                   f"Breakthroughs: {test_state.breakthrough_count}, "
                   f"Self-Mod: {test_state.self_modification_count}, "
                   f"Errors: {len(test_state.errors)}")


async def main_test_loop(cfg):
    """Main autonomous test loop"""
    logger.info("=" * 80)
    logger.info("ICEBURG AUTONOMOUS STRESS TEST STARTING")
    logger.info("=" * 80)
    
    # Start health monitor
    health_task = asyncio.create_task(health_monitor())
    
    try:
        while test_state.active:
            test_state.cycle_count += 1
            logger.info(f"\n{'='*80}")
            logger.info(f"TEST CYCLE {test_state.cycle_count}")
            logger.info(f"{'='*80}\n")
            
            # Phase 1: Force emergence detection
            logger.info("[PHASE 1] Forcing emergence detection...")
            emergence_queries = [
                "What novel patterns emerge when connecting quantum mechanics, consciousness, and information theory?",
                "How do bioelectric fields interact with electromagnetic fields to create coherent information structures?",
                "What breakthrough discoveries are possible when synthesizing suppressed research across domains?",
            ]
            
            for query in emergence_queries:
                if not test_state.active:
                    break
                result = await force_emergence_detection(cfg, query)
                test_state.results.append(result)
                await asyncio.sleep(5)  # Rate limiting
            
            # Phase 2: Test self-modification
            logger.info("\n[PHASE 2] Testing self-modification...")
            if test_state.active:
                mod_result = await test_self_modification(cfg)
                test_state.results.append(mod_result)
            
            # Phase 3: Stress test intelligence gaps
            logger.info("\n[PHASE 3] Stress testing intelligence gaps...")
            if test_state.active:
                stress_result = await stress_test_intelligence(cfg)
                test_state.results.append(stress_result)
            
            # Phase 4: Autonomous schematic studies
            logger.info("\n[PHASE 4] Running autonomous schematic studies...")
            if test_state.active:
                auto_result = await autonomous_schematic_studies(cfg)
                test_state.results.append(auto_result)
            
            # Save results
            results_file = Path("data/autonomous_test_results.jsonl")
            results_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_file, "a") as f:
                for result in test_state.results[-10:]:  # Save last 10 results
                    f.write(json.dumps(result) + "\n")
            
            test_state.results = []  # Clear to avoid memory issues
            
            # Wait before next cycle
            if test_state.active:
                logger.info(f"\n[WAIT] Waiting 60 seconds before next cycle...")
                await asyncio.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error in main loop: {e}", exc_info=True)
    finally:
        test_state.active = False
        health_task.cancel()
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("TEST SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total cycles: {test_state.cycle_count}")
        logger.info(f"Emergence detected: {test_state.emergence_count}")
        logger.info(f"Breakthroughs: {test_state.breakthrough_count}")
        logger.info(f"Self-modification tests: {test_state.self_modification_count}")
        logger.info(f"Total errors: {len(test_state.errors)}")
        
        if test_state.errors:
            logger.info("\nErrors encountered:")
            for error in test_state.errors[-10:]:  # Last 10 errors
                logger.info(f"  - {error}")


def main():
    """Main entry point"""
    # Load configuration
    cfg = load_config()
    
    # Create output directories
    Path("data/logs").mkdir(parents=True, exist_ok=True)
    Path("data/autonomous_test").mkdir(parents=True, exist_ok=True)
    
    # Run async main loop
    try:
        asyncio.run(main_test_loop(cfg))
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()

