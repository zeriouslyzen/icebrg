#!/usr/bin/env python3
"""
Overnight Unbounded Research Run: Retrieval → Labs → Synthesis
Compares emergence deltas to baseline from today's runs.

Structure:
1. Retrieval Phase: Populate vector store with targeted corpus
2. Labs Phase: Run Real Scientific Research + Virtual Ecosystem + Hypothesis Testing
3. Synthesis Phase: Full protocol with Synthesist/Oracle
4. Emergence Delta: Compare emergence scores before/after
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import sys
import os

# Add project root and src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from iceburg.config import load_config
from iceburg.protocol.legacy.protocol_legacy import iceberg_protocol
from iceburg.memory.unified_memory import UnifiedMemory

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(str(Path(__file__).parent.parent / f"data/logs/overnight_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
            logging.StreamHandler()
        ]
)
logger = logging.getLogger(__name__)


class OvernightUnboundedRun:
    """Manages unbounded research cycle with retrieval → labs → synthesis."""
    
    def __init__(self, base_query: str, config: Dict[str, Any] = None):
        self.base_query = base_query
        self.config = config or {}
        self.cfg = load_config()
        self.memory = UnifiedMemory(self.cfg)
        
        # Track emergence scores
        self.baseline_emergence = None
        self.final_emergence = None
        self.emergence_delta = None
        
        # Results from each phase
        self.retrieval_results = {}
        self.lab_results = {}
        self.synthesis_results = {}
        
        # Output directory
        self.output_dir = Path(self.cfg.data_dir) / "overnight_runs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Initialized overnight run: {self.run_id}")
    
    def capture_baseline_emergence(self) -> Dict[str, Any]:
        """Capture current emergence state as baseline."""
        logger.info("[BASELINE] Capturing emergence baseline...")
        
        try:
            # Query vector store for existing emergence signals
            baseline_hits = self.memory.search("emergence", "quantum biology bioelectricity emergence breakthrough", k=10)
            
            # Calculate baseline emergence score
            emergence_count = len(baseline_hits)
            avg_confidence = sum(h.get("distance", 1.0) for h in baseline_hits) / max(1, emergence_count)
            
            baseline = {
                "emergence_signals": emergence_count,
                "avg_confidence": 1.0 - avg_confidence,  # Invert distance to confidence
                "timestamp": datetime.now().isoformat(),
                "run_id": self.run_id
            }
            
            self.baseline_emergence = baseline
            logger.info(f"[BASELINE] Emergence signals: {emergence_count}, avg confidence: {baseline['avg_confidence']:.3f}")
            
            return baseline
            
        except Exception as e:
            logger.error(f"[BASELINE] Error capturing baseline: {e}")
            return {"error": str(e)}
    
    async def phase_retrieval(self) -> Dict[str, Any]:
        """Phase 1: Retrieval-first - populate vector store with targeted corpus."""
        logger.info("[PHASE 1: RETRIEVAL] Starting retrieval phase...")
        
        retrieval_queries = [
            f"{self.base_query} - literature search and evidence curation",
            "quantum biology membrane potential ion channels",
            "bioelectricity emergence patterns coherence",
            "CIM stack consciousness integration metrics",
            "hypothesis testing lab protocols experimental design",
        ]
        
        results = {}
        for i, query in enumerate(retrieval_queries, 1):
            logger.info(f"[RETRIEVAL] Query {i}/{len(retrieval_queries)}: {query[:80]}...")
            
            try:
                # Run fast protocol to generate retrievable content
                result = iceberg_protocol(
                    query,
                    fast=True,
                    hybrid=False,
                    verbose=False
                )
                
                # Index result into memory
                self.memory.log_and_index(
                    run_id=self.run_id,
                    agent_id="retrieval_phase",
                    task_id=f"retrieval_{i}",
                    event_type="retrieval",
                    text=result,
                    meta={"query": query, "phase": "retrieval"}
                )
                
                results[f"retrieval_{i}"] = {
                    "query": query,
                    "result_length": len(result),
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"[RETRIEVAL] Indexed {len(result)} chars from query {i}")
                await asyncio.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"[RETRIEVAL] Error on query {i}: {e}")
                results[f"retrieval_{i}"] = {"error": str(e)}
        
        self.retrieval_results = results
        logger.info(f"[PHASE 1: RETRIEVAL] ✅ Completed: {len(results)} queries indexed")
        
        return results
    
    async def phase_labs(self) -> Dict[str, Any]:
        """Phase 2: Labs - run Real Scientific Research + Virtual Ecosystem + Hypothesis Testing."""
        logger.info("[PHASE 2: LABS] Starting labs phase...")
        
        lab_queries = [
            f"{self.base_query} - design experimental protocol with N, metrics, thresholds",
            f"{self.base_query} - generate virtual population and equipment for hypothesis testing",
            f"{self.base_query} - run statistical tests on bioelectricity emergence hypotheses",
        ]
        
        results = {}
        
        # Lab 1: Real Scientific Research
        try:
            logger.info("[LABS] Running Real Scientific Research...")
            from iceburg.agents.real_scientific_research import run_real_scientific_research
            
            research_result = run_real_scientific_research(
                self.cfg,
                lab_queries[0],
                None,
                verbose=False
            )
            
            results["real_research"] = research_result
            logger.info(f"[LABS] Real Research: {len(str(research_result))} chars")
            
        except Exception as e:
            logger.error(f"[LABS] Real Research error: {e}")
            results["real_research"] = {"error": str(e)}
        
        # Lab 2: Virtual Scientific Ecosystem
        try:
            logger.info("[LABS] Running Virtual Scientific Ecosystem...")
            from iceburg.agents.virtual_scientific_ecosystem import run
            
            ecosystem_result = run(
                self.cfg,
                lab_queries[1],
                None,
                verbose=False
            )
            
            results["virtual_ecosystem"] = ecosystem_result
            logger.info(f"[LABS] Virtual Ecosystem: {len(str(ecosystem_result))} chars")
            
        except Exception as e:
            logger.error(f"[LABS] Virtual Ecosystem error: {e}")
            results["virtual_ecosystem"] = {"error": str(e)}
        
        # Lab 3: Hypothesis Testing
        try:
            logger.info("[LABS] Running Hypothesis Testing Lab...")
            from iceburg.agents.hypothesis_testing_laboratory import HypothesisTestingLaboratory
            
            lab = HypothesisTestingLaboratory(self.cfg)
            hypothesis_query = f"{self.base_query} - test bioelectricity emergence coherence hypotheses"
            
            # Use the run() method which is available
            test_result = lab.run(self.cfg, hypothesis_query, None, verbose=False)
            results["hypothesis_test"] = {
                "query": hypothesis_query,
                "result": test_result,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"[LABS] Hypothesis Test: {str(test_result)[:100]}...")
            
        except Exception as e:
            logger.error(f"[LABS] Hypothesis Test error: {e}")
            results["hypothesis_test"] = {"error": str(e)}
        
        # Index lab results
        for lab_name, lab_result in results.items():
            self.memory.log_and_index(
                run_id=self.run_id,
                agent_id="lab_phase",
                task_id=lab_name,
                event_type="lab_result",
                text=str(lab_result),
                meta={"phase": "labs", "lab_type": lab_name}
            )
        
        self.lab_results = results
        logger.info(f"[PHASE 2: LABS] ✅ Completed: {len(results)} labs executed")
        
        return results
    
    async def phase_synthesis(self) -> Dict[str, Any]:
        """Phase 3: Synthesis - full protocol with Synthesist/Oracle."""
        logger.info("[PHASE 3: SYNTHESIS] Starting synthesis phase...")
        
        try:
            # Run full protocol (not fast mode) for deep synthesis
            synthesis_result = iceberg_protocol(
                self.base_query,
                fast=False,
                hybrid=False,
                verbose=True
            )
            
            # Index synthesis
            self.memory.log_and_index(
                run_id=self.run_id,
                agent_id="synthesis_phase",
                task_id="full_synthesis",
                event_type="synthesis",
                text=synthesis_result,
                meta={"phase": "synthesis", "protocol_mode": "full"}
            )
            
            self.synthesis_results = {
                "synthesis": synthesis_result,
                "length": len(synthesis_result),
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"[PHASE 3: SYNTHESIS] ✅ Completed: {len(synthesis_result)} chars")
            
            return self.synthesis_results
            
        except Exception as e:
            logger.error(f"[SYNTHESIS] Error: {e}")
            return {"error": str(e)}
    
    def capture_final_emergence(self) -> Dict[str, Any]:
        """Capture final emergence state after all phases."""
        logger.info("[FINAL] Capturing final emergence state...")
        
        try:
            # Query vector store again for emergence signals
            final_hits = self.memory.search("emergence", "quantum biology bioelectricity emergence breakthrough", k=10)
            
            emergence_count = len(final_hits)
            avg_confidence = sum(h.get("distance", 1.0) for h in final_hits) / max(1, emergence_count)
            
            final = {
                "emergence_signals": emergence_count,
                "avg_confidence": 1.0 - avg_confidence,
                "timestamp": datetime.now().isoformat(),
                "run_id": self.run_id
            }
            
            self.final_emergence = final
            
            # Calculate delta
            if self.baseline_emergence:
                self.emergence_delta = {
                    "signal_delta": final["emergence_signals"] - self.baseline_emergence["emergence_signals"],
                    "confidence_delta": final["avg_confidence"] - self.baseline_emergence["avg_confidence"],
                    "percent_change_signals": ((final["emergence_signals"] - self.baseline_emergence["emergence_signals"]) / max(1, self.baseline_emergence["emergence_signals"])) * 100,
                    "percent_change_confidence": ((final["avg_confidence"] - self.baseline_emergence["avg_confidence"]) / max(0.01, self.baseline_emergence["avg_confidence"])) * 100,
                }
            
            logger.info(f"[FINAL] Emergence signals: {emergence_count}, avg confidence: {final['avg_confidence']:.3f}")
            if self.emergence_delta:
                logger.info(f"[DELTA] Signal delta: {self.emergence_delta['signal_delta']}, Confidence delta: {self.emergence_delta['confidence_delta']:.3f}")
            
            return final
            
        except Exception as e:
            logger.error(f"[FINAL] Error capturing final emergence: {e}")
            return {"error": str(e)}
    
    def save_results(self):
        """Save all results to JSON file."""
        output_file = self.output_dir / f"unbounded_run_{self.run_id}.json"
        
        results = {
            "run_id": self.run_id,
            "base_query": self.base_query,
            "baseline_emergence": self.baseline_emergence,
            "final_emergence": self.final_emergence,
            "emergence_delta": self.emergence_delta,
            "retrieval_results": self.retrieval_results,
            "lab_results": self.lab_results,
            "synthesis_results": self.synthesis_results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"[SAVE] Results saved to {output_file}")
        return output_file
    
    async def run_full_cycle(self):
        """Execute full unbounded cycle: retrieval → labs → synthesis."""
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info("OVERNIGHT UNBOUNDED RESEARCH CYCLE STARTING")
        logger.info("=" * 80)
        
        # Baseline
        baseline = self.capture_baseline_emergence()
        
        # Phase 1: Retrieval
        await self.phase_retrieval()
        
        # Phase 2: Labs
        await self.phase_labs()
        
        # Phase 3: Synthesis
        await self.phase_synthesis()
        
        # Final emergence
        final = self.capture_final_emergence()
        
        # Save results
        output_file = self.save_results()
        
        total_time = time.time() - start_time
        
        logger.info("=" * 80)
        logger.info("OVERNIGHT UNBOUNDED RESEARCH CYCLE COMPLETED")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Results: {output_file}")
        logger.info("=" * 80)
        
        if self.emergence_delta:
            logger.info("\nEMERGENCE DELTA SUMMARY:")
            logger.info(f"  Signal delta: {self.emergence_delta['signal_delta']:+d}")
            logger.info(f"  Confidence delta: {self.emergence_delta['confidence_delta']:+.3f}")
            logger.info(f"  Signal % change: {self.emergence_delta['percent_change_signals']:+.1f}%")
            logger.info(f"  Confidence % change: {self.emergence_delta['percent_change_confidence']:+.1f}%")
        
        return {
            "run_id": self.run_id,
            "total_time_minutes": total_time / 60,
            "output_file": str(output_file),
            "emergence_delta": self.emergence_delta
        }


async def main():
    """Main entry point for overnight unbounded run."""
    base_query = os.getenv(
        "ICEBURG_OVERNIGHT_QUERY",
        "Full research cycle on quantum biology + bioelectricity using CIM stack, emergence detection, hypothesis testing lab, and grounding layer. Autonomously design experiments, run analyses, detect emergence, and report evidence/confidence/next steps."
    )
    
    config = {
        "max_concurrent_queries": 3,
        "research_cycle_interval": 300,
    }
    
    runner = OvernightUnboundedRun(base_query, config)
    await runner.run_full_cycle()


if __name__ == "__main__":
    asyncio.run(main())

