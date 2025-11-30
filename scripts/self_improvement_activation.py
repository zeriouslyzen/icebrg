#!/usr/bin/env python3
"""
ICEBURG Self-Improvement Activation Script
Activates self-improvement agents to identify issues and propose upgrades
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from iceburg.protocol.config import ProtocolConfig, load_config
try:
    from iceburg.protocol.execution.agents.self_redesign_engine import run as self_redesign_run
    SELF_REDESIGN_AVAILABLE = True
except ImportError:
    SELF_REDESIGN_AVAILABLE = False
    self_redesign_run = None

try:
    from iceburg.protocol.execution.agents.unbounded_learning_engine import run as unbounded_learning_run
    UNBOUNDED_LEARNING_AVAILABLE = True
except ImportError:
    UNBOUNDED_LEARNING_AVAILABLE = False
    unbounded_learning_run = None

try:
    from iceburg.protocol.execution.agents.novel_intelligence_creator import run as novel_intelligence_run
    NOVEL_INTELLIGENCE_AVAILABLE = True
except ImportError:
    NOVEL_INTELLIGENCE_AVAILABLE = False
    novel_intelligence_run = None

try:
    from iceburg.protocol.execution.agents.autonomous_goal_formation import run as autonomous_goals_run
    AUTONOMOUS_GOALS_AVAILABLE = True
except ImportError:
    AUTONOMOUS_GOALS_AVAILABLE = False
    autonomous_goals_run = None

try:
    from iceburg.agents.capability_gap_detector import CapabilityGapDetector
    CAPABILITY_GAP_AVAILABLE = True
except ImportError:
    CAPABILITY_GAP_AVAILABLE = False
    CapabilityGapDetector = None

try:
    from iceburg.config import IceburgConfig
    ICEBURG_CONFIG_AVAILABLE = True
except ImportError:
    ICEBURG_CONFIG_AVAILABLE = False
    IceburgConfig = None

try:
    from iceburg.evolution.evolution_pipeline import EvolutionPipeline
    EVOLUTION_AVAILABLE = True
except ImportError:
    EVOLUTION_AVAILABLE = False
    EvolutionPipeline = None
import json
from datetime import datetime


def main():
    """Main self-improvement activation"""
    print("=" * 80)
    print("ICEBURG SELF-IMPROVEMENT ACTIVATION")
    print("=" * 80)
    print()
    
    # Load configuration
    cfg = load_config()
    iceburg_cfg = None
    if ICEBURG_CONFIG_AVAILABLE:
        try:
            # Try to create IceburgConfig with defaults
            iceburg_cfg = IceburgConfig(
                data_dir="data",
                surveyor_model="llama3.1:8b",
                dissident_model="llama3.1:8b",
                synthesist_model="llama3.1:8b",
                oracle_model="llama3.1:8b",
                embed_model="nomic-embed-text"
            )
        except Exception as e:
            print(f"Warning: Could not create IceburgConfig: {e}")
            iceburg_cfg = None
    
    # Query for self-improvement
    query = "Analyze ICEBURG's current architecture, identify bottlenecks and limitations, and propose specific upgrades and improvements"
    
    print("=" * 80)
    print("STEP 1: SELF-REDESIGN ENGINE")
    print("=" * 80)
    print()
    
    # Run Self-Redesign Engine
    if SELF_REDESIGN_AVAILABLE:
        try:
            redesign_result = self_redesign_run(
                cfg=cfg,
                query=query,
                context=None,
                verbose=True
            )
            print(redesign_result)
            print()
            
            # Save result
            output_dir = Path("data/self_improvement")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"self_redesign_{int(datetime.now().timestamp())}.txt", "w") as f:
                f.write(redesign_result)
        
        except Exception as e:
            print(f"Error in Self-Redesign Engine: {e}")
    else:
        print("Self-Redesign Engine not available (import error)")
        print()
    
    print("=" * 80)
    print("STEP 2: CAPABILITY GAP DETECTOR")
    print("=" * 80)
    print()
    
    # Run Capability Gap Detector
    if CAPABILITY_GAP_AVAILABLE and ICEBURG_CONFIG_AVAILABLE:
        try:
            gap_detector = CapabilityGapDetector(iceburg_cfg)
            
            # Create sample reasoning chain
            reasoning_chain = {
                "surveyor": {"sources": ["source1", "source2"], "analysis": "Sample analysis"},
                "dissident": {"contradictions": [], "challenges": []},
                "synthesist": {"synthesis": "Sample synthesis"},
                "oracle": {"decision": "Sample decision"}
            }
            
            gap_analysis = gap_detector.run(
                reasoning_chain=reasoning_chain,
                initial_query=query,
                verbose=True
            )
            
            print(json.dumps(gap_analysis, indent=2))
            print()
            
            # Save result
            output_dir = Path("data/self_improvement")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"capability_gaps_{int(datetime.now().timestamp())}.json", "w") as f:
                json.dump(gap_analysis, f, indent=2)
        
        except Exception as e:
            print(f"Error in Capability Gap Detector: {e}")
    else:
        print("Capability Gap Detector not available (import error)")
        print()
    
    print("=" * 80)
    print("STEP 3: UNBOUNDED LEARNING ENGINE")
    print("=" * 80)
    print()
    
    # Run Unbounded Learning Engine
    if UNBOUNDED_LEARNING_AVAILABLE:
        try:
            learning_result = unbounded_learning_run(
                cfg=cfg,
                query="Learn about the latest AI research and identify improvements for ICEBURG",
                context=None,
                verbose=True
            )
            print(learning_result)
            print()
            
            # Save result
            output_dir = Path("data/self_improvement")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"unbounded_learning_{int(datetime.now().timestamp())}.txt", "w") as f:
                f.write(learning_result)
        
        except Exception as e:
            print(f"Error in Unbounded Learning Engine: {e}")
    else:
        print("Unbounded Learning Engine not available (import error)")
        print()
    
    print("=" * 80)
    print("STEP 4: NOVEL INTELLIGENCE CREATOR")
    print("=" * 80)
    print()
    
    # Run Novel Intelligence Creator
    if NOVEL_INTELLIGENCE_AVAILABLE:
        try:
            intelligence_result = novel_intelligence_run(
                cfg=cfg,
                query="Create novel intelligence architectures for ICEBURG improvements",
                context=None,
                verbose=True
            )
            print(intelligence_result)
            print()
            
            # Save result
            output_dir = Path("data/self_improvement")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"novel_intelligence_{int(datetime.now().timestamp())}.txt", "w") as f:
                f.write(intelligence_result)
        
        except Exception as e:
            print(f"Error in Novel Intelligence Creator: {e}")
    else:
        print("Novel Intelligence Creator not available (import error)")
        print()
    
    print("=" * 80)
    print("STEP 5: AUTONOMOUS GOAL FORMATION")
    print("=" * 80)
    print()
    
    # Run Autonomous Goal Formation
    if AUTONOMOUS_GOALS_AVAILABLE:
        try:
            goals_result = autonomous_goals_run(
                cfg=cfg,
                query="Form autonomous goals for ICEBURG self-improvement",
                context=None,
                verbose=True
            )
            print(goals_result)
            print()
            
            # Save result
            output_dir = Path("data/self_improvement")
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / f"autonomous_goals_{int(datetime.now().timestamp())}.txt", "w") as f:
                f.write(goals_result)
        
        except Exception as e:
            print(f"Error in Autonomous Goal Formation: {e}")
    else:
        print("Autonomous Goal Formation not available (import error)")
        print()
    
    print("=" * 80)
    print("STEP 6: EVOLUTION PIPELINE")
    print("=" * 80)
    print()
    
    # Run Evolution Pipeline
    if EVOLUTION_AVAILABLE:
        try:
            evolution = EvolutionPipeline()
            
            # Create improvement specification
            improvement_spec = {
                "name": "ICEBURG Self-Improvement",
                "description": "Improvements identified by self-improvement agents",
                "priority": "high",
                "estimated_benefit": "Significant performance and capability improvements"
            }
            
            print("Evolution pipeline initialized")
            print("To run evolution, use: evolution.evolve(improvement_spec)")
            print()
        
        except Exception as e:
            print(f"Error in Evolution Pipeline: {e}")
    else:
        print("Evolution Pipeline not available (import error)")
        print()
    
    print("=" * 80)
    print("SELF-IMPROVEMENT ACTIVATION COMPLETE")
    print("=" * 80)
    print()
    print("Results saved to: data/self_improvement/")
    print()
    print("Next steps:")
    print("1. Review self-improvement results")
    print("2. Implement recommended improvements")
    print("3. Run evolution pipeline to deploy improvements")


if __name__ == "__main__":
    main()

