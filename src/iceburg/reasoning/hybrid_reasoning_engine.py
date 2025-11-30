"""
Hybrid Reasoning Engine - COCONUT + ICEBURG Integration
Combines COCONUT-style latent reasoning with ICEBURG's emergence detection and deliberation pauses

This module creates the hybrid reasoning cycle that integrates:
1. COCONUT-style silent reasoning in latent space
2. ICEBURG's existing deliberation pauses for emergence
3. Enhanced dual-layer emergence detection
4. Seamless integration with existing ICEBURG agents

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

from __future__ import annotations

import json
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from ..config import IceburgConfig
from ..agents.deliberation_agent import add_deliberation_pause, detect_emergence
from .coconut_latent_reasoning import COCONUTLatentReasoning, LatentReasoningResult
from ..emergence.quantum_emergence_detector import QuantumEmergenceDetector
from ..emergence.temporal_emergence_detector import TemporalEmergenceDetector
from ..global_workspace import GlobalWorkspace, ThoughtType, ThoughtPriority

logger = logging.getLogger(__name__)

@dataclass
class HybridReasoningStep:
    """Represents a step in the hybrid reasoning process"""
    
    step_id: str
    agent_name: str
    reasoning_type: str  # "coconut", "deliberation", "emergence"
    coconut_result: Optional[LatentReasoningResult] = None
    deliberation_result: Optional[Dict[str, Any]] = None
    emergence_result: Optional[Dict[str, Any]] = None
    duration: float = 0.0
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class HybridReasoningResult:
    """Result of hybrid reasoning process"""
    
    query: str
    reasoning_steps: List[HybridReasoningStep]
    final_emergence_signals: List[Dict[str, Any]]
    total_duration: float
    overall_confidence: float
    coconut_contributions: int
    deliberation_contributions: int
    emergence_contributions: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class HybridReasoningEngine:
    """
    Hybrid Reasoning Engine combining COCONUT and ICEBURG approaches
    
    This engine orchestrates the hybrid reasoning cycle:
    1. Agent processes input (Surveyor, Dissident, etc.)
    2. COCONUT performs silent reasoning on agent output
    3. Deliberation pause allows emergence to unlock
    4. Enhanced emergence detection analyzes both latent and text patterns
    5. Results feed into next agent in the cycle
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        
        # Initialize COCONUT reasoning engine
        self.coconut_engine = COCONUTLatentReasoning(cfg)
        
        # Initialize ICEBURG emergence detectors
        self.quantum_detector = QuantumEmergenceDetector()
        self.temporal_detector = TemporalEmergenceDetector()
        
        # Initialize curiosity engine
        from ..curiosity import CuriosityEngine
        self.curiosity_engine = CuriosityEngine(cfg)
        
        # Initialize dynamic agent factory
        from ..agents.dynamic_agent_factory import DynamicAgentFactory
        self.dynamic_agent_factory = DynamicAgentFactory(cfg)
        
        # Initialize global workspace for coordination
        self.global_workspace = GlobalWorkspace(verbose=getattr(cfg, "verbose", False))
        
        # Hybrid reasoning configuration
        self.enable_coconut = getattr(cfg, "enable_coconut_reasoning", True)
        self.enable_deliberation_pauses = getattr(cfg, "enable_deliberation_pauses", True)
        self.enable_dual_emergence = getattr(cfg, "enable_dual_emergence_detection", True)
        
        # Reasoning state
        self.current_reasoning: Optional[HybridReasoningResult] = None
        self.reasoning_history: List[HybridReasoningResult] = []
        
        logger.info("🚀 Hybrid Reasoning Engine initialized (COCONUT + ICEBURG)")
    
    async def process_agent_output(
        self,
        agent_name: str,
        agent_output: str,
        query: str,
        context: Dict[str, Any],
        reasoning_type: str = "analysis",
        verbose: bool = False
    ) -> HybridReasoningStep:
        """
        Process agent output through hybrid reasoning cycle
        
        Args:
            agent_name: Name of the agent (Surveyor, Dissident, etc.)
            agent_output: Output from the agent
            query: Original query
            context: Context from previous reasoning steps
            reasoning_type: Type of reasoning for COCONUT
            verbose: Enable verbose logging
            
        Returns:
            HybridReasoningStep with all reasoning results
        """
        step_start_time = time.time()
        step_id = f"hybrid_{agent_name}_{int(time.time())}"
        
        if verbose:
            logger.info(f"🔄 Processing {agent_name} output through hybrid reasoning")
        
        try:
            # Step 1: COCONUT Silent Reasoning
            coconut_result = None
            if self.enable_coconut:
                coconut_result = await self._perform_coconut_reasoning(
                    agent_output, context, reasoning_type, verbose
                )
            
            # Step 2: Deliberation Pause (ICEBURG's emergence unlock)
            deliberation_result = None
            if self.enable_deliberation_pauses:
                deliberation_result = await self._perform_deliberation_pause(
                    agent_name, agent_output, query, verbose
                )
            
            # Step 3: Enhanced Emergence Detection
            emergence_result = None
            if self.enable_dual_emergence:
                emergence_result = await self._perform_dual_emergence_detection(
                    agent_output, coconut_result, context, verbose
                )
            
            # Step 4: Broadcast to Global Workspace
            await self._broadcast_to_workspace(
                agent_name, agent_output, coconut_result, emergence_result, verbose
            )
            
            # Calculate overall confidence
            confidence = self._calculate_step_confidence(
                coconut_result, deliberation_result, emergence_result
            )
            
            # Create hybrid reasoning step
            step = HybridReasoningStep(
                step_id=step_id,
                agent_name=agent_name,
                reasoning_type=reasoning_type,
                coconut_result=coconut_result,
                deliberation_result=deliberation_result,
                emergence_result=emergence_result,
                duration=time.time() - step_start_time,
                confidence=confidence,
                metadata={
                    "agent_output_length": len(agent_output),
                    "coconut_enabled": self.enable_coconut,
                    "deliberation_enabled": self.enable_deliberation_pauses,
                    "emergence_enabled": self.enable_dual_emergence
                }
            )
            
            if verbose:
                logger.info(f"✅ Hybrid reasoning completed for {agent_name}: "
                          f"{step.duration:.2f}s, confidence: {confidence:.3f}")
            
            return step
            
        except Exception as e:
            logger.error(f"❌ Hybrid reasoning failed for {agent_name}: {e}")
            raise
    
    async def _perform_coconut_reasoning(
        self,
        agent_output: str,
        context: Dict[str, Any],
        reasoning_type: str,
        verbose: bool = False
    ) -> LatentReasoningResult:
        """Perform COCONUT-style silent reasoning"""
        if verbose:
            logger.info(f"🧠 Performing COCONUT reasoning: {reasoning_type}")
        
        # Prepare context for COCONUT
        coconut_context = {
            "agent_output": agent_output,
            "reasoning_type": reasoning_type,
            **context
        }
        
        # Perform silent reasoning
        result = self.coconut_engine.reason_silently(
            query=agent_output,
            context=coconut_context,
            reasoning_type=reasoning_type,
            verbose=verbose
        )
        
        if verbose:
            logger.info(f"🧠 COCONUT reasoning completed: {result.reasoning_duration:.2f}s, "
                      f"{result.iteration_count} iterations, confidence: {result.confidence_score:.3f}")
        
        return result
    
    async def _perform_deliberation_pause(
        self,
        agent_name: str,
        agent_output: str,
        query: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Perform ICEBURG's deliberation pause for emergence unlock"""
        if verbose:
            logger.info(f"⏸️ Performing deliberation pause for {agent_name}")
        
        # Use ICEBURG's existing deliberation pause system
        deliberation_result = add_deliberation_pause(
            cfg=self.cfg,
            layer_name=agent_name,
            findings=agent_output,
            query=query,
            verbose=verbose
        )
        
        if verbose:
            logger.info(f"⏸️ Deliberation pause completed: {deliberation_result['duration_seconds']}s")
        
        return deliberation_result
    
    async def _perform_dual_emergence_detection(
        self,
        agent_output: str,
        coconut_result: Optional[LatentReasoningResult],
        context: Dict[str, Any],
        verbose: bool = False
    ) -> Dict[str, Any]:
        """Perform dual-layer emergence detection (latent + text)"""
        if verbose:
            logger.info("🌟 Performing dual-layer emergence detection")
        
        emergence_signals = []
        
        # Layer 1: Latent space emergence (from COCONUT)
        if coconut_result and coconut_result.emergence_signals:
            for signal in coconut_result.emergence_signals:
                emergence_signals.append({
                    "source": "latent_space",
                    "type": signal["type"],
                    "confidence": signal["confidence"],
                    "novelty": signal["novelty"],
                    "metadata": signal["metadata"]
                })
        
        # Layer 2: Text-based emergence (ICEBURG's existing system)
        try:
            # Use ICEBURG's existing emergence detection
            text_emergence = detect_emergence(
                cfg=self.cfg,
                all_findings=[{"output": agent_output, "context": context}],
                query=agent_output,
                verbose=verbose
            )
            
            if text_emergence and "emergence_detected" in text_emergence:
                emergence_signals.append({
                    "source": "text_analysis",
                    "type": "text_emergence",
                    "confidence": text_emergence.get("confidence", 0.5),
                    "novelty": text_emergence.get("novelty_score", 0.5),
                    "metadata": text_emergence
                })
        except Exception as e:
            logger.warning(f"Text-based emergence detection failed: {e}")
        
        # Layer 3: Quantum and Temporal emergence
        try:
            # Use the new quantum emergence detector
            quantum_result = self.quantum_detector.detect({
                "content": agent_output,
                "context": context,
                "agent_name": agent_name
            })
            
            if quantum_result.get("emergence_detected", False):
                emergence_signals.append({
                    "source": "quantum_detector",
                    "type": "quantum_emergence",
                    "confidence": quantum_result.get("confidence", 0.0),
                    "quantum_signature": quantum_result.get("quantum_signature"),
                    "coherence_level": quantum_result.get("coherence_level"),
                    "entanglement_indicators": quantum_result.get("entanglement_indicators", []),
                    "metadata": quantum_result.get("metadata", {})
                })
            
            # Use the new temporal emergence detector
            temporal_result = self.temporal_detector.detect({
                "content": agent_output,
                "context": context,
                "agent_name": agent_name
            })
            
            if temporal_result.get("temporal_emergence_detected", False):
                emergence_signals.append({
                    "source": "temporal_detector",
                    "type": "temporal_emergence",
                    "confidence": temporal_result.get("confidence", 0.0),
                    "temporal_signature": temporal_result.get("temporal_signature"),
                    "evolution_stage": temporal_result.get("evolution_stage"),
                    "time_scale": temporal_result.get("time_scale"),
                    "causality_indicators": temporal_result.get("causality_indicators", []),
                    "metadata": temporal_result.get("metadata", {})
                })
        except Exception as e:
            logger.warning(f"Quantum/Temporal emergence detection failed: {e}")
        
        # Store breakthroughs if high-confidence emergence detected
        if emergence_signals:
            try:
                from ..emergence.breakthrough_storage import BreakthroughStorage
                storage = BreakthroughStorage()
                
                # Find highest confidence emergence signal
                best_signal = max(emergence_signals, key=lambda x: x.get("confidence", 0.0))
                
                if best_signal.get("confidence", 0.0) > 0.7:  # High confidence threshold
                    breakthrough = storage.store_breakthrough_from_emergence(
                        emergence_data=best_signal,
                        agent_output=agent_output,
                        source_agent=agent_name
                    )
                    
                    if breakthrough:
                        if verbose:
                            logger.info(f"🎯 Stored breakthrough discovery: {breakthrough.title}")
                        
                        # Trigger dynamic agent creation for breakthrough discoveries
                        try:
                            agent_creation_result = self.dynamic_agent_factory.analyze_emergence_for_agent_creation(
                                emergence_data=best_signal,
                                breakthrough=breakthrough,
                                agent_output=agent_output
                            )
                            
                            if agent_creation_result and agent_creation_result.get("agent_created", False):
                                if verbose:
                                    logger.info(f"🤖 Created dynamic agent: {agent_creation_result.get('agent_name', 'unknown')}")
                        except Exception as e:
                            logger.warning(f"Dynamic agent creation failed: {e}")
            except Exception as e:
                logger.warning(f"Breakthrough storage failed: {e}")
        
        # Generate curiosity-driven queries for high-uncertainty content
        try:
            uncertainty_score = self.curiosity_engine.analyze_uncertainty(agent_output, context)
            novelty_score = self.curiosity_engine.analyze_novelty(agent_output, context)
            
            if uncertainty_score > 0.6 or novelty_score > 0.7:
                curiosity_queries = self.curiosity_engine.generate_curiosity_queries(
                    agent_output, context, max_queries=3
                )
                
                if curiosity_queries:
                    # Store curiosity queries for future exploration
                    self.curiosity_engine.exploration_history.extend(curiosity_queries)
                    
                    if verbose:
                        logger.info(f"🧠 Generated {len(curiosity_queries)} curiosity-driven queries")
                        for query in curiosity_queries[:2]:  # Show first 2
                            logger.info(f"   - {query.query_text}")
        except Exception as e:
            logger.warning(f"Curiosity engine failed: {e}")
        
        # Combine and analyze all emergence signals
        combined_emergence = {
            "emergence_detected": len(emergence_signals) > 0,
            "total_signals": len(emergence_signals),
            "signals": emergence_signals,
            "combined_confidence": self._calculate_combined_emergence_confidence(emergence_signals),
            "emergence_types": list(set(signal["type"] for signal in emergence_signals)),
            "detection_sources": list(set(signal["source"] for signal in emergence_signals))
        }
        
        if verbose:
            logger.info(f"🌟 Dual emergence detection completed: {len(emergence_signals)} signals, "
                      f"confidence: {combined_emergence['combined_confidence']:.3f}")
        
        return combined_emergence
    
    async def _broadcast_to_workspace(
        self,
        agent_name: str,
        agent_output: str,
        coconut_result: Optional[LatentReasoningResult],
        emergence_result: Optional[Dict[str, Any]],
        verbose: bool = False
    ):
        """Broadcast reasoning results to Global Workspace"""
        if verbose:
            logger.info(f"📡 Broadcasting {agent_name} results to Global Workspace")
        
        # Broadcast agent output
        self.global_workspace.broadcast_thought(
            agent=agent_name,
            thought_type=ThoughtType.INSIGHT,
            content=agent_output[:200] + "..." if len(agent_output) > 200 else agent_output,
            priority=ThoughtPriority.NORMAL,
            metadata={
                "full_output": agent_output,
                "agent": agent_name,
                "timestamp": time.time()
            }
        )
        
        # Broadcast COCONUT results if available
        if coconut_result:
            self.global_workspace.broadcast_thought(
                agent="COCONUT",
                thought_type=ThoughtType.HYPOTHESIS,
                content=f"Latent reasoning: {coconut_result.confidence_score:.3f} confidence, "
                       f"{coconut_result.iteration_count} iterations",
                priority=ThoughtPriority.IMPORTANT if coconut_result.confidence_score > 0.8 else ThoughtPriority.NORMAL,
                metadata={
                    "coconut_result": coconut_result,
                    "reasoning_type": coconut_result.metadata.get("reasoning_type"),
                    "timestamp": time.time()
                }
            )
        
        # Broadcast emergence signals if available
        if emergence_result and emergence_result.get("emergence_detected"):
            self.global_workspace.broadcast_thought(
                agent="EmergenceDetector",
                thought_type=ThoughtType.INSIGHT,
                content=f"Emergence detected: {emergence_result['total_signals']} signals, "
                       f"{emergence_result['combined_confidence']:.3f} confidence",
                priority=ThoughtPriority.CRITICAL,
                metadata={
                    "emergence_result": emergence_result,
                    "timestamp": time.time()
                }
            )
    
    def _calculate_step_confidence(
        self,
        coconut_result: Optional[LatentReasoningResult],
        deliberation_result: Optional[Dict[str, Any]],
        emergence_result: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate overall confidence for hybrid reasoning step"""
        confidences = []
        
        # COCONUT confidence
        if coconut_result:
            confidences.append(coconut_result.confidence_score)
        
        # Deliberation confidence (based on pause completion)
        if deliberation_result:
            deliberation_confidence = 0.8 if deliberation_result.get("status") == "paused" else 0.5
            confidences.append(deliberation_confidence)
        
        # Emergence confidence
        if emergence_result:
            confidences.append(emergence_result.get("combined_confidence", 0.5))
        
        # Return average confidence
        return sum(confidences) / len(confidences) if confidences else 0.5
    
    def _calculate_combined_emergence_confidence(self, emergence_signals: List[Dict[str, Any]]) -> float:
        """Calculate combined confidence from all emergence signals"""
        if not emergence_signals:
            return 0.0
        
        confidences = [signal.get("confidence", 0.0) for signal in emergence_signals]
        
        # Weight by signal type
        weighted_confidences = []
        for signal in emergence_signals:
            confidence = signal.get("confidence", 0.0)
            signal_type = signal.get("type", "")
            
            # Weight different signal types
            if "latent" in signal_type:
                weight = 1.2  # Latent signals are more reliable
            elif "quantum" in signal_type:
                weight = 1.1  # Quantum signals are interesting
            elif "temporal" in signal_type:
                weight = 1.0  # Temporal signals are standard
            else:
                weight = 0.9  # Default weight
            
            weighted_confidences.append(confidence * weight)
        
        return sum(weighted_confidences) / len(weighted_confidences)
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of hybrid reasoning state"""
        if not self.current_reasoning:
            return {"status": "no_reasoning"}
        
        return {
            "status": "active",
            "hybrid_reasoning": {
                "total_steps": len(self.current_reasoning.reasoning_steps),
                "total_duration": self.current_reasoning.total_duration,
                "overall_confidence": self.current_reasoning.overall_confidence,
                "coconut_contributions": self.current_reasoning.coconut_contributions,
                "deliberation_contributions": self.current_reasoning.deliberation_contributions,
                "emergence_contributions": self.current_reasoning.emergence_contributions
            },
            "coconut_engine": self.coconut_engine.get_reasoning_summary(),
            "global_workspace": {
                "active_thoughts": len(self.global_workspace.active_thoughts),
                "consciousness_level": self.global_workspace.consciousness_level,
                "emergence_events": self.global_workspace.emergence_events
            }
        }
    
    def clear_reasoning_state(self):
        """Clear current reasoning state"""
        self.current_reasoning = None
        self.coconut_engine.clear_emergence_signals()
        logger.info("🧹 Hybrid reasoning state cleared")
