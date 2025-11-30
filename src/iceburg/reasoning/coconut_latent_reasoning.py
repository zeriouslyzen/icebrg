"""
COCONUT-Style Latent Reasoning Engine for ICEBURG
Implements Chain of Continuous Thought (COCONUT) paradigm for silent reasoning

This module integrates COCONUT-style latent reasoning with ICEBURG's existing
emergence detection and deliberation pause system.

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

from __future__ import annotations

import json
import time
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from ..config import IceburgConfig
from ..llm import embed_texts, chat_complete
from ..latent_space_controller import LatentSpaceController

logger = logging.getLogger(__name__)

@dataclass
class LatentReasoningStep:
    """Represents a single step in latent space reasoning"""
    
    step_id: str
    iteration: int
    input_vector: List[float]
    hidden_state: List[float]
    attention_weights: Dict[str, float]
    reasoning_type: str  # "analysis", "synthesis", "validation", "emergence"
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class LatentReasoningResult:
    """Result of latent space reasoning process"""
    
    final_hidden_state: List[float]
    reasoning_steps: List[LatentReasoningStep]
    emergence_signals: List[Dict[str, Any]]
    confidence_score: float
    reasoning_duration: float
    iteration_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class COCONUTLatentReasoning:
    """
    COCONUT-Style Latent Reasoning Engine
    
    Implements Chain of Continuous Thought (COCONUT) paradigm:
    - Silent reasoning in continuous latent space
    - Internal iterations with hidden state feedback
    - Integration with ICEBURG's emergence detection
    - Compatible with existing deliberation pauses
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.latent_controller = LatentSpaceController()
        
        # COCONUT configuration
        self.max_iterations = int(getattr(cfg, "coconut_max_iterations", 5))
        self.convergence_threshold = float(getattr(cfg, "coconut_convergence_threshold", 0.95))
        self.emergence_threshold = float(getattr(cfg, "coconut_emergence_threshold", 0.8))
        
        # Reasoning state
        self.current_reasoning: Optional[LatentReasoningResult] = None
        self.reasoning_history: List[LatentReasoningResult] = []
        
        # Integration with ICEBURG systems
        self.emergence_signals: List[Dict[str, Any]] = []
        self.deliberation_context: Dict[str, Any] = {}
        
        logger.info("🧠 COCONUT Latent Reasoning Engine initialized")
    
    def reason_silently(
        self, 
        query: str, 
        context: Dict[str, Any],
        reasoning_type: str = "analysis",
        verbose: bool = False
    ) -> LatentReasoningResult:
        """
        Perform silent reasoning in latent space using COCONUT paradigm
        
        Args:
            query: Input query for reasoning
            context: Context from ICEBURG agents
            reasoning_type: Type of reasoning (analysis, synthesis, validation, emergence)
            verbose: Enable verbose logging
            
        Returns:
            LatentReasoningResult with reasoning steps and emergence signals
        """
        start_time = time.time()
        
        if verbose:
            logger.info(f"🧠 Starting COCONUT latent reasoning for: {reasoning_type}")
        
        try:
            # Initialize reasoning state
            initial_vector = self._initialize_reasoning_vector(query, context)
            reasoning_steps = []
            current_hidden_state = initial_vector.copy()
            
            # COCONUT reasoning iterations
            for iteration in range(self.max_iterations):
                if verbose:
                    logger.info(f"🔄 COCONUT iteration {iteration + 1}/{self.max_iterations}")
                
                # Perform single reasoning step
                step_result = self._perform_reasoning_step(
                    current_hidden_state, 
                    context, 
                    reasoning_type, 
                    iteration,
                    verbose
                )
                
                reasoning_steps.append(step_result)
                current_hidden_state = step_result.hidden_state
                
                # Check for convergence
                if self._check_convergence(reasoning_steps):
                    if verbose:
                        logger.info(f"✅ COCONUT converged after {iteration + 1} iterations")
                    break
                
                # Check for emergence signals
                emergence_signal = self._detect_emergence_signal(step_result, context)
                if emergence_signal:
                    self.emergence_signals.append(emergence_signal)
                    if verbose:
                        logger.info(f"🌟 Emergence signal detected: {emergence_signal['type']}")
            
            # Create final result
            reasoning_duration = time.time() - start_time
            confidence_score = self._calculate_confidence_score(reasoning_steps)
            
            result = LatentReasoningResult(
                final_hidden_state=current_hidden_state,
                reasoning_steps=reasoning_steps,
                emergence_signals=self.emergence_signals.copy(),
                confidence_score=confidence_score,
                reasoning_duration=reasoning_duration,
                iteration_count=len(reasoning_steps),
                metadata={
                    "reasoning_type": reasoning_type,
                    "query": query,
                    "convergence_achieved": len(reasoning_steps) < self.max_iterations
                }
            )
            
            # Store in history
            self.current_reasoning = result
            self.reasoning_history.append(result)
            
            if verbose:
                logger.info(f"🎯 COCONUT reasoning completed: {reasoning_duration:.2f}s, "
                          f"{len(reasoning_steps)} steps, confidence: {confidence_score:.3f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ COCONUT reasoning failed: {e}")
            raise
    
    def _initialize_reasoning_vector(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> List[float]:
        """Initialize reasoning vector from query and context"""
        try:
            # Use ICEBURG's existing embedding system
            query_embedding = embed_texts(self.cfg.embed_model, [query])[0]
            
            # Enhance with context if available
            if context and "agent_outputs" in context:
                context_text = " ".join(str(output) for output in context["agent_outputs"].values())
                context_embedding = embed_texts(self.cfg.embed_model, [context_text])[0]
                
                # Combine query and context embeddings
                combined_vector = np.array(query_embedding) + 0.3 * np.array(context_embedding)
                return combined_vector.tolist()
            
            return query_embedding
            
        except Exception as e:
            logger.warning(f"Failed to initialize reasoning vector: {e}")
            # Fallback to random vector
            return np.random.normal(0, 1, 1536).tolist()
    
    def _perform_reasoning_step(
        self,
        input_vector: List[float],
        context: Dict[str, Any],
        reasoning_type: str,
        iteration: int,
        verbose: bool = False
    ) -> LatentReasoningStep:
        """Perform a single reasoning step in latent space"""
        
        # Get attention weights from latent controller
        attention_weights = self.latent_controller.optimize_attention(
            [input_vector], 
            ["engineering", "science", "factual"]
        )
        
        # Apply reasoning transformation based on type
        if reasoning_type == "analysis":
            hidden_state = self._apply_analysis_transformation(input_vector, attention_weights)
        elif reasoning_type == "synthesis":
            hidden_state = self._apply_synthesis_transformation(input_vector, attention_weights)
        elif reasoning_type == "validation":
            hidden_state = self._apply_validation_transformation(input_vector, attention_weights)
        elif reasoning_type == "emergence":
            hidden_state = self._apply_emergence_transformation(input_vector, attention_weights)
        else:
            hidden_state = self._apply_default_transformation(input_vector, attention_weights)
        
        # Calculate confidence based on attention weights and transformation
        confidence = self._calculate_step_confidence(hidden_state, attention_weights)
        
        return LatentReasoningStep(
            step_id=f"coconut_step_{iteration}_{int(time.time())}",
            iteration=iteration,
            input_vector=input_vector,
            hidden_state=hidden_state,
            attention_weights=attention_weights,
            reasoning_type=reasoning_type,
            confidence=confidence,
            metadata={
                "transformation_type": reasoning_type,
                "attention_sum": sum(attention_weights.values()),
                "vector_magnitude": np.linalg.norm(hidden_state)
            }
        )
    
    def _apply_analysis_transformation(
        self, 
        input_vector: List[float], 
        attention_weights: Dict[str, float]
    ) -> List[float]:
        """Apply analysis transformation to input vector"""
        vector = np.array(input_vector)
        
        # Apply attention-weighted transformation
        engineering_weight = attention_weights.get("engineering", 0.5)
        factual_weight = attention_weights.get("factual", 0.5)
        
        # Create transformation vector that matches input vector length
        vector_length = len(vector)
        analysis_transform = np.ones(vector_length)
        
        # Apply weights to different sections of the vector
        quarter = vector_length // 4
        analysis_transform[:quarter] *= engineering_weight * 0.8  # Technical aspects
        analysis_transform[quarter:2*quarter] *= factual_weight * 1.2  # Factual analysis
        analysis_transform[2*quarter:3*quarter] *= 0.3  # Reduce abstraction
        analysis_transform[3*quarter:] *= 0.7  # Maintain logical structure
        
        # Apply transformation
        transformed = vector * analysis_transform
        
        return transformed.tolist()
    
    def _apply_synthesis_transformation(
        self, 
        input_vector: List[float], 
        attention_weights: Dict[str, float]
    ) -> List[float]:
        """Apply synthesis transformation to input vector"""
        vector = np.array(input_vector)
        
        # Create transformation vector that matches input vector length
        vector_length = len(vector)
        synthesis_transform = np.ones(vector_length)
        
        # Apply weights to different sections of the vector
        quarter = vector_length // 4
        synthesis_transform[:quarter] *= 0.6  # Moderate technical focus
        synthesis_transform[quarter:2*quarter] *= 0.8  # Strong factual basis
        synthesis_transform[2*quarter:3*quarter] *= 0.9  # Allow some abstraction for integration
        synthesis_transform[3*quarter:] *= 1.1  # Enhance logical connections
        
        transformed = vector * synthesis_transform
        
        return transformed.tolist()
    
    def _apply_validation_transformation(
        self, 
        input_vector: List[float], 
        attention_weights: Dict[str, float]
    ) -> List[float]:
        """Apply validation transformation to input vector"""
        vector = np.array(input_vector)
        
        # Create transformation vector that matches input vector length
        vector_length = len(vector)
        validation_transform = np.ones(vector_length)
        
        # Apply weights to different sections of the vector
        quarter = vector_length // 4
        validation_transform[:quarter] *= 1.0  # Full technical focus
        validation_transform[quarter:2*quarter] *= 1.2  # Strong factual emphasis
        validation_transform[2*quarter:3*quarter] *= 0.2  # Minimal abstraction
        validation_transform[3*quarter:] *= 0.9  # Maintain logical structure
        
        transformed = vector * validation_transform
        
        return transformed.tolist()
    
    def _apply_emergence_transformation(
        self, 
        input_vector: List[float], 
        attention_weights: Dict[str, float]
    ) -> List[float]:
        """Apply emergence transformation to input vector"""
        vector = np.array(input_vector)
        
        # Create transformation vector that matches input vector length
        vector_length = len(vector)
        emergence_transform = np.ones(vector_length)
        
        # Apply weights to different sections of the vector
        quarter = vector_length // 4
        emergence_transform[:quarter] *= 0.7  # Moderate technical focus
        emergence_transform[quarter:2*quarter] *= 0.6  # Moderate factual basis
        emergence_transform[2*quarter:3*quarter] *= 1.3  # Allow abstraction for novel patterns
        emergence_transform[3*quarter:] *= 1.2  # Enhance creative connections
        
        transformed = vector * emergence_transform
        
        return transformed.tolist()
    
    def _apply_default_transformation(
        self, 
        input_vector: List[float], 
        attention_weights: Dict[str, float]
    ) -> List[float]:
        """Apply default transformation to input vector"""
        vector = np.array(input_vector)
        
        # Create transformation vector that matches input vector length
        vector_length = len(vector)
        default_transform = np.ones(vector_length) * 0.8  # Default balanced transformation
        
        transformed = vector * default_transform
        
        return transformed.tolist()
    
    def _calculate_step_confidence(
        self, 
        hidden_state: List[float], 
        attention_weights: Dict[str, float]
    ) -> float:
        """Calculate confidence score for reasoning step"""
        try:
            # Base confidence from attention weights
            attention_confidence = sum(attention_weights.values()) / len(attention_weights)
            
            # Vector stability confidence
            vector_magnitude = np.linalg.norm(hidden_state)
            stability_confidence = min(1.0, vector_magnitude / 10.0)  # Normalize
            
            # Combined confidence
            confidence = (attention_confidence * 0.6) + (stability_confidence * 0.4)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception:
            return 0.5  # Default confidence
    
    def _check_convergence(self, reasoning_steps: List[LatentReasoningStep]) -> bool:
        """Check if reasoning has converged"""
        if len(reasoning_steps) < 2:
            return False
        
        # Check if last two steps are similar
        last_step = reasoning_steps[-1]
        prev_step = reasoning_steps[-2]
        
        # Calculate similarity between hidden states
        similarity = np.corrcoef(last_step.hidden_state, prev_step.hidden_state)[0, 1]
        
        return similarity > self.convergence_threshold
    
    def _detect_emergence_signal(
        self, 
        step: LatentReasoningStep, 
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Detect emergence signals in reasoning step"""
        try:
            # Check for high confidence with novel patterns
            if step.confidence > self.emergence_threshold:
                # Check for novel vector patterns
                vector_novelty = self._calculate_vector_novelty(step.hidden_state)
                
                if vector_novelty > 0.7:
                    return {
                        "type": "latent_emergence",
                        "confidence": step.confidence,
                        "novelty": vector_novelty,
                        "reasoning_type": step.reasoning_type,
                        "timestamp": step.timestamp,
                        "metadata": {
                            "step_id": step.step_id,
                            "iteration": step.iteration,
                            "attention_weights": step.attention_weights
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to detect emergence signal: {e}")
            return None
    
    def _calculate_vector_novelty(self, vector: List[float]) -> float:
        """Calculate novelty score for vector"""
        try:
            # Compare with historical vectors
            if not self.reasoning_history:
                return 1.0  # First vector is maximally novel
            
            # Calculate similarity with previous vectors
            similarities = []
            for result in self.reasoning_history[-5:]:  # Check last 5 results
                for step in result.reasoning_steps:
                    similarity = np.corrcoef(vector, step.hidden_state)[0, 1]
                    similarities.append(similarity)
            
            if similarities:
                avg_similarity = np.mean(similarities)
                novelty = 1.0 - avg_similarity
                return max(0.0, min(1.0, novelty))
            
            return 1.0
            
        except Exception:
            return 0.5  # Default novelty
    
    def _calculate_confidence_score(self, reasoning_steps: List[LatentReasoningStep]) -> float:
        """Calculate overall confidence score for reasoning result"""
        if not reasoning_steps:
            return 0.0
        
        # Average confidence across steps
        step_confidences = [step.confidence for step in reasoning_steps]
        avg_confidence = np.mean(step_confidences)
        
        # Bonus for convergence
        convergence_bonus = 0.1 if len(reasoning_steps) < self.max_iterations else 0.0
        
        # Bonus for emergence signals
        emergence_bonus = 0.1 if self.emergence_signals else 0.0
        
        final_confidence = avg_confidence + convergence_bonus + emergence_bonus
        
        return min(1.0, max(0.0, final_confidence))
    
    def get_emergence_signals(self) -> List[Dict[str, Any]]:
        """Get current emergence signals"""
        return self.emergence_signals.copy()
    
    def clear_emergence_signals(self):
        """Clear current emergence signals"""
        self.emergence_signals.clear()
    
    def get_reasoning_summary(self) -> Dict[str, Any]:
        """Get summary of current reasoning state"""
        if not self.current_reasoning:
            return {"status": "no_reasoning"}
        
        return {
            "status": "active",
            "current_reasoning": {
                "type": self.current_reasoning.metadata.get("reasoning_type"),
                "confidence": self.current_reasoning.confidence_score,
                "duration": self.current_reasoning.reasoning_duration,
                "iterations": self.current_reasoning.iteration_count,
                "emergence_signals": len(self.current_reasoning.emergence_signals)
            },
            "history": {
                "total_reasoning_sessions": len(self.reasoning_history),
                "total_emergence_signals": sum(len(r.emergence_signals) for r in self.reasoning_history)
            }
        }
