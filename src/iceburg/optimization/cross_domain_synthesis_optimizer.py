"""
Cross-Domain Synthesis Optimization Engine
Optimizes cross-domain knowledge synthesis for speed and accuracy

© 2025 Praxis Research & Engineering Inc. All rights reserved.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from collections import defaultdict

from ..config import IceburgConfig

logger = logging.getLogger(__name__)

@dataclass
class SynthesisPattern:
    """Represents a successful synthesis pattern"""
    pattern_id: str
    domain_combination: Tuple[str, str]
    synthesis_type: str  # "conceptual", "methodological", "theoretical", "practical"
    success_rate: float
    avg_processing_time: float
    quality_score: float
    usage_count: int
    last_used: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DomainConnection:
    """Represents a connection between two domains"""
    domain_a: str
    domain_b: str
    connection_strength: float
    connection_type: str  # "direct", "indirect", "conceptual", "methodological"
    evidence_count: int
    last_updated: float = field(default_factory=time.time)

@dataclass
class SynthesisOptimization:
    """Optimization recommendation for synthesis"""
    optimization_id: str
    optimization_type: str  # "pattern_caching", "domain_mapping", "template_matching", "parallel_synthesis"
    expected_improvement: float
    implementation_effort: str  # "low", "medium", "high"
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class CrossDomainSynthesisOptimizer:
    """
    Optimizes cross-domain synthesis for speed and accuracy
    """
    
    def __init__(self, cfg: IceburgConfig):
        self.cfg = cfg
        self.data_dir = Path("data/optimization/cross_domain_synthesis")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Storage files
        self.patterns_file = self.data_dir / "synthesis_patterns.json"
        self.connections_file = self.data_dir / "domain_connections.json"
        self.optimizations_file = self.data_dir / "synthesis_optimizations.json"
        
        # Data structures
        self.synthesis_patterns: Dict[str, SynthesisPattern] = {}
        self.domain_connections: Dict[str, DomainConnection] = {}
        self.synthesis_optimizations: List[SynthesisOptimization] = []
        
        # Performance tracking
        self.synthesis_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Load existing data
        self._load_data()
        
        logger.info("🔗 Cross-Domain Synthesis Optimizer initialized")
    
    def record_synthesis_attempt(
        self,
        domains: List[str],
        synthesis_type: str,
        processing_time: float,
        quality_score: float,
        success: bool,
        metadata: Dict[str, Any] = None
    ) -> None:
        """Record a synthesis attempt for optimization analysis"""
        if metadata is None:
            metadata = {}
        
        record = {
            "timestamp": time.time(),
            "domains": domains,
            "synthesis_type": synthesis_type,
            "processing_time": processing_time,
            "quality_score": quality_score,
            "success": success,
            "metadata": metadata
        }
        
        self.synthesis_history.append(record)
        
        # Keep only last 1000 records
        if len(self.synthesis_history) > 1000:
            self.synthesis_history = self.synthesis_history[-1000:]
        
        # Update performance metrics
        self.performance_metrics["processing_time"].append(processing_time)
        self.performance_metrics["quality_score"].append(quality_score)
        self.performance_metrics["success_rate"].append(1.0 if success else 0.0)
        
        # Analyze for patterns
        self._analyze_synthesis_patterns(domains, synthesis_type, processing_time, quality_score, success)
        
        # Update domain connections
        self._update_domain_connections(domains, success)
        
        # Save data
        self._save_data()
        
        logger.info(f"📝 Recorded synthesis attempt: {domains} -> {synthesis_type} ({'success' if success else 'failed'})")
    
    def _analyze_synthesis_patterns(
        self,
        domains: List[str],
        synthesis_type: str,
        processing_time: float,
        quality_score: float,
        success: bool
    ) -> None:
        """Analyze synthesis patterns for optimization opportunities"""
        if len(domains) < 2:
            return
        
        # Create domain combination key
        domain_key = tuple(sorted(domains))
        pattern_id = f"{'_'.join(domain_key)}_{synthesis_type}"
        
        # Update or create pattern
        if pattern_id in self.synthesis_patterns:
            pattern = self.synthesis_patterns[pattern_id]
            pattern.usage_count += 1
            pattern.last_used = time.time()
            
            # Update success rate (exponential moving average)
            alpha = 0.1
            pattern.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * pattern.success_rate
            
            # Update processing time (exponential moving average)
            pattern.avg_processing_time = alpha * processing_time + (1 - alpha) * pattern.avg_processing_time
            
            # Update quality score (exponential moving average)
            pattern.quality_score = alpha * quality_score + (1 - alpha) * pattern.quality_score
            
        else:
            # Create new pattern
            pattern = SynthesisPattern(
                pattern_id=pattern_id,
                domain_combination=domain_key,
                synthesis_type=synthesis_type,
                success_rate=1.0 if success else 0.0,
                avg_processing_time=processing_time,
                quality_score=quality_score,
                usage_count=1,
                metadata={"created_from": "synthesis_analysis"}
            )
            
            self.synthesis_patterns[pattern_id] = pattern
    
    def _update_domain_connections(self, domains: List[str], success: bool) -> None:
        """Update domain connection strengths based on synthesis success"""
        if len(domains) < 2:
            return
        
        # Update all pairwise connections
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain_a, domain_b = domains[i], domains[j]
                connection_key = f"{domain_a}_{domain_b}" if domain_a < domain_b else f"{domain_b}_{domain_a}"
                
                if connection_key in self.domain_connections:
                    connection = self.domain_connections[connection_key]
                    connection.evidence_count += 1
                    connection.last_updated = time.time()
                    
                    # Update connection strength (exponential moving average)
                    alpha = 0.1
                    success_value = 1.0 if success else 0.0
                    connection.connection_strength = alpha * success_value + (1 - alpha) * connection.connection_strength
                    
                else:
                    # Create new connection
                    connection = DomainConnection(
                        domain_a=domain_a,
                        domain_b=domain_b,
                        connection_strength=1.0 if success else 0.0,
                        connection_type="direct",
                        evidence_count=1
                    )
                    
                    self.domain_connections[connection_key] = connection
    
    def get_optimization_recommendations(self) -> List[SynthesisOptimization]:
        """Get optimization recommendations based on analysis"""
        recommendations = []
        
        # Pattern caching optimization
        high_success_patterns = [
            pattern for pattern in self.synthesis_patterns.values()
            if pattern.success_rate > 0.8 and pattern.usage_count > 5
        ]
        
        if high_success_patterns:
            recommendations.append(SynthesisOptimization(
                optimization_id="pattern_caching_001",
                optimization_type="pattern_caching",
                expected_improvement=0.3,  # 30% speed improvement
                implementation_effort="low",
                description=f"Cache {len(high_success_patterns)} high-success synthesis patterns for instant reuse",
                metadata={"patterns_count": len(high_success_patterns)}
            ))
        
        # Domain mapping optimization
        strong_connections = [
            conn for conn in self.domain_connections.values()
            if conn.connection_strength > 0.7 and conn.evidence_count > 3
        ]
        
        if strong_connections:
            recommendations.append(SynthesisOptimization(
                optimization_id="domain_mapping_001",
                optimization_type="domain_mapping",
                expected_improvement=0.2,  # 20% accuracy improvement
                implementation_effort="medium",
                description=f"Pre-compute {len(strong_connections)} strong domain connections for faster synthesis",
                metadata={"connections_count": len(strong_connections)}
            ))
        
        # Template matching optimization
        frequent_combinations = [
            pattern for pattern in self.synthesis_patterns.values()
            if pattern.usage_count > 10
        ]
        
        if frequent_combinations:
            recommendations.append(SynthesisOptimization(
                optimization_id="template_matching_001",
                optimization_type="template_matching",
                expected_improvement=0.25,  # 25% speed improvement
                implementation_effort="medium",
                description=f"Create templates for {len(frequent_combinations)} frequently used domain combinations",
                metadata={"templates_count": len(frequent_combinations)}
            ))
        
        # Parallel synthesis optimization
        if len(self.synthesis_history) > 50:
            avg_processing_time = np.mean(self.performance_metrics["processing_time"])
            if avg_processing_time > 2.0:  # More than 2 seconds
                recommendations.append(SynthesisOptimization(
                    optimization_id="parallel_synthesis_001",
                    optimization_type="parallel_synthesis",
                    expected_improvement=0.4,  # 40% speed improvement
                    implementation_effort="high",
                    description="Implement parallel synthesis for complex multi-domain queries",
                    metadata={"avg_processing_time": avg_processing_time}
                ))
        
        self.synthesis_optimizations = recommendations
        self._save_data()
        
        logger.info(f"🎯 Generated {len(recommendations)} optimization recommendations")
        
        return recommendations
    
    def get_synthesis_strategy(
        self,
        domains: List[str],
        synthesis_type: str = "conceptual"
    ) -> Dict[str, Any]:
        """Get optimized synthesis strategy for given domains"""
        if len(domains) < 2:
            return {"strategy": "single_domain", "optimization": "none"}
        
        # Check for existing patterns
        domain_key = tuple(sorted(domains))
        pattern_id = f"{'_'.join(domain_key)}_{synthesis_type}"
        
        if pattern_id in self.synthesis_patterns:
            pattern = self.synthesis_patterns[pattern_id]
            
            if pattern.success_rate > 0.8:
                return {
                    "strategy": "cached_pattern",
                    "pattern_id": pattern_id,
                    "expected_success_rate": pattern.success_rate,
                    "expected_processing_time": pattern.avg_processing_time,
                    "optimization": "pattern_reuse"
                }
        
        # Check for strong domain connections
        strong_connections = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain_a, domain_b = domains[i], domains[j]
                connection_key = f"{domain_a}_{domain_b}" if domain_a < domain_b else f"{domain_b}_{domain_a}"
                
                if connection_key in self.domain_connections:
                    connection = self.domain_connections[connection_key]
                    if connection.connection_strength > 0.6:
                        strong_connections.append(connection)
        
        if strong_connections:
            return {
                "strategy": "connection_based",
                "strong_connections": len(strong_connections),
                "avg_connection_strength": np.mean([c.connection_strength for c in strong_connections]),
                "optimization": "domain_mapping"
            }
        
        # Default strategy
        return {
            "strategy": "exploratory",
            "optimization": "none",
            "recommendation": "Collect more data for optimization"
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary and metrics"""
        if not self.performance_metrics["processing_time"]:
            return {"status": "no_data"}
        
        return {
            "total_synthesis_attempts": len(self.synthesis_history),
            "avg_processing_time": np.mean(self.performance_metrics["processing_time"]),
            "avg_quality_score": np.mean(self.performance_metrics["quality_score"]),
            "overall_success_rate": np.mean(self.performance_metrics["success_rate"]),
            "patterns_discovered": len(self.synthesis_patterns),
            "domain_connections": len(self.domain_connections),
            "optimization_recommendations": len(self.synthesis_optimizations),
            "recent_performance": {
                "last_10_avg_time": np.mean(self.performance_metrics["processing_time"][-10:]),
                "last_10_avg_quality": np.mean(self.performance_metrics["quality_score"][-10:]),
                "last_10_success_rate": np.mean(self.performance_metrics["success_rate"][-10:])
            }
        }
    
    def _load_data(self) -> None:
        """Load data from storage files"""
        try:
            # Load synthesis patterns
            if self.patterns_file.exists():
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                    self.synthesis_patterns = {
                        pattern_id: SynthesisPattern(**pattern_data)
                        for pattern_id, pattern_data in data.items()
                    }
            
            # Load domain connections
            if self.connections_file.exists():
                with open(self.connections_file, 'r') as f:
                    data = json.load(f)
                    self.domain_connections = {
                        conn_id: DomainConnection(**conn_data)
                        for conn_id, conn_data in data.items()
                    }
            
            # Load optimizations
            if self.optimizations_file.exists():
                with open(self.optimizations_file, 'r') as f:
                    data = json.load(f)
                    self.synthesis_optimizations = [
                        SynthesisOptimization(**opt_data)
                        for opt_data in data
                    ]
            
            logger.info(f"📁 Loaded synthesis data: {len(self.synthesis_patterns)} patterns, {len(self.domain_connections)} connections")
            
        except Exception as e:
            logger.warning(f"Failed to load synthesis data: {e}")
    
    def _save_data(self) -> None:
        """Save data to storage files"""
        try:
            # Save synthesis patterns
            patterns_data = {
                pattern_id: {
                    "pattern_id": pattern.pattern_id,
                    "domain_combination": pattern.domain_combination,
                    "synthesis_type": pattern.synthesis_type,
                    "success_rate": pattern.success_rate,
                    "avg_processing_time": pattern.avg_processing_time,
                    "quality_score": pattern.quality_score,
                    "usage_count": pattern.usage_count,
                    "last_used": pattern.last_used,
                    "metadata": pattern.metadata
                }
                for pattern_id, pattern in self.synthesis_patterns.items()
            }
            
            with open(self.patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save domain connections
            connections_data = {
                conn_id: {
                    "domain_a": conn.domain_a,
                    "domain_b": conn.domain_b,
                    "connection_strength": conn.connection_strength,
                    "connection_type": conn.connection_type,
                    "evidence_count": conn.evidence_count,
                    "last_updated": conn.last_updated
                }
                for conn_id, conn in self.domain_connections.items()
            }
            
            with open(self.connections_file, 'w') as f:
                json.dump(connections_data, f, indent=2)
            
            # Save optimizations
            optimizations_data = [
                {
                    "optimization_id": opt.optimization_id,
                    "optimization_type": opt.optimization_type,
                    "expected_improvement": opt.expected_improvement,
                    "implementation_effort": opt.implementation_effort,
                    "description": opt.description,
                    "metadata": opt.metadata
                }
                for opt in self.synthesis_optimizations
            ]
            
            with open(self.optimizations_file, 'w') as f:
                json.dump(optimizations_data, f, indent=2)
            
            logger.debug("💾 Saved synthesis data to storage")
            
        except Exception as e:
            logger.error(f"Failed to save synthesis data: {e}")


# Helper functions for integration
def create_synthesis_optimizer(cfg: IceburgConfig) -> CrossDomainSynthesisOptimizer:
    """Create cross-domain synthesis optimizer instance"""
    return CrossDomainSynthesisOptimizer(cfg)

def record_synthesis_attempt(
    optimizer: CrossDomainSynthesisOptimizer,
    domains: List[str],
    synthesis_type: str,
    processing_time: float,
    quality_score: float,
    success: bool,
    metadata: Dict[str, Any] = None
) -> None:
    """Record synthesis attempt for optimization"""
    optimizer.record_synthesis_attempt(domains, synthesis_type, processing_time, quality_score, success, metadata)

def get_synthesis_strategy(
    optimizer: CrossDomainSynthesisOptimizer,
    domains: List[str],
    synthesis_type: str = "conceptual"
) -> Dict[str, Any]:
    """Get optimized synthesis strategy"""
    return optimizer.get_synthesis_strategy(domains, synthesis_type)
