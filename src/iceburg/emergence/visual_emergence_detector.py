"""
Visual Emergence Detector
Detects emergent patterns in generated visual UIs
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from pathlib import Path


@dataclass
class EmergenceSignal:
    """Represents an emergence signal"""
    type: str
    confidence: float
    description: str
    metadata: Dict[str, Any]
    timestamp: str


class VisualEmergenceDetector:
    """Detects emergence in visual generation"""
    
    def __init__(self):
        self.pattern_history: List[Dict[str, Any]] = []
        self.novelty_threshold = 0.7
        self.optimization_threshold = 0.3
    
    def detect_emergence(
        self,
        visual_result: Any  # VisualGenerationResult
    ) -> List[EmergenceSignal]:
        """
        Detect emergence signals in visual generation result
        
        Args:
            visual_result: VisualGenerationResult from VisualArchitect
            
        Returns:
            List of emergence signals
        """
        signals = []
        
        # Check for novel visual patterns
        if self._is_novel_visual_pattern(visual_result):
            novelty_score = self._calculate_novelty_score(visual_result)
            signals.append(EmergenceSignal(
                type="novel_visual_pattern",
                confidence=novelty_score,
                description=f"Generated novel UI pattern with {novelty_score:.2%} novelty",
                metadata={
                    "component_types": self._get_component_types(visual_result),
                    "layout_type": self._get_layout_type(visual_result)
                },
                timestamp=datetime.now().isoformat()
            ))
        
        # Check for cross-domain visual synthesis
        if self._is_cross_domain_synthesis(visual_result):
            signals.append(EmergenceSignal(
                type="cross_domain_visual_synthesis",
                confidence=0.8,
                description="Combined visual patterns from multiple design paradigms",
                metadata={
                    "backends": visual_result.metadata.get("backends_compiled", [])
                },
                timestamp=datetime.now().isoformat()
            ))
        
        # Check for optimization breakthroughs
        optimization_improvement = self._calculate_optimization_improvement(visual_result)
        if optimization_improvement > self.optimization_threshold:
            signals.append(EmergenceSignal(
                type="optimization_breakthrough",
                confidence=optimization_improvement,
                description=f"Achieved {optimization_improvement*100:.1f}% optimization improvement",
                metadata={
                    "rules_applied": visual_result.optimization.applied_rules,
                    "improvements": visual_result.optimization.improvements
                },
                timestamp=datetime.now().isoformat()
            ))
        
        # Check for accessibility innovations
        if self._has_accessibility_innovations(visual_result):
            signals.append(EmergenceSignal(
                type="accessibility_innovation",
                confidence=0.75,
                description="Generated UI with enhanced accessibility features",
                metadata={
                    "wcag_compliance": "AA",
                    "features": ["keyboard_navigation", "screen_reader", "aria_labels"]
                },
                timestamp=datetime.now().isoformat()
            ))
        
        # Check for performance optimizations
        if self._has_performance_optimizations(visual_result):
            signals.append(EmergenceSignal(
                type="performance_optimization",
                confidence=0.7,
                description="Generated highly optimized UI with excellent performance metrics",
                metadata={
                    "estimated_load_ms": visual_result.ir.performance_metrics.get("estimated_load_ms", 0),
                    "estimated_bundle_kb": visual_result.ir.performance_metrics.get("estimated_bundle_kb", 0)
                },
                timestamp=datetime.now().isoformat()
            ))
        
        # Record pattern for future novelty detection
        self._record_pattern(visual_result)
        
        return signals
    
    def _is_novel_visual_pattern(self, visual_result: Any) -> bool:
        """Check if the visual pattern is novel"""
        # Compare with historical patterns
        current_pattern = self._extract_pattern(visual_result)
        
        for historical_pattern in self.pattern_history:
            similarity = self._calculate_pattern_similarity(current_pattern, historical_pattern)
            if similarity > 0.8:  # Too similar to existing pattern
                return False
        
        return True
    
    def _calculate_novelty_score(self, visual_result: Any) -> float:
        """Calculate novelty score"""
        if not self.pattern_history:
            return 0.9  # First generation is highly novel
        
        current_pattern = self._extract_pattern(visual_result)
        
        # Calculate average dissimilarity from historical patterns
        dissimilarities = []
        for historical_pattern in self.pattern_history:
            similarity = self._calculate_pattern_similarity(current_pattern, historical_pattern)
            dissimilarities.append(1.0 - similarity)
        
        return sum(dissimilarities) / len(dissimilarities)
    
    def _extract_pattern(self, visual_result: Any) -> Dict[str, Any]:
        """Extract pattern signature from visual result"""
        return {
            "component_types": self._get_component_types(visual_result),
            "layout_type": self._get_layout_type(visual_result),
            "interaction_count": len(visual_result.ir.interaction_graph.edges),
            "style_count": len(visual_result.ir.style_graph.styles),
            "backends": visual_result.metadata.get("backends_compiled", [])
        }
    
    def _calculate_pattern_similarity(
        self,
        pattern1: Dict[str, Any],
        pattern2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two patterns"""
        # Simple similarity based on component types
        types1 = set(pattern1.get("component_types", []))
        types2 = set(pattern2.get("component_types", []))
        
        if not types1 or not types2:
            return 0.0
        
        intersection = len(types1 & types2)
        union = len(types1 | types2)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_component_types(self, visual_result: Any) -> List[str]:
        """Get list of component types in the result"""
        types = []
        for component in visual_result.ir.ui_components.values():
            types.append(component.type.value)
        return list(set(types))
    
    def _get_layout_type(self, visual_result: Any) -> str:
        """Get layout type"""
        if visual_result.ir.layout_graph:
            return visual_result.ir.layout_graph.layout_type.value
        return "unknown"
    
    def _is_cross_domain_synthesis(self, visual_result: Any) -> bool:
        """Check if result synthesizes multiple domains"""
        # Check if compiled to multiple backends
        backends = visual_result.metadata.get("backends_compiled", [])
        return len(backends) >= 3
    
    def _calculate_optimization_improvement(self, visual_result: Any) -> float:
        """Calculate optimization improvement ratio"""
        improvements = visual_result.optimization.improvements
        
        # Calculate improvement ratio
        components_optimized = improvements.get("components_optimized", 0)
        styles_optimized = improvements.get("styles_optimized", 0)
        rules_applied = improvements.get("rules_applied", 0)
        
        total_optimizations = components_optimized + styles_optimized + rules_applied
        
        # Normalize to 0-1 range
        if total_optimizations > 0:
            return min(total_optimizations / 10.0, 1.0)
        
        return 0.0
    
    def _has_accessibility_innovations(self, visual_result: Any) -> bool:
        """Check for accessibility innovations"""
        # Check if accessibility requirements are met
        accessibility_report = visual_result.ir.accessibility_report
        
        if not accessibility_report:
            return False
        
        # Check for comprehensive accessibility
        return (
            accessibility_report.get("wcag_level") == "AA" and
            accessibility_report.get("keyboard_navigable") and
            accessibility_report.get("screen_reader_compatible")
        )
    
    def _has_performance_optimizations(self, visual_result: Any) -> bool:
        """Check for performance optimizations"""
        perf_metrics = visual_result.ir.performance_metrics
        
        if not perf_metrics:
            return False
        
        budget = perf_metrics.get("budget", {})
        estimated_load = perf_metrics.get("estimated_load_ms", 999)
        estimated_size = perf_metrics.get("estimated_bundle_kb", 999)
        
        # Check if within budget
        max_load = budget.get("max_load_time_ms", 100)
        max_size = budget.get("max_bundle_size_kb", 50)
        
        return estimated_load <= max_load and estimated_size <= max_size
    
    def _record_pattern(self, visual_result: Any) -> None:
        """Record pattern for future novelty detection"""
        pattern = self._extract_pattern(visual_result)
        pattern["timestamp"] = datetime.now().isoformat()
        
        self.pattern_history.append(pattern)
        
        # Keep only recent patterns (last 100)
        if len(self.pattern_history) > 100:
            self.pattern_history = self.pattern_history[-100:]
    
    def save_pattern_history(self, filepath: Path) -> None:
        """Save pattern history to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.pattern_history, f, indent=2)
    
    def load_pattern_history(self, filepath: Path) -> None:
        """Load pattern history from file"""
        filepath = Path(filepath)
        
        if filepath.exists():
            with open(filepath, 'r') as f:
                self.pattern_history = json.load(f)


if __name__ == "__main__":
    # Example usage
    detector = VisualEmergenceDetector()
    
    print("Visual Emergence Detector initialized")
    print(f"Novelty threshold: {detector.novelty_threshold}")
    print(f"Optimization threshold: {detector.optimization_threshold}")
    print(f"Pattern history size: {len(detector.pattern_history)}")

