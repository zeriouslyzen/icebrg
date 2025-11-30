"""
LLM-Enhanced Bottleneck Detector
Bridge between rule-based detector and LLM analysis with caching
"""

import logging
import time
from typing import Dict, Any, Optional
from .bottleneck_detector import BottleneckAlert, Severity
from .analysis_cache import AnalysisCache

logger = logging.getLogger(__name__)


class LLMEnhancedDetector:
    """Bridge between rule-based detector and LLM analysis with caching."""
    
    def __init__(self, cfg=None, cache_dir=None):
        """
        Initialize LLM-enhanced detector.
        
        Args:
            cfg: ICEBURG configuration
            cache_dir: Cache directory for analyses
        """
        self.cfg = cfg
        self.cache = AnalysisCache(cache_dir=cache_dir)
        self.self_redesign_engine = None
        self._load_self_redesign_engine()
    
    def _load_self_redesign_engine(self):
        """Load SelfRedesignEngine for LLM analysis."""
        try:
            from ..protocol.execution.agents.self_redesign_engine import run as self_redesign_run
            from ..config import load_config
            
            if self.cfg is None:
                self.cfg = load_config()
            
            # Store the run function
            self.self_redesign_run = self_redesign_run
            logger.info("Loaded SelfRedesignEngine for LLM analysis")
        except Exception as e:
            logger.warning(f"Failed to load SelfRedesignEngine: {e}")
            self.self_redesign_run = None
    
    async def analyze_bottleneck_with_llm(
        self,
        alert: BottleneckAlert,
        system_metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze bottleneck using LLM (only for HIGH/CRITICAL severity).
        
        Args:
            alert: Bottleneck alert
            system_metrics: Current system metrics
            
        Returns:
            LLM analysis result or None if not applicable/cached
        """
        # Only analyze HIGH/CRITICAL severity bottlenecks
        if alert.severity not in [Severity.HIGH, Severity.CRITICAL]:
            logger.debug(f"Skipping LLM analysis for {alert.severity.value} severity alert")
            return None
        
        # Convert alert to dict for caching
        alert_dict = {
            "alert_id": alert.alert_id,
            "bottleneck_type": alert.bottleneck_type.value,
            "severity": alert.severity.value,
            "threshold": alert.threshold,
            "current_value": alert.current_value,
            "description": alert.description
        }
        
        # Check cache first
        cached_analysis = self.cache.get(alert_dict)
        if cached_analysis:
            logger.info(f"Using cached LLM analysis for alert {alert.alert_id}")
            return cached_analysis
        
        # No cache hit - perform LLM analysis
        if self.self_redesign_run is None:
            logger.warning("SelfRedesignEngine not available, skipping LLM analysis")
            return None
        
        try:
            # Build context for LLM analysis
            context = {
                "alert": alert_dict,
                "system_metrics": system_metrics or {}
            }
            
            # Create query for LLM analysis
            query = (
                f"Analyze bottleneck: {alert.description}\n"
                f"Type: {alert.bottleneck_type.value}\n"
                f"Severity: {alert.severity.value}\n"
                f"Current value: {alert.current_value:.2f} (threshold: {alert.threshold:.2f})"
            )
            
            # Run LLM analysis
            logger.info(f"Running LLM analysis for alert {alert.alert_id}")
            analysis_result = self.self_redesign_run(
                self.cfg,
                query,
                context=context,
                verbose=False
            )
            
            # Parse analysis result
            analysis = {
                "root_cause": self._extract_root_cause(analysis_result),
                "recommendations": self._extract_recommendations(analysis_result),
                "confidence": self._extract_confidence(analysis_result),
                "full_analysis": analysis_result,
                "timestamp": time.time()
            }
            
            # Cache the analysis
            self.cache.set(alert_dict, analysis)
            
            logger.info(f"LLM analysis completed for alert {alert.alert_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis failed for alert {alert.alert_id}: {e}")
            return None
    
    def _extract_root_cause(self, analysis_text: str) -> str:
        """Extract root cause from LLM analysis."""
        # Simple extraction - look for "root cause" or "cause" keywords
        lines = analysis_text.split('\n')
        for i, line in enumerate(lines):
            if 'root cause' in line.lower() or 'cause:' in line.lower():
                # Return next few lines as root cause
                return '\n'.join(lines[i:min(i+3, len(lines))])
        return "Root cause analysis not found in LLM response"
    
    def _extract_recommendations(self, analysis_text: str) -> list:
        """Extract recommendations from LLM analysis."""
        recommendations = []
        lines = analysis_text.split('\n')
        
        # Look for numbered or bulleted recommendations
        for line in lines:
            line = line.strip()
            if line.startswith(('1.', '2.', '3.', '-', '*', '•')):
                recommendations.append(line)
            elif 'recommendation' in line.lower() and len(line) > 20:
                recommendations.append(line)
        
        return recommendations[:5]  # Limit to top 5
    
    def _extract_confidence(self, analysis_text: str) -> float:
        """Extract confidence level from LLM analysis."""
        # Look for confidence keywords
        text_lower = analysis_text.lower()
        if 'high confidence' in text_lower or 'very confident' in text_lower:
            return 0.9
        elif 'medium confidence' in text_lower or 'moderate confidence' in text_lower:
            return 0.7
        elif 'low confidence' in text_lower or 'uncertain' in text_lower:
            return 0.5
        else:
            return 0.7  # Default medium confidence
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.get_stats()

